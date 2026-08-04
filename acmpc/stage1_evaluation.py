"""Stage 1 (near_low_speed_catch) multi-seed success-rate/safety evaluation.

Pure evaluation -- does NOT touch controller/cost/threshold/geometry, and does
NOT narrow Stage 1's domain range. Each seed independently draws its own
mass/friction/launch_velocity/target_flight_time via the stage's own
`sample_stage_domain` (unchanged), then runs a fresh `run_box_catch` episode
(use_launch_fixture=False, online_learning=False -- no PPO update/rollout-
buffer persistence). Unlike Stage 0, Stage 1 keeps the real (force AND linear
AND angular) success condition -- this script must not, and does not, apply
Stage 0's diagnostic-only angular treatment.

`_stable_hold_reset_reason` (main_acmpc_box_catch.py) no longer reports
"angular" as a reset cause (removed when Stage 0's success condition changed).
For Stage 1 we do NOT touch that function -- instead we read
`stable_angular_velocity_condition`/`angular_speed_exceeded_diagnostic`
directly off each HOLD3 trace row to attribute resets to angular ourselves,
without altering shared production logic.

Usage:
    python3 acmpc/stage1_evaluation.py --checkpoint 20
    python3 acmpc/stage1_evaluation.py --checkpoint 50
    python3 acmpc/stage1_evaluation.py --checkpoint 100
    python3 acmpc/stage1_evaluation.py --viewer-seed 1001
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from acmpc.main_acmpc_box_catch import AcmpcBoxCatchConfig, run_box_catch
from acmpc.stage0_evaluation import wilson_bounds
from control.squeeze.config import DynamicSideSqueezeConfig
from control.squeeze.progressive_curriculum import (
    progressive_catch_curriculum,
    sample_stage_domain,
)

STAGE_NAME = "near_low_speed_catch"
STAGE_INDEX = 1
CACHE_PATH = ROOT / "sweep_results" / "stage1_evaluation_cache.json"

_HOLD3_RE = re.compile(
    r"HOLD3 t=(?P<t>[\d.]+) phase=(?P<phase>\S+) "
    r"left_contact_active=(?P<left_active>True|False) right_contact_active=(?P<right_active>True|False) "
    r"left_normal_force_n=(?P<left_force>[\d.]+) right_normal_force_n=(?P<right_force>[\d.]+) "
    r"required_grip_force_n=(?P<required_force>[\d.]+) .*?"
    r"box_linear_speed_n=(?P<lin_speed>[\d.]+) .*?box_angular_speed_n=(?P<ang_speed>[\d.]+) "
    r"stable_contact_condition=(?P<contact_ok>\S+) stable_force_condition=(?P<force_ok>\S+) "
    r"stable_velocity_condition=(?P<lin_ok>\S+) stable_angular_velocity_condition=(?P<ang_ok>\S+) "
    r"stable_hold_condition=(?P<stable_ok>\S+) stable_hold_duration_s=(?P<stable_dur>[\d.]+) "
    r"stable_hold_reset_reason='(?P<reset_reason>[^']*)'"
)
_PHASE_RE = re.compile(
    r"\[Stage 0\] PHASE (?P<from>\S+) -> (?P<to>\S+) reason=(?P<reason>\S+) at t=(?P<t>[\d.]+)s"
)

FAILURE_CATEGORIES = (
    "PHASE_TRANSITION_FAILURE",
    "INTERCEPT_TRACKING_FAILURE",
    "CONTACT_LOSS",
    "INSUFFICIENT_GRIP_FORCE",
    "LINEAR_SPEED_UNSTABLE",
    "ANGULAR_SPEED_UNSTABLE",
    "EMERGENCY_FORCE",
    "TIMEOUT",
    "OTHER",
)


@dataclass
class SeedResult:
    seed: int
    mass_kg: float
    friction: float
    launch_velocity_mps: tuple
    target_flight_time_s: float
    success: bool
    failure_reason: str
    pre_impact_entered: bool
    capture_entered: bool
    hold_entered: bool
    first_contact_time_s: Optional[float]
    hold_entry_time_s: Optional[float]
    stable_hold_duration_s: float
    max_continuous_stable_s: float
    first_contact_peak_force_n: float
    episode_peak_force_n: float
    emergency_occurred: bool
    contact_lost: bool
    grip_force_violation_count: int
    linear_violation_count: int
    angular_violation_count: int
    angular_caused_reset_count: int
    counterfactual_max_stable_excluding_angular_s: float
    hold_row_count: int
    primary_failure_category: str
    secondary_failure_categories: tuple = field(default_factory=tuple)


def _run_one(seed: int, *, viewer: bool = False):
    os.environ["STAGE0_DEBUG_TRACE"] = "1"
    rng = np.random.default_rng(seed)
    stage = progressive_catch_curriculum()[STAGE_INDEX]
    assert stage.name == STAGE_NAME
    base_squeeze = DynamicSideSqueezeConfig(random_seed=seed)
    resolved = sample_stage_domain(stage, rng, STAGE_INDEX, base_squeeze)
    cfg = AcmpcBoxCatchConfig(
        seed=seed,
        device="cpu",
        viewer=viewer,
        online_learning=False,
        use_launch_fixture=False,
        domain_parameters=resolved.domain,
        squeeze=resolved.squeeze_config,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary = run_box_catch(cfg)
    return cfg, resolved, summary, buf.getvalue()


def _classify_failure(cfg, summary, hold3_rows, phase_rows, contact_lost: bool) -> tuple:
    reason = summary.failure_reason.lower()
    categories = []

    if "emergency" in reason:
        categories.append("EMERGENCY_FORCE")
    if not phase_rows and not summary.success:
        categories.append("PHASE_TRANSITION_FAILURE")
    elif summary.intercept_dwell_s > 0 and summary.pre_impact_dwell_s == 0:
        categories.append("INTERCEPT_TRACKING_FAILURE")
    if summary.first_contact_peak_force_n > cfg.squeeze.first_contact_force_limit:
        categories.append("INSUFFICIENT_GRIP_FORCE")
    if contact_lost:
        categories.append("CONTACT_LOSS")
    if hold3_rows:
        grip_violations = sum(
            1 for r in hold3_rows
            if r["left_active"] and r["left_force"] < r["required_force"]
            or r["right_active"] and r["right_force"] < r["required_force"]
        )
        if grip_violations / len(hold3_rows) > 0.3:
            categories.append("INSUFFICIENT_GRIP_FORCE")
        linear_violations = sum(1 for r in hold3_rows if not r["lin_ok"])
        if linear_violations / len(hold3_rows) > 0.3:
            categories.append("LINEAR_SPEED_UNSTABLE")
        angular_violations = sum(1 for r in hold3_rows if not r["ang_ok"])
        if angular_violations / len(hold3_rows) > 0.3:
            categories.append("ANGULAR_SPEED_UNSTABLE")
    if not categories and reason == "timeout":
        categories.append("TIMEOUT")
    if not categories:
        categories.append("OTHER")

    return categories[0], tuple(categories[1:])


def _analyze(seed: int, cfg, resolved, summary, trace_text: str) -> SeedResult:
    hold3_rows = []
    for line in trace_text.splitlines():
        m = _HOLD3_RE.match(line)
        if m:
            hold3_rows.append(
                {
                    "t": float(m.group("t")),
                    "phase": m.group("phase"),
                    "left_active": m.group("left_active") == "True",
                    "right_active": m.group("right_active") == "True",
                    "left_force": float(m.group("left_force")),
                    "right_force": float(m.group("right_force")),
                    "required_force": float(m.group("required_force")),
                    "lin_speed": float(m.group("lin_speed")),
                    "ang_speed": float(m.group("ang_speed")),
                    "contact_ok": m.group("contact_ok") == "True",
                    "force_ok": m.group("force_ok") == "True",
                    "lin_ok": m.group("lin_ok") == "True",
                    "ang_ok": m.group("ang_ok") == "True",
                    "stable_ok": m.group("stable_ok") == "True",
                    "stable_dur": float(m.group("stable_dur")),
                    "reset_reason": m.group("reset_reason"),
                }
            )
    phase_rows = [m.groupdict() for m in _PHASE_RE.finditer(trace_text)]
    # HOLD3 fires every control step regardless of phase (the trace gate was
    # generalized for Stage 1 use) -- restrict all HOLD-window statistics to
    # phase=="hold" rows explicitly, not the raw hold3_rows list, or violation
    # counts can exceed the row count they're supposedly a fraction of.
    hold_rows = [r for r in hold3_rows if r["phase"] == "hold"]

    contact_lost = any(not (r["left_active"] and r["right_active"]) for r in hold_rows)
    episode_peak = max((max(r["left_force"], r["right_force"]) for r in hold3_rows), default=0.0)
    grip_violation_count = sum(
        1 for r in hold_rows
        if r["left_active"] and r["left_force"] < r["required_force"]
        or r["right_active"] and r["right_force"] < r["required_force"]
    )
    linear_violation_count = sum(1 for r in hold_rows if not r["lin_ok"])
    angular_violation_count = sum(1 for r in hold_rows if not r["ang_ok"])

    # Reset attribution: `_stable_hold_reset_reason` no longer names angular
    # directly (removed for Stage 0), so we attribute a reset to angular
    # ourselves whenever the timer visibly dropped (stable_dur decreased
    # step-to-step) while angular was the only failing condition among
    # contact/force/linear/angular at that row.
    angular_caused_reset_count = 0
    prev_dur = None
    for r in hold_rows:
        if prev_dur is not None and r["stable_dur"] < prev_dur:
            if not r["ang_ok"] and r["contact_ok"] and r["force_ok"] and r["lin_ok"]:
                angular_caused_reset_count += 1
        prev_dur = r["stable_dur"]

    # Counterfactual (analysis-only): longest continuous run of hold rows
    # satisfying contact+force+linear, ignoring angular entirely. Does NOT
    # feed back into success/failure_reason/primary_failure_category.
    counterfactual_max = 0.0
    run_start = None
    for r in hold_rows:
        ok_without_angular = r["contact_ok"] and r["force_ok"] and r["lin_ok"]
        if ok_without_angular:
            if run_start is None:
                run_start = r["t"]
            counterfactual_max = max(counterfactual_max, r["t"] - run_start)
        else:
            run_start = None

    primary, secondary = _classify_failure(cfg, summary, hold3_rows, phase_rows, contact_lost)

    return SeedResult(
        seed=seed,
        mass_kg=resolved.domain.mass,
        friction=resolved.domain.friction,
        launch_velocity_mps=tuple(float(v) for v in resolved.squeeze_config.launch_velocity_low),
        target_flight_time_s=resolved.squeeze_config.target_flight_time_s,
        success=summary.success,
        failure_reason=summary.failure_reason,
        pre_impact_entered=summary.pre_impact_dwell_s > 0,
        capture_entered=summary.capture_dwell_s > 0,
        hold_entered=summary.hold_dwell_s > 0,
        first_contact_time_s=summary.first_contact_time_s,
        hold_entry_time_s=hold_rows[0]["t"] if hold_rows else None,
        stable_hold_duration_s=summary.hold_time_s,
        max_continuous_stable_s=summary.strict_hold_time_s,
        first_contact_peak_force_n=summary.first_contact_peak_force_n,
        episode_peak_force_n=episode_peak,
        emergency_occurred="emergency" in summary.failure_reason.lower(),
        contact_lost=contact_lost,
        grip_force_violation_count=grip_violation_count,
        linear_violation_count=linear_violation_count,
        angular_violation_count=angular_violation_count,
        angular_caused_reset_count=angular_caused_reset_count,
        counterfactual_max_stable_excluding_angular_s=counterfactual_max,
        hold_row_count=len(hold_rows),
        primary_failure_category="" if summary.success else primary,
        secondary_failure_categories=() if summary.success else secondary,
    )


def evaluate(seeds: list) -> list:
    results = []
    for seed in seeds:
        cfg, resolved, summary, trace_text = _run_one(seed)
        results.append(_analyze(seed, cfg, resolved, summary, trace_text))
    return results


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def evaluate_with_cache(seeds: list) -> list:
    """Reuse already-evaluated seeds from a prior (smaller) checkpoint run
    instead of re-running them -- same pattern as stage0_evaluation's
    escalating checkpoints must not duplicate seeds."""

    cache = _load_cache()
    results = []
    dirty = False
    for seed in seeds:
        key = str(seed)
        if key in cache:
            results.append(SeedResult(**cache[key]))
            continue
        cfg, resolved, summary, trace_text = _run_one(seed)
        r = _analyze(seed, cfg, resolved, summary, trace_text)
        results.append(r)
        cache[key] = r.__dict__
        dirty = True
    if dirty:
        _save_cache(cache)
    return results


def print_seed_table(results: list) -> None:
    print(
        f"{'seed':>5} {'mass':>5} {'fric':>5} {'vx':>6} {'tflight':>7} "
        f"{'success':>7} {'hold_ent':>8} {'max_stable_s':>12} {'first_peak_N':>12} "
        f"{'ep_peak_N':>9} {'lost':>5} {'grip_v':>6} {'lin_v':>5} {'ang_v':>5} "
        f"{'ang_reset':>9} {'cf_stable_s':>11} {'emerg':>6}  failure"
    )
    for r in results:
        print(
            f"{r.seed:5d} {r.mass_kg:5.2f} {r.friction:5.2f} {r.launch_velocity_mps[0]:6.2f} "
            f"{r.target_flight_time_s:7.2f} {str(r.success):>7} {str(r.hold_entered):>8} "
            f"{r.max_continuous_stable_s:12.3f} {r.first_contact_peak_force_n:12.2f} "
            f"{r.episode_peak_force_n:9.2f} {str(r.contact_lost):>5} {r.grip_force_violation_count:6d} "
            f"{r.linear_violation_count:5d} {r.angular_violation_count:5d} "
            f"{r.angular_caused_reset_count:9d} {r.counterfactual_max_stable_excluding_angular_s:11.3f} "
            f"{str(r.emergency_occurred):>6}  {r.primary_failure_category or ''} {r.failure_reason}"
        )


def print_checkpoint_summary(checkpoint: int, results: list) -> dict:
    successes = sum(1 for r in results if r.success)
    trials = len(results)
    observed_rate = successes / trials
    lower, upper = wilson_bounds(successes, trials, confidence=0.95)
    emergency_count = sum(1 for r in results if r.emergency_occurred)
    emergency_seeds = [r.seed for r in results if r.emergency_occurred]

    failure_counts = Counter(r.primary_failure_category for r in results if not r.success)
    dominant = failure_counts.most_common(1)[0][0] if failure_counts else "NONE"

    total_angular_resets = sum(r.angular_caused_reset_count for r in results)
    total_hold_rows = sum(r.hold_row_count for r in results)
    angular_violation_rows = sum(r.angular_violation_count for r in results)
    angular_dominant = dominant == "ANGULAR_SPEED_UNSTABLE" or (
        total_hold_rows > 0 and angular_violation_rows / total_hold_rows > 0.3
    )

    if emergency_count > 0:
        decision = "NOT_READY"
    elif observed_rate >= 0.80 and lower >= 0.75:
        decision = "PROMOTION_READY"
    elif upper < 0.75:
        decision = "NOT_READY"
    else:
        decision = "INCONCLUSIVE"

    print(f"\ncheckpoint={checkpoint}")
    print(f"successes={successes}")
    print(f"trials={trials}")
    print(f"observed_rate={observed_rate:.3f}")
    print(f"wilson_lower={lower:.3f}")
    print(f"wilson_upper={upper:.3f}")
    print(f"emergency_count={emergency_count}  seeds={emergency_seeds}")
    print(f"dominant_failure_reason={dominant}")
    print(f"angular_caused_reset_total={total_angular_resets}  angular_violation_rows={angular_violation_rows}/{total_hold_rows}")
    print(f"angular_condition_dominant={'YES' if angular_dominant else 'NO'}")
    print(f"decision={decision}")

    return {
        "checkpoint": checkpoint,
        "successes": successes,
        "trials": trials,
        "observed_rate": observed_rate,
        "wilson_lower": lower,
        "wilson_upper": upper,
        "emergency_count": emergency_count,
        "emergency_seeds": emergency_seeds,
        "dominant_failure_reason": dominant,
        "angular_condition_dominant": angular_dominant,
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=int, default=20, choices=(20, 50, 100))
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument(
        "--viewer-seed",
        type=int,
        help="replay one Stage 1 seed in the MuJoCo viewer instead of running a sweep",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.viewer_seed is not None:
        cfg, resolved, summary, trace_text = _run_one(args.viewer_seed, viewer=True)
        print_seed_table([_analyze(args.viewer_seed, cfg, resolved, summary, trace_text)])
        return
    seeds = list(range(args.base_seed, args.base_seed + args.checkpoint))
    results = evaluate_with_cache(seeds)
    print_seed_table(results)
    print_checkpoint_summary(args.checkpoint, results)


if __name__ == "__main__":
    main()
