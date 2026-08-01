"""Stage 0 (static_grasp_bootstrap) multi-seed success-rate/safety evaluation.

Pure evaluation -- does not touch controller/cost/threshold/geometry. Reuses
the exact fixed config already validated at seed=7 (stage0_fixture_box_z_
offset_m=0.04 default, engineered priors, online_learning=False so no PPO
update/rollout-buffer persistence happens). Each seed is a fully independent
run_box_catch call (fresh MjModel/MjData every call, already the existing
behavior -- no explicit reset code needed).

Reuses the STAGE0_DEBUG_TRACE per-step text instrumentation (main_acmpc_
box_catch.py) the same way stage0_fixture_height_experiment.py does, rather
than adding new permanent BoxCatchSummary fields for a one-off evaluation.

Usage:
    python3 acmpc/stage0_evaluation.py --checkpoint 20
    python3 acmpc/stage0_evaluation.py --checkpoint 50
    python3 acmpc/stage0_evaluation.py --checkpoint 100
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.main_acmpc_box_catch import AcmpcBoxCatchConfig, run_box_catch

_HOLD3_RE = re.compile(
    r"HOLD3 t=(?P<t>[\d.]+) phase=(?P<phase>\S+) "
    r"left_contact_active=(?P<left_active>True|False) right_contact_active=(?P<right_active>True|False) "
    r"left_normal_force_n=(?P<left_force>[\d.]+) right_normal_force_n=(?P<right_force>[\d.]+) "
    r".*?box_linear_speed_n=(?P<lin_speed>[\d.]+) .*?box_angular_speed_n=(?P<ang_speed>[\d.]+)"
)

FAILURE_CATEGORIES = (
    "PHASE_TRANSITION_FAILURE",
    "FIXTURE_RELEASE_FAILURE",
    "CONTACT_LOSS",
    "INSUFFICIENT_GRIP_FORCE",
    "LINEAR_SPEED_UNSTABLE",
    "EMERGENCY_FORCE",
    "TIMEOUT",
    "OTHER",
)


@dataclass
class SeedResult:
    seed: int
    success: bool
    failure_reason: str
    pre_impact_entered: bool
    capture_entered: bool
    fixture_released: bool
    hold_entered: bool
    first_contact_time_s: Optional[float]
    fixture_release_time_s: Optional[float]
    hold_entry_time_s: Optional[float]
    stable_hold_duration_s: float
    max_continuous_stable_s: float
    first_contact_peak_force_n: float
    episode_peak_force_n: float
    emergency_occurred: bool
    post_release_bilateral_duration_s: float
    contact_lost_after_release: bool
    final_box_linear_speed_mps: float
    box_angular_speed_mean: float
    box_angular_speed_max: float
    primary_failure_category: str
    secondary_failure_categories: tuple = field(default_factory=tuple)


def _run_one(seed: int):
    os.environ["STAGE0_DEBUG_TRACE"] = "1"
    cfg = AcmpcBoxCatchConfig(
        seed=seed,
        device="cpu",
        online_learning=False,
        use_launch_fixture=True,
        release_fixture_on_bilateral_contact=True,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary = run_box_catch(cfg)
    return cfg, summary, buf.getvalue()


def _classify_failure(cfg, summary, hold3_rows) -> tuple[str, tuple]:
    reason = summary.failure_reason.lower()
    categories = []

    if "emergency" in reason:
        categories.append("EMERGENCY_FORCE")
    if not summary.fixture_released:
        if "fixture release force threshold exceeds" in reason or "fixture release timeout" in reason:
            categories.append("FIXTURE_RELEASE_FAILURE")
    if summary.intercept_dwell_s > 0 and summary.pre_impact_dwell_s == 0 and summary.capture_dwell_s == 0:
        categories.append("PHASE_TRANSITION_FAILURE")
    if summary.first_contact_peak_force_n > cfg.squeeze.first_contact_force_limit:
        categories.append("INSUFFICIENT_GRIP_FORCE")  # excessive first-contact impact, grouped with grip issues
    if summary.fixture_released and hold3_rows:
        post_release = [r for r in hold3_rows if r["t"] >= (summary.fixture_release_time_s or 0.0)]
        if post_release:
            lost = any(not (r["left_active"] and r["right_active"]) for r in post_release)
            if lost:
                categories.append("CONTACT_LOSS")
            linear_violations = sum(1 for r in post_release if r["lin_speed"] > cfg.strict_box_speed_max_mps)
            if linear_violations / len(post_release) > 0.3:
                categories.append("LINEAR_SPEED_UNSTABLE")
    if not categories and reason == "timeout":
        categories.append("TIMEOUT")
    if not categories:
        categories.append("OTHER")

    primary = categories[0]
    secondary = tuple(categories[1:])
    return primary, secondary


def _analyze(seed: int, cfg, summary, trace_text: str) -> SeedResult:
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
                    "lin_speed": float(m.group("lin_speed")),
                    "ang_speed": float(m.group("ang_speed")),
                }
            )

    release_time = summary.fixture_release_time_s
    post_release_rows = [r for r in hold3_rows if release_time is not None and r["t"] >= release_time]
    post_release_bilateral_duration = 0.0
    contact_lost = False
    if post_release_rows:
        start_t = post_release_rows[0]["t"]
        for r in post_release_rows:
            if r["left_active"] and r["right_active"]:
                post_release_bilateral_duration = r["t"] - start_t
            else:
                contact_lost = True
                break

    episode_peak = max((max(r["left_force"], r["right_force"]) for r in hold3_rows), default=0.0)
    ang_speeds = [r["ang_speed"] for r in hold3_rows] or [0.0]
    hold_rows = [r for r in hold3_rows if r["phase"] == "hold"]
    hold_entry_time = hold_rows[0]["t"] if hold_rows else None

    primary, secondary = _classify_failure(cfg, summary, hold3_rows)

    return SeedResult(
        seed=seed,
        success=summary.success,
        failure_reason=summary.failure_reason,
        pre_impact_entered=summary.pre_impact_dwell_s > 0,
        capture_entered=summary.capture_dwell_s > 0,
        fixture_released=summary.fixture_released,
        hold_entered=summary.hold_dwell_s > 0,
        first_contact_time_s=summary.first_contact_time_s,
        fixture_release_time_s=release_time,
        hold_entry_time_s=hold_entry_time,
        stable_hold_duration_s=summary.hold_time_s,
        max_continuous_stable_s=summary.strict_hold_time_s,
        first_contact_peak_force_n=summary.first_contact_peak_force_n,
        episode_peak_force_n=episode_peak,
        emergency_occurred="emergency" in summary.failure_reason.lower(),
        post_release_bilateral_duration_s=post_release_bilateral_duration,
        contact_lost_after_release=contact_lost,
        final_box_linear_speed_mps=summary.final_box_speed_mps,
        box_angular_speed_mean=sum(ang_speeds) / len(ang_speeds),
        box_angular_speed_max=max(ang_speeds),
        primary_failure_category="" if summary.success else primary,
        secondary_failure_categories=() if summary.success else secondary,
    )


def wilson_bounds(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """One-sided (z computed from `confidence` as a one-sided quantile)
    Wilson score interval, returned as (lower, upper) using the same z for
    both edges of the standard two-sided-form Wilson formula."""

    if trials == 0:
        return 0.0, 1.0
    p_hat = successes / trials
    # one-sided z (e.g. 0.95 -> 1.645), via inverse-normal-CDF rational
    # approximation (Acklam) -- no scipy dependency for one number.
    z = _inverse_normal_cdf(confidence)
    denom = 1.0 + z * z / trials
    center = p_hat + z * z / (2 * trials)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials))
    lower = max(0.0, (center - margin) / denom)
    upper = min(1.0, (center + margin) / denom)
    return lower, upper


def _inverse_normal_cdf(p: float) -> float:
    # Rational approximation (Peter Acklam), accurate to ~1e-9 -- avoids a
    # scipy dependency for a single one-sided z value (0.95 -> ~1.6449).
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def evaluate(seeds: list) -> list:
    results = []
    for seed in seeds:
        cfg, summary, trace_text = _run_one(seed)
        results.append(_analyze(seed, cfg, summary, trace_text))
    return results


def print_seed_table(results: list) -> None:
    print(
        f"{'seed':>4} {'success':>7} {'hold_ent':>8} {'max_stable_s':>12} "
        f"{'first_peak_N':>12} {'ep_peak_N':>9} {'post_rel_s':>10} {'lost':>5} {'emerg':>6}  failure_reason"
    )
    for r in results:
        print(
            f"{r.seed:4d} {str(r.success):>7} {str(r.hold_entered):>8} "
            f"{r.max_continuous_stable_s:12.3f} {r.first_contact_peak_force_n:12.2f} "
            f"{r.episode_peak_force_n:9.2f} {r.post_release_bilateral_duration_s:10.3f} "
            f"{str(r.contact_lost_after_release):>5} {str(r.emergency_occurred):>6}  "
            f"{r.primary_failure_category or ''} {r.failure_reason}"
        )


def print_checkpoint_summary(checkpoint: int, results: list) -> dict:
    successes = sum(1 for r in results if r.success)
    trials = len(results)
    observed_rate = successes / trials
    lower, upper = wilson_bounds(successes, trials, confidence=0.95)
    emergency_count = sum(1 for r in results if r.emergency_occurred)
    emergency_seeds = [r.seed for r in results if r.emergency_occurred]

    from collections import Counter
    failure_counts = Counter(r.primary_failure_category for r in results if not r.success)
    dominant = failure_counts.most_common(1)[0][0] if failure_counts else "NONE"

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
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=int, default=20, choices=(20, 50, 100))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = list(range(args.checkpoint))
    results = evaluate(seeds)
    print_seed_table(results)
    print_checkpoint_summary(args.checkpoint, results)


if __name__ == "__main__":
    main()
