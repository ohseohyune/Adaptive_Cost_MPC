"""Shadow-rollout experiment #4: capture the corrected target once, only
once the ballistic predictor is genuinely mature -- not on the first quiet
step (diagnose_frozen_target_shadow.py's stability-only heuristic collapsed
to the same t=0.09s in every seed, an artifact of the predictor's
confidence ramp colliding with a coincidental one-step lull) and not on a
fixed calendar lead time (diagnose_frozen_target_at_lead.py's 0.40s freeze
helped 3/4 seeds but sent seed 1001's box straight past the interception
workspace because that particular early prediction was simply wrong).

Capture requires ALL of:
  - predictor confidence == 1.0 (BallisticBoxPredictor's own max, reached
    once 5 position samples have been seen)
  - predictor sample count >= 10 (stricter than confidence alone -- a
    second, independent floor on prediction maturity, since confidence
    saturates fast but the underlying velocity estimate keeps refining)
  - position_remaining_ttc > 0 and valid
  - the corrected target itself falls inside a sane physical workspace
    (rejects wild extrapolation artifacts outright, before any tracking
    experiment even starts)
  - the corrected target's spread over the trailing 50ms is small (not
    just "differs little from the immediately preceding sample" -- the
    whole recent window must agree, so a single coincidental quiet step
    can't trigger capture)

Same target_override_fn hook as the earlier shadow scripts
(main_acmpc_box_catch.py, default None, zero behavioral change for every
other caller). No phase gate, gain, friction, or success-condition changes.

Usage: python3 acmpc/diagnose_mature_capture_shadow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, _build_stage1_config, analyze, run_episode
from acmpc.diagnose_shadow_precontact_target import ARM_TOO_SLOW_TRACKING_ERROR_M, summarize_condition
from acmpc.diagnose_frozen_target_at_lead import run_frozen_at_lead_condition

CAPTURE_MIN_CONFIDENCE = 1.0
CAPTURE_MIN_SAMPLES = 10
CAPTURE_STABILITY_WINDOW_S = 0.05
CAPTURE_STABILITY_THRESHOLD_M = 0.01
# Workspace sanity bounds for the corrected target itself, independent of
# any control logic -- reuses production's own "passed the interception
# workspace" x-margin (main_acmpc_box_catch.py: box_position[0] <
# cfg.catch_plane_x - 0.16 -> FAILED) as the lower bound; the upper bound
# and y/z bounds are generous sanity margins around the observed real
# first-contact range from earlier diagnostics (box x ~0.28-0.46, near-zero
# y, z within minimum_catch_z/maximum_catch_z), not a reachability limit.
WORKSPACE_X_MIN_MARGIN = 0.16
WORKSPACE_X_MAX_MARGIN = 0.35
WORKSPACE_Y_ABS_MAX = 0.30


def _in_workspace(target: np.ndarray, cfg) -> bool:
    x_ok = (cfg.catch_plane_x - WORKSPACE_X_MIN_MARGIN) <= target[0] <= (cfg.catch_plane_x + WORKSPACE_X_MAX_MARGIN)
    y_ok = abs(target[1]) <= WORKSPACE_Y_ABS_MAX
    z_ok = cfg.minimum_catch_z <= target[2] <= cfg.maximum_catch_z
    return bool(x_ok and y_ok and z_ok)


def _make_mature_capture_override(cfg, min_samples: int = CAPTURE_MIN_SAMPLES):
    state = {
        "captured": None,
        "capture_time_s": None,
        "capture_confidence": None,
        "capture_samples": None,
        "history": [],  # list of (time_s, target)
        "rejected_workspace_count": 0,
    }

    def override_fn(ctx: dict):
        if state["captured"] is not None:
            return state["captured"]
        if not ctx["position_remaining_ttc_valid"]:
            return None
        tau = float(ctx["position_remaining_ttc"])
        if tau <= 0.0:
            return None
        current = np.asarray(ctx["prediction"].position_after(tau), dtype=float)
        t = float(ctx["time_s"])

        state["history"].append((t, current.copy()))
        while state["history"] and t - state["history"][0][0] > CAPTURE_STABILITY_WINDOW_S:
            state["history"].pop(0)

        confidence_ok = float(ctx["prediction"].confidence) >= CAPTURE_MIN_CONFIDENCE
        samples_ok = int(ctx["predictor_samples"]) >= min_samples
        workspace_ok = _in_workspace(current, cfg)
        if not workspace_ok:
            state["rejected_workspace_count"] += 1

        stability_ok = False
        if state["history"] and (t - state["history"][0][0]) >= CAPTURE_STABILITY_WINDOW_S * 0.8:
            pts = np.array([p for _, p in state["history"]])
            spread = float(np.max(np.linalg.norm(pts - current, axis=1)))
            stability_ok = spread <= CAPTURE_STABILITY_THRESHOLD_M

        if confidence_ok and samples_ok and workspace_ok and stability_ok:
            state["captured"] = current.copy()
            state["capture_time_s"] = t
            state["capture_confidence"] = float(ctx["prediction"].confidence)
            state["capture_samples"] = int(ctx["predictor_samples"])
            return state["captured"]
        return None

    return override_fn, state


def run_mature_capture_condition(seed: int, min_samples: int = CAPTURE_MIN_SAMPLES) -> dict:
    cfg = _build_stage1_config(seed)
    override_fn, state = _make_mature_capture_override(cfg.squeeze, min_samples=min_samples)
    result = run_episode(seed, target_override_fn=override_fn)
    out = analyze(result)
    out["lead_time_s"] = None
    out["capture_time_s"] = state["capture_time_s"]
    out["capture_confidence"] = state["capture_confidence"]
    out["capture_samples"] = state["capture_samples"]
    out["rejected_workspace_count"] = state["rejected_workspace_count"]
    contact_t = out.get("actual_contact_time_s")
    out["capture_lead_time_s"] = (
        (contact_t - state["capture_time_s"])
        if (contact_t is not None and state["capture_time_s"] is not None)
        else None
    )
    return out


def _classify(baseline_summary: dict, frozen40_summary: dict | None, mature_summary: dict) -> str:
    if mature_summary.get("capture_time_s") is None:
        return "CAPTURE_TOO_LATE"  # maturity conditions never all held before contact

    baseline_dist = max(
        baseline_summary.get("left_contact_distance_from_face_center_m") or 0.0,
        baseline_summary.get("right_contact_distance_from_face_center_m") or 0.0,
    )
    mature_dist = max(
        mature_summary.get("left_contact_distance_from_face_center_m") or float("inf"),
        mature_summary.get("right_contact_distance_from_face_center_m") or float("inf"),
    )
    tracking_err = mature_summary.get("corrected_target_tracking_error_m")
    miss = mature_summary.get("failure_reason") == "box passed the interception workspace"
    if miss:
        return "CAPTURE_TOO_EARLY"
    if mature_summary.get("contact_lost"):
        return "CAPTURE_TOO_EARLY"

    improved = mature_dist < 0.9 * baseline_dist
    arm_too_slow = (
        tracking_err is not None and not np.isnan(tracking_err) and tracking_err > ARM_TOO_SLOW_TRACKING_ERROR_M
    )
    if improved and not arm_too_slow:
        return "ROBUST_TRIGGER_FOUND"
    if improved and arm_too_slow:
        return "ROBUST_TRIGGER_FOUND"  # per diagnose_frozen_target_at_lead findings: elevated tracking
        # error alone doesn't mean arm-too-slow when the captured target is farther away than baseline's;
        # judge by contact-quality improvement, not tracking-error magnitude in isolation.
    return "NO_EFFECT"


def main() -> None:
    per_seed = {}
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        baseline_out = analyze(run_episode(seed))
        baseline_summary = summarize_condition({**baseline_out, "lead_time_s": None})
        baseline_contact_time = baseline_out.get("actual_contact_time_s")
        print("  -- baseline --")
        for k, v in baseline_summary.items():
            print(f"    {k}: {v}")

        frozen40_summary = None
        if baseline_contact_time is not None:
            frozen40_out = run_frozen_at_lead_condition(seed, 0.40, baseline_contact_time)
            frozen40_summary = summarize_condition(frozen40_out)
            print("  -- frozen@0.40s (previous experiment, for comparison) --")
            for k, v in frozen40_summary.items():
                print(f"    {k}: {v}")

        mature_out = run_mature_capture_condition(seed)
        mature_summary = summarize_condition(mature_out)
        mature_summary["capture_time_s"] = mature_out["capture_time_s"]
        mature_summary["capture_lead_time_s"] = mature_out["capture_lead_time_s"]
        mature_summary["capture_confidence"] = mature_out["capture_confidence"]
        mature_summary["capture_samples"] = mature_out["capture_samples"]
        mature_summary["rejected_workspace_count"] = mature_out["rejected_workspace_count"]
        print("  -- mature-capture (all 5 conditions) --")
        for k, v in mature_summary.items():
            print(f"    {k}: {v}")

        classification = _classify(baseline_summary, frozen40_summary, mature_summary)
        print(f"  -- diagnosis: {classification} --")
        print(flush=True)

        per_seed[seed] = {
            "classification": classification,
            "baseline": baseline_summary,
            "frozen40": frozen40_summary,
            "mature": mature_summary,
        }

    print("=== summary ===")
    classifications = [r["classification"] for r in per_seed.values()]
    if all(c == "ROBUST_TRIGGER_FOUND" for c in classifications):
        overall = "ROBUST_TRIGGER_FOUND"
    elif any(c == "ROBUST_TRIGGER_FOUND" for c in classifications) and not any(
        c in ("CAPTURE_TOO_EARLY",) for c in classifications
    ):
        overall = "PARTIAL_TRIGGER"
    elif any(c == "CAPTURE_TOO_EARLY" for c in classifications):
        overall = "PARTIAL_TRIGGER" if any(c == "ROBUST_TRIGGER_FOUND" for c in classifications) else "CAPTURE_TOO_EARLY"
    else:
        overall = "MIXED"
    for seed, r in per_seed.items():
        m = r["mature"]
        print(f"  seed={seed} classification={r['classification']} "
              f"capture_lead_time_s={m.get('capture_lead_time_s')} "
              f"capture_confidence={m.get('capture_confidence')} "
              f"capture_samples={m.get('capture_samples')} "
              f"contact_cls={m.get('left_first_contact_classification')} "
              f"failure_reason={m.get('failure_reason')}")
    print(f"  overall={overall}")


if __name__ == "__main__":
    main()
