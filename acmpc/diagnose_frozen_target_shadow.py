"""Shadow-rollout experiment #2: capture the corrected contact target once
(not every control step), the first time it has stabilized, and command that
single frozen point for the rest of the approach.

Motivation (per review of diagnose_shadow_precontact_target.py's result):
that experiment kept re-evaluating the corrected target every control step,
so a large tracking error could mean either "arm too slow" or "target kept
moving under it" -- those two causes were confounded. This experiment
removes the confound: if tracking error to a *frozen* target is still large,
that is real arm/impedance response limitation (ARM_RESPONSE_TOO_SLOW). If
it's small, the earlier result was really TARGET_UNSTABLE, not arm speed.

Stabilization is defined the moment BOTH hold for one control step to the
next:
  - predictor confidence >= CONFIDENCE_THRESHOLD (1.0 -- the predictor's own
    max, reached after 5 position samples, ~0.04-0.05s into flight)
  - the corrected target moved less than STEP_CHANGE_THRESHOLD_M between
    consecutive control steps (10ms apart)

Same target_override_fn hook as diagnose_shadow_precontact_target.py
(main_acmpc_box_catch.py, default None, zero behavioral change for every
other caller). No phase gate, gain, friction, or success-condition changes.

Usage: python3 acmpc/diagnose_frozen_target_shadow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, analyze, run_episode
from acmpc.diagnose_shadow_precontact_target import ARM_TOO_SLOW_TRACKING_ERROR_M, summarize_condition

CONFIDENCE_THRESHOLD = 1.0
STEP_CHANGE_THRESHOLD_M = 0.01  # 1cm/10ms = 1 m/s implied target speed -- see module docstring


def _make_frozen_override(confidence_threshold: float = CONFIDENCE_THRESHOLD,
                           step_change_threshold_m: float = STEP_CHANGE_THRESHOLD_M):
    state = {"captured": None, "prev_target": None, "capture_time_s": None, "n_pre_capture_steps": 0}

    def override_fn(ctx: dict):
        if state["captured"] is not None:
            return state["captured"]
        if not ctx["position_remaining_ttc_valid"]:
            return None
        tau = max(0.0, float(ctx["position_remaining_ttc"]))
        current = np.asarray(ctx["prediction"].position_after(tau), dtype=float)

        confidence_ok = float(ctx["prediction"].confidence) >= confidence_threshold
        step_ok = False
        if state["prev_target"] is not None:
            step_ok = float(np.linalg.norm(current - state["prev_target"])) <= step_change_threshold_m
        state["prev_target"] = current.copy()
        state["n_pre_capture_steps"] += 1

        if confidence_ok and step_ok:
            state["captured"] = current.copy()
            state["capture_time_s"] = float(ctx["time_s"])
            return state["captured"]
        return None  # not yet stable -- fall back to production's own target_center this step

    return override_fn, state


def _make_frozen_at_lead_override(lead_time_s: float, baseline_contact_time_s: float):
    """Capture once, at a *prescribed* lead time before contact -- not a
    stability heuristic. Isolates "can the arm track a genuinely static
    point given this much time" from any capture-timing confound.
    """
    state = {"captured": None, "capture_time_s": None}

    def override_fn(ctx: dict):
        if state["captured"] is not None:
            return state["captured"]
        if not ctx["position_remaining_ttc_valid"]:
            return None
        if ctx["time_s"] < baseline_contact_time_s - lead_time_s:
            return None
        tau = max(0.0, float(ctx["position_remaining_ttc"]))
        current = np.asarray(ctx["prediction"].position_after(tau), dtype=float)
        state["captured"] = current.copy()
        state["capture_time_s"] = float(ctx["time_s"])
        return state["captured"]

    return override_fn, state


def run_frozen_at_lead_condition(seed: int, lead_time_s: float, baseline_contact_time_s: float) -> dict:
    override_fn, state = _make_frozen_at_lead_override(lead_time_s, baseline_contact_time_s)
    result = run_episode(seed, target_override_fn=override_fn)
    out = analyze(result)
    out["lead_time_s"] = lead_time_s
    out["capture_time_s"] = state["capture_time_s"]
    contact_t = out.get("actual_contact_time_s")
    out["capture_lead_time_s"] = (
        (contact_t - state["capture_time_s"])
        if (contact_t is not None and state["capture_time_s"] is not None)
        else None
    )
    return out


def run_frozen_condition(seed: int) -> dict:
    override_fn, state = _make_frozen_override()
    result = run_episode(seed, target_override_fn=override_fn)
    out = analyze(result)
    out["lead_time_s"] = None  # not prescribed -- capture time is whatever stabilization produced
    out["capture_time_s"] = state["capture_time_s"]
    out["n_pre_capture_control_steps"] = state["n_pre_capture_steps"]
    contact_t = out.get("actual_contact_time_s")
    out["capture_lead_time_s"] = (
        (contact_t - state["capture_time_s"])
        if (contact_t is not None and state["capture_time_s"] is not None)
        else None
    )
    return out


def main() -> None:
    all_out = []
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        baseline_out = analyze(run_episode(seed))
        baseline_summary = summarize_condition({**baseline_out, "lead_time_s": None})
        print("  -- baseline --")
        for k, v in baseline_summary.items():
            print(f"    {k}: {v}")

        frozen_out = run_frozen_condition(seed)
        frozen_summary = summarize_condition(frozen_out)
        frozen_summary["capture_time_s"] = frozen_out["capture_time_s"]
        frozen_summary["capture_lead_time_s"] = frozen_out["capture_lead_time_s"]
        frozen_summary["n_pre_capture_control_steps"] = frozen_out["n_pre_capture_control_steps"]
        print("  -- frozen (captured once, held to contact) --")
        for k, v in frozen_summary.items():
            print(f"    {k}: {v}")

        baseline_dist = max(
            baseline_summary.get("left_contact_distance_from_face_center_m") or 0.0,
            baseline_summary.get("right_contact_distance_from_face_center_m") or 0.0,
        )
        frozen_dist = max(
            frozen_summary.get("left_contact_distance_from_face_center_m") or float("inf"),
            frozen_summary.get("right_contact_distance_from_face_center_m") or float("inf"),
        )
        tracking_err = frozen_summary["corrected_target_tracking_error_m"]
        arm_too_slow = (
            tracking_err is not None
            and not np.isnan(tracking_err)
            and tracking_err > ARM_TOO_SLOW_TRACKING_ERROR_M
        )
        improved_contact = frozen_dist < 0.9 * baseline_dist
        if frozen_out["capture_time_s"] is None:
            classification = "NEVER_STABILIZED"
        elif arm_too_slow:
            classification = "ARM_RESPONSE_TOO_SLOW"
        elif improved_contact:
            classification = "TARGET_UNSTABLE_CONFIRMED"  # moving target was the real problem
        else:
            classification = "NO_CLEAR_IMPROVEMENT"

        print(f"  -- diagnosis: {classification} "
              f"(tracking_err={tracking_err}, baseline_dist={baseline_dist:.4f}, "
              f"frozen_dist={frozen_dist:.4f}, capture_lead_time_s={frozen_out['capture_lead_time_s']}) --")
        print(flush=True)

        all_out.append({
            "seed": seed,
            "classification": classification,
            "baseline": baseline_summary,
            "frozen": frozen_summary,
        })

    print("=== summary ===")
    for r in all_out:
        f = r["frozen"]
        print(f"  seed={r['seed']} classification={r['classification']} "
              f"capture_lead_time_s={f.get('capture_lead_time_s')} "
              f"tracking_error_m={f.get('corrected_target_tracking_error_m')} "
              f"contact_cls={f.get('left_first_contact_classification')}")


if __name__ == "__main__":
    main()
