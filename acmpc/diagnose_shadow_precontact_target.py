"""Shadow-rollout experiment: how early would the corrected (geometrically
accurate) contact target need to be commanded for the arms to actually reach
it before real contact?

Does NOT change baseline behavior -- run_box_catch's target_override_fn
parameter (main_acmpc_box_catch.py) defaults to None and is a no-op for
every existing caller, including the plain diagnostic runs in
diagnose_precontact_grasp_offset.py. This script is the first caller that
passes a non-None override, and only inside its own experimental episodes.

For each seed, we first run the untouched baseline once to find its actual
first-contact time (hindsight only, used to schedule *when* the override
switches on -- the override itself never looks into the future). Then, for
each lead time in {0.40, 0.30, 0.20, 0.10}s, we rerun the same seed with an
override that starts commanding the corrected ballistic-contact-plane target
(the same position_remaining_ttc-based prediction from
diagnose_precontact_grasp_offset.py / diagnose_reachability_margin.py, never
wired into production) starting at (baseline_contact_time - lead_time), and
record what actually happens under that shadow control.

Usage: python3 acmpc/diagnose_shadow_precontact_target.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, analyze, run_episode

LEAD_TIMES_S = [0.40, 0.30, 0.20, 0.10]
# Tracking-error scale baseline itself achieves when *not* being asked to
# chase a late-arriving/moving target (see diagnose_precontact_grasp_offset
# report: baseline left/right_pad_target_tracking_error_m ~= 0.026-0.029m).
# Used only as a diagnostic yardstick for ARM_RESPONSE_TOO_SLOW, not a
# control threshold.
ARM_TOO_SLOW_TRACKING_ERROR_M = 0.04


def _make_override(lead_time_s: float, baseline_contact_time_s: float):
    def override_fn(ctx: dict):
        if not ctx["position_remaining_ttc_valid"]:
            return None
        if ctx["time_s"] < baseline_contact_time_s - lead_time_s:
            return None
        tau = max(0.0, float(ctx["position_remaining_ttc"]))
        return ctx["prediction"].position_after(tau)

    return override_fn


def run_condition(seed: int, lead_time_s: float | None, baseline_contact_time_s: float | None) -> dict:
    if lead_time_s is None:
        result = run_episode(seed)
    else:
        override_fn = _make_override(lead_time_s, baseline_contact_time_s)
        result = run_episode(seed, target_override_fn=override_fn)
    out = analyze(result)
    out["lead_time_s"] = lead_time_s
    return out


def summarize_condition(out: dict) -> dict:
    tracking_err = max(
        out.get("left_pad_target_tracking_error_m") or float("nan"),
        out.get("right_pad_target_tracking_error_m") or float("nan"),
    )
    return {
        "lead_time_s": out.get("lead_time_s"),
        "success": out.get("success"),
        "corrected_target_tracking_error_m": tracking_err,
        "left_first_contact_box_local_xyz": out.get("left_first_contact_box_local_xyz"),
        "right_first_contact_box_local_xyz": out.get("right_first_contact_box_local_xyz"),
        "left_contact_distance_from_face_center_m": out.get("left_contact_distance_from_face_center_m"),
        "right_contact_distance_from_face_center_m": out.get("right_contact_distance_from_face_center_m"),
        "left_first_contact_classification": out.get("left_first_contact_classification"),
        "right_first_contact_classification": out.get("right_first_contact_classification"),
        "actual_grasp_center_to_com_offset_m": out.get("actual_grasp_center_to_com_offset_m"),
        "gravity_torque_y_at_bilateral_contact_nm": out.get("gravity_torque_y_at_bilateral_contact_nm"),
        "mean_box_omega_y_radps_hold": out.get("mean_box_omega_y_radps_hold"),
        "cumulative_rotation_deg_hold": out.get("cumulative_rotation_deg_hold"),
        "first_contact_peak_force_n": out.get("first_contact_peak_force_n"),
        "contact_lost": out.get("contact_lost"),
        "failure_reason": out.get("failure_reason"),
    }


def _classify_seed(baseline_summary: dict, condition_summaries: list[dict]) -> dict:
    baseline_dist = max(
        baseline_summary.get("left_contact_distance_from_face_center_m") or 0.0,
        baseline_summary.get("right_contact_distance_from_face_center_m") or 0.0,
    )
    baseline_gravity = baseline_summary.get("gravity_torque_y_at_bilateral_contact_nm")
    baseline_rotation = baseline_summary.get("cumulative_rotation_deg_hold")

    min_required_lead = None
    per_lead = {}
    for cs in condition_summaries:
        lt = cs["lead_time_s"]
        dist = max(
            cs.get("left_contact_distance_from_face_center_m") or float("inf"),
            cs.get("right_contact_distance_from_face_center_m") or float("inf"),
        )
        tracking_err = cs["corrected_target_tracking_error_m"]
        improved_contact = dist < 0.9 * baseline_dist
        improved_rotation = (
            baseline_rotation is not None
            and cs.get("cumulative_rotation_deg_hold") is not None
            and cs["cumulative_rotation_deg_hold"] < 0.9 * baseline_rotation
        )
        improved_gravity = (
            baseline_gravity is not None
            and cs.get("gravity_torque_y_at_bilateral_contact_nm") is not None
            and cs["gravity_torque_y_at_bilateral_contact_nm"] < 0.9 * baseline_gravity
        )
        arm_too_slow = tracking_err is not None and not np.isnan(tracking_err) and tracking_err > ARM_TOO_SLOW_TRACKING_ERROR_M
        per_lead[lt] = {
            "improved_contact": improved_contact,
            "improved_rotation": improved_rotation,
            "improved_gravity": improved_gravity,
            "arm_too_slow": arm_too_slow,
            "tracking_error_m": tracking_err,
            "distance_from_face_center_m": dist,
        }
        if improved_contact and (improved_rotation or improved_gravity) and not arm_too_slow:
            if min_required_lead is None or lt < min_required_lead:
                min_required_lead = lt

    if min_required_lead is not None:
        classification = "PHASE_GATE_TOO_LATE"
    elif all(v["arm_too_slow"] for v in per_lead.values()):
        classification = "ARM_RESPONSE_TOO_SLOW"
    else:
        classification = "NO_CLEAR_IMPROVEMENT"

    return {
        "classification": classification,
        "min_required_lead_time_s": min_required_lead,
        "per_lead_detail": per_lead,
    }


def main() -> None:
    all_results = {}
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        baseline_out = run_condition(seed, None, None)
        baseline_summary = summarize_condition(baseline_out)
        baseline_contact_time = baseline_out.get("actual_contact_time_s")
        print("  -- baseline --")
        for k, v in baseline_summary.items():
            print(f"    {k}: {v}")

        condition_summaries = []
        if baseline_contact_time is not None:
            for lead_time in LEAD_TIMES_S:
                out = run_condition(seed, lead_time, baseline_contact_time)
                summ = summarize_condition(out)
                condition_summaries.append(summ)
                print(f"  -- lead_time={lead_time:.2f}s --")
                for k, v in summ.items():
                    print(f"    {k}: {v}")

        diag = _classify_seed(baseline_summary, condition_summaries) if condition_summaries else {
            "classification": "INCONCLUSIVE", "min_required_lead_time_s": None, "per_lead_detail": {}
        }
        print(f"  -- diagnosis: {diag['classification']} "
              f"min_required_lead_time_s={diag['min_required_lead_time_s']} --")
        for lt, d in diag["per_lead_detail"].items():
            print(f"    lead={lt:.2f}s: {d}")
        print(flush=True)

        all_results[seed] = {
            "baseline": baseline_summary,
            "conditions": condition_summaries,
            "diagnosis": diag,
        }

    print("=== summary ===")
    for seed, r in all_results.items():
        print(f"  seed={seed} diagnosis={r['diagnosis']['classification']} "
              f"min_required_lead_time_s={r['diagnosis']['min_required_lead_time_s']}")


if __name__ == "__main__":
    main()
