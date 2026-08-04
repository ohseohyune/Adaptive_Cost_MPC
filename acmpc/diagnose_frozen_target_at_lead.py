"""Shadow-rollout experiment #3: freeze the corrected target once, at a
*prescribed* lead time before contact (not an auto-detected "stability"
moment -- diagnose_frozen_target_shadow.py's stability heuristic collapsed
to the same fixed t=0.09s in all 4 seeds, which is an artifact of the
predictor's confidence ramp-up, not real trajectory-dependent settling, and
captured too early to be geometrically accurate).

This is the clean, non-confounded version of the original question: for
each lead time in {0.40, 0.30, 0.20, 0.10}s, capture the corrected target
exactly once (at contact_time - lead_time) and hold that single point fixed
until contact, instead of re-evaluating it every control step
(diagnose_shadow_precontact_target.py) or triggering capture on a noisy
stability signal (diagnose_frozen_target_shadow.py).

If tracking error to the frozen point is small and contact quality
improves at some lead time -> TARGET_UNSTABLE was the real problem, and
that lead time is the minimum required. If tracking error stays large even
to a frozen point -> ARM_RESPONSE_TOO_SLOW / reachability. If tracking is
fine but contact quality still doesn't improve -> the captured value itself
isn't accurate enough that early (long ballistic-extrapolation horizon
error), a separate problem from both target instability and arm speed.

No phase gate, gain, friction, or success-condition changes.

Usage: python3 acmpc/diagnose_frozen_target_at_lead.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, analyze, run_episode
from acmpc.diagnose_shadow_precontact_target import ARM_TOO_SLOW_TRACKING_ERROR_M, LEAD_TIMES_S, summarize_condition
from acmpc.diagnose_frozen_target_shadow import run_frozen_at_lead_condition


def main() -> None:
    all_results = {}
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        baseline_out = analyze(run_episode(seed))
        baseline_summary = summarize_condition({**baseline_out, "lead_time_s": None})
        baseline_contact_time = baseline_out.get("actual_contact_time_s")
        print("  -- baseline --")
        for k, v in baseline_summary.items():
            print(f"    {k}: {v}")

        baseline_dist = max(
            baseline_summary.get("left_contact_distance_from_face_center_m") or 0.0,
            baseline_summary.get("right_contact_distance_from_face_center_m") or 0.0,
        )

        condition_summaries = []
        min_required_lead = None
        if baseline_contact_time is not None:
            for lead_time in LEAD_TIMES_S:
                out = run_frozen_at_lead_condition(seed, lead_time, baseline_contact_time)
                summ = summarize_condition(out)
                summ["capture_lead_time_s"] = out["capture_lead_time_s"]
                condition_summaries.append(summ)
                print(f"  -- frozen-at-lead={lead_time:.2f}s --")
                for k, v in summ.items():
                    print(f"    {k}: {v}")

                dist = max(
                    summ.get("left_contact_distance_from_face_center_m") or float("inf"),
                    summ.get("right_contact_distance_from_face_center_m") or float("inf"),
                )
                tracking_err = summ["corrected_target_tracking_error_m"]
                arm_too_slow = (
                    tracking_err is not None
                    and not np.isnan(tracking_err)
                    and tracking_err > ARM_TOO_SLOW_TRACKING_ERROR_M
                )
                improved_contact = dist < 0.9 * baseline_dist
                if improved_contact and not arm_too_slow:
                    if min_required_lead is None or lead_time < min_required_lead:
                        min_required_lead = lead_time

        if min_required_lead is not None:
            classification = "TARGET_UNSTABLE_CONFIRMED"
        elif condition_summaries and all(
            (s["corrected_target_tracking_error_m"] or 0) > ARM_TOO_SLOW_TRACKING_ERROR_M
            for s in condition_summaries
        ):
            classification = "ARM_RESPONSE_TOO_SLOW"
        elif condition_summaries and all(
            (s["corrected_target_tracking_error_m"] or 0) <= ARM_TOO_SLOW_TRACKING_ERROR_M
            for s in condition_summaries
        ):
            classification = "TARGET_ACCURACY_LIMITED"  # tracks fine, contact still doesn't improve
        else:
            classification = "MIXED"

        print(f"  -- diagnosis: {classification} min_required_lead_time_s={min_required_lead} --")
        print(flush=True)
        all_results[seed] = {"classification": classification, "min_required_lead_time_s": min_required_lead,
                              "conditions": condition_summaries}

    print("=== summary ===")
    for seed, r in all_results.items():
        print(f"  seed={seed} classification={r['classification']} "
              f"min_required_lead_time_s={r['min_required_lead_time_s']}")


if __name__ == "__main__":
    main()
