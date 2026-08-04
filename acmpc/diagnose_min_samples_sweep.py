"""Sweep CAPTURE_MIN_SAMPLES in {5, 6, 7} for the mature-capture shadow
rollout (diagnose_mature_capture_shadow.py), holding every other capture
condition fixed (confidence==1.0, position_remaining_ttc>0, workspace
sanity check, <1cm target spread over the trailing 50ms).

Motivation: at MIN_SAMPLES=10, capture converged to nearly the same t~0.14-
0.16s in all 4 seeds and mostly erased the improvement seen when freezing
much earlier (diagnose_frozen_target_at_lead.py's 0.40s-lead freeze helped
3/4 seeds) -- confidence already saturates at 5 samples, so the sample
floor, not confidence or stability, was the binding constraint. This sweep
checks whether a lower floor still avoids seed 1001's workspace-miss
failure (already-verified safe via the workspace check rejecting 3 early
candidates every seed) while recovering more of the improvement.

Selection rule (in order):
  1. no workspace miss and no contact regression in any of the 4 seeds
  2. contact centering and rotation improved vs baseline in every seed
  3. among values satisfying 1-2, pick the smallest MIN_SAMPLES

Does not touch production PRE_IMPACT gate/control logic -- shadow only.

Usage: python3 acmpc/diagnose_min_samples_sweep.py
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
from acmpc.diagnose_mature_capture_shadow import run_mature_capture_condition

MIN_SAMPLES_SWEEP = [5, 6, 7]


def run_condition_for_samples(seed: int, min_samples: int) -> dict:
    out = run_mature_capture_condition(seed, min_samples=min_samples)
    return out


def _seed_ok(baseline_summary: dict, cond_summary: dict) -> tuple[bool, bool]:
    """Returns (safe, improved)."""
    miss = cond_summary.get("failure_reason") == "box passed the interception workspace"
    lost = bool(cond_summary.get("contact_lost"))
    emergency = bool(cond_summary.get("failure_reason") == "emergency contact force exceeded")
    safe = not miss and not lost and not emergency

    baseline_dist = max(
        baseline_summary.get("left_contact_distance_from_face_center_m") or 0.0,
        baseline_summary.get("right_contact_distance_from_face_center_m") or 0.0,
    )
    cond_dist = max(
        cond_summary.get("left_contact_distance_from_face_center_m") or float("inf"),
        cond_summary.get("right_contact_distance_from_face_center_m") or float("inf"),
    )
    baseline_rot = baseline_summary.get("cumulative_rotation_deg_hold")
    cond_rot = cond_summary.get("cumulative_rotation_deg_hold")
    dist_improved = cond_dist < 0.9 * baseline_dist
    rot_improved = (
        baseline_rot is not None and cond_rot is not None and cond_rot < 0.9 * baseline_rot
    )
    improved = dist_improved and rot_improved
    return safe, improved


def main() -> None:
    baselines = {}
    for seed in STAGE1_SEEDS:
        baselines[seed] = summarize_condition({**analyze(run_episode(seed)), "lead_time_s": None})

    results = {}  # min_samples -> {seed -> summary}
    for min_samples in MIN_SAMPLES_SWEEP:
        print(f"=== MIN_SAMPLES={min_samples} ===", flush=True)
        results[min_samples] = {}
        for seed in STAGE1_SEEDS:
            out = run_condition_for_samples(seed, min_samples)
            summ = summarize_condition(out)
            summ["capture_time_s"] = out["capture_time_s"]
            summ["capture_lead_time_s"] = out["capture_lead_time_s"]
            summ["capture_confidence"] = out["capture_confidence"]
            summ["capture_samples"] = out["capture_samples"]
            summ["rejected_workspace_count"] = out["rejected_workspace_count"]
            results[min_samples][seed] = summ
            safe, improved = _seed_ok(baselines[seed], summ)
            print(f"  -- seed {seed} --")
            for k, v in summ.items():
                print(f"    {k}: {v}")
            print(f"    safe={safe} improved_vs_baseline={improved}")
        print(flush=True)

    print("=== per-MIN_SAMPLES verdict ===")
    passing = []
    for min_samples in MIN_SAMPLES_SWEEP:
        all_safe = True
        all_improved = True
        detail = []
        for seed in STAGE1_SEEDS:
            summ = results[min_samples][seed]
            safe, improved = _seed_ok(baselines[seed], summ)
            all_safe = all_safe and safe
            all_improved = all_improved and improved
            detail.append((seed, safe, improved))
        print(f"  MIN_SAMPLES={min_samples}: all_safe={all_safe} all_improved={all_improved} detail={detail}")
        if all_safe and all_improved:
            passing.append(min_samples)

    if passing:
        chosen = min(passing)
        classification = "ROBUST_TRIGGER_FOUND"
        print(f"\nchosen MIN_SAMPLES={chosen} (smallest passing both criteria)")
    else:
        chosen = None
        # distinguish CAPTURE_TOO_EARLY (unsafe somewhere) vs
        # CAPTURE_TOO_LATE/NO_EFFECT (safe everywhere but never improves)
        any_unsafe = any(
            not _seed_ok(baselines[seed], results[ms][seed])[0]
            for ms in MIN_SAMPLES_SWEEP for seed in STAGE1_SEEDS
        )
        any_improved_anywhere = any(
            _seed_ok(baselines[seed], results[ms][seed])[1]
            for ms in MIN_SAMPLES_SWEEP for seed in STAGE1_SEEDS
        )
        if any_unsafe and not any_improved_anywhere:
            classification = "CAPTURE_TOO_EARLY"
        elif not any_improved_anywhere:
            classification = "NO_EFFECT"
        else:
            classification = "PARTIAL_TRIGGER"
        print(f"\nno MIN_SAMPLES in {MIN_SAMPLES_SWEEP} satisfies both criteria for all 4 seeds")

    print(f"final classification: {classification}")


if __name__ == "__main__":
    main()
