"""Sweep the frozen-target capture's stability threshold (window x max
change), with the workspace gate removed (diagnose_gate_ablation_shadow.py
showed it was redundant given stability -- B==C exactly) and MIN_SAMPLES
fixed at 5 (diagnose_min_samples_sweep.py showed 5/6/7/10 were identical;
5 is simply the smallest already-verified-equivalent floor).

Sweep:
  stability_window_ms  in {20, 30, 40, 50}
  max_target_change_m  in {0.005, 0.01, 0.02}
  (12 combinations x 4 seeds = 48 shadow episodes)

Fixed: confidence == 1.0, position_remaining_ttc > 0, MIN_SAMPLES = 5, no
workspace gate. Capture happens once, at the first time the corrected
target's spread over the trailing stability_window_ms is <=
max_target_change_m, and is held fixed to contact.

Selection priority:
  1. no miss/contact_lost/emergency in any of the 4 tuning seeds
  2. contact centering + rotation improved vs baseline in every one of the
     4 tuning seeds (not just some -- partial improvement does not qualify
     as ROBUST_TRIGGER_FOUND)
  3. among candidates satisfying 1-2, pick the earliest (smallest mean
     capture time, i.e. most permissive combination still safe)

If a ROBUST_TRIGGER_FOUND candidate exists, validate it on 20 fresh random
seeds not used for tuning, compare against baseline, and only call it a
real candidate if safety + improvement reproduce there too -- otherwise
report NO_GENERALIZATION and stop.

Does not touch production PRE_IMPACT gate/control logic -- shadow only.

Usage: python3 acmpc/diagnose_stability_threshold_sweep.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, analyze, run_episode
from acmpc.diagnose_shadow_precontact_target import summarize_condition

MIN_SAMPLES = 5
STABILITY_WINDOW_MS_SWEEP = [20, 30, 40, 50]
MAX_TARGET_CHANGE_M_SWEEP = [0.005, 0.01, 0.02]
N_VALIDATION_SEEDS = 20
VALIDATION_SEED_RNG_SEED = 424242  # separate from tuning seeds {1000,1017,1005,1001}


def _make_stability_capture_override(min_samples: int, window_s: float, threshold_m: float):
    state = {"captured": None, "capture_time_s": None, "history": []}

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
        while state["history"] and t - state["history"][0][0] > window_s:
            state["history"].pop(0)

        confidence_ok = float(ctx["prediction"].confidence) >= 1.0
        samples_ok = int(ctx["predictor_samples"]) >= min_samples

        stability_ok = False
        if state["history"] and (t - state["history"][0][0]) >= window_s * 0.8:
            pts = np.array([p for _, p in state["history"]])
            spread = float(np.max(np.linalg.norm(pts - current, axis=1)))
            stability_ok = spread <= threshold_m

        if confidence_ok and samples_ok and stability_ok:
            state["captured"] = current.copy()
            state["capture_time_s"] = t
            return state["captured"]
        return None

    return override_fn, state


def run_condition(seed: int, window_ms: float, threshold_m: float) -> dict:
    override_fn, state = _make_stability_capture_override(MIN_SAMPLES, window_ms / 1000.0, threshold_m)
    result = run_episode(seed, target_override_fn=override_fn)
    out = analyze(result)
    out["capture_time_s"] = state["capture_time_s"]
    contact_t = out.get("actual_contact_time_s")
    out["capture_lead_time_s"] = (
        (contact_t - state["capture_time_s"])
        if (contact_t is not None and state["capture_time_s"] is not None)
        else None
    )
    return out


def _seed_verdict(baseline_summary: dict, cond_summary: dict) -> tuple[bool, bool]:
    miss = cond_summary.get("failure_reason") == "box passed the interception workspace"
    lost = bool(cond_summary.get("contact_lost"))
    emergency = cond_summary.get("failure_reason") == "emergency contact force exceeded"
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
    rot_improved = baseline_rot is not None and cond_rot is not None and cond_rot < 0.9 * baseline_rot
    return safe, (dist_improved and rot_improved)


def run_tuning_sweep():
    baselines = {seed: summarize_condition({**analyze(run_episode(seed)), "lead_time_s": None}) for seed in STAGE1_SEEDS}

    results = {}
    for window_ms in STABILITY_WINDOW_MS_SWEEP:
        for threshold_m in MAX_TARGET_CHANGE_M_SWEEP:
            key = (window_ms, threshold_m)
            print(f"=== window={window_ms}ms threshold={threshold_m}m ===", flush=True)
            results[key] = {}
            for seed in STAGE1_SEEDS:
                out = run_condition(seed, window_ms, threshold_m)
                summ = summarize_condition(out)
                summ["capture_time_s"] = out["capture_time_s"]
                summ["capture_lead_time_s"] = out["capture_lead_time_s"]
                safe, improved = _seed_verdict(baselines[seed], summ)
                summ["safe"] = safe
                summ["improved"] = improved
                results[key][seed] = summ
                print(f"  -- seed {seed} --")
                for k, v in summ.items():
                    print(f"    {k}: {v}")
            print(flush=True)

    print("=== verdict per combination ===")
    candidates = []
    for key in results:
        window_ms, threshold_m = key
        details = results[key]
        all_safe = all(d["safe"] for d in details.values())
        all_improved = all(d["improved"] for d in details.values())
        capture_times = [d["capture_time_s"] for d in details.values() if d["capture_time_s"] is not None]
        mean_capture_time = float(np.mean(capture_times)) if capture_times else float("inf")
        print(f"  window={window_ms}ms threshold={threshold_m}m: all_safe={all_safe} all_improved={all_improved} "
              f"mean_capture_time_s={mean_capture_time:.3f} "
              f"detail={[(s, details[s]['safe'], details[s]['improved']) for s in STAGE1_SEEDS]}")
        if all_safe and all_improved:
            candidates.append((mean_capture_time, key))

    classification = None
    chosen_key = None
    if candidates:
        candidates.sort(key=lambda x: x[0])
        chosen_key = candidates[0][1]
        classification = "ROBUST_TRIGGER_FOUND"
        print(f"\nchosen candidate: window={chosen_key[0]}ms threshold={chosen_key[1]}m "
              f"(earliest mean capture time among safe+all-improved combinations)")
    else:
        any_unsafe = any(not d["safe"] for details in results.values() for d in details.values())
        any_improved = any(d["improved"] for details in results.values() for d in details.values())
        if any_unsafe and not any_improved:
            classification = "CAPTURE_TOO_EARLY"
        elif not any_improved:
            classification = "NO_EFFECT"
        else:
            classification = "PARTIAL_TRIGGER"
        print(f"\nno combination is safe AND fully-improved across all 4 tuning seeds")

    print(f"tuning classification: {classification}")
    return classification, chosen_key, baselines


def run_validation(window_ms: float, threshold_m: float):
    rng = random.Random(VALIDATION_SEED_RNG_SEED)
    tuning_seeds = set(STAGE1_SEEDS)
    val_seeds = []
    while len(val_seeds) < N_VALIDATION_SEEDS:
        candidate = rng.randint(2000, 9999)
        if candidate not in tuning_seeds and candidate not in val_seeds:
            val_seeds.append(candidate)

    print(f"\n=== 20-seed validation: window={window_ms}ms threshold={threshold_m}m ===")
    print(f"validation seeds: {val_seeds}")

    baseline_rows = []
    cond_rows = []
    for seed in val_seeds:
        b_out = analyze(run_episode(seed))
        b_summ = summarize_condition({**b_out, "lead_time_s": None})
        c_out = run_condition(seed, window_ms, threshold_m)
        c_summ = summarize_condition(c_out)
        safe, improved = _seed_verdict(b_summ, c_summ)
        c_summ["safe"] = safe
        c_summ["improved"] = improved
        baseline_rows.append(b_summ)
        cond_rows.append(c_summ)
        print(f"  seed {seed}: baseline_success={b_summ.get('success')} cond_success={c_summ.get('success')} "
              f"cond_failure_reason={c_summ.get('failure_reason')} safe={safe} improved={improved} "
              f"dist_L={c_summ.get('left_contact_distance_from_face_center_m')} "
              f"rot={c_summ.get('cumulative_rotation_deg_hold')}")

    def _dist(row):
        return max(
            row.get("left_contact_distance_from_face_center_m") or float("nan"),
            row.get("right_contact_distance_from_face_center_m") or float("nan"),
        )

    success_count = sum(1 for r in cond_rows if r.get("success"))
    miss_count = sum(1 for r in cond_rows if r.get("failure_reason") == "box passed the interception workspace")
    loss_count = sum(1 for r in cond_rows if r.get("contact_lost"))
    emergency_count = sum(1 for r in cond_rows if r.get("failure_reason") == "emergency contact force exceeded")
    dists = np.array([_dist(r) for r in cond_rows if np.isfinite(_dist(r))])
    rots = np.array([r.get("cumulative_rotation_deg_hold") for r in cond_rows if r.get("cumulative_rotation_deg_hold") is not None])
    forces = np.array([r.get("first_contact_peak_force_n") for r in cond_rows if r.get("first_contact_peak_force_n") is not None])

    print("\n=== 20-seed validation summary ===")
    print(f"  success_count: {success_count}/{N_VALIDATION_SEEDS}")
    print(f"  workspace_miss_count: {miss_count}")
    print(f"  contact_loss_count: {loss_count}")
    print(f"  emergency_count: {emergency_count}")
    print(f"  mean_face_center_distance_m: {float(np.mean(dists)) if dists.size else None}")
    print(f"  p95_face_center_distance_m: {float(np.percentile(dists, 95)) if dists.size else None}")
    print(f"  mean_cumulative_rotation_deg: {float(np.mean(rots)) if rots.size else None}")
    print(f"  p95_cumulative_rotation_deg: {float(np.percentile(rots, 95)) if rots.size else None}")
    print(f"  mean_first_contact_peak_force_n: {float(np.mean(forces)) if forces.size else None}")

    baseline_dists = np.array([_dist(r) for r in baseline_rows if np.isfinite(_dist(r))])
    baseline_rots = np.array([r.get("cumulative_rotation_deg_hold") for r in baseline_rows if r.get("cumulative_rotation_deg_hold") is not None])
    print("\n=== 20-seed baseline (for comparison) ===")
    print(f"  success_count: {sum(1 for r in baseline_rows if r.get('success'))}/{N_VALIDATION_SEEDS}")
    print(f"  mean_face_center_distance_m: {float(np.mean(baseline_dists)) if baseline_dists.size else None}")
    print(f"  mean_cumulative_rotation_deg: {float(np.mean(baseline_rots)) if baseline_rots.size else None}")

    generalizes = (
        miss_count == 0
        and loss_count == 0
        and emergency_count == 0
        and dists.size
        and baseline_dists.size
        and float(np.mean(dists)) < 0.9 * float(np.mean(baseline_dists))
        and rots.size
        and baseline_rots.size
        and float(np.mean(rots)) < 0.9 * float(np.mean(baseline_rots))
    )
    final = "ROBUST_TRIGGER_FOUND" if generalizes else "NO_GENERALIZATION"
    print(f"\nfinal validation classification: {final}")
    return final


def main() -> None:
    classification, chosen_key, _ = run_tuning_sweep()
    if classification == "ROBUST_TRIGGER_FOUND" and chosen_key is not None:
        window_ms, threshold_m = chosen_key
        run_validation(window_ms, threshold_m)
    else:
        print(f"\nNo tuning candidate reached ROBUST_TRIGGER_FOUND ({classification}) -- skipping 20-seed validation.")


if __name__ == "__main__":
    main()
