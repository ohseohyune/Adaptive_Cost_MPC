"""Extends diagnose_unit_audit_and_n20_revalidation.py with the two pieces
that script didn't cover yet:

  3. Full cross-resolution (N=5/10/20/40) snapshot comparison at every
     requested checkpoint, left AND right pad separately, using the single
     shared acmpc/diagnose_contact_geometry_common.py definition.
  4. Refined-margin (N20/N40) trajectory re-evaluation restricted to the
     0.15s >= (t_contact - t) >= 0.02s confidence window.
  5. Shadow-only early-warning evaluation: NUMERICALLY_UNCERTAIN /
     TIP_EDGE_RISK / FACE_CONTACT_EXPECTED, requiring the margin to stay on
     one side of a threshold for >=3 consecutive ~10ms control steps before
     it counts as a warning (not a single noisy sample). Never wired to any
     controller -- diagnostic output only.

Read-only: does not touch production target, TTC predictor, trigger, phase
gate, pad orientation, controller gain, or trajectory.

Usage: python3 acmpc/diagnose_shadow_warning_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS
from acmpc.diagnose_grid_convergence import run_poses
from acmpc.diagnose_3d_contact_geometry import analyze_snapshot, run_and_snapshot
from acmpc.diagnose_contact_geometry_common import compute_contact_mode_geometry

RESOLUTIONS = [5, 10, 20, 40]
CHECKPOINT_LEADS_S = [0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
CONTROL_STEP_S = 0.010
CONSECUTIVE_STEPS_REQUIRED = 3
THRESHOLDS_M = {"m<0": 0.0, "m<2mm": 0.002, "m<5mm": 0.005, "m<10mm": 0.010}
CONFIDENCE_WINDOW_S = (0.02, 0.15)  # 0.15s >= (contact_t - t) >= 0.02s


# ---------------------------------------------------------------------------
# Part 3: cross-resolution snapshot comparison
# ---------------------------------------------------------------------------

def snapshot_ground_truth(seed: int) -> dict:
    snap_result = run_and_snapshot(seed)
    bilateral = None
    for key, snap in snap_result["snapshots"].items():
        if key == "bilateral":
            bilateral = analyze_snapshot(seed, key, snap, snap_result["ids"]["box_geom"])
    contact_t = bilateral.get("time_s") if bilateral else None
    actual_mode = "TIP_EDGE" if (bilateral and bilateral.get("normal_vs_world_x_deg", 0) > 45) else "FACE_FACE"
    return {"contact_t": contact_t, "actual_mode": actual_mode}


def cross_resolution_checkpoint(poses, t_target: float) -> dict:
    times = np.array([p.time_s for p in poses])
    idx = int(np.argmin(np.abs(times - t_target)))
    pose = poses[idx]
    out = {"time_s": pose.time_s, "sides": {}}
    for side in ("left", "right"):
        side_out = {}
        for n in RESOLUTIONS:
            g = compute_contact_mode_geometry(
                pose.box_center, pose.box_rot, pose.box_half,
                pose.pad_center[side], pose.pad_rot[side], pose.pad_half[side],
                side, resolution=n,
            )
            side_out[n] = {
                "d_A_m": g.dist_a_m,
                "d_B_m": g.dist_b_m,
                "margin_m": g.margin_m,
                "predicted_mode": "TIP_EDGE" if g.margin_m < 0 else "FACE",
                "closest_a_world": g.closest_a_world.tolist(),
                "closest_b_world": g.closest_b_world.tolist(),
                "grid_index_a": g.grid_index_a,
                "grid_index_b": g.grid_index_b,
            }
        out["sides"][side] = side_out
    # aggregate: mean of left/right margin at each resolution
    out["aggregate"] = {
        n: 0.5 * (out["sides"]["left"][n]["margin_m"] + out["sides"]["right"][n]["margin_m"])
        for n in RESOLUTIONS
    }
    return out


def part3_cross_resolution(seed: int, poses, contact_t: float) -> dict:
    checkpoints = {}
    for lead in CHECKPOINT_LEADS_S:
        checkpoints[f"lead_{lead:.2f}s"] = cross_resolution_checkpoint(poses, contact_t - lead)
    checkpoints["last_pre_contact_step"] = cross_resolution_checkpoint(poses, contact_t - 1e-6)
    return checkpoints


# ---------------------------------------------------------------------------
# Part 4: refined margin trajectory, confidence-window re-evaluation
# ---------------------------------------------------------------------------

def build_margin_series(poses, resolution: int, side: str = "left", window_s: float = 0.60) -> tuple[np.ndarray, np.ndarray]:
    # covers the full pre-contact trace for every Stage-1 seed tested here
    # (longest observed ~0.48s) -- a prior 0.30s restriction produced a
    # left-edge artifact: resampling started mid-transient (the pad-settling
    # blip already found around 0.20s pre-contact), which spuriously
    # satisfied the 3-consecutive-step-below-threshold warning condition
    # right at the window boundary for 3/4 seeds (all reporting the same
    # suspicious lead=0.28s, one sample after window start). Full history
    # removes that artifact.
    contact_t = poses[-1].time_s
    poses = [p for p in poses if contact_t - p.time_s <= window_s]
    times = np.array([p.time_s for p in poses])
    m = np.zeros(len(poses))
    for i, p in enumerate(poses):
        g = compute_contact_mode_geometry(
            p.box_center, p.box_rot, p.box_half,
            p.pad_center[side], p.pad_rot[side], p.pad_half[side],
            side, resolution=resolution,
        )
        m[i] = g.margin_m
    return times, m


def part4_confidence_window(times, m, contact_t: float) -> dict:
    lo, hi = CONFIDENCE_WINDOW_S
    mask = (contact_t - times >= lo) & (contact_t - times <= hi)
    window_m = m[mask]
    if window_m.size == 0:
        return {"error": "empty confidence window"}
    return {
        "n_samples": int(window_m.size),
        "min_m": float(np.min(window_m)),
        "max_m": float(np.max(window_m)),
        "mean_m": float(np.mean(window_m)),
        "all_negative": bool(np.all(window_m < 0)),
        "all_positive": bool(np.all(window_m > 0)),
        "sign_changes": int(np.sum(np.diff(np.sign(window_m)) != 0)),
    }


# ---------------------------------------------------------------------------
# Part 5: shadow-only persistence-gated warning evaluation
# ---------------------------------------------------------------------------

BURN_IN_S = 0.08  # excludes the post-launch pad-settling transient (predictor
# confidence itself only reaches 1.0 by ~0.04-0.05s into flight, per every
# earlier capture-gating diagnostic this session) -- without this, all
# thresholds (even m<0) spuriously "warn" at episode start for every seed,
# safe and risky alike, because early pad/box velocity estimates are noisy
# regardless of true geometric risk.


def resample_to_control_grid(times: np.ndarray, m: np.ndarray, contact_t: float) -> tuple[np.ndarray, np.ndarray]:
    start = times[0] + BURN_IN_S
    grid_times = np.arange(start, contact_t, CONTROL_STEP_S)
    grid_m = np.array([m[np.argmin(np.abs(times - t))] for t in grid_times])
    return grid_times, grid_m


def evaluate_threshold(grid_times: np.ndarray, grid_m: np.ndarray, contact_t: float,
                        threshold: float, actual_mode: str, restrict_to_confidence_window: bool = True) -> dict:
    n = len(grid_m)
    warn = np.zeros(n, dtype=bool)
    lo, hi = CONFIDENCE_WINDOW_S
    for i in range(CONSECUTIVE_STEPS_REQUIRED - 1, n):
        if restrict_to_confidence_window:
            lead_i = contact_t - grid_times[i]
            if not (lo <= lead_i <= hi):
                continue
        window = grid_m[i - CONSECUTIVE_STEPS_REQUIRED + 1 : i + 1]
        if np.all(window < threshold):
            warn[i] = True

    warned_indices = np.where(warn)[0]
    ever_warned = warned_indices.size > 0
    first_warn_time = float(grid_times[warned_indices[0]]) if ever_warned else None
    earliest_lead_s = (contact_t - first_warn_time) if first_warn_time is not None else None

    predicted_tip_edge = ever_warned
    predicted_mode = "TIP_EDGE" if predicted_tip_edge else "FACE"
    true_positive = predicted_mode == "TIP_EDGE" and actual_mode == "TIP_EDGE"
    false_positive = predicted_mode == "TIP_EDGE" and actual_mode == "FACE_FACE"
    false_negative = predicted_mode == "FACE" and actual_mode == "TIP_EDGE"

    # consecutive-step stability: once warned, does it stay warned (no flip-flop)?
    if ever_warned:
        post_warn = warn[warned_indices[0]:]
        stability = float(np.mean(post_warn))
    else:
        stability = None

    return {
        "threshold_m": threshold,
        "ever_warned": ever_warned,
        "first_warn_lead_s": earliest_lead_s,
        "predicted_mode": predicted_mode,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "post_warn_stability_fraction": stability,
    }


def part5_shadow_warning(seed: int, poses, contact_t: float, actual_mode: str) -> dict:
    results = {}
    for n in [20, 40]:
        times, m = build_margin_series(poses, n, side="left")
        grid_times, grid_m = resample_to_control_grid(times, m, contact_t)
        results[n] = {
            thr_name: evaluate_threshold(grid_times, grid_m, contact_t, thr_val, actual_mode)
            for thr_name, thr_val in THRESHOLDS_M.items()
        }
    # N20 vs N40 agreement per threshold
    agreement = {}
    for thr_name in THRESHOLDS_M:
        agreement[thr_name] = (
            results[20][thr_name]["predicted_mode"] == results[40][thr_name]["predicted_mode"]
        )
    return {"seed": seed, "by_resolution": results, "n20_n40_agreement": agreement}


# ---------------------------------------------------------------------------
def main() -> None:
    all_part3 = {}
    all_part4 = {}
    all_part5 = {}

    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        gt = snapshot_ground_truth(seed)
        contact_t, actual_mode = gt["contact_t"], gt["actual_mode"]
        ctx, poses = run_poses(seed)
        if not poses or contact_t is None:
            print("  no data, skipping")
            continue
        print(f"  actual_mode={actual_mode} contact_t={contact_t:.4f}")

        # --- Part 3 ---
        cp3 = part3_cross_resolution(seed, poses, contact_t)
        all_part3[seed] = cp3
        for cp_name, cp in cp3.items():
            left20 = cp["sides"]["left"][20]
            right20 = cp["sides"]["right"][20]
            print(f"  -- {cp_name} (N=20) -- L: d_A={left20['d_A_m']:.5f} d_B={left20['d_B_m']:.5f} "
                  f"m={left20['margin_m']:.5f} mode={left20['predicted_mode']} | "
                  f"R: d_A={right20['d_A_m']:.5f} d_B={right20['d_B_m']:.5f} m={right20['margin_m']:.5f} "
                  f"mode={right20['predicted_mode']} | agg={cp['aggregate'][20]:.5f}")

        # --- Part 4 ---
        times20, m20 = build_margin_series(poses, 20, "left")
        times40, m40 = build_margin_series(poses, 40, "left")
        cw20 = part4_confidence_window(times20, m20, contact_t)
        cw40 = part4_confidence_window(times40, m40, contact_t)
        all_part4[seed] = {"n20": cw20, "n40": cw40}
        print(f"  confidence_window[0.02-0.15s] N20: {cw20}")
        print(f"  confidence_window[0.02-0.15s] N40: {cw40}")

        # --- Part 5 ---
        p5 = part5_shadow_warning(seed, poses, contact_t, actual_mode)
        all_part5[seed] = p5
        for n in [20, 40]:
            for thr_name, res in p5["by_resolution"][n].items():
                print(f"  N={n} {thr_name}: warned={res['ever_warned']} lead={res['first_warn_lead_s']} "
                      f"mode={res['predicted_mode']} TP={res['true_positive']} FP={res['false_positive']} "
                      f"FN={res['false_negative']} stability={res['post_warn_stability_fraction']}")
        print(f"  N20/N40 agreement: {p5['n20_n40_agreement']}")
        print(flush=True)

    print("=== cross-seed summary: threshold performance ===")
    for thr_name in THRESHOLDS_M:
        print(f"  -- {thr_name} (N=20) --")
        for seed in STAGE1_SEEDS:
            if seed not in all_part5:
                continue
            res = all_part5[seed]["by_resolution"][20][thr_name]
            print(f"    seed={seed} TP={res['true_positive']} FP={res['false_positive']} "
                  f"FN={res['false_negative']} lead={res['first_warn_lead_s']}")


if __name__ == "__main__":
    main()
