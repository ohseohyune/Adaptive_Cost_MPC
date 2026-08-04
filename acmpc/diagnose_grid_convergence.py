"""Read-only validation: is the FACE vs TIP_EDGE contact-mode margin
m(t) = d_B(t) - d_A(t) (diagnose_contact_mode_race.py /
diagnose_margin_predictability.py) a real 3D geometric quantity, or a
grid-sampling artifact of the 5x5 point-cloud approximation used so far?

Runs each seed's baseline episode exactly once, storing the raw box/pad
geom world pose (center, rotation, half-extents) every substep -- cheap,
no physics rerun needed. All requested grid resolutions (5, 10, 20, 40,
and 80 as a "continuous" stand-in reference) are then recomputed purely as
post-processing from these stored poses.

Feature-switch classification: the existing candidate-B feature choice
(which pad z-face, which box x/z-edge) is *decided once per step* by a
proximity sign heuristic (_candidate_clearances_indexed's pad_z_sign /
box_z_sign) -- i.e. there is exactly one hypothesized tip-face and one
hypothesized edge at any instant, never multiple alternatives. A "physical-
feature switch" is therefore only possible if one of those *signs* flips
between consecutive steps; any index change with the *same* signs is by
construction a within-feature grid-sample switch, not a different physical
feature.

Does not touch target, TTC predictor, trigger, phase gate, pad
orientation, controller gain, or production trajectory.

Usage: python3 acmpc/diagnose_grid_convergence.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, _build_stage1_config, seed_everything
from acmpc.diagnose_contact_mode_race import _edge_grid, _face_grid
from acmpc.diagnose_3d_contact_geometry import analyze_snapshot, run_and_snapshot
from acmpc.main_acmpc_box_catch import run_box_catch

RESOLUTIONS = [5, 10, 20]
CONTINUOUS_N = 40  # finest resolution used as the "continuous" stand-in reference;
# 80 would need a (6400x6400x3) pairwise-distance tensor (~1GB) per call across
# hundreds of checkpoints -- 40 (1600x1600 pairs) is the largest that stays fast
# and memory-safe while still being 8x the coarse baseline.
CHECKPOINT_LEADS_S = [0.20, 0.15, 0.10, 0.05, 0.02, 0.01]


@dataclass
class GeomPose:
    time_s: float
    box_center: np.ndarray
    box_rot: np.ndarray
    box_half: np.ndarray
    pad_center: dict  # side -> np.ndarray
    pad_rot: dict
    pad_half: dict


def run_poses(seed: int) -> tuple[dict, list[GeomPose]]:
    seed_everything(seed)
    cfg = _build_stage1_config(seed)
    ids = {}
    poses: list[GeomPose] = []

    def on_step(ctx: dict) -> None:
        model, data = ctx["model"], ctx["data"]
        if not ids:
            ids["box_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
            ids["pad_geom"] = {
                s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_catch_pad") for s in ("left", "right")
            }
        contact = ctx["contact"]
        if contact.left.active and contact.right.active:
            return
        box_center = data.geom_xpos[ids["box_geom"]].copy()
        box_rot = data.geom_xmat[ids["box_geom"]].reshape(3, 3).copy()
        box_half = model.geom_size[ids["box_geom"]].copy()
        pad_center, pad_rot, pad_half = {}, {}, {}
        for side in ("left", "right"):
            pad_center[side] = data.geom_xpos[ids["pad_geom"][side]].copy()
            pad_rot[side] = data.geom_xmat[ids["pad_geom"][side]].reshape(3, 3).copy()
            pad_half[side] = model.geom_size[ids["pad_geom"][side]].copy()
        poses.append(GeomPose(ctx["time_s"], box_center, box_rot, box_half, pad_center, pad_rot, pad_half))

    summary = run_box_catch(cfg, step_callback=on_step)
    snap_result = run_and_snapshot(seed)
    return {"summary": summary, "snap_result": snap_result, "ids": ids}, poses


def _clearances_at_resolution(pose: GeomPose, side: str, n: int) -> dict:
    box_center, box_rot, box_half = pose.box_center, pose.box_rot, pose.box_half
    pad_center, pad_rot, pad_half = pose.pad_center[side], pose.pad_rot[side], pose.pad_half[side]
    y_sign = 1.0 if side == "left" else -1.0

    face_plus = pad_center + pad_rot[:, 0] * pad_half[0]
    face_minus = pad_center - pad_rot[:, 0] * pad_half[0]
    pad_x_sign = 1.0 if abs(face_plus[0] - box_center[0]) < abs(face_minus[0] - box_center[0]) else -1.0
    pad_wide_face = _face_grid(pad_center, pad_rot, 0, pad_x_sign * pad_half[0], (1, 2), (pad_half[1], pad_half[2]), n=n)
    box_grasp_face = _face_grid(box_center, box_rot, 1, y_sign * box_half[1], (0, 2), (box_half[0], box_half[2]), n=n)
    diff_a = pad_wide_face[:, None, :] - box_grasp_face[None, :, :]
    dist_a_mat = np.linalg.norm(diff_a, axis=2)
    ia, ja = np.unravel_index(np.argmin(dist_a_mat), dist_a_mat.shape)
    dist_a = float(dist_a_mat[ia, ja])
    pt_a_pad, pt_a_box = pad_wide_face[ia], box_grasp_face[ja]

    tip_plus = pad_center + pad_rot[:, 2] * pad_half[2]
    tip_minus = pad_center - pad_rot[:, 2] * pad_half[2]
    pad_z_sign = 1.0 if abs(tip_plus[2] - box_center[2]) < abs(tip_minus[2] - box_center[2]) else -1.0
    pad_tip_face = _face_grid(pad_center, pad_rot, 2, pad_z_sign * pad_half[2], (0, 1), (pad_half[0], pad_half[1]), n=n)

    box_near_x_sign = -1.0
    box_edge_plus = box_center + box_rot[:, 2] * box_half[2] + box_rot[:, 0] * (box_near_x_sign * box_half[0])
    box_edge_minus = box_center - box_rot[:, 2] * box_half[2] + box_rot[:, 0] * (box_near_x_sign * box_half[0])
    box_z_sign = 1.0 if abs(box_edge_plus[2] - pad_center[2]) < abs(box_edge_minus[2] - pad_center[2]) else -1.0
    box_edge = _edge_grid(
        box_center, box_rot, (0, 2), (box_near_x_sign * box_half[0], box_z_sign * box_half[2]), 1, box_half[1], n=n
    )
    diff_b = pad_tip_face[:, None, :] - box_edge[None, :, :]
    dist_b_mat = np.linalg.norm(diff_b, axis=2)
    ib, jb = np.unravel_index(np.argmin(dist_b_mat), dist_b_mat.shape)
    dist_b = float(dist_b_mat[ib, jb])
    pt_b_pad, pt_b_box = pad_tip_face[ib], box_edge[jb]

    return {
        "n": n,
        "dist_a": dist_a,
        "dist_b": dist_b,
        "m": dist_b - dist_a,
        "pad_x_sign": pad_x_sign,
        "pad_z_sign": pad_z_sign,
        "box_z_sign": box_z_sign,
        "closest_a_world": (pt_a_pad + pt_a_box) / 2,
        "closest_b_world": (pt_b_pad + pt_b_box) / 2,
        "grid_index_a": (int(ia), int(ja)),
        "grid_index_b": (int(ib), int(jb)),
    }


def convergence_at_time(poses: list[GeomPose], t_target: float, side: str = "left") -> dict:
    times = np.array([p.time_s for p in poses])
    idx = int(np.argmin(np.abs(times - t_target)))
    pose = poses[idx]

    results = {}
    for n in RESOLUTIONS + [CONTINUOUS_N]:
        results[n] = _clearances_at_resolution(pose, side, n)

    ref = results[CONTINUOUS_N]
    for n in RESOLUTIONS:
        results[n]["m_error_vs_continuous"] = results[n]["m"] - ref["m"]
        results[n]["sign_disagrees_with_continuous"] = (
            np.sign(results[n]["m"]) != np.sign(ref["m"]) and abs(ref["m"]) > 1e-9 and abs(results[n]["m"]) > 1e-9
        )

    return {"time_s": pose.time_s, "resolutions": results}


def analyze_seed(seed: int) -> dict:
    ctx, poses = run_poses(seed)
    if not poses:
        return {"seed": seed, "error": "no poses"}

    snap_result = ctx["snap_result"]
    bilateral = None
    for key, snap in snap_result["snapshots"].items():
        if key == "bilateral":
            bilateral = analyze_snapshot(seed, key, snap, snap_result["ids"]["box_geom"])
    contact_t = bilateral.get("time_s") if bilateral else None
    actual_mode = (
        "TIP_EDGE" if bilateral and bilateral.get("normal_vs_world_x_deg", 0) > 45 else "FACE_FACE"
    )

    checkpoints = {}
    if contact_t is not None:
        for lead in CHECKPOINT_LEADS_S:
            checkpoints[f"lead_{lead:.2f}s"] = convergence_at_time(poses, contact_t - lead)
        checkpoints["last_pre_contact_step"] = convergence_at_time(poses, contact_t - 1e-6)

    # full-resolution-ladder error stats over the whole final 0.25s window
    times = np.array([p.time_s for p in poses])
    window_start = (contact_t - 0.25) if contact_t is not None else times[0]
    window_poses = [p for p in poses if p.time_s >= window_start and (contact_t is None or p.time_s <= contact_t)]
    per_res_errors = {n: [] for n in RESOLUTIONS}
    sign_disagree_counts = {n: 0 for n in RESOLUTIONS}
    for p in window_poses[:: max(1, len(window_poses) // 60)]:  # subsample for speed
        conv = convergence_at_time([p], p.time_s)
        for n in RESOLUTIONS:
            per_res_errors[n].append(conv["resolutions"][n]["m_error_vs_continuous"])
            if conv["resolutions"][n]["sign_disagrees_with_continuous"]:
                sign_disagree_counts[n] += 1

    error_stats = {}
    for n in RESOLUTIONS:
        errs = np.array(per_res_errors[n])
        error_stats[n] = {
            "max_abs_error_m": float(np.max(np.abs(errs))) if errs.size else None,
            "rms_error_m": float(np.sqrt(np.mean(errs**2))) if errs.size else None,
            "sign_disagreement_count": sign_disagree_counts[n],
            "n_samples": len(errs),
        }

    # physical vs grid-index switch classification over final 100ms
    recent = [p for p in poses if contact_t is not None and contact_t - p.time_s <= 0.10]
    physical_switches = []
    grid_switches = []
    prev = None
    for p in recent:
        cur = _clearances_at_resolution(p, "left", 5)
        if prev is not None:
            signs_same = (
                cur["pad_x_sign"] == prev["pad_x_sign"]
                and cur["pad_z_sign"] == prev["pad_z_sign"]
                and cur["box_z_sign"] == prev["box_z_sign"]
            )
            index_changed = cur["grid_index_b"] != prev["grid_index_b"]
            if index_changed:
                jump = float(np.linalg.norm(cur["closest_b_world"] - prev["closest_b_world"]))
                record = {
                    "time_s": p.time_s,
                    "prev_index": prev["grid_index_b"],
                    "new_index": cur["grid_index_b"],
                    "closest_point_jump_m": jump,
                    "margin_jump_m": cur["m"] - prev["m"],
                    "signs_unchanged": signs_same,
                }
                if signs_same:
                    grid_switches.append(record)
                else:
                    physical_switches.append(record)
        prev = cur

    final_pose = poses[-1]
    final_ref = _clearances_at_resolution(final_pose, "left", CONTINUOUS_N)
    final_coarse = _clearances_at_resolution(final_pose, "left", RESOLUTIONS[0])

    return {
        "seed": seed,
        "actual_contact_mode": actual_mode,
        "actual_contact_time_s": contact_t,
        "checkpoints": checkpoints,
        "window_error_stats": error_stats,
        "physical_switches_last_100ms": physical_switches,
        "grid_switches_last_100ms": grid_switches,
        "final_m_continuous_n80": final_ref["m"],
        "final_m_coarse_n5": final_coarse["m"],
        "final_m_error_coarse_vs_continuous": final_coarse["m"] - final_ref["m"],
    }


def main() -> None:
    all_results = []
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        a = analyze_seed(seed)
        all_results.append(a)
        print(f"  actual_contact_mode: {a.get('actual_contact_mode')}")
        print(f"  actual_contact_time_s: {a.get('actual_contact_time_s')}")
        print(f"  final_m_continuous_n80: {a.get('final_m_continuous_n80')}")
        print(f"  final_m_coarse_n5: {a.get('final_m_coarse_n5')}")
        print(f"  final_m_error_coarse_vs_continuous: {a.get('final_m_error_coarse_vs_continuous')}")
        print(f"  window_error_stats: {a.get('window_error_stats')}")
        print(f"  physical_switches_last_100ms: {a.get('physical_switches_last_100ms')}")
        print(f"  grid_switches_last_100ms (count={len(a.get('grid_switches_last_100ms', []))}): "
              f"{a.get('grid_switches_last_100ms')}")
        for cp_name, cp in a.get("checkpoints", {}).items():
            if cp_name == "last_pre_contact_step":
                continue
            res5 = cp["resolutions"][5]
            res80 = cp["resolutions"][CONTINUOUS_N]
            print(f"  -- {cp_name}: m_n5={res5['m']:.5f} m_n80={res80['m']:.5f} "
                  f"error={res5['m_error_vs_continuous']:.5f} sign_disagree={res5['sign_disagrees_with_continuous']} --")
        print(flush=True)

    print("=== cross-seed final-margin robustness ===")
    for a in all_results:
        print(f"  seed={a['seed']} mode={a.get('actual_contact_mode')} "
              f"m_n5={a.get('final_m_coarse_n5')} m_n80={a.get('final_m_continuous_n80')} "
              f"error={a.get('final_m_error_coarse_vs_continuous')}")


if __name__ == "__main__":
    main()
