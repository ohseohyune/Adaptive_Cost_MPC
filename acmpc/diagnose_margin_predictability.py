"""Read-only diagnosis: can the FACE vs TIP_EDGE contact-mode margin
m(t) = d_B(t) - d_A(t) (see diagnose_contact_mode_race.py) be predicted to
cross zero 50-200ms *before* actual contact, using only causal (past-only)
data -- or does it only resolve in the last few ms, too late for a 10ms
control cycle to act on?

Reuses the exact same OBB feature distances (d_A: pad-face<->box-face,
d_B: pad-tip<->box-edge) as diagnose_contact_mode_race.py. Adds:
  - causal derivative/second-derivative estimates (short trailing linear/
    quadratic regression, no future data)
  - a linear crossing-time predictor T_cross = -m/m_dot, gated by
    m>0, m_dot<0, T_cross>0, |m_dot| not too small
  - a quadratic refinement compared against the linear predictor
  - supporting relative-geometry variables (z offset, normal misalignment,
    footprint containment, closing speeds) at the same checkpoints
  - closest-grid-point index tracking to flag discontinuous feature
    switches vs a continuous trend

Does not touch target, TTC predictor, trigger, phase gate, pad
orientation, controller gain, or production trajectory -- purely reads
the existing baseline (no target_override_fn) run.

Usage: python3 acmpc/diagnose_margin_predictability.py
"""

from __future__ import annotations

import sys
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

CHECKPOINT_LEADS_S = [0.20, 0.15, 0.10, 0.05]
MIN_CONTROL_CYCLE_S = 0.010
DERIV_WINDOW = 5  # causal regression window, in control-step samples
MIN_ABS_MDOT_MPS = 0.02


def _min_pairwise_distance_indexed(pts_a: np.ndarray, pts_b: np.ndarray):
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    flat_idx = int(np.argmin(dist))
    i, j = np.unravel_index(flat_idx, dist.shape)
    return float(dist[i, j]), int(i), int(j), pts_a[i], pts_b[j]


def _candidate_clearances_indexed(box_center, box_rot, box_half, pad_center, pad_rot, pad_half, side: str) -> dict:
    y_sign = 1.0 if side == "left" else -1.0
    face_plus = pad_center + pad_rot[:, 0] * pad_half[0]
    face_minus = pad_center - pad_rot[:, 0] * pad_half[0]
    pad_x_sign = 1.0 if abs(face_plus[0] - box_center[0]) < abs(face_minus[0] - box_center[0]) else -1.0

    pad_wide_face = _face_grid(pad_center, pad_rot, 0, pad_x_sign * pad_half[0], (1, 2), (pad_half[1], pad_half[2]))
    box_grasp_face = _face_grid(box_center, box_rot, 1, y_sign * box_half[1], (0, 2), (box_half[0], box_half[2]))
    dist_a, ia, ja, pt_a_pad, pt_a_box = _min_pairwise_distance_indexed(pad_wide_face, box_grasp_face)

    tip_plus = pad_center + pad_rot[:, 2] * pad_half[2]
    tip_minus = pad_center - pad_rot[:, 2] * pad_half[2]
    pad_z_sign = 1.0 if abs(tip_plus[2] - box_center[2]) < abs(tip_minus[2] - box_center[2]) else -1.0
    pad_tip_face = _face_grid(pad_center, pad_rot, 2, pad_z_sign * pad_half[2], (0, 1), (pad_half[0], pad_half[1]))

    box_near_x_sign = -1.0
    box_edge_plus = box_center + box_rot[:, 2] * box_half[2] + box_rot[:, 0] * (box_near_x_sign * box_half[0])
    box_edge_minus = box_center - box_rot[:, 2] * box_half[2] + box_rot[:, 0] * (box_near_x_sign * box_half[0])
    box_z_sign = 1.0 if abs(box_edge_plus[2] - pad_center[2]) < abs(box_edge_minus[2] - pad_center[2]) else -1.0
    box_edge = _edge_grid(
        box_center, box_rot, (0, 2), (box_near_x_sign * box_half[0], box_z_sign * box_half[2]), 1, box_half[1]
    )
    dist_b, ib, jb, pt_b_pad, pt_b_box = _min_pairwise_distance_indexed(pad_tip_face, box_edge)

    # supporting relative-geometry variables
    z_offset = float(box_center[2] - pad_center[2])
    pad_face_normal = pad_x_sign * pad_rot[:, 0]
    box_face_normal = y_sign * box_rot[:, 1]
    normal_angle_deg = float(np.degrees(np.arccos(np.clip(pad_face_normal @ box_face_normal, -1.0, 1.0))))
    tip_edge_height_diff = float(pt_b_pad[2] - pt_b_box[2])

    # relative roll/pitch: angle of the relative rotation pad_rot^T @ box_rot
    r_rel = pad_rot.T @ box_rot
    trace = np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0)
    misalignment_deg = float(np.degrees(np.arccos(trace)))

    # pad tip footprint inside/outside box face footprint (project tip
    # point onto box-local x,z and compare to box half extents)
    tip_box_local = box_rot.T @ (pt_b_pad - box_center)
    tip_inside_box_face = bool(abs(tip_box_local[0]) <= box_half[0] and abs(tip_box_local[2]) <= box_half[2])

    return {
        "dist_a": dist_a,
        "dist_b": dist_b,
        "grid_index_a": (ia, ja),
        "grid_index_b": (ib, jb),
        "z_offset_box_minus_pad": z_offset,
        "normal_angle_deg": normal_angle_deg,
        "tip_edge_height_diff_m": tip_edge_height_diff,
        "misalignment_deg": misalignment_deg,
        "tip_inside_box_face_footprint": tip_inside_box_face,
    }


def run_series(seed: int) -> dict:
    seed_everything(seed)
    cfg = _build_stage1_config(seed)
    ids = {}
    rows = {"left": [], "right": []}

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

        for side in ("left", "right"):
            pad_center = data.geom_xpos[ids["pad_geom"][side]].copy()
            pad_rot = data.geom_xmat[ids["pad_geom"][side]].reshape(3, 3).copy()
            pad_half = model.geom_size[ids["pad_geom"][side]].copy()
            extra = _candidate_clearances_indexed(box_center, box_rot, box_half, pad_center, pad_rot, pad_half, side)
            rows[side].append({"time_s": ctx["time_s"], **extra})

    summary = run_box_catch(cfg, step_callback=on_step)
    snap_result = run_and_snapshot(seed)
    return {"summary": summary, "rows": rows, "snap_result": snap_result, "ids": ids}


def _causal_linreg(times: np.ndarray, values: np.ndarray, i: int, window: int, degree: int):
    lo = max(0, i - window + 1)
    t = times[lo : i + 1]
    v = values[lo : i + 1]
    if len(t) < degree + 1:
        return None
    t0 = t - t[-1]  # center on current time so coefficients are local slope/curvature at t[i]
    coeffs = np.polyfit(t0, v, degree)
    return coeffs  # highest degree first


def build_predictions(rows: list[dict]) -> list[dict]:
    times = np.array([r["time_s"] for r in rows])
    dist_a = np.array([r["dist_a"] for r in rows])
    dist_b = np.array([r["dist_b"] for r in rows])
    m = dist_b - dist_a

    out = []
    for i in range(len(rows)):
        lin_a = _causal_linreg(times, dist_a, i, DERIV_WINDOW, 1)
        lin_b = _causal_linreg(times, dist_b, i, DERIV_WINDOW, 1)
        dot_a = float(lin_a[0]) if lin_a is not None else 0.0
        dot_b = float(lin_b[0]) if lin_b is not None else 0.0
        m_dot = dot_b - dot_a

        quad_m = _causal_linreg(times, m, i, DERIV_WINDOW, 2)
        m_ddot = float(2.0 * quad_m[0]) if quad_m is not None else None

        m_t = float(m[i])
        valid = m_t > 0 and m_dot < 0 and abs(m_dot) >= MIN_ABS_MDOT_MPS
        t_cross_linear = (-m_t / m_dot) if valid else None
        if t_cross_linear is not None and t_cross_linear <= 0:
            t_cross_linear = None
            valid = False

        t_cross_quadratic = None
        if valid and m_ddot is not None and abs(m_ddot) > 1e-6:
            # solve m + m_dot*tau + 0.5*m_ddot*tau^2 = 0 for smallest positive tau
            a, b, c = 0.5 * m_ddot, m_dot, m_t
            disc = b * b - 4 * a * c
            if disc >= 0:
                sq = np.sqrt(disc)
                roots = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
                positive_roots = [r for r in roots if r > 0]
                if positive_roots:
                    t_cross_quadratic = float(min(positive_roots))

        out.append(
            {
                **rows[i],
                "m": m_t,
                "dot_dist_a": dot_a,
                "dot_dist_b": dot_b,
                "m_dot": m_dot,
                "m_ddot": m_ddot,
                "t_cross_linear_valid": valid,
                "t_cross_linear_s": t_cross_linear,
                "t_cross_quadratic_s": t_cross_quadratic,
            }
        )
    return out


def analyze(seed: int, result: dict) -> dict:
    rows = result["rows"]["left"]
    if not rows:
        return {"seed": seed, "error": "no series"}
    preds = build_predictions(rows)
    times = np.array([p["time_s"] for p in preds])

    snap_result = result["snap_result"]
    actual_rows = {
        key: analyze_snapshot(seed, key, snap, snap_result["ids"]["box_geom"])
        for key, snap in snap_result["snapshots"].items()
    }
    bilateral = actual_rows.get("bilateral", {})
    contact_t = bilateral.get("time_s")
    normal_deg = bilateral.get("normal_vs_world_x_deg")
    actual_mode = "TIP_EDGE" if (normal_deg is not None and normal_deg > 45) else "FACE_FACE"

    checkpoints = {}
    for lead in CHECKPOINT_LEADS_S:
        if contact_t is None:
            break
        target_t = contact_t - lead
        idx = int(np.argmin(np.abs(times - target_t)))
        p = preds[idx]
        predicted_crossing_before_contact = (
            p["t_cross_linear_valid"]
            and p["t_cross_linear_s"] is not None
            and p["t_cross_linear_s"] < (contact_t - p["time_s"])
            and p["t_cross_linear_s"] >= MIN_CONTROL_CYCLE_S
        )
        predicted_margin_at_contact = (
            (p["m"] + p["m_dot"] * (contact_t - p["time_s"]))
            if p["time_s"] <= contact_t
            else None
        )
        predicted_mode = "TIP_EDGE" if (predicted_margin_at_contact is not None and predicted_margin_at_contact < 0) else "FACE"
        checkpoints[f"lead_{lead:.2f}s"] = {
            "time_s": p["time_s"],
            "d_A": p["dist_a"],
            "d_B": p["dist_b"],
            "m": p["m"],
            "dot_d_A": p["dot_dist_a"],
            "dot_d_B": p["dot_dist_b"],
            "m_dot": p["m_dot"],
            "m_ddot": p["m_ddot"],
            "z_offset_box_minus_pad": p["z_offset_box_minus_pad"],
            "normal_angle_deg": p["normal_angle_deg"],
            "misalignment_deg": p["misalignment_deg"],
            "tip_edge_height_diff_m": p["tip_edge_height_diff_m"],
            "tip_inside_box_face_footprint": p["tip_inside_box_face_footprint"],
            "grid_index_a": p["grid_index_a"],
            "grid_index_b": p["grid_index_b"],
            "t_cross_linear_valid": p["t_cross_linear_valid"],
            "t_cross_linear_s": p["t_cross_linear_s"],
            "t_cross_quadratic_s": p["t_cross_quadratic_s"],
            "predicted_margin_at_contact": predicted_margin_at_contact,
            "predicted_crossing_before_contact_with_margin": predicted_crossing_before_contact,
            "predicted_mode_at_this_lead": predicted_mode,
        }

    # last control step before contact (~10ms)
    if contact_t is not None:
        idx_last = int(np.argmin(np.abs(times - (contact_t - MIN_CONTROL_CYCLE_S))))
        p = preds[idx_last]
        checkpoints["last_control_step"] = {
            "time_s": p["time_s"],
            "d_A": p["dist_a"],
            "d_B": p["dist_b"],
            "m": p["m"],
            "m_dot": p["m_dot"],
        }

    # feature-switch discontinuity check: how often does grid_index_b jump
    # between consecutive samples in the final 100ms before contact?
    switch_count = 0
    if contact_t is not None:
        recent = [p for p in preds if contact_t - p["time_s"] <= 0.10]
        for i in range(1, len(recent)):
            if recent[i]["grid_index_b"] != recent[i - 1]["grid_index_b"]:
                switch_count += 1

    final_m = preds[-1]["m"] if preds else None

    return {
        "seed": seed,
        "actual_contact_mode": actual_mode,
        "actual_contact_time_s": contact_t,
        "final_m_at_last_sample": final_m,
        "grid_index_b_switch_count_last_100ms": switch_count,
        "checkpoints": checkpoints,
    }


def main() -> None:
    all_results = []
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        result = run_series(seed)
        a = analyze(seed, result)
        all_results.append(a)
        print(f"  actual_contact_mode: {a.get('actual_contact_mode')}")
        print(f"  actual_contact_time_s: {a.get('actual_contact_time_s')}")
        print(f"  final_m_at_last_sample: {a.get('final_m_at_last_sample')}")
        print(f"  grid_index_b_switch_count_last_100ms: {a.get('grid_index_b_switch_count_last_100ms')}")
        for cp_name, cp in a.get("checkpoints", {}).items():
            print(f"  -- {cp_name} --")
            for k, v in cp.items():
                print(f"    {k}: {v}")
        print(flush=True)

    print("=== cross-seed summary (0.10s lead checkpoint) ===")
    for a in all_results:
        cp = a.get("checkpoints", {}).get("lead_0.10s", {})
        print(
            f"  seed={a['seed']} actual={a.get('actual_contact_mode')} "
            f"m={cp.get('m')} m_dot={cp.get('m_dot')} "
            f"t_cross_linear={cp.get('t_cross_linear_s')} "
            f"predicted_mode={cp.get('predicted_mode_at_this_lead')} "
            f"crossing_before_contact={cp.get('predicted_crossing_before_contact_with_margin')}"
        )


if __name__ == "__main__":
    main()
