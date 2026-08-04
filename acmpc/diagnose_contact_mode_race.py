"""Read-only pre-contact diagnosis: which collision feature-pair reaches
zero clearance first -- the intended pad wide-face <-> box grasp-face
contact (A), or the unintended pad tip/edge <-> box edge contact (B) --
and does that predicted "race" match the actual first-contact mode?

Uses real OBB geometry (grid-sampled point clouds over each candidate
feature, transformed by the geom's actual world pose every step), not a
global x-plane or geom centers:

  A. pad wide face  (pad_local x = +/-pad_half_x, spans y,z)
     vs box grasp face (box_local y = +/-box_half_y, spans x,z)
  B. pad tip face   (pad_local z = +/-pad_half_z, spans x,y -- the pad's
     thin z-extreme, found in diagnose_3d_contact_geometry.py to be what
     actually caught seed 1017's box edge)
     vs box edge     (box_local x = -box_half_x AND z = +/-box_half_z,
     varies along y only -- a true 1D edge)

Per-candidate minimum clearance is the smallest pairwise distance between
the two sampled point clouds; closing speed is its finite-difference rate
of change; TTC_candidate = clearance / closing_speed when closing.

Compares against the actual first-contact geom pair/point/normal (reusing
diagnose_3d_contact_geometry.py's support-point-based mode classification).

Does not touch TTC model, target, pad orientation, phase gate, or
controller gain code -- purely reads raw geom pose/size every substep of
the existing baseline run (no target_override_fn).

Usage: python3 acmpc/diagnose_contact_mode_race.py
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
from acmpc.diagnose_3d_contact_geometry import _box_support_point, run_and_snapshot, analyze_snapshot
from acmpc.main_acmpc_box_catch import run_box_catch

GRID_N = 5


def _face_grid(center, rot, fixed_axis, fixed_value, span_axes, spans, n=GRID_N) -> np.ndarray:
    a0, a1 = span_axes
    h0, h1 = spans
    u = np.linspace(-h0, h0, n)
    v = np.linspace(-h1, h1, n)
    U, V = np.meshgrid(u, v)
    local = np.zeros((n * n, 3))
    local[:, fixed_axis] = fixed_value
    local[:, a0] = U.ravel()
    local[:, a1] = V.ravel()
    return center + local @ rot.T


def _edge_grid(center, rot, fixed_axes, fixed_values, span_axis, span_half, n=GRID_N) -> np.ndarray:
    a0, a1 = fixed_axes
    v0, v1 = fixed_values
    u = np.linspace(-span_half, span_half, n)
    local = np.zeros((n, 3))
    local[:, a0] = v0
    local[:, a1] = v1
    local[:, span_axis] = u
    return center + local @ rot.T


def _min_pairwise_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return float(dist[i, j]), pts_a[i], pts_b[j]


def _candidate_clearances(box_center, box_rot, box_half, pad_center, pad_rot, pad_half, side: str) -> dict:
    y_sign = 1.0 if side == "left" else -1.0

    # A: pad wide face (box-facing) vs box grasp face
    pad_x_sign = 1.0 if (pad_center + pad_rot[:, 0] * pad_half[0])[0] < (pad_center - pad_rot[:, 0] * pad_half[0])[0] else -1.0
    # choose the pad-face sign nearer the box along world x
    face_plus = pad_center + pad_rot[:, 0] * pad_half[0]
    face_minus = pad_center - pad_rot[:, 0] * pad_half[0]
    pad_x_sign = 1.0 if abs(face_plus[0] - box_center[0]) < abs(face_minus[0] - box_center[0]) else -1.0

    pad_wide_face = _face_grid(pad_center, pad_rot, 0, pad_x_sign * pad_half[0], (1, 2), (pad_half[1], pad_half[2]))
    box_grasp_face = _face_grid(box_center, box_rot, 1, y_sign * box_half[1], (0, 2), (box_half[0], box_half[2]))
    dist_a, pt_a_pad, pt_a_box = _min_pairwise_distance(pad_wide_face, box_grasp_face)

    # B: pad tip face (z-extreme nearer box) vs box edge (near/-x face x z-extreme nearer pad)
    tip_plus = pad_center + pad_rot[:, 2] * pad_half[2]
    tip_minus = pad_center - pad_rot[:, 2] * pad_half[2]
    pad_z_sign = 1.0 if abs(tip_plus[2] - box_center[2]) < abs(tip_minus[2] - box_center[2]) else -1.0
    pad_tip_face = _face_grid(pad_center, pad_rot, 2, pad_z_sign * pad_half[2], (0, 1), (pad_half[0], pad_half[1]))

    box_near_x_sign = -1.0  # box approaches from +x, its -x face is the leading edge
    box_edge_plus = box_center + box_rot[:, 2] * box_half[2] + box_rot[:, 0] * (box_near_x_sign * box_half[0])
    box_edge_minus = box_center - box_rot[:, 2] * box_half[2] + box_rot[:, 0] * (box_near_x_sign * box_half[0])
    box_z_sign = 1.0 if abs(box_edge_plus[2] - pad_center[2]) < abs(box_edge_minus[2] - pad_center[2]) else -1.0
    box_edge = _edge_grid(
        box_center, box_rot, (0, 2), (box_near_x_sign * box_half[0], box_z_sign * box_half[2]), 1, box_half[1]
    )
    dist_b, pt_b_pad, pt_b_box = _min_pairwise_distance(pad_tip_face, box_edge)

    return {
        "dist_a": dist_a,
        "dist_b": dist_b,
        "feature_a_pad_point": pt_a_pad,
        "feature_a_box_point": pt_a_box,
        "feature_b_pad_point": pt_b_pad,
        "feature_b_box_point": pt_b_box,
    }


def run_race(seed: int) -> dict:
    seed_everything(seed)
    cfg = _build_stage1_config(seed)
    ids = {}
    series = {"left": [], "right": []}

    def on_step(ctx: dict) -> None:
        model, data = ctx["model"], ctx["data"]
        if not ids:
            ids["box_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
            ids["pad_geom"] = {
                s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_catch_pad") for s in ("left", "right")
            }

        contact = ctx["contact"]
        if contact.left.active and contact.right.active:
            return  # stop once bilateral contact has formed; race is a pre-contact question

        box_center = data.geom_xpos[ids["box_geom"]].copy()
        box_rot = data.geom_xmat[ids["box_geom"]].reshape(3, 3).copy()
        box_half = model.geom_size[ids["box_geom"]].copy()

        for side in ("left", "right"):
            pad_center = data.geom_xpos[ids["pad_geom"][side]].copy()
            pad_rot = data.geom_xmat[ids["pad_geom"][side]].reshape(3, 3).copy()
            pad_half = model.geom_size[ids["pad_geom"][side]].copy()
            clearances = _candidate_clearances(box_center, box_rot, box_half, pad_center, pad_rot, pad_half, side)
            series[side].append({"time_s": ctx["time_s"], **clearances})

    summary = run_box_catch(cfg, step_callback=on_step)
    snapshot_result = run_and_snapshot(seed)  # separate deterministic rerun for actual-contact ground truth
    return {"summary": summary, "series": series, "snapshot_result": snapshot_result, "ids": ids}


def _ttc_series(rows: list[dict], key_dist: str) -> list[dict]:
    times = np.array([r["time_s"] for r in rows])
    dist = np.array([r[key_dist] for r in rows])
    closing_speed = np.zeros(len(dist))
    for i in range(1, len(dist)):
        dt = times[i] - times[i - 1]
        if dt > 1e-9:
            closing_speed[i] = -(dist[i] - dist[i - 1]) / dt
    if len(dist) > 1:
        closing_speed[0] = closing_speed[1]
    ttc = np.where(closing_speed > 1e-4, dist / np.maximum(closing_speed, 1e-9), np.inf)
    out = []
    for i, r in enumerate(rows):
        out.append({"time_s": r["time_s"], "dist": float(dist[i]), "closing_speed": float(closing_speed[i]), "ttc": float(ttc[i])})
    return out


def analyze(seed: int, result: dict) -> dict:
    side = "left"
    rows = result["series"][side]
    if not rows:
        return {"seed": seed, "error": "no pre-contact samples"}

    ttc_a = _ttc_series(rows, "dist_a")
    ttc_b = _ttc_series(rows, "dist_b")

    # first time B's TTC drops below A's TTC (tip-edge predicted to win the race).
    # Noisy right after episode start (pad settling motion produces spurious
    # transient closing speeds), so also track the more robust raw-clearance
    # leader (dist_b < dist_a) and report the *last* switch before contact,
    # which is far less sensitive to that initial transient.
    crossover_t = None
    for i in range(len(ttc_a)):
        if ttc_b[i]["ttc"] < ttc_a[i]["ttc"]:
            crossover_t = ttc_a[i]["time_s"]
            break

    clearance_leader_b = [r["dist_b"] < r["dist_a"] for r in rows]
    last_clearance_switch_t = None
    for i in range(1, len(rows)):
        if clearance_leader_b[i] != clearance_leader_b[i - 1]:
            last_clearance_switch_t = rows[i]["time_s"]
    stable_b_leads_from_t = None
    for i in range(len(rows) - 1, -1, -1):
        if not clearance_leader_b[i]:
            stable_b_leads_from_t = rows[i + 1]["time_s"] if i + 1 < len(rows) else None
            break
    else:
        stable_b_leads_from_t = rows[0]["time_s"] if rows else None

    final_a = ttc_a[-1]
    final_b = ttc_b[-1]
    predicted_mode = "TIP_EDGE_CONTACT_PREDICTED" if final_b["dist"] < final_a["dist"] else "FACE_CONTACT_PREDICTED"

    # actual mode from the independent ground-truth rerun
    snap_result = result["snapshot_result"]
    actual_rows = {}
    for key, snap in snap_result["snapshots"].items():
        row = analyze_snapshot(seed, key, snap, snap_result["ids"]["box_geom"])
        actual_rows[key] = row

    bilateral = actual_rows.get("bilateral", {})
    normal_deg = bilateral.get("normal_vs_world_x_deg")
    if normal_deg is not None and normal_deg > 45:
        actual_mode = "TIP_EDGE"
    else:
        actual_mode = "FACE_FACE"

    match = (
        (predicted_mode == "TIP_EDGE_CONTACT_PREDICTED" and actual_mode == "TIP_EDGE")
        or (predicted_mode == "FACE_CONTACT_PREDICTED" and actual_mode == "FACE_FACE")
    )

    return {
        "seed": seed,
        "final_dist_a_face_m": final_a["dist"],
        "final_dist_b_tip_edge_m": final_b["dist"],
        "final_ttc_a_face_s": final_a["ttc"],
        "final_ttc_b_tip_edge_s": final_b["ttc"],
        "tip_edge_predicted_lead_s": (final_a["ttc"] - final_b["ttc"]) if np.isfinite(final_a["ttc"]) and np.isfinite(final_b["ttc"]) else None,
        "crossover_time_s_ttc_based_noisy": crossover_t,
        "stable_clearance_b_leads_from_time_s": stable_b_leads_from_t,
        "last_clearance_leader_switch_time_s": last_clearance_switch_t,
        "predicted_first_contact_mode": predicted_mode,
        "actual_first_contact_mode": actual_mode,
        "actual_normal_vs_world_x_deg": normal_deg,
        "actual_contact_classification": bilateral.get("contact_classification"),
        "predicted_matches_actual": match,
    }


def main() -> None:
    all_results = []
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        result = run_race(seed)
        summary = analyze(seed, result)
        all_results.append(summary)
        for k, v in summary.items():
            print(f"  {k}: {v}")

        rows = result["series"]["left"]
        ttc_a = _ttc_series(rows, "dist_a")
        ttc_b = _ttc_series(rows, "dist_b")
        print("  -- clearance/TTC race (left pad, every 20ms) --")
        print(f"  {'time_s':>7} {'dist_A':>8} {'dist_B':>8} {'ttc_A':>8} {'ttc_B':>8} {'leader':>10}")
        last_t = None
        for i in range(len(rows)):
            t = rows[i]["time_s"]
            if last_t is not None and t - last_t < 0.02:
                continue
            last_t = t
            leader = "B(tip)" if ttc_b[i]["ttc"] < ttc_a[i]["ttc"] else "A(face)"
            print(
                f"  {t:7.3f} {ttc_a[i]['dist']:8.4f} {ttc_b[i]['dist']:8.4f} "
                f"{(ttc_a[i]['ttc'] if np.isfinite(ttc_a[i]['ttc']) else -1):8.3f} "
                f"{(ttc_b[i]['ttc'] if np.isfinite(ttc_b[i]['ttc']) else -1):8.3f} {leader:>10}"
            )
        print(flush=True)

    print("=== cross-seed summary ===")
    for s in all_results:
        print(f"  seed={s['seed']} predicted={s.get('predicted_first_contact_mode')} "
              f"actual={s.get('actual_first_contact_mode')} match={s.get('predicted_matches_actual')} "
              f"crossover_t={s.get('crossover_time_s')}")


if __name__ == "__main__":
    main()
