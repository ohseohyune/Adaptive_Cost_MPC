"""Read-only diagnosis of how badly the current 1D pad_plane_x contact
model misrepresents the true 3D collision geometry.

At each side's actual first-contact instant (and at bilateral contact),
records the true MuJoCo contact (position, normal, geom pair) plus both
collision geoms' full world pose/size, then computes two contact-gap
measures:

  d_x = x_box_surface - x_pad_plane          (current 1D model, x-axis only,
                                               using global x-extents)
  d_n = n . (p_box_support - p_pad_support)  (true geometry: support points
                                               of each box-shaped geom along
                                               the *actual* contact normal,
                                               not geom center or a global
                                               x-corner)

Box/pad support point along a direction d (standard box support function):
  local_d = R^T @ d
  support  = center + R @ (sign(local_d) * half_extents)

Does not touch production control, predictor, trigger, phase gate, target,
or controller code -- purely reads contact.{left,right}.contacts (already
computed by read_bilateral_pad_contact) and raw MuJoCo geom pose/size at
the instant of contact.

Usage: python3 acmpc/diagnose_3d_contact_geometry.py
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
from acmpc.main_acmpc_box_catch import run_box_catch

EDGE_RATIO = 0.7  # matches diagnose_precontact_grasp_offset.py's _classify_contact threshold


def _geom_pose(model, data, geom_id: int):
    center = data.geom_xpos[geom_id].copy()
    rot = data.geom_xmat[geom_id].reshape(3, 3).copy()
    half = model.geom_size[geom_id].copy()
    return center, rot, half


def _box_support_point(center: np.ndarray, rot: np.ndarray, half: np.ndarray, direction: np.ndarray) -> np.ndarray:
    local_dir = rot.T @ direction
    signs = np.sign(local_dir)
    signs[signs == 0] = 1.0
    return center + rot @ (signs * half)


def _geom_world_xrange(center: np.ndarray, rot: np.ndarray, half: np.ndarray) -> tuple[float, float]:
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    world_corners = center + (signs * half) @ rot.T
    xs = world_corners[:, 0]
    return float(xs.min()), float(xs.max())


def _classify(local_xz: np.ndarray, half_x: float, half_z: float) -> str:
    near_x = abs(local_xz[0]) >= EDGE_RATIO * half_x
    near_z = abs(local_xz[1]) >= EDGE_RATIO * half_z
    if near_x and near_z:
        return "CORNER"
    if near_x or near_z:
        return "EDGE"
    return "FACE"


@dataclass
class ContactSnapshot:
    time_s: float
    contact_pos: np.ndarray
    contact_normal: np.ndarray  # geom1 -> geom2, world frame
    geom1: int
    geom2: int
    box_center: np.ndarray
    box_rot: np.ndarray
    box_half: np.ndarray
    pad_center: np.ndarray
    pad_rot: np.ndarray
    pad_half: np.ndarray
    pad_plane_x: float


def run_and_snapshot(seed: int) -> dict:
    seed_everything(seed)
    cfg = _build_stage1_config(seed)
    ids = {}
    snapshots: dict[str, ContactSnapshot] = {}

    def on_step(ctx: dict) -> None:
        model, data = ctx["model"], ctx["data"]
        if not ids:
            ids["box_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
            ids["pad_geom"] = {
                side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_catch_pad")
                for side in ("left", "right")
            }

        contact = ctx["contact"]
        for side, measurement in (("left", contact.left), ("right", contact.right)):
            key = f"first_{side}"
            if key not in snapshots and measurement.active and measurement.contacts:
                c = measurement.contacts[0]
                box_center, box_rot, box_half = _geom_pose(model, data, ids["box_geom"])
                pad_center, pad_rot, pad_half = _geom_pose(model, data, ids["pad_geom"][side])
                snapshots[key] = ContactSnapshot(
                    time_s=ctx["time_s"],
                    contact_pos=c.pos.copy(),
                    contact_normal=c.normal.copy(),
                    geom1=c.geom1,
                    geom2=c.geom2,
                    box_center=box_center,
                    box_rot=box_rot,
                    box_half=box_half,
                    pad_center=pad_center,
                    pad_rot=pad_rot,
                    pad_half=pad_half,
                    pad_plane_x=float(ctx["pad_plane_x"]),
                )

        if "bilateral" not in snapshots and contact.left.active and contact.right.active:
            # box-local midpoint of both contact patches, using the side
            # whose contacts list is non-empty (both should be non-empty
            # once bilateral by construction of read_bilateral_pad_contact).
            side = "left" if contact.left.contacts else "right"
            c = contact.left.contacts[0] if contact.left.contacts else contact.right.contacts[0]
            box_center, box_rot, box_half = _geom_pose(model, data, ids["box_geom"])
            pad_center, pad_rot, pad_half = _geom_pose(model, data, ids["pad_geom"][side])
            snapshots["bilateral"] = ContactSnapshot(
                time_s=ctx["time_s"],
                contact_pos=c.pos.copy(),
                contact_normal=c.normal.copy(),
                geom1=c.geom1,
                geom2=c.geom2,
                box_center=box_center,
                box_rot=box_rot,
                box_half=box_half,
                pad_center=pad_center,
                pad_rot=pad_rot,
                pad_half=pad_half,
                pad_plane_x=float(ctx["pad_plane_x"]),
            )

    summary = run_box_catch(cfg, step_callback=on_step)
    return {"summary": summary, "snapshots": snapshots, "cfg": cfg, "ids": ids}


def analyze_snapshot(seed: int, key: str, snap: ContactSnapshot, box_geom_id: int) -> dict:
    box_is_geom1 = snap.geom1 == box_geom_id
    # canonicalize: n_box_to_pad points from box toward pad.
    n_box_to_pad = snap.contact_normal if box_is_geom1 else -snap.contact_normal
    n_box_to_pad = n_box_to_pad / (np.linalg.norm(n_box_to_pad) + 1e-12)

    box_support = _box_support_point(snap.box_center, snap.box_rot, snap.box_half, n_box_to_pad)
    pad_support = _box_support_point(snap.pad_center, snap.pad_rot, snap.pad_half, -n_box_to_pad)
    d_n = float(n_box_to_pad @ (box_support - pad_support))

    box_xmin, box_xmax = _geom_world_xrange(snap.box_center, snap.box_rot, snap.box_half)
    x_box_surface = box_xmin  # box approaches from +x, its near/-x face is what the pad model targets
    d_x = x_box_surface - snap.pad_plane_x

    contact_box_local = snap.box_rot.T @ (snap.contact_pos - snap.box_center)
    contact_pad_local = snap.pad_rot.T @ (snap.contact_pos - snap.pad_center)

    classification = _classify(
        np.array([contact_box_local[0], contact_box_local[2]]), snap.box_half[0], snap.box_half[2]
    )

    pad_xmin, pad_xmax = _geom_world_xrange(snap.pad_center, snap.pad_rot, snap.pad_half)
    pad_surface_x_toward_box = pad_xmax if abs(pad_xmax - snap.box_center[0]) < abs(pad_xmin - snap.box_center[0]) else pad_xmin
    global_x_extreme_vs_actual_contact_x = pad_surface_x_toward_box - float(snap.contact_pos[0])

    world_x_axis = np.array([1.0, 0.0, 0.0])
    normal_vs_world_x_deg = float(np.degrees(np.arccos(np.clip(abs(n_box_to_pad @ world_x_axis), -1.0, 1.0))))

    return {
        "seed": seed,
        "event": key,
        "time_s": snap.time_s,
        "contact_pos_world_xyz": snap.contact_pos.tolist(),
        "contact_pos_box_local_xyz": contact_box_local.tolist(),
        "contact_pos_pad_local_xyz": contact_pad_local.tolist(),
        "contact_normal_box_to_pad": n_box_to_pad.tolist(),
        "box_center": snap.box_center.tolist(),
        "box_half_extents": snap.box_half.tolist(),
        "pad_center": snap.pad_center.tolist(),
        "pad_half_extents": snap.pad_half.tolist(),
        "pad_plane_x_used": snap.pad_plane_x,
        "pad_geom_world_xmin": pad_xmin,
        "pad_geom_world_xmax": pad_xmax,
        "box_geom_world_xmin": box_xmin,
        "box_geom_world_xmax": box_xmax,
        "box_support_point": box_support.tolist(),
        "pad_support_point": pad_support.tolist(),
        "d_x_1d_model_m": d_x,
        "d_n_true_normal_gap_m": d_n,
        "global_x_extreme_vs_actual_contact_x_m": global_x_extreme_vs_actual_contact_x,
        "contact_classification": classification,
        "normal_vs_world_x_deg": normal_vs_world_x_deg,
    }


def main() -> None:
    all_rows = []
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        result = run_and_snapshot(seed)
        for key, snap in result["snapshots"].items():
            row = analyze_snapshot(seed, key, snap, result["ids"]["box_geom"])
            all_rows.append(row)
            print(f"  -- {key} --")
            for k, v in row.items():
                print(f"    {k}: {v}")
        print(flush=True)

    print("=== cross-seed comparison (bilateral event) ===")
    for row in all_rows:
        if row["event"] != "bilateral":
            continue
        print(
            f"  seed={row['seed']} d_x={row['d_x_1d_model_m']:.4f} d_n={row['d_n_true_normal_gap_m']:.4f} "
            f"global_x_vs_actual={row['global_x_extreme_vs_actual_contact_x_m']:.4f} "
            f"class={row['contact_classification']} normal_vs_x_deg={row['normal_vs_world_x_deg']:.2f}"
        )


if __name__ == "__main__":
    main()
