"""Geometric decomposition of the ~4cm x-axis contact-prediction bias that
persists to contact for seeds 1000/1005/1001 but converges to ~0 for seed
1017 (see diagnose_target_accuracy_hindsight.py).

Read-only: does not touch production control, the predictor, or the
PRE_IMPACT gate. Records, every physics substep, the actual runtime
geometry (box body COM, box collision geom's true world x-extent via its
rotated corners -- not just COM +/- half_x, catch pad geoms' true world
x-extent the same way, EE/pad site positions, the predictor's own
box-half-width and pad-plane-x inputs) and the actual first-contact geom
pair/point, then decomposes:

  x_predicted_contact = x_pad_plane + h_box  (the corrected-target formula
                                               diagnose_precontact_grasp_offset.py
                                               and the shadow-rollout
                                               scripts have used throughout)
  e_x = x_predicted_contact - x_actual_contact

into named geometric components by comparing against the true collision
geometry read directly from MuJoCo at the contact instant.

Usage: python3 acmpc/diagnose_x_bias_geometry.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, _build_stage1_config, seed_everything
from acmpc.main_acmpc_box_catch import run_box_catch


def _geom_world_xrange(model, data, geom_id: int) -> tuple[float, float]:
    center = data.geom_xpos[geom_id]
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    half = model.geom_size[geom_id]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local_corners = signs * half
    world_corners = center + local_corners @ rot.T
    xs = world_corners[:, 0]
    return float(xs.min()), float(xs.max())


@dataclass
class Sample:
    time_s: float
    phase: str
    box_com_x: float
    box_geom_center_x: float
    box_half_x_runtime: float
    box_world_xmin: float
    box_world_xmax: float
    left_ee_x: float
    right_ee_x: float
    left_pad_site_x: float
    right_pad_site_x: float
    left_pad_world_xmin: float
    left_pad_world_xmax: float
    right_pad_world_xmin: float
    right_pad_world_xmax: float
    pad_thickness_half: float
    predictor_half_x: float
    pad_plane_x: float
    left_active: bool
    right_active: bool
    left_contact_x: float | None
    right_contact_x: float | None
    left_contact_geom1: int | None
    left_contact_geom2: int | None


@dataclass
class Recording:
    samples: list = field(default_factory=list)


def run_episode_geometry(seed: int) -> tuple[dict, Recording]:
    seed_everything(seed)
    cfg = _build_stage1_config(seed)
    rec = Recording()
    ids = {}

    def on_step(ctx: dict) -> None:
        model, data = ctx["model"], ctx["data"]
        if not ids:
            ids["box_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
            ids["box_body"] = ctx["box_body_id"]
            ids["pad_geom"] = {
                side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_catch_pad")
                for side in ("left", "right")
            }

        contact = ctx["contact"]
        box_xmin, box_xmax = _geom_world_xrange(model, data, ids["box_geom"])
        left_pad_xmin, left_pad_xmax = _geom_world_xrange(model, data, ids["pad_geom"]["left"])
        right_pad_xmin, right_pad_xmax = _geom_world_xrange(model, data, ids["pad_geom"]["right"])

        left_contact_x = None
        left_g1 = left_g2 = None
        if contact.left.active and contact.left.contacts:
            left_contact_x = float(np.mean([c.pos[0] for c in contact.left.contacts]))
            left_g1 = contact.left.contacts[0].geom1
            left_g2 = contact.left.contacts[0].geom2
        right_contact_x = None
        if contact.right.active and contact.right.contacts:
            right_contact_x = float(np.mean([c.pos[0] for c in contact.right.contacts]))

        rec.samples.append(
            Sample(
                time_s=ctx["time_s"],
                phase=ctx["phase"].value,
                box_com_x=float(data.xipos[ids["box_body"]][0]),
                box_geom_center_x=float(data.geom_xpos[ids["box_geom"]][0]),
                box_half_x_runtime=float(model.geom_size[ids["box_geom"]][0]),
                box_world_xmin=box_xmin,
                box_world_xmax=box_xmax,
                left_ee_x=float(ctx["left_ee"][0]),
                right_ee_x=float(ctx["right_ee"][0]),
                left_pad_site_x=float(data.site_xpos[ctx["catch_pad_site_ids"]["left"]][0]),
                right_pad_site_x=float(data.site_xpos[ctx["catch_pad_site_ids"]["right"]][0]),
                left_pad_world_xmin=left_pad_xmin,
                left_pad_world_xmax=left_pad_xmax,
                right_pad_world_xmin=right_pad_xmin,
                right_pad_world_xmax=right_pad_xmax,
                pad_thickness_half=float(model.geom_size[ids["pad_geom"]["left"]][0]),
                predictor_half_x=float(cfg.domain_parameters.half_size[0]) if cfg.domain_parameters is not None else float("nan"),
                pad_plane_x=float(ctx["pad_plane_x"]),
                left_active=bool(contact.left.active),
                right_active=bool(contact.right.active),
                left_contact_x=left_contact_x,
                right_contact_x=right_contact_x,
                left_contact_geom1=left_g1,
                left_contact_geom2=left_g2,
            )
        )

    summary = run_box_catch(cfg, step_callback=on_step)
    return {"summary": summary, "cfg": cfg}, rec


def analyze_geometry(seed: int, result: dict, rec: Recording) -> dict:
    samples = rec.samples
    model_names_needed = True  # resolved lazily below via a fresh model load only for name lookup
    first_left = next((s for s in samples if s.left_active), None)
    first_right = next((s for s in samples if s.right_active), None)
    bilateral = next((s for s in samples if s.left_active and s.right_active), None)
    if bilateral is None:
        return {"seed": seed, "error": "no bilateral contact"}

    ref = bilateral  # decompose at the actual contact instant

    # actual contact plane (near/-x box face touching the pad): use the
    # measured contact point x directly (ground truth, no model needed).
    x_actual_contact_left = first_left.left_contact_x if first_left else None
    x_actual_contact_right = first_right.right_contact_x if first_right else None

    # predicted contact plane, exactly as diagnose_precontact_grasp_offset.py
    # / the shadow scripts compute it: pad_plane_x (avg of live pad SITE x)
    # + predictor's box half-width, evaluated at the same instant (bilateral
    # contact) for an apples-to-apples hindsight comparison.
    x_predicted_contact = ref.pad_plane_x + ref.predictor_half_x

    # Ground truth for x_predicted_contact must be the actual box COM at
    # contact, not the contact *point* -- x_predicted_contact represents a
    # predicted box-center position (position_after(tau)'s output feeds
    # target_center, the box-center reference), so the contact point
    # (offset from box COM by ~box_half_x, a box-radius) is not the right
    # comparison and was an earlier bug in this script (produced a
    # misleadingly small ~1.4cm "error" by comparing box-center prediction
    # against a box-surface ground truth).
    x_actual = ref.box_com_x
    e_x_total = x_predicted_contact - x_actual if x_actual is not None else None

    # also report the pad-surface-corrected prediction (using the pad's
    # TRUE world xmax -- its collision-facing surface -- instead of its
    # site/geom-center x) for direct comparison, still evaluated in
    # hindsight only, not fed to any controller.
    pad_surface_avg_xmax = 0.5 * (ref.left_pad_world_xmax + ref.right_pad_world_xmax)
    x_predicted_contact_surface_corrected = pad_surface_avg_xmax + ref.predictor_half_x
    e_x_surface_corrected = x_predicted_contact_surface_corrected - x_actual

    # component decomposition, all evaluated at the bilateral-contact instant:
    box_size_mismatch = ref.box_half_x_runtime - ref.predictor_half_x  # should be ~0 (domain_parameters IS what sizes the geom)
    body_geom_offset = ref.box_geom_center_x - ref.box_com_x  # should be ~0 (no inertial offset)
    ee_to_pad_site_offset_left = ref.left_pad_site_x - ref.left_ee_x
    ee_to_pad_site_offset_right = ref.right_pad_site_x - ref.right_ee_x
    pad_site_to_geom_surface_left = (ref.left_pad_world_xmax - ref.left_pad_world_xmin) / 2.0 - ref.pad_thickness_half
    # true pad geom world x-half-extent vs its declared "thickness" half --
    # nonzero here means the pad's tilt (ffw_sg2.xml quat) is projecting a
    # *different* local axis onto world x, i.e. the true x-footprint isn't
    # just the declared 0.006m thickness.
    pad_geom_actual_half_x_left = (ref.left_pad_world_xmax - ref.left_pad_world_xmin) / 2.0
    pad_geom_actual_half_x_right = (ref.right_pad_world_xmax - ref.right_pad_world_xmin) / 2.0

    # box geom's true world half-x vs its declared half_x (should match --
    # nonzero would indicate box rotation already projecting extra x-extent
    # pre-contact, which we've established is ~0 since omega_y~0 pre-contact).
    box_geom_true_half_x = (ref.box_world_xmax - ref.box_world_xmin) / 2.0
    box_rotation_x_projection = box_geom_true_half_x - ref.box_half_x_runtime

    # pad_plane_x used the pad SITE (declared geom center), not the pad's
    # true collision surface facing the box (its world xmax, the face on
    # the +x/box-approach side, since box approaches from +x).
    pad_true_surface_left = ref.left_pad_world_xmax
    pad_true_surface_right = ref.right_pad_world_xmax
    pad_plane_vs_true_surface = ref.pad_plane_x - 0.5 * (pad_true_surface_left + pad_true_surface_right)

    return {
        "seed": seed,
        "bilateral_contact_time_s": ref.time_s,
        "box_com_x": ref.box_com_x,
        "box_geom_center_x": ref.box_geom_center_x,
        "box_half_x_runtime": ref.box_half_x_runtime,
        "box_world_xmin": ref.box_world_xmin,
        "box_world_xmax": ref.box_world_xmax,
        "box_geom_true_half_x": box_geom_true_half_x,
        "left_ee_x": ref.left_ee_x,
        "right_ee_x": ref.right_ee_x,
        "left_pad_site_x": ref.left_pad_site_x,
        "right_pad_site_x": ref.right_pad_site_x,
        "left_pad_world_xmin": ref.left_pad_world_xmin,
        "left_pad_world_xmax": ref.left_pad_world_xmax,
        "right_pad_world_xmin": ref.right_pad_world_xmin,
        "right_pad_world_xmax": ref.right_pad_world_xmax,
        "pad_thickness_half_declared": ref.pad_thickness_half,
        "pad_geom_actual_half_x_left": pad_geom_actual_half_x_left,
        "pad_geom_actual_half_x_right": pad_geom_actual_half_x_right,
        "predictor_half_x": ref.predictor_half_x,
        "pad_plane_x": ref.pad_plane_x,
        "x_actual_contact_left": x_actual_contact_left,
        "x_actual_contact_right": x_actual_contact_right,
        "x_predicted_contact": x_predicted_contact,
        "e_x_total_m": e_x_total,
        "x_predicted_contact_surface_corrected": x_predicted_contact_surface_corrected,
        "e_x_surface_corrected_m": e_x_surface_corrected,
        "component_box_size_mismatch_m": box_size_mismatch,
        "component_body_geom_offset_m": body_geom_offset,
        "component_ee_to_pad_site_offset_left_m": ee_to_pad_site_offset_left,
        "component_ee_to_pad_site_offset_right_m": ee_to_pad_site_offset_right,
        "component_pad_site_to_geom_surface_extra_half_x_m": pad_site_to_geom_surface_left,
        "component_box_rotation_x_projection_m": box_rotation_x_projection,
        "component_pad_plane_vs_true_surface_m": pad_plane_vs_true_surface,
        "first_left_contact_geom_pair": (ref.left_contact_geom1, ref.left_contact_geom2),
        "first_left_contact_time_s": first_left.time_s if first_left else None,
        "first_right_contact_time_s": first_right.time_s if first_right else None,
    }


def main() -> None:
    results = []
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        result, rec = run_episode_geometry(seed)
        analysis = analyze_geometry(seed, result, rec)
        results.append(analysis)
        for k, v in analysis.items():
            print(f"  {k}: {v}")
        print(flush=True)

    print("=== cross-seed comparison ===")
    for a in results:
        if "error" in a:
            continue
        print(
            f"  seed={a['seed']} e_x_total={a['e_x_total_m']:.4f} "
            f"box_half_x={a['box_half_x_runtime']:.4f} "
            f"pad_plane_x={a['pad_plane_x']:.4f} "
            f"pad_geom_actual_half_x_L={a['pad_geom_actual_half_x_left']:.4f} "
            f"pad_site_to_geom_surface_extra={a['component_pad_site_to_geom_surface_extra_half_x_m']:.4f} "
            f"pad_plane_vs_true_surface={a['component_pad_plane_vs_true_surface_m']:.4f}"
        )


if __name__ == "__main__":
    main()
