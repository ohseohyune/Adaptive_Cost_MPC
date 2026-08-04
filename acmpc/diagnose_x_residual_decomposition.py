"""Decompose the x-axis residual that remains after the pad-surface
geometry fix (main_acmpc_box_catch.py's _pad_plane_x now uses the pad
collision geom's true box-facing world-x surface instead of its site x --
see diagnose_x_bias_geometry.py / the fix applied in response to that
diagnosis).

Read-only: does not touch production control, the predictor, trigger, or
capture-timing logic. Reruns the same mature-capture shadow condition
(confidence==1.0, samples>=5, workspace, 50ms/1cm stability -- unchanged)
used to validate the geometry fix, but also keeps the full per-substep
dense trace to extract exactly what happens between capture and actual
contact.

Decomposition (all exact algebraic identities, no fitting):

  e_x^total = x_target_captured - x_actual_box_com_at_contact

  Timing/velocity split (pred(tau) = p0 + v0*tau, x has no gravity term):
    e_x^TTC      = pred(ttc_at_capture) - pred(dt_true)   [x-component]
    e_x^velocity = pred(dt_true) - x_actual_box_com_at_contact

  Tracking/geometry split (equally exact, different cut of the same total):
    e_x^tracking = x_target_captured - (x_actual_pad_at_contact + h_box)
                 = x_pad_plane_at_capture - x_actual_pad_at_contact
      (the pad's own physical x drifting/lagging between capture and
      contact -- since the frozen target commands the pad to *box_half_x*
      further out than where it was sitting at capture, this is really
      "did the pad close that gap in time", i.e. the same quantity as
      left/right_pad_target_tracking_error_m already reported, restricted
      to x)
    e_x^geometry = (x_actual_pad_at_contact + h_box) - x_actual_box_com_at_contact
      (leftover after accounting for tracking -- tests whether "pad's own
      actual final position + box half-width" still correctly predicts the
      box's true COM at contact; should be small now that the surface fix
      is in place)

  e_x^other = e_x^total - (e_x^TTC + e_x^velocity)  [should be ~0 by
              construction; reported only as a sanity residual]

Usage: python3 acmpc/diagnose_x_residual_decomposition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, _build_stage1_config, analyze, run_episode
from acmpc.diagnose_mature_capture_shadow import CAPTURE_MIN_SAMPLES, _make_mature_capture_override

GRAVITY = np.array([0.0, 0.0, -9.81])


def run_and_capture(seed: int, min_samples: int = CAPTURE_MIN_SAMPLES):
    cfg = _build_stage1_config(seed)
    override_fn, state = _make_mature_capture_override(cfg.squeeze, min_samples=min_samples)
    result = run_episode(seed, target_override_fn=override_fn)
    out = analyze(result)
    return result, out, state, cfg


def decompose(seed: int) -> dict:
    result, out, state, cfg = run_and_capture(seed)
    dense = result["rec"].dense
    capture_t = state["capture_time_s"]
    contact_t = out.get("bilateral_contact_time_s") or out.get("actual_contact_time_s")
    if capture_t is None or contact_t is None or not dense:
        return {"seed": seed, "error": "no capture or no contact"}

    capture_row = min(dense, key=lambda r: abs(r.time_s - capture_t))
    contact_row = min(dense, key=lambda r: abs(r.time_s - contact_t))

    # box half-width along the closing (x) axis -- domain_parameters.half_size[0]
    h_box_x = (
        float(cfg.domain_parameters.half_size[0])
        if cfg.domain_parameters is not None
        else float(capture_row.box_com[0])  # unreachable fallback; domain_parameters always set for stage1
    )

    tau = float(capture_row.position_remaining_ttc)
    dt_true = contact_t - capture_t
    predicted_contact_time = capture_t + tau
    e_t = predicted_contact_time - contact_t

    def pred(t: float) -> np.ndarray:
        return capture_row.prediction_position + capture_row.prediction_velocity * t + 0.5 * GRAVITY * t**2

    pred_at_ttc = pred(tau)
    pred_at_true_dt = pred(dt_true)
    x_target_captured = float(pred_at_ttc[0])
    x_actual_box_com = float(contact_row.box_com[0])

    e_x_total = x_target_captured - x_actual_box_com
    e_x_ttc = float(pred_at_ttc[0] - pred_at_true_dt[0])
    e_x_velocity = float(pred_at_true_dt[0] - x_actual_box_com)
    e_x_other = e_x_total - (e_x_ttc + e_x_velocity)  # sanity, should be ~0

    # true collision-facing surface, not the site/geom-center x (that
    # site-vs-surface gap is exactly what the earlier pad-surface fix
    # targeted in _pad_plane_x -- reusing site x here would silently
    # reintroduce the same ~3.7cm artifact into this decomposition).
    x_actual_pad_at_contact = 0.5 * (contact_row.left_pad_surface_x + contact_row.right_pad_surface_x)
    e_x_tracking = x_target_captured - (x_actual_pad_at_contact + h_box_x)
    e_x_geometry = (x_actual_pad_at_contact + h_box_x) - x_actual_box_com

    return {
        "seed": seed,
        "capture_time_s": capture_t,
        "actual_bilateral_contact_time_s": contact_t,
        "predicted_contact_time_s": predicted_contact_time,
        "ttc_error_s": e_t,
        "box_position_at_capture": capture_row.box_com.tolist(),
        "box_velocity_estimate_at_capture": capture_row.prediction_velocity.tolist(),
        "box_half_x": h_box_x,
        "predicted_box_position_at_contact": pred_at_ttc.tolist(),
        "actual_box_position_at_contact": contact_row.box_com.tolist(),
        "x_target_captured": x_target_captured,
        "x_actual_pad_at_contact_avg": x_actual_pad_at_contact,
        "left_pad_surface_x_at_contact": contact_row.left_pad_surface_x,
        "right_pad_surface_x_at_contact": contact_row.right_pad_surface_x,
        "left_pad_site_x_at_contact": float(contact_row.left_pad_site_pos[0]),
        "right_pad_site_x_at_contact": float(contact_row.right_pad_site_pos[0]),
        "target_tracking_error_m": out.get("left_pad_target_tracking_error_m"),
        "left_first_contact_geom_pair": None,  # resolved via contact.left.contacts below if needed
        "e_x_total_m": e_x_total,
        "e_x_TTC_m": e_x_ttc,
        "e_x_velocity_m": e_x_velocity,
        "e_x_tracking_m": e_x_tracking,
        "e_x_geometry_m": e_x_geometry,
        "e_x_other_m": e_x_other,
        "left_contact_distance_from_face_center_m": out.get("left_contact_distance_from_face_center_m"),
        "left_first_contact_classification": out.get("left_first_contact_classification"),
        "cumulative_rotation_deg_hold": out.get("cumulative_rotation_deg_hold"),
        "success": out.get("success"),
    }


def main() -> None:
    all_results = []
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        d = decompose(seed)
        all_results.append(d)
        for k, v in d.items():
            print(f"  {k}: {v}")
        print(flush=True)

    print("=== cross-seed comparison ===")
    total = sum(abs(d["e_x_total_m"]) for d in all_results if "e_x_total_m" in d)
    for d in all_results:
        if "e_x_total_m" not in d:
            continue
        print(
            f"  seed={d['seed']} total={d['e_x_total_m']:.4f} "
            f"TTC={d['e_x_TTC_m']:.4f} velocity={d['e_x_velocity_m']:.4f} "
            f"tracking={d['e_x_tracking_m']:.4f} geometry={d['e_x_geometry_m']:.4f} "
            f"other={d['e_x_other_m']:.4f}"
        )


if __name__ == "__main__":
    main()
