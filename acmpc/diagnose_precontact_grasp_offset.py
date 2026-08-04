"""Pre-contact root-cause diagnosis for the Stage 1 near_low_speed_catch
box-edge-rotation bug.

Symptom (already established, see box_catch_mass_sweep_eval.py /
diagnose_hold_rotation.py): desired_grasp_center tracks box COM to within
4-12mm, but the actual pad-box contact center ends up 31-48mm off COM, so
the box hangs off one edge during HOLD and rotates under gravity.

This script does NOT touch HOLD-phase targets, hand translation, K/D,
friction, condim, MPC weights, angular-speed thresholds, success
conditions, or box mass/inertia. It only *measures* where and when the
first real contact forms, from INTERCEPT through the start of HOLD, to
find out why the pad ends up biting an edge instead of the box face.

Usage: python3 acmpc/diagnose_precontact_grasp_offset.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.main_acmpc_box_catch import AcmpcBoxCatchConfig, CatchPhase, _pad_box_facing_surface_x, run_box_catch
from acmpc.diagnose_hold_rotation import _mat2quat, _quat_angle_deg  # reuse, pure helpers
from control.squeeze.config import DynamicSideSqueezeConfig
from control.squeeze.progressive_curriculum import progressive_catch_curriculum, sample_stage_domain

STAGE1_SEEDS = [1000, 1017, 1005, 1001]
POST_BILATERAL_WINDOW_S = 0.15  # dense recording continues this long past bilateral contact
PRE_CONTACT_LOOKBACK_S = 0.10   # window A


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_stage1_config(seed: int) -> AcmpcBoxCatchConfig:
    stage = progressive_catch_curriculum()[1]  # near_low_speed_catch
    rng = np.random.default_rng(seed)
    base_squeeze = DynamicSideSqueezeConfig(random_seed=seed)
    resolved = sample_stage_domain(stage, rng, 1, base_squeeze)
    return AcmpcBoxCatchConfig(
        seed=seed,
        device="cpu",
        online_learning=False,
        use_launch_fixture=False,
        domain_parameters=resolved.domain,
        squeeze=resolved.squeeze_config,
    )


@dataclass
class Row:
    step: int
    time_s: float
    phase: str
    box_com: np.ndarray
    box_quat: np.ndarray
    box_rotation: np.ndarray
    box_lin_vel: np.ndarray
    box_ang_vel: np.ndarray
    target_center: np.ndarray
    left_active: bool
    right_active: bool
    left_force: float
    right_force: float
    left_contact_pos: np.ndarray | None
    right_contact_pos: np.ndarray | None
    left_pad_site_pos: np.ndarray
    right_pad_site_pos: np.ndarray
    left_pad_surface_x: float
    right_pad_surface_x: float
    left_pad_normal: np.ndarray
    right_pad_normal: np.ndarray
    left_ee: np.ndarray
    right_ee: np.ndarray
    left_pad_target: np.ndarray
    right_pad_target: np.ndarray
    y_axis: np.ndarray
    prediction_position: np.ndarray
    prediction_velocity: np.ndarray
    remaining_ttc: float
    remaining_ttc_valid: bool
    position_remaining_ttc: float
    position_remaining_ttc_valid: bool
    prediction_confidence: float
    predictor_samples: int


@dataclass
class LightRow:
    time_s: float
    phase: str
    omega_y: float
    box_quat: np.ndarray


@dataclass
class EpisodeRecording:
    dense: list = field(default_factory=list)
    light: list = field(default_factory=list)
    box_half_extents: np.ndarray | None = None
    bilateral_first_time: float | None = None


def run_episode(seed: int, target_override_fn=None) -> dict:
    seed_everything(seed)
    cfg = _build_stage1_config(seed)
    rec = EpisodeRecording()

    ids = {}  # lazily-cached geom/site ids, filled on first callback

    def on_step(ctx: dict) -> None:
        model, data = ctx["model"], ctx["data"]
        if not ids:
            ids["box_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
            ids["pad_geom"] = {
                side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_catch_pad")
                for side in ("left", "right")
            }
            rec.box_half_extents = model.geom_size[ids["box_geom"]].copy()

        contact = ctx["contact"]
        box_body_id = ctx["box_body_id"]
        box_dof_address = ctx["box_dof_address"]
        box_com = data.xipos[box_body_id].copy()
        box_quat = data.xquat[box_body_id].copy()
        box_rotation = data.xmat[box_body_id].reshape(3, 3).copy()
        box_ang_vel = data.qvel[box_dof_address + 3 : box_dof_address + 6].copy()
        phase_name = ctx["phase"].value

        rec.light.append(
            LightRow(
                time_s=ctx["time_s"],
                phase=phase_name,
                omega_y=float(box_ang_vel[1]),
                box_quat=box_quat,
            )
        )

        # Dense recording window: from episode start through
        # bilateral-contact + POST_BILATERAL_WINDOW_S. Once that window has
        # closed we stop paying the per-substep dense-row cost (episode can
        # run another ~5s of HOLD after this).
        if contact.left.active and contact.right.active and rec.bilateral_first_time is None:
            rec.bilateral_first_time = ctx["time_s"]
        if rec.bilateral_first_time is not None and (
            ctx["time_s"] - rec.bilateral_first_time > POST_BILATERAL_WINDOW_S
        ):
            return

        catch_pad_site_ids = ctx["catch_pad_site_ids"]
        left_normal = data.site_xmat[catch_pad_site_ids["left"]].reshape(3, 3)[:, 0].copy()
        right_normal = data.site_xmat[catch_pad_site_ids["right"]].reshape(3, 3)[:, 0].copy()

        rec.dense.append(
            Row(
                step=ctx["step"],
                time_s=ctx["time_s"],
                phase=phase_name,
                box_com=box_com,
                box_quat=box_quat,
                box_rotation=box_rotation,
                box_lin_vel=data.qvel[box_dof_address : box_dof_address + 3].copy(),
                box_ang_vel=box_ang_vel,
                target_center=ctx["target_center"],
                left_active=bool(contact.left.active),
                right_active=bool(contact.right.active),
                left_force=float(contact.left.normal_force),
                right_force=float(contact.right.normal_force),
                left_contact_pos=(contact.left.mean_position.copy() if contact.left.active else None),
                right_contact_pos=(contact.right.mean_position.copy() if contact.right.active else None),
                left_pad_site_pos=data.site_xpos[catch_pad_site_ids["left"]].copy(),
                right_pad_site_pos=data.site_xpos[catch_pad_site_ids["right"]].copy(),
                left_pad_surface_x=_pad_box_facing_surface_x(model, data, ids["pad_geom"]["left"], float(ctx["box_position"][0])),
                right_pad_surface_x=_pad_box_facing_surface_x(model, data, ids["pad_geom"]["right"], float(ctx["box_position"][0])),
                left_pad_normal=left_normal,
                right_pad_normal=right_normal,
                left_ee=ctx["left_ee"],
                right_ee=ctx["right_ee"],
                left_pad_target=ctx["left_pad_target"],
                right_pad_target=ctx["right_pad_target"],
                y_axis=ctx["y_axis"],
                prediction_position=ctx["prediction_position"],
                prediction_velocity=ctx["prediction_velocity"],
                remaining_ttc=ctx["remaining_ttc"],
                remaining_ttc_valid=ctx["remaining_ttc_valid"],
                position_remaining_ttc=ctx["position_remaining_ttc"],
                position_remaining_ttc_valid=ctx["position_remaining_ttc_valid"],
                prediction_confidence=float(ctx["prediction_confidence"]),
                predictor_samples=int(ctx["predictor_samples"]),
            )
        )

    summary = run_box_catch(cfg, step_callback=on_step, target_override_fn=target_override_fn)
    return {"summary": summary, "rec": rec, "seed": seed, "cfg": cfg}


def _box_local(p_world: np.ndarray, box_com: np.ndarray, box_rotation: np.ndarray) -> np.ndarray:
    return box_rotation.T @ (p_world - box_com)


def _classify_contact(local_xz: np.ndarray, half_x: float, half_z: float, edge_ratio: float = 0.7) -> str:
    near_x_edge = abs(local_xz[0]) >= edge_ratio * half_x
    near_z_edge = abs(local_xz[1]) >= edge_ratio * half_z
    if near_x_edge and near_z_edge:
        return "CORNER_CONTACT"
    if near_x_edge or near_z_edge:
        return "EDGE_CONTACT"
    return "FACE_CENTER_CONTACT"


def analyze(result: dict) -> dict:
    seed, summary, rec = result["seed"], result["summary"], result["rec"]
    cfg = result["cfg"]
    dense = rec.dense
    light = rec.light
    hx, hy, hz = rec.box_half_extents if rec.box_half_extents is not None else (np.nan, np.nan, np.nan)

    out: dict = {"seed": seed, "success": summary.success, "failure_reason": summary.failure_reason}

    if not dense:
        out["error"] = "no dense rows recorded (episode ended before contact)"
        return out

    def first_idx(pred):
        for i, r in enumerate(dense):
            if pred(r):
                return i
        return None

    left_idx = first_idx(lambda r: r.left_active)
    right_idx = first_idx(lambda r: r.right_active)
    bilat_idx = first_idx(lambda r: r.left_active and r.right_active)

    left_t = dense[left_idx].time_s if left_idx is not None else None
    right_t = dense[right_idx].time_s if right_idx is not None else None
    bilat_t = dense[bilat_idx].time_s if bilat_idx is not None else None

    out["predicted_contact_time_s"] = None  # ballistic predictor internal, not exposed via ctx
    out["actual_contact_time_s"] = left_t if left_t is not None and (right_t is None or left_t <= right_t) else right_t
    out["left_first_contact_time_s"] = left_t
    out["right_first_contact_time_s"] = right_t
    out["bilateral_contact_time_s"] = bilat_t
    out["left_right_contact_time_difference_s"] = (
        (right_t - left_t) if (left_t is not None and right_t is not None) else None
    )
    if left_t is not None and right_t is not None:
        dt_lr = right_t - left_t
        if abs(dt_lr) < 0.004:
            out["contact_order"] = "SIMULTANEOUS_CONTACT"
        elif dt_lr > 0:
            out["contact_order"] = "LEFT_FIRST"
        else:
            out["contact_order"] = "RIGHT_FIRST"
    else:
        out["contact_order"] = "LEFT_FIRST" if left_t is not None else ("RIGHT_FIRST" if right_t is not None else "NONE")

    def local_xyz(row: Row, side: str) -> np.ndarray | None:
        pos = row.left_contact_pos if side == "left" else row.right_contact_pos
        if pos is None:
            return None
        return _box_local(pos, row.box_com, row.box_rotation)

    left_local = local_xyz(dense[left_idx], "left") if left_idx is not None else None
    right_local = local_xyz(dense[right_idx], "right") if right_idx is not None else None
    bilat_local = None
    if bilat_idx is not None:
        r = dense[bilat_idx]
        pts = [p for p in (r.left_contact_pos, r.right_contact_pos) if p is not None]
        if pts:
            bilat_local = _box_local(np.mean(pts, axis=0), r.box_com, r.box_rotation)

    out["box_half_extents_xyz"] = [float(hx), float(hy), float(hz)]
    out["left_first_contact_box_local_xyz"] = None if left_local is None else left_local.tolist()
    out["right_first_contact_box_local_xyz"] = None if right_local is None else right_local.tolist()
    out["bilateral_contact_center_box_local_xyz"] = None if bilat_local is None else bilat_local.tolist()

    def classify_and_dist(local: np.ndarray | None):
        if local is None:
            return None, None
        xz = np.array([local[0], local[2]])
        cls = _classify_contact(xz, hx, hz)
        dist_from_face_center = float(np.linalg.norm(xz))
        return cls, dist_from_face_center

    left_cls, left_dist = classify_and_dist(left_local)
    right_cls, right_dist = classify_and_dist(right_local)
    out["left_first_contact_classification"] = left_cls
    out["right_first_contact_classification"] = right_cls
    out["left_contact_distance_from_face_center_m"] = left_dist
    out["right_contact_distance_from_face_center_m"] = right_dist

    # EE-site-to-pad-surface offset at bilateral contact (world + pad-local).
    if bilat_idx is not None:
        r = dense[bilat_idx]
        ee_offsets = {}
        for side, ee, pad_site, pad_normal in (
            ("left", r.left_ee, r.left_pad_site_pos, r.left_pad_normal),
            ("right", r.right_ee, r.right_pad_site_pos, r.right_pad_normal),
        ):
            delta_world = pad_site - ee
            normal_component = float(np.dot(delta_world, pad_normal))
            ee_offsets[side] = {
                "world_xyz": delta_world.tolist(),
                "along_pad_normal_m": normal_component,
            }
        out["ee_site_to_pad_surface_offset"] = ee_offsets

        # desired pad target vs actual pad site (control tracking error)
        out["left_pad_target_tracking_error_m"] = float(np.linalg.norm(r.left_pad_target - r.left_pad_site_pos))
        out["right_pad_target_tracking_error_m"] = float(np.linalg.norm(r.right_pad_target - r.right_pad_site_pos))

        # desired_grasp_center (target_center) vs COM, and actual grasp
        # center (mean of both pad contact points) vs COM.
        out["desired_grasp_center_to_com_offset_m"] = float(np.linalg.norm(r.target_center - r.box_com))
        actual_center = np.mean([p for p in (r.left_contact_pos, r.right_contact_pos) if p is not None], axis=0)
        out["actual_grasp_center_to_com_offset_m"] = float(np.linalg.norm(actual_center - r.box_com))

        # pad-normal vs box-face-normal alignment (front/twisted/oblique).
        for side, pad_normal in (("left", r.left_pad_normal), ("right", r.right_pad_normal)):
            cos_theta = float(np.clip(abs(np.dot(pad_normal, r.y_axis)), -1.0, 1.0))
            out[f"{side}_pad_normal_vs_box_face_normal_deg"] = float(np.degrees(np.arccos(cos_theta)))

        mass = float(cfg.domain_parameters.mass) if cfg.domain_parameters is not None else 0.5
        gravity = np.array([0.0, 0.0, -9.81])
        gravity_moment = np.cross(r.box_com - actual_center, mass * gravity)
        out["gravity_torque_y_at_bilateral_contact_nm"] = float(gravity_moment[1])
    else:
        out["ee_site_to_pad_surface_offset"] = None
        out["left_pad_target_tracking_error_m"] = None
        out["right_pad_target_tracking_error_m"] = None
        out["desired_grasp_center_to_com_offset_m"] = None
        out["actual_grasp_center_to_com_offset_m"] = None
        out["gravity_torque_y_at_bilateral_contact_nm"] = None

    # omega_y before/after contact (section 9).
    def omega_near(t: float | None, before: bool, dt: float = 0.02) -> float | None:
        if t is None:
            return None
        target_t = t - dt if before else t + dt
        best = min(light, key=lambda lr: abs(lr.time_s - target_t))
        return float(best.omega_y)

    out["omega_y_100ms_before_first_contact"] = omega_near(left_t if left_t and (not right_t or left_t <= right_t) else right_t, True, dt=0.10)
    out["omega_y_immediately_before_first_contact"] = omega_near(left_t if left_t and (not right_t or left_t <= right_t) else right_t, True, dt=0.005)
    out["omega_y_immediately_after_first_contact"] = omega_near(left_t if left_t and (not right_t or left_t <= right_t) else right_t, False, dt=0.005)
    out["omega_y_after_bilateral_contact"] = omega_near(bilat_t, False, dt=0.02)

    pre = out["omega_y_100ms_before_first_contact"]
    just_before = out["omega_y_immediately_before_first_contact"]
    just_after = out["omega_y_immediately_after_first_contact"]
    if pre is not None and just_before is not None and abs(pre) > 0.15:
        out["rotation_source"] = "PREEXISTING_ROTATION"
    elif just_before is not None and just_after is not None and abs(just_after) > abs(just_before) + 0.15:
        out["rotation_source"] = "CONTACT_INDUCED_ROTATION"
    else:
        out["rotation_source"] = "GRAVITY_AFTER_CONTACT"

    # HOLD-phase aggregate (cheap, from the full light list -- no dense-row cost).
    hold_omega = [lr.omega_y for lr in light if lr.phase == "hold"]
    hold_quats = [lr.box_quat for lr in light if lr.phase == "hold"]
    if hold_omega:
        out["mean_box_omega_y_radps_hold"] = float(np.mean(hold_omega))
        out["p95_abs_box_omega_y_radps_hold"] = float(np.percentile(np.abs(hold_omega), 95))
    else:
        out["mean_box_omega_y_radps_hold"] = None
        out["p95_abs_box_omega_y_radps_hold"] = None
    if len(hold_quats) >= 2:
        out["cumulative_rotation_deg_hold"] = _quat_angle_deg(hold_quats[0], hold_quats[-1])
    else:
        out["cumulative_rotation_deg_hold"] = None

    out["first_contact_peak_force_n"] = summary.first_contact_peak_force_n
    out["contact_lost"] = bool(any((not lr_.left_active or not lr_.right_active) for lr_ in dense[bilat_idx:]) ) if bilat_idx is not None else None

    return out


def decompose_prediction_error(result: dict, out: dict) -> dict:
    """Numerically split the closing-axis position error at contact into a
    timing (remaining_ttc) component and a velocity/model-residual
    component, using the exact identity

        pred(tau) = p_i + v_i*tau + 0.5*g*tau**2
        error     = pred(remaining_ttc_i) - p_actual
                  = [pred(remaining_ttc_i) - pred(dt)]   (timing_contribution)
                  + [pred(dt) - p_actual]                 (residual_contribution)

    where dt = actual_contact_time - t_i is the *true* time-to-contact from
    sample i, so pred(dt) is what the same velocity/gravity model would have
    predicted had remaining_ttc been exactly correct. No approximation.
    """

    dense = result["rec"].dense
    contact_t = out.get("actual_contact_time_s")
    if contact_t is None or not dense:
        return {"error": "no contact recorded"}

    # actual box position at the moment of first contact (nearest row).
    contact_row = min(dense, key=lambda r: abs(r.time_s - contact_t))
    p_actual = contact_row.box_com

    # sample: last pre-contact row (largest time_s strictly before contact).
    pre_rows = [r for r in dense if r.time_s < contact_t and r.position_remaining_ttc_valid]
    if not pre_rows:
        return {"error": "no valid pre-contact TTC sample"}
    sample = max(pre_rows, key=lambda r: r.time_s)
    dt = contact_t - sample.time_s

    predicted_contact_time = sample.time_s + sample.position_remaining_ttc
    e_t = predicted_contact_time - contact_t

    def pred(tau: float) -> np.ndarray:
        gravity = np.array([0.0, 0.0, -9.81])
        return sample.prediction_position + sample.prediction_velocity * tau + 0.5 * gravity * tau**2

    pred_at_ttc = pred(sample.position_remaining_ttc)
    pred_at_true_dt = pred(dt)
    total_error = pred_at_ttc - p_actual
    timing_contribution = pred_at_ttc - pred_at_true_dt
    residual_contribution = pred_at_true_dt - p_actual

    # window-A sample (~100ms before contact) for trend context.
    window_a = [r for r in pre_rows if contact_t - r.time_s <= PRE_CONTACT_LOOKBACK_S + 0.02]

    return {
        "sample_time_s": sample.time_s,
        "actual_contact_time_s": contact_t,
        "predicted_contact_time_s": predicted_contact_time,
        "contact_time_prediction_error_s": e_t,
        "current_box_position_at_sample": sample.prediction_position.tolist(),
        "current_box_velocity_estimate_at_sample": sample.prediction_velocity.tolist(),
        "remaining_ttc_s": sample.position_remaining_ttc,
        "actual_time_remaining_s": dt,
        "predicted_contact_position_xyz": pred_at_ttc.tolist(),
        "actual_contact_position_xyz": p_actual.tolist(),
        "position_prediction_error_xyz": total_error.tolist(),
        "closing_axis_x_error_m": float(total_error[0]),
        "x_timing_contribution_m": float(timing_contribution[0]),
        "x_velocity_residual_contribution_m": float(residual_contribution[0]),
        "z_error_m": float(total_error[2]),
        "z_timing_contribution_m": float(timing_contribution[2]),
        "z_velocity_residual_contribution_m": float(residual_contribution[2]),
        "e_p_x_predicted_by_vx_times_e_t": float(sample.prediction_velocity[0] * e_t),
        "num_window_a_samples": len(window_a),
    }


def _root_cause(out: dict) -> str:
    left_cls = out.get("left_first_contact_classification")
    right_cls = out.get("right_first_contact_classification")
    order = out.get("contact_order")
    align_deg = max(
        out.get("left_pad_normal_vs_box_face_normal_deg") or 0.0,
        out.get("right_pad_normal_vs_box_face_normal_deg") or 0.0,
    )
    tracking_err = max(
        out.get("left_pad_target_tracking_error_m") or 0.0,
        out.get("right_pad_target_tracking_error_m") or 0.0,
    )
    rotation_source = out.get("rotation_source")

    reasons = []
    if left_cls in ("EDGE_CONTACT", "CORNER_CONTACT") or right_cls in ("EDGE_CONTACT", "CORNER_CONTACT"):
        if tracking_err is not None and tracking_err >= 0.015:
            reasons.append("D")  # CONTACT_TIMING_ERROR / tracking lag large enough to reach an edge
        if align_deg >= 8.0:
            reasons.append("B")  # PAD_GEOMETRY_ORIENTATION -- pad not parallel to box face at contact
        if order in ("LEFT_FIRST", "RIGHT_FIRST") and out.get("left_right_contact_time_difference_s") not in (None,) and abs(out.get("left_right_contact_time_difference_s") or 0) >= 0.02:
            reasons.append("E")  # LEFT_RIGHT_ASYNCHRONY
        if rotation_source == "PREEXISTING_ROTATION":
            reasons.append("G")
    if not reasons:
        reasons.append("J")
    return "MULTIPLE_CAUSES" if len(reasons) > 1 else reasons[0], reasons


def main() -> None:
    all_out = []
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        result = run_episode(seed)
        out = analyze(result)
        cause, reasons = _root_cause(out) if "error" not in out else ("N/A", [])
        out["dominant_root_cause"] = cause
        out["root_cause_candidates"] = reasons
        decomp = decompose_prediction_error(result, out)
        out["prediction_error_decomposition"] = decomp
        all_out.append(out)
        for k, v in out.items():
            print(f"  {k}: {v}")
        print(flush=True)

    print("=== summary across seeds ===")
    for out in all_out:
        print(
            f"  seed={out['seed']} success={out.get('success')} "
            f"left_cls={out.get('left_first_contact_classification')} "
            f"right_cls={out.get('right_first_contact_classification')} "
            f"order={out.get('contact_order')} "
            f"rotation_source={out.get('rotation_source')} "
            f"root_cause={out.get('dominant_root_cause')} ({out.get('root_cause_candidates')})"
        )


if __name__ == "__main__":
    main()
