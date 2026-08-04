"""Verify + evaluate the HOLD grasp-center x-alignment fix for the persistent
y-axis box rotation (see AcmpcBoxCatchConfig.hold_grasp_com_align).

Three conditions, same seeds:
  baseline            hold_grasp_com_align=False (today's behavior, exact no-op)
  diagnostic_instant   hold_grasp_com_align=True, ramp_s ~= one control step
                       (near-instant snap -- causal validation only, see
                       run_box_catch's docstring-comment: never use this
                       ramp value outside a diagnostic run)
  smooth               hold_grasp_com_align=True, ramp_s=1.0 (conservative
                       point in the requested 0.5-1.5s range)

Usage: python3 acmpc/diagnose_hold_rotation.py
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

from acmpc.main_acmpc_box_catch import AcmpcBoxCatchConfig, CatchPhase, run_box_catch
from control.clik.contact import get_contacts_between
from control.squeeze import apply_box_domain_randomization
from control.squeeze.config import DynamicSideSqueezeConfig
from control.squeeze.progressive_curriculum import progressive_catch_curriculum, sample_stage_domain

STAGE1_SEEDS = {"1000": "success", "1017": "success", "1005": "failure", "1001": "failure"}
STAGE0_SEED = 7  # representative Stage-0 static_grasp_bootstrap seed

CONDITIONS = {
    "baseline": dict(hold_grasp_com_align=False),
    "diagnostic_instant": dict(
        hold_grasp_com_align=True, hold_grasp_com_align_ramp_s=0.01, hold_grasp_com_align_max_shift_m=0.03
    ),
    "smooth": dict(
        hold_grasp_com_align=True, hold_grasp_com_align_ramp_s=1.0, hold_grasp_com_align_max_shift_m=0.03
    ),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mat2quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    return q


def _quat_angle_deg(q_from: np.ndarray, q_to: np.ndarray) -> float:
    q_from_conj = q_from * np.array([1, -1, -1, -1])
    q_rel = np.zeros(4)
    mujoco.mju_mulQuat(q_rel, q_to, q_from_conj)
    w = np.clip(abs(q_rel[0]), -1.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(w)))


@dataclass
class HoldRecord:
    rows: list = field(default_factory=list)


def _contact_wrench_world(model, data, pad_geom_id: int, box_geom_id: int, box_com: np.ndarray):
    f_world_total = np.zeros(3)
    tau_about_com_total = np.zeros(3)
    tau_spin_total = np.zeros(3)
    n_contacts = 0
    mean_pos = np.zeros(3)
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        if pair != {pad_geom_id, box_geom_id}:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        frame = c.frame.reshape(3, 3)
        rot_local_to_world = frame.T
        f_world = rot_local_to_world @ force[:3]
        tau_spin_world = rot_local_to_world @ force[3:6]
        pos = c.pos.copy()
        tau_arm_world = np.cross(pos - box_com, f_world)
        f_world_total += f_world
        tau_about_com_total += tau_arm_world + tau_spin_world
        tau_spin_total += tau_spin_world
        mean_pos += pos
        n_contacts += 1
    if n_contacts:
        mean_pos /= n_contacts
    return f_world_total, tau_about_com_total, tau_spin_total, n_contacts, mean_pos


def _build_config(seed: int, condition: str, *, stage0: bool = False) -> AcmpcBoxCatchConfig:
    overrides = CONDITIONS[condition]
    if stage0:
        cfg = AcmpcBoxCatchConfig(
            seed=seed,
            device="cpu",
            online_learning=False,
            use_launch_fixture=True,
            domain_parameters=None,
            **overrides,
        )
    else:
        stage = progressive_catch_curriculum()[1]  # near_low_speed_catch
        rng = np.random.default_rng(seed)
        base_squeeze = DynamicSideSqueezeConfig(random_seed=seed)
        resolved = sample_stage_domain(stage, rng, 1, base_squeeze)
        cfg = AcmpcBoxCatchConfig(
            seed=seed,
            device="cpu",
            online_learning=False,
            use_launch_fixture=False,
            domain_parameters=resolved.domain,
            squeeze=resolved.squeeze_config,
            **overrides,
        )
    return cfg


def run_condition(seed: int, condition: str, *, stage0: bool = False) -> dict:
    seed_everything(seed)
    cfg = _build_config(seed, condition, stage0=stage0)

    record = HoldRecord()
    box_geom_id = None
    pad_geom_ids = None

    def on_step(ctx: dict) -> None:
        nonlocal box_geom_id, pad_geom_ids
        if ctx["phase"] is not CatchPhase.HOLD:
            return
        model, data = ctx["model"], ctx["data"]
        if box_geom_id is None:
            box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
            pad_geom_ids = {
                side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_catch_pad")
                for side in ("left", "right")
            }

        box_body_id = ctx["box_body_id"]
        box_dof_address = ctx["box_dof_address"]
        box_com = data.xipos[box_body_id].copy()
        box_quat = data.xquat[box_body_id].copy()
        box_angular_velocity = data.qvel[box_dof_address + 3 : box_dof_address + 6].copy()

        pads = {}
        f_total = np.zeros(3)
        tau_about_com_total = np.zeros(3)
        tau_spin_total = np.zeros(3)
        for side in ("left", "right"):
            f_w, tau_com, tau_spin, n_c, mean_pos = _contact_wrench_world(
                model, data, pad_geom_ids[side], box_geom_id, box_com
            )
            f_total += f_w
            tau_about_com_total += tau_com
            tau_spin_total += tau_spin
            pads[side] = {"n_contacts": n_c, "contact_mean_pos": mean_pos}
        actual_grasp_center = None
        if pads["left"]["n_contacts"] and pads["right"]["n_contacts"]:
            actual_grasp_center = 0.5 * (pads["left"]["contact_mean_pos"] + pads["right"]["contact_mean_pos"])

        mass = float(model.body_mass[box_body_id])
        gravity = np.array(model.opt.gravity, dtype=float)
        grasp_ref = actual_grasp_center if actual_grasp_center is not None else ctx["target_center"]
        offset = box_com - grasp_ref
        gravity_moment = np.cross(offset, mass * gravity)

        record.rows.append(
            {
                "time_s": ctx["time_s"],
                "box_com": box_com,
                "box_quat": box_quat,
                "box_angular_velocity": box_angular_velocity,
                "actual_grasp_center": actual_grasp_center,
                "desired_grasp_center": ctx["target_center"],
                "com_grasp_offset": offset,
                "gravity_moment_y": float(gravity_moment[1]),
                "contact_torque_y": float(tau_about_com_total[1] - tau_spin_total[1]),
                "torsional_torque_y": float(tau_spin_total[1]),
                "align_delta_x": ctx["align_delta_x"],
                "align_elapsed_s": ctx["align_elapsed_s"],
                "left_active": bool(pads["left"]["n_contacts"] > 0),
                "right_active": bool(pads["right"]["n_contacts"] > 0),
                "emergency": ctx["impact_emergency"],
                "first_contact_peak_force_n": ctx["first_contact_peak_force_n"],
            }
        )

    summary = run_box_catch(cfg, step_callback=on_step)
    return {"summary": summary, "record": record, "config": cfg, "seed": seed, "condition": condition}


def summarize(result: dict) -> dict:
    summary = result["summary"]
    rows = result["record"].rows
    out = {
        "seed": result["seed"],
        "condition": result["condition"],
        "success": summary.success,
        "failure_reason": summary.failure_reason,
        "hold_time_s": summary.hold_time_s,
        "first_contact_peak_force_n": summary.first_contact_peak_force_n,
        "n_hold_rows": len(rows),
    }
    if not rows:
        out["error"] = "HOLD never reached; no rows recorded"
        return out

    box_quats = np.array([r["box_quat"] for r in rows])
    omega_y = np.array([r["box_angular_velocity"][1] for r in rows])
    offset_x = np.array([r["com_grasp_offset"][0] for r in rows])
    gmoment_y = np.array([r["gravity_moment_y"] for r in rows])
    ctorque_y = np.array([r["contact_torque_y"] for r in rows])
    tspin_y = np.array([r["torsional_torque_y"] for r in rows])
    left_active = np.array([r["left_active"] for r in rows])
    right_active = np.array([r["right_active"] for r in rows])
    emergency = np.array([r["emergency"] for r in rows])
    align_delta_x = next((r["align_delta_x"] for r in rows if r["align_delta_x"] is not None), None)

    out.update(
        mean_com_grasp_offset_x_m=float(np.mean(offset_x)),
        max_com_grasp_offset_x_m=float(np.max(np.abs(offset_x))),
        mean_gravity_torque_y_nm=float(np.mean(gmoment_y)),
        mean_torsional_resistance_y_nm=float(np.mean(tspin_y)),
        mean_residual_torque_y_nm=float(np.mean(gmoment_y + tspin_y + ctorque_y)),
        mean_box_omega_y_radps=float(np.mean(omega_y)),
        p95_box_omega_y_radps=float(np.percentile(np.abs(omega_y), 95)),
        max_box_omega_y_radps=float(np.max(np.abs(omega_y))),
        cumulative_rotation_deg=_quat_angle_deg(box_quats[0], box_quats[-1]),
        final_orientation_change_deg=_quat_angle_deg(box_quats[0], box_quats[-1]),
        contact_lost=bool(np.any(~left_active) or np.any(~right_active)),
        emergency_contact=bool(np.any(emergency)),
        align_delta_x_captured_m=align_delta_x,
        align_delta_x_clipped=(
            None if align_delta_x is None else bool(abs(align_delta_x) >= 0.0299)
        ),
    )
    return out


def check_reproducibility(seed: int) -> bool:
    r1 = run_condition(seed, "baseline")
    r2 = run_condition(seed, "baseline")
    s1, s2 = r1["summary"], r2["summary"]
    q1 = r1["record"].rows[0]["box_quat"] if r1["record"].rows else None
    q2 = r2["record"].rows[0]["box_quat"] if r2["record"].rows else None
    ok = (
        s1.success == s2.success
        and abs(s1.hold_time_s - s2.hold_time_s) < 1e-9
        and (q1 is None) == (q2 is None)
        and (q1 is None or np.allclose(q1, q2, atol=1e-9))
    )
    print(f"[reproducibility] seed={seed}: run1(success={s1.success}, hold={s1.hold_time_s:.4f}) "
          f"run2(success={s2.success}, hold={s2.hold_time_s:.4f}) -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> None:
    print("=== reproducibility check ===")
    repro_ok = check_reproducibility(1005)
    print()

    all_results = {}
    for seed_str in STAGE1_SEEDS:
        seed = int(seed_str)
        for condition in CONDITIONS:
            print(f"--- seed {seed} / {condition} ---", flush=True)
            result = run_condition(seed, condition)
            summ = summarize(result)
            all_results[(seed, condition)] = summ
            for k, v in summ.items():
                print(f"  {k}: {v}")
            print(flush=True)

    print(f"--- Stage 0 seed {STAGE0_SEED} / baseline ---", flush=True)
    stage0_baseline = summarize(run_condition(STAGE0_SEED, "baseline", stage0=True))
    for k, v in stage0_baseline.items():
        print(f"  {k}: {v}")
    print(flush=True)

    print(f"--- Stage 0 seed {STAGE0_SEED} / smooth ---", flush=True)
    stage0_smooth = summarize(run_condition(STAGE0_SEED, "smooth", stage0=True))
    for k, v in stage0_smooth.items():
        print(f"  {k}: {v}")
    print(flush=True)

    print("=== reduction % vs baseline (per seed, smooth condition) ===")
    for seed_str in STAGE1_SEEDS:
        seed = int(seed_str)
        base = all_results[(seed, "baseline")]
        smooth = all_results[(seed, "smooth")]
        if base.get("error") or smooth.get("error"):
            print(f"  seed {seed}: baseline or smooth run has no HOLD rows, skipping")
            continue

        def pct(b, s):
            return 100.0 * (b - s) / b if b else float("nan")

        print(
            f"  seed {seed}: gravity_torque_reduction%={pct(base['mean_gravity_torque_y_nm'], smooth['mean_gravity_torque_y_nm']):.1f} "
            f"angular_speed_reduction%={pct(base['mean_box_omega_y_radps'], smooth['mean_box_omega_y_radps']):.1f} "
            f"cumulative_rotation_reduction%={pct(base['cumulative_rotation_deg'], smooth['cumulative_rotation_deg']):.1f} "
            f"peak_force_delta={smooth['first_contact_peak_force_n'] - base['first_contact_peak_force_n']:.2f} "
            f"contact_lost_base={base['contact_lost']} contact_lost_smooth={smooth['contact_lost']} "
            f"success_base={base['success']} success_smooth={smooth['success']}"
        )

    print(f"\nreproducibility_fixed: {'YES' if repro_ok else 'NO'}")


if __name__ == "__main__":
    main()
