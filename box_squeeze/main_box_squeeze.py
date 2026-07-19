"""Milestone 1: hold a stationary box with broad gripper-side pads.

The gripper fingers stay fully open.  Both arms approach fixed opposite box
faces, then independent normal-force admittance offsets squeeze the box while
Cartesian impedance holds tangential position and orientation.  Once bilateral
force is stable, the world fixture is disabled; success requires frictional
side contact to support the free box against gravity for at least one second.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.clik import build_serial_arm, get_ee_transform
from control.clik.impedance import CartesianImpedanceConfig, CartesianImpedanceController
from control.squeeze import (
    HybridSqueezeController,
    SideSqueezeConfig,
    read_bilateral_pad_contact,
)
from robot.ffw_config import FFW_ARMS, FFW_GRIPPERS

SCENE = ROOT / "model" / "robotis_ffw" / "scene_ffw_sg2_fixed_base_box_squeeze.xml"


RIGHT_HOME_Q = np.array(
    [-0.534804, -0.460945, -1.278942, 0.871459, -0.281033, -0.379103, 0.050507]
)
LEFT_HOME_Q = np.array(
    [-0.534802, 0.460945, 1.278943, 0.871459, 0.281032, -0.379103, -0.050513]
)


class SqueezePhase(Enum):
    SETTLE = "settle"
    APPROACH = "approach"
    SQUEEZE = "squeeze"
    FREE_HOLD = "free_hold"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class RunConfig:
    viewer: bool = False
    log_path: Optional[str] = None
    squeeze: SideSqueezeConfig = field(default_factory=SideSqueezeConfig)


@dataclass
class SqueezeSummary:
    success: bool
    final_phase: str
    simulated_time_s: float
    bilateral_contact_time_s: Optional[float]
    fixture_release_time_s: Optional[float]
    free_hold_time_s: float
    final_left_force_n: float
    final_right_force_n: float
    peak_total_force_n: float
    peak_force_imbalance_n: float
    final_box_drop_m: float
    final_box_lateral_drift_m: float
    grippers_remained_open: bool


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _require_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(value)


def _actuator_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [_require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names],
        dtype=int,
    )


def _configure_impedance_axes(
    controller: CartesianImpedanceController, config: SideSqueezeConfig
) -> None:
    # World-y is normal to the selected fixed opposite faces. World-x/z are
    # tangential. Rotational axes remain isotropic.
    controller.K[:] = np.diag(
        [config.rotational_stiffness] * 3
        + [config.tangential_stiffness, config.normal_stiffness, config.tangential_stiffness]
    )
    controller.D[:] = np.diag(
        [config.rotational_damping] * 3
        + [config.tangential_damping, config.normal_damping, config.tangential_damping]
    )


def run_side_squeeze(config: Optional[RunConfig] = None) -> SqueezeSummary:
    config = config or RunConfig()
    cfg = config.squeeze
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    arms = {name: build_serial_arm(model, FFW_ARMS[name]) for name in ("left", "right")}
    homes = {"left": LEFT_HOME_Q.copy(), "right": RIGHT_HOME_Q.copy()}
    gripper_ids = {
        name: _actuator_ids(model, FFW_GRIPPERS[name].actuator_names)
        for name in ("left", "right")
    }
    for name, arm in arms.items():
        data.qpos[arm.qpos_indices] = homes[name]
        data.ctrl[arm.actuator_ids] = homes[name]
        data.ctrl[gripper_ids[name]] = FFW_GRIPPERS[name].open_ctrl
    mujoco.mj_forward(model, data)

    box_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY, "flying_box")
    fixture_id = _require_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "box_fixture")
    initial_box_position = data.xpos[box_body_id].copy()
    locked_box_center = initial_box_position.copy()
    initial_transforms = {
        name: get_ee_transform(data, arms[name]).copy() for name in ("left", "right")
    }
    desired_transforms = {
        name: initial_transforms[name].copy() for name in ("left", "right")
    }

    impedance = {}
    for name in ("left", "right"):
        controller = CartesianImpedanceController(
            CartesianImpedanceConfig(
                K_pos=cfg.tangential_stiffness,
                K_rot=cfg.rotational_stiffness,
                D_pos=cfg.tangential_damping,
                D_rot=cfg.rotational_damping,
                tau_limit=280.0,
                Kp_ns=6.0,
                Kd_ns=2.5,
            ),
            arms[name],
            model,
            homes[name],
        )
        _configure_impedance_axes(controller, cfg)
        impedance[name] = controller

    force_controller = HybridSqueezeController(cfg)
    empty_contact = read_bilateral_pad_contact(model, data)
    precontact_command = force_controller.update(
        box_center=locked_box_center,
        contact=empty_contact,
        dt=0.0,
    )
    precontact_positions = {
        "left": precontact_command.left_position.copy(),
        "right": precontact_command.right_position.copy(),
    }

    phase = SqueezePhase.SETTLE
    bilateral_contact_time: Optional[float] = None
    fixture_release_time: Optional[float] = None
    stable_timer = 0.0
    free_hold_timer = 0.0
    peak_total_force = 0.0
    peak_force_imbalance = 0.0
    log_rows: list[dict[str, float | str | int]] = []
    grippers_remained_open = True

    viewer = None
    if config.viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)

    total_steps = int(np.ceil(cfg.timeout_s / dt))
    try:
        for step in range(total_steps):
            if viewer is not None and not viewer.is_running():
                break
            time_s = float(data.time)
            contact = read_bilateral_pad_contact(model, data)
            total_force = contact.left.normal_force + contact.right.normal_force
            peak_total_force = max(peak_total_force, total_force)
            peak_force_imbalance = max(peak_force_imbalance, contact.force_imbalance)
            box_position = data.xpos[box_body_id].copy()
            box_drop = float(initial_box_position[2] - box_position[2])
            lateral_drift = float(np.linalg.norm(box_position[:2] - initial_box_position[:2]))

            if phase is SqueezePhase.SETTLE:
                if time_s >= cfg.initial_settle_s:
                    phase = SqueezePhase.APPROACH

            if phase is SqueezePhase.APPROACH:
                progress = (time_s - cfg.initial_settle_s) / max(cfg.approach_s, dt)
                blend = _smoothstep(progress)
                for name in ("left", "right"):
                    desired_transforms[name][:3, 3] = (
                        (1.0 - blend) * initial_transforms[name][:3, 3]
                        + blend * precontact_positions[name]
                    )
                if progress >= 1.0:
                    phase = SqueezePhase.SQUEEZE

            elif phase in {SqueezePhase.SQUEEZE, SqueezePhase.FREE_HOLD}:
                command = force_controller.update(
                    box_center=locked_box_center,
                    contact=contact,
                    dt=dt,
                )
                desired_transforms["left"][:3, 3] = command.left_position
                desired_transforms["right"][:3, 3] = command.right_position

                contact_is_stable = (
                    contact.left.normal_force >= cfg.minimum_contact_force
                    and contact.right.normal_force >= cfg.minimum_contact_force
                    and contact.force_imbalance <= cfg.maximum_force_imbalance
                    and contact.left.normal_force <= cfg.maximum_safe_force
                    and contact.right.normal_force <= cfg.maximum_safe_force
                )
                if phase is SqueezePhase.SQUEEZE:
                    stable_timer = stable_timer + dt if contact_is_stable else 0.0
                    if bilateral_contact_time is None and contact.bilateral:
                        bilateral_contact_time = time_s
                    if stable_timer >= cfg.bilateral_stable_s:
                        data.eq_active[fixture_id] = 0
                        mujoco.mj_forward(model, data)
                        fixture_release_time = time_s
                        phase = SqueezePhase.FREE_HOLD
                        free_hold_timer = 0.0
                else:
                    safe_hold = (
                        contact_is_stable
                        and box_drop <= cfg.maximum_box_drop
                        and lateral_drift <= cfg.maximum_box_lateral_drift
                    )
                    free_hold_timer = free_hold_timer + dt if safe_hold else 0.0
                    if box_drop > cfg.maximum_box_drop or lateral_drift > cfg.maximum_box_lateral_drift:
                        phase = SqueezePhase.FAILED
                    elif free_hold_timer >= cfg.required_free_hold_s:
                        phase = SqueezePhase.SUCCESS

            if (
                contact.left.normal_force > cfg.maximum_safe_force * 1.5
                or contact.right.normal_force > cfg.maximum_safe_force * 1.5
            ):
                phase = SqueezePhase.FAILED

            for name in ("left", "right"):
                impedance[name].apply(model, data, arms[name], desired_transforms[name])
                # Side squeezing must never use finger closure.
                data.ctrl[gripper_ids[name]] = FFW_GRIPPERS[name].open_ctrl
                grippers_remained_open = grippers_remained_open and bool(
                    np.allclose(data.ctrl[gripper_ids[name]], 0.0, atol=1e-12)
                )

            if step % 5 == 0:
                log_rows.append(
                    {
                        "time_s": time_s,
                        "phase": phase.value,
                        "left_force_n": contact.left.normal_force,
                        "right_force_n": contact.right.normal_force,
                        "force_imbalance_n": contact.force_imbalance,
                        "left_contact_count": contact.left.count,
                        "right_contact_count": contact.right.count,
                        "box_x": box_position[0],
                        "box_y": box_position[1],
                        "box_z": box_position[2],
                        "box_drop_m": box_drop,
                        "free_hold_s": free_hold_timer,
                        "fixture_active": int(data.eq_active[fixture_id]),
                    }
                )

            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if phase in {SqueezePhase.SUCCESS, SqueezePhase.FAILED}:
                break
    finally:
        if viewer is not None:
            viewer.close()

    final_contact = read_bilateral_pad_contact(model, data)
    final_box_position = data.xpos[box_body_id].copy()
    final_drop = float(initial_box_position[2] - final_box_position[2])
    final_lateral = float(
        np.linalg.norm(final_box_position[:2] - initial_box_position[:2])
    )
    if config.log_path and log_rows:
        path = Path(config.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    return SqueezeSummary(
        success=phase is SqueezePhase.SUCCESS,
        final_phase=phase.value,
        simulated_time_s=float(data.time),
        bilateral_contact_time_s=bilateral_contact_time,
        fixture_release_time_s=fixture_release_time,
        free_hold_time_s=float(free_hold_timer),
        final_left_force_n=float(final_contact.left.normal_force),
        final_right_force_n=float(final_contact.right.normal_force),
        peak_total_force_n=float(peak_total_force),
        peak_force_imbalance_n=float(peak_force_imbalance),
        final_box_drop_m=final_drop,
        final_box_lateral_drift_m=final_lateral,
        grippers_remained_open=grippers_remained_open,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--log",
        default=str(ROOT / "sweep_results" / "box_side_squeeze.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_side_squeeze(RunConfig(viewer=args.viewer, log_path=args.log))
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    if not summary.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
