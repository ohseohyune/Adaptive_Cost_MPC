"""Milestone 2: intercept and side-squeeze a randomly launched ballistic box."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from control.clik import build_serial_arm, get_ee_transform
from control.clik.catching import adaptive_stiffness
from control.clik.impedance import CartesianImpedanceConfig, CartesianImpedanceController
from control.squeeze import (
    BallisticBoxPredictor,
    BimanualWrenchAllocator,
    BoxDomainParameters,
    BoxFaceInterceptionPlanner,
    DynamicSideSqueezeConfig,
    FirstContactForceLimiter,
    HybridSqueezeController,
    QuaternionAngularVelocityPredictor,
    RotatingSideSqueezeConfig,
    SE3BoxFaceTargetPlanner,
    adaptive_impact_command,
    apply_box_domain_randomization,
    read_bilateral_pad_contact,
    minimum_symmetric_squeeze_force,
    resolve_ballistic_launch_position,
    resolve_ballistic_launch_velocity,
    rotation_exp,
    rotation_to_quaternion,
)
from main_box_squeeze import LEFT_HOME_Q, RIGHT_HOME_Q
from robot.ffw_config import FFW_ARMS, FFW_GRIPPERS


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "model/robotis_ffw/scene_ffw_sg2_fixed_base_box_dynamic_squeeze.xml"
HAND_CAMERA_COLLISION_BIT = 32


class DynamicSqueezePhase(Enum):
    PREPARE = "prepare"
    TRACK = "track"
    SINGLE_CONTACT = "single_contact"
    IMPACT = "impact"
    VELOCITY_MATCH = "velocity_match"
    CAPTURE = "capture"
    HOLD = "hold"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class DynamicRunConfig:
    viewer: bool = False
    viewer_show_prepare: bool = False
    log_path: Optional[str] = None
    squeeze: DynamicSideSqueezeConfig = field(default_factory=DynamicSideSqueezeConfig)
    domain_parameters: Optional[BoxDomainParameters] = None
    collision_mode: str = "miss_backstop"


@dataclass
class DynamicSqueezeSummary:
    success: bool
    final_phase: str
    random_seed: int
    launch_position_m: tuple[float, float, float]
    launch_velocity_mps: tuple[float, float, float]
    launch_angular_velocity_radps: tuple[float, float, float]
    box_half_size_m: tuple[float, float, float]
    box_mass_kg: float
    box_friction: float
    initial_ttc_s: float
    simultaneous_start: bool
    collision_mode: str
    robot_collision_enabled: bool
    simulated_time_s: float
    first_contact_time_s: Optional[float]
    first_nonpad_contact_time_s: Optional[float]
    first_nonpad_contact_object: str
    bilateral_contact_time_s: Optional[float]
    first_contact_peak_force_n: float
    first_contact_force_limit_n: float
    peak_total_force_n: float
    minimum_no_slip_force_per_pad_n: float
    final_hold_force_per_pad_n: float
    minimum_ttc_stiffness_npm: float
    final_box_speed_mps: float
    final_angular_speed_radps: float
    peak_angular_speed_radps: float
    prediction_velocity_error_mps: float
    interception_center_error_m: float
    reachable_intercept_seen: bool
    angular_prediction_error_radps: float
    wrench_qp_status: str
    wrench_tracking_error: float
    slip_cost: float
    angular_velocity_cost: float
    dynamic_hold_time_s: float
    capture_drop_m: float
    capture_drift_m: float
    grippers_remained_open: bool
    failure_reason: str


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


def _disable_duplicate_end_effector_collisions(
    model: mujoco.MjModel,
    pad_ids: set[int],
) -> None:
    """Let broad pads represent the distal mesh collision geometry.

    The link-7/hand meshes geometrically overlap the purpose-built broad pad.
    Allowing both to collide with the box creates a duplicate contact before
    the controller can establish bilateral pad contact. Removing ordinary
    bit-1 *contype* from those geoms preserves their other collision direction
    while preventing box-affinity bit 1 from selecting them. The named camera
    proxies use dedicated bit 32, so they remain physical.
    """

    for geom_id in range(model.ngeom):
        if geom_id in pad_ids:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        )
        if body_name in {"arm_l_link7", "arm_r_link7"} or body_name.startswith(
            ("gripper_l_", "gripper_r_")
        ):
            model.geom_contype[geom_id] &= ~1


def _set_impedance_axes(
    controller: CartesianImpedanceController,
    config: DynamicSideSqueezeConfig,
    *,
    tangential_stiffness: float,
    normal_stiffness: float,
    rotational_stiffness: float,
) -> None:
    controller.K[:] = np.diag(
        [rotational_stiffness] * 3
        + [tangential_stiffness, normal_stiffness, tangential_stiffness]
    )
    controller.D[:] = np.diag(
        [config.rotational_damping] * 3
        + [config.tangential_damping, config.normal_damping, config.tangential_damping]
    )


def _oracle_plane_intersection(
    position: np.ndarray,
    velocity: np.ndarray,
    gravity: np.ndarray,
    plane_x: float,
) -> np.ndarray:
    if abs(float(velocity[0])) < 1e-9:
        return position.copy()
    ttc = max(0.0, (plane_x - float(position[0])) / float(velocity[0]))
    return position + velocity * ttc + 0.5 * gravity * ttc**2


def _limit_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm <= 1e-12:
        return vector
    return vector * (maximum / norm)


def _move_toward(current: np.ndarray, target: np.ndarray, maximum_step: float) -> np.ndarray:
    delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance <= maximum_step or distance <= 1e-12:
        return np.asarray(target, dtype=float).copy()
    return np.asarray(current, dtype=float) + delta * (maximum_step / distance)


def _ee_rotations_for_box(
    box_rotation: np.ndarray,
    left_ee_to_pad: np.ndarray,
    right_ee_to_pad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_axis, y_axis, z_axis = box_rotation[:, 0], box_rotation[:, 1], box_rotation[:, 2]
    left_pad = np.column_stack([y_axis, -x_axis, z_axis])
    right_pad = np.column_stack([-y_axis, x_axis, z_axis])
    return left_pad @ left_ee_to_pad.T, right_pad @ right_ee_to_pad.T


def run_dynamic_side_squeeze(
    config: Optional[DynamicRunConfig] = None,
) -> DynamicSqueezeSummary:
    config = config or DynamicRunConfig()
    if config.collision_mode not in {"pad_only", "miss_backstop", "full"}:
        raise ValueError(
            "collision_mode must be pad_only, miss_backstop, or full"
        )
    cfg = config.squeeze
    rng = np.random.default_rng(cfg.random_seed)
    sampled_launch_velocity = rng.uniform(
        cfg.launch_velocity_low, cfg.launch_velocity_high
    )
    launch_position = resolve_ballistic_launch_position(
        cfg, sampled_launch_velocity
    )
    launch_velocity = resolve_ballistic_launch_velocity(
        cfg, sampled_launch_velocity, launch_position=launch_position
    )
    rotating_cfg = cfg if isinstance(cfg, RotatingSideSqueezeConfig) else None
    launch_angular_velocity = (
        rng.uniform(
            rotating_cfg.random_angular_velocity_low,
            rotating_cfg.random_angular_velocity_high,
        )
        if rotating_cfg is not None
        else np.zeros(3)
    )
    gravity = np.asarray(cfg.gravity, dtype=float)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    if config.domain_parameters is not None:
        if not np.isclose(cfg.box_half_y, config.domain_parameters.half_size[1]):
            raise ValueError(
                "controller box_half_y must match randomized physical half-size"
            )
        apply_box_domain_randomization(model, config.domain_parameters)
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

    box_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_box")
    box_geom_id = _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
    # Bit 1 is used by ordinary robot/floor geoms. Bit 32 is reserved for the
    # hand-camera protrusions and remains active from launch in both physical
    # modes, independently of the duplicate distal-mesh suppression below.
    if config.collision_mode != "pad_only":
        model.geom_conaffinity[box_geom_id] |= HAND_CAMERA_COLLISION_BIT
    else:
        model.geom_conaffinity[box_geom_id] &= ~HAND_CAMERA_COLLISION_BIT
    if config.collision_mode == "full":
        model.geom_conaffinity[box_geom_id] |= 1
    else:
        model.geom_conaffinity[box_geom_id] &= ~1
    robot_collision_enabled = config.collision_mode == "full"
    box_joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "dynamic_box_joint")
    box_qpos_address = int(model.jnt_qposadr[box_joint_id])
    box_dof_address = int(model.jnt_dofadr[box_joint_id])
    fixture_id = _require_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "box_launch_fixture")
    pad_ids = {
        name: _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_squeeze_pad")
        for name in ("left", "right")
    }
    pad_box_pair_ids = tuple(
        _require_id(model, mujoco.mjtObj.mjOBJ_PAIR, name)
        for name in ("dynamic_left_pad_box", "dynamic_right_pad_box")
    )
    impact_pair_time_constants = {
        pair_id: float(model.pair_solref[pair_id, 0])
        for pair_id in pad_box_pair_ids
    }
    data.qpos[box_qpos_address : box_qpos_address + 3] = np.asarray(
        launch_position, dtype=float
    )
    mujoco.mj_forward(model, data)
    if config.collision_mode != "pad_only":
        _disable_duplicate_end_effector_collisions(
            model, set(pad_ids.values())
        )

    initial_transforms = {
        name: get_ee_transform(data, arms[name]).copy() for name in ("left", "right")
    }
    pad_relative_rotations = {
        name: initial_transforms[name][:3, :3].T
        @ data.geom_xmat[pad_ids[name]].reshape(3, 3)
        for name in ("left", "right")
    }
    desired_transforms = {
        name: initial_transforms[name].copy() for name in ("left", "right")
    }
    launch_ttc = (cfg.catch_plane_x - launch_position[0]) / launch_velocity[0]
    nominal_catch_center = (
        launch_position + launch_velocity * launch_ttc + 0.5 * gravity * launch_ttc**2
    )

    force_controller = HybridSqueezeController(cfg)
    nominal_command = force_controller.update(
        box_center=nominal_catch_center,
        contact=read_bilateral_pad_contact(model, data, box_geom_name="dynamic_box_geom"),
        dt=0.0,
    )
    ready_positions = {
        "left": nominal_command.left_position.copy(),
        "right": nominal_command.right_position.copy(),
    }

    impedance: dict[str, CartesianImpedanceController] = {}
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
        impedance[name] = controller

    predictor = BallisticBoxPredictor(
        gravity=gravity,
        velocity_alpha=cfg.predictor_velocity_alpha,
        max_speed=cfg.predictor_max_speed,
    )
    planner = BoxFaceInterceptionPlanner(cfg)
    rotational_predictor = (
        QuaternionAngularVelocityPredictor(rotating_cfg)
        if rotating_cfg is not None
        else None
    )
    se3_planner = (
        SE3BoxFaceTargetPlanner(
            rotating_cfg,
            left_ee_to_pad_rotation=pad_relative_rotations["left"],
            right_ee_to_pad_rotation=pad_relative_rotations["right"],
        )
        if rotating_cfg is not None
        else None
    )
    wrench_allocator = (
        BimanualWrenchAllocator(rotating_cfg)
        if rotating_cfg is not None
        else None
    )
    limiter = FirstContactForceLimiter(cfg)

    phase = DynamicSqueezePhase.PREPARE
    launch_time: float | None = None
    bilateral_contact_time: float | None = None
    first_nonpad_contact_time: float | None = None
    first_nonpad_contact_object = ""
    capture_center: np.ndarray | None = None
    capture_rotation: np.ndarray | None = None
    capture_timer = 0.0
    velocity_match_timer = 0.0
    hold_timer = 0.0
    hold_unsafe_timer = 0.0
    single_contact_start_time: float | None = None
    locked_interception_time_s: float | None = None
    capture_reference_velocity = np.zeros(3)
    capture_reference_angular_velocity = np.zeros(3)
    peak_total_force = 0.0
    peak_angular_speed = float(np.linalg.norm(launch_angular_velocity))
    minimum_stiffness = cfg.tangential_stiffness
    velocity_errors: list[float] = []
    interception_errors: list[float] = []
    angular_velocity_errors: list[float] = []
    reachable_intercept_seen = False
    grippers_remained_open = True
    failure_reason = ""
    wrench_qp_status = "disabled"
    wrench_tracking_error = 0.0
    slip_cost = 0.0
    angular_velocity_cost = 0.0
    wrench_feedforward = {
        "left": np.zeros(6),
        "right": np.zeros(6),
    }
    previous_pad_positions = {
        name: data.geom_xpos[pad_ids[name]].copy() for name in ("left", "right")
    }
    log_rows: list[dict[str, float | str | int]] = []

    object_mass = (
        float(config.domain_parameters.mass)
        if config.domain_parameters is not None
        else float(rotating_cfg.box_mass if rotating_cfg is not None else 0.50)
    )
    object_friction = (
        float(config.domain_parameters.friction)
        if config.domain_parameters is not None
        else 1.20
    )
    minimum_hold_force = minimum_symmetric_squeeze_force(
        mass=object_mass,
        friction=object_friction,
        gravity=gravity,
        calibration_factor=cfg.minimum_hold_force_calibration,
    )
    minimum_hold_force = float(
        np.clip(
            minimum_hold_force,
            cfg.minimum_contact_force,
            cfg.maximum_hold_normal_force,
        )
    )
    adaptive_hold_force = minimum_hold_force

    if cfg.simultaneous_start:
        data.eq_active[fixture_id] = 0
        data.qvel[box_dof_address : box_dof_address + 3] = launch_velocity
        data.qvel[box_dof_address + 3 : box_dof_address + 6] = (
            launch_angular_velocity
        )
        mujoco.mj_forward(model, data)
        predictor.reset(initial_velocity=launch_velocity)
        if rotational_predictor is not None:
            rotational_predictor.reset(initial_velocity=launch_velocity)
        force_controller.reset()
        launch_time = float(data.time)
        phase = DynamicSqueezePhase.TRACK

    viewer = None
    if config.viewer and (config.viewer_show_prepare or cfg.simultaneous_start):
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)

    total_steps = int(np.ceil(cfg.timeout_s / dt))
    try:
        for step in range(total_steps):
            if viewer is not None and not viewer.is_running():
                failure_reason = "viewer closed"
                break
            time_s = float(data.time)
            box_position = data.xpos[box_body_id].copy()
            box_velocity = data.qvel[box_dof_address : box_dof_address + 3].copy()
            box_rotation = data.xmat[box_body_id].reshape(3, 3).copy()
            box_angular_velocity = data.qvel[
                box_dof_address + 3 : box_dof_address + 6
            ].copy()
            pad_positions = {
                name: data.geom_xpos[pad_ids[name]].copy()
                for name in ("left", "right")
            }
            pad_velocities = {
                name: (pad_positions[name] - previous_pad_positions[name]) / dt
                for name in ("left", "right")
            }
            previous_pad_positions = {
                name: position.copy() for name, position in pad_positions.items()
            }
            impact_y_axis = box_rotation[:, 1]
            left_contact_velocity = box_velocity + np.cross(
                box_angular_velocity, cfg.box_half_y * impact_y_axis
            )
            right_contact_velocity = box_velocity + np.cross(
                box_angular_velocity, -cfg.box_half_y * impact_y_axis
            )
            relative_normal_speed = max(
                abs(
                    float(
                        np.dot(
                            left_contact_velocity - pad_velocities["left"],
                            -impact_y_axis,
                        )
                    )
                ),
                abs(
                    float(
                        np.dot(
                            right_contact_velocity - pad_velocities["right"],
                            impact_y_axis,
                        )
                    )
                ),
            )
            if config.domain_parameters is not None:
                half_x, _, half_z = config.domain_parameters.half_size
            else:
                half_x, half_z = 0.055, 0.055
            impact_command = adaptive_impact_command(
                cfg,
                object_mass=object_mass,
                contact_face_area=4.0 * half_x * half_z,
                relative_normal_speed=relative_normal_speed,
            )
            if launch_time is not None:
                peak_angular_speed = max(
                    peak_angular_speed,
                    float(np.linalg.norm(box_angular_velocity)),
                )
            contact = read_bilateral_pad_contact(
                model, data, box_geom_name="dynamic_box_geom"
            )
            if first_nonpad_contact_time is None:
                for raw_contact in data.contact[: data.ncon]:
                    geom1, geom2 = int(raw_contact.geom1), int(raw_contact.geom2)
                    if box_geom_id not in (geom1, geom2):
                        continue
                    other_geom = geom2 if geom1 == box_geom_id else geom1
                    if other_geom in pad_ids.values():
                        continue
                    other_body = int(model.geom_bodyid[other_geom])
                    first_nonpad_contact_time = time_s
                    first_nonpad_contact_object = (
                        mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                        )
                        or mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_BODY, other_body
                        )
                        or f"geom_{other_geom}"
                    )
                    break
            impact = limiter.update(time_s, contact)
            if impact.first_contact_time_s is not None:
                # The XML pair is deliberately soft during impact. Keeping
                # that compliance for the full hold causes artificial
                # tangential creep. Ramp to the hold contact after the
                # force-limited window instead of switching discontinuously.
                transition_elapsed = max(
                    0.0,
                    time_s
                    - impact.first_contact_time_s
                    - cfg.first_contact_window_s,
                )
                contact_blend = _smoothstep(
                    transition_elapsed / max(cfg.hold_contact_transition_s, dt)
                )
                for pair_id in pad_box_pair_ids:
                    model.pair_solref[pair_id, 0] = (
                        (1.0 - contact_blend)
                        * impact_pair_time_constants[pair_id]
                        + contact_blend * cfg.hold_contact_time_constant_s
                    )
                    model.pair_solref[pair_id, 1] = cfg.hold_contact_damping_ratio
            if (
                config.collision_mode == "miss_backstop"
                and not robot_collision_enabled
                and launch_time is not None
                and box_position[0] <= cfg.catch_plane_x - 0.02
            ):
                model.geom_conaffinity[box_geom_id] |= 1
                robot_collision_enabled = True
            peak_total_force = max(
                peak_total_force,
                contact.left.normal_force + contact.right.normal_force,
            )

            if phase is DynamicSqueezePhase.PREPARE:
                blend = _smoothstep(time_s / max(cfg.prepare_s, dt))
                for name in ("left", "right"):
                    desired_transforms[name][:3, 3] = (
                        (1.0 - blend) * initial_transforms[name][:3, 3]
                        + blend * ready_positions[name]
                    )
                if time_s >= cfg.prepare_s:
                    data.eq_active[fixture_id] = 0
                    data.qvel[box_dof_address : box_dof_address + 3] = launch_velocity
                    data.qvel[box_dof_address + 3 : box_dof_address + 6] = (
                        launch_angular_velocity
                    )
                    mujoco.mj_forward(model, data)
                    predictor.reset(initial_velocity=launch_velocity)
                    if rotational_predictor is not None:
                        rotational_predictor.reset(
                            initial_velocity=launch_velocity
                        )
                    force_controller.reset()
                    launch_time = time_s
                    phase = DynamicSqueezePhase.TRACK
                    # PREPARE belongs to environment reset. Unless explicitly
                    # requested, open the viewer at the exact step where box
                    # release and active arm tracking start together.
                    if config.viewer and viewer is None:
                        from mujoco import viewer as mj_viewer

                        viewer = mj_viewer.launch_passive(model, data)

            if phase in {
                DynamicSqueezePhase.TRACK,
                DynamicSqueezePhase.SINGLE_CONTACT,
                DynamicSqueezePhase.IMPACT,
                DynamicSqueezePhase.VELOCITY_MATCH,
                DynamicSqueezePhase.CAPTURE,
                DynamicSqueezePhase.HOLD,
            }:
                relative_time = time_s - float(launch_time)
                prediction = predictor.update(relative_time, box_position)
                plan = planner.plan(
                    prediction,
                    left_pad_position=data.geom_xpos[pad_ids["left"]],
                    right_pad_position=data.geom_xpos[pad_ids["right"]],
                )
                plan_ttc = plan.time_to_contact_s
                plan_center = plan.box_center
                plan_left_position = plan.left_pad_target
                plan_right_position = plan.right_pad_target
                planned_rotations: tuple[np.ndarray, np.ndarray] | None = None
                if rotational_predictor is not None and se3_planner is not None:
                    rotating_prediction = rotational_predictor.update(
                        relative_time,
                        box_position,
                        rotation_to_quaternion(box_rotation),
                    )
                    se3_plan = se3_planner.plan(
                        rotating_prediction,
                        left_pad_position=data.geom_xpos[pad_ids["left"]],
                        right_pad_position=data.geom_xpos[pad_ids["right"]],
                        target_ttc_s=(
                            max(
                                cfg.minimum_intercept_ttc,
                                locked_interception_time_s - relative_time,
                            )
                            if locked_interception_time_s is not None
                            else None
                        ),
                    )
                    if (
                        locked_interception_time_s is None
                        and relative_time >= cfg.interception_lock_delay_s
                        and se3_plan.confidence >= 0.5
                        and se3_plan.reachable
                        and np.isfinite(se3_plan.time_to_contact_s)
                    ):
                        biased_ttc = float(
                            np.clip(
                                se3_plan.time_to_contact_s + cfg.catch_ttc_offset_s,
                                cfg.minimum_intercept_ttc,
                                cfg.maximum_intercept_ttc,
                            )
                        )
                        locked_interception_time_s = relative_time + biased_ttc
                    elif (
                        locked_interception_time_s is not None
                        and impact.first_contact_time_s is None
                        and not se3_plan.reachable
                    ):
                        # The committed catch point/time is no longer
                        # reachable before contact (e.g. the box decelerated
                        # less than predicted). Unlock immediately so the next
                        # control step re-searches candidate catch points via
                        # ``_select_candidate_ttc`` instead of continuing to
                        # chase a target the pads cannot reach in time.
                        locked_interception_time_s = None
                    plan_ttc = se3_plan.time_to_contact_s
                    plan_center = se3_plan.box_center
                    plan_left_position = se3_plan.left_pad_transform[:3, 3]
                    plan_right_position = se3_plan.right_pad_transform[:3, 3]
                    planned_rotations = (
                        se3_plan.left_pad_transform[:3, :3],
                        se3_plan.right_pad_transform[:3, :3],
                    )
                    reachable_intercept_seen = (
                        reachable_intercept_seen or se3_plan.reachable
                    )
                    if (
                        rotating_prediction.angular_confidence >= 0.5
                        and impact.first_contact_time_s is None
                    ):
                        angular_velocity_errors.append(
                            float(
                                np.linalg.norm(
                                    rotating_prediction.angular_velocity
                                    - box_angular_velocity
                                )
                            )
                        )
                else:
                    reachable_intercept_seen = reachable_intercept_seen or plan.reachable
                if (
                    prediction.confidence >= 0.5
                    and impact.first_contact_time_s is None
                    and np.isfinite(plan_ttc)
                ):
                    velocity_errors.append(float(np.linalg.norm(prediction.velocity - box_velocity)))
                    oracle = _oracle_plane_intersection(
                        box_position, box_velocity, gravity, cfg.catch_plane_x
                    )
                    interception_errors.append(float(np.linalg.norm(plan_center - oracle)))

                if (
                    phase is DynamicSqueezePhase.VELOCITY_MATCH
                    and capture_center is not None
                ):
                    capture_reference_velocity *= np.exp(
                        -dt / max(cfg.capture_linear_time_constant_s, dt)
                    )
                    capture_reference_angular_velocity *= np.exp(
                        -dt / max(cfg.capture_angular_time_constant_s, dt)
                    )
                    capture_center = (
                        capture_center + capture_reference_velocity * dt
                    )
                    if capture_rotation is not None:
                        capture_rotation = (
                            rotation_exp(
                                capture_reference_angular_velocity * dt
                            )
                            @ capture_rotation
                        )
                    velocity_match_timer += dt

                # The compliant first-contact window deliberately uses low
                # impact forces (impact_command.desired_force), so the box
                # keeps falling under gravity largely unarrested for up to
                # first_contact_window_s. Freezing the pad target at the
                # instant of first touch leaves the pads chasing a static
                # point while the real box keeps dropping/advancing past it,
                # which is what tore contact loose and spun the box up.
                # Keep following the live prediction through single contact
                # and the compliant impact window; only snap to the fixed
                # post-arrest reference once VELOCITY_MATCH takes over.
                tracking_live_prediction = bool(
                    capture_center is None
                    or (
                        phase
                        in {
                            DynamicSqueezePhase.SINGLE_CONTACT,
                            DynamicSqueezePhase.IMPACT,
                        }
                        and impact.in_first_contact_window
                    )
                )
                if tracking_live_prediction:
                    target_center = plan_center
                else:
                    target_center = capture_center
                    if phase is DynamicSqueezePhase.HOLD:
                        # Static pad targets allow compliant contact to creep
                        # over a multi-second hold. Move both pads together in
                        # the contact tangent plane to restore the captured
                        # box center, while leaving normal-force control alone.
                        normal_axis = (
                            capture_rotation[:, 1]
                            if capture_rotation is not None
                            else np.array([0.0, 1.0, 0.0])
                        )
                        center_error = capture_center - box_position
                        tangent_error = center_error - normal_axis * np.dot(
                            normal_axis, center_error
                        )
                        target_center = capture_center + _limit_norm(
                            cfg.hold_slip_recovery_gain * tangent_error,
                            cfg.maximum_hold_recovery_offset,
                        )

                allocation = None
                if wrench_allocator is not None and rotating_cfg is not None:
                    y_axis = box_rotation[:, 1]
                    left_contact_position = (
                        contact.left.mean_position
                        if contact.left.active
                        else box_position + rotating_cfg.box_half_y * y_axis
                    )
                    right_contact_position = (
                        contact.right.mean_position
                        if contact.right.active
                        else box_position - rotating_cfg.box_half_y * y_axis
                    )
                    allocation = wrench_allocator.solve(
                        box_rotation=box_rotation,
                        left_contact_position=left_contact_position,
                        right_contact_position=right_contact_position,
                        box_center=box_position,
                        linear_velocity=box_velocity,
                        angular_velocity=box_angular_velocity,
                        object_mass=object_mass,
                        box_half_size=(half_x, cfg.box_half_y, half_z),
                        friction_coefficient=object_friction,
                        dt=dt,
                        capture_phase=phase
                        in {
                            DynamicSqueezePhase.SINGLE_CONTACT,
                            DynamicSqueezePhase.IMPACT,
                            DynamicSqueezePhase.VELOCITY_MATCH,
                            DynamicSqueezePhase.CAPTURE,
                        }
                        or (
                            phase is DynamicSqueezePhase.HOLD
                            and np.linalg.norm(box_angular_velocity)
                            > 0.8 * rotating_cfg.maximum_capture_angular_speed
                        ),
                    )
                    wrench_qp_status = allocation.status
                    wrench_tracking_error = allocation.tracking_error
                    slip_cost = allocation.slip_cost
                    angular_velocity_cost = allocation.angular_velocity_cost
                    if (
                        impact.first_contact_time_s is not None
                        and not impact.in_first_contact_window
                    ):
                        wrench_feedforward["left"] = -rotating_cfg.wrench_feedforward_scale * np.concatenate(
                            [allocation.left_wrench[3:], allocation.left_wrench[:3]]
                        )
                        wrench_feedforward["right"] = -rotating_cfg.wrench_feedforward_scale * np.concatenate(
                            [allocation.right_wrench[3:], allocation.right_wrench[:3]]
                        )

                squeeze_now = bool(
                    contact.left.active
                    or contact.right.active
                    or (
                        np.isfinite(plan_ttc)
                        and plan_ttc <= cfg.squeeze_lead_s
                    )
                )
                if phase is DynamicSqueezePhase.HOLD:
                    gravity_norm = max(float(np.linalg.norm(gravity)), 1e-9)
                    gravity_direction = gravity / gravity_norm
                    left_relative = (
                        left_contact_velocity - pad_velocities["left"]
                    )
                    right_relative = (
                        right_contact_velocity - pad_velocities["right"]
                    )
                    left_tangent = left_relative - impact_y_axis * np.dot(
                        impact_y_axis, left_relative
                    )
                    right_tangent = right_relative - impact_y_axis * np.dot(
                        impact_y_axis, right_relative
                    )
                    contact_slip_speed = max(
                        float(np.linalg.norm(left_tangent)),
                        float(np.linalg.norm(right_tangent)),
                    )
                    downward_slip_speed = max(
                        0.0, float(np.dot(box_velocity, gravity_direction))
                    )
                    slip_speed = max(contact_slip_speed, downward_slip_speed)
                    measured_tangential_force = max(
                        contact.left.tangential_force,
                        contact.right.tangential_force,
                    )
                    friction_force = min(
                        cfg.maximum_hold_normal_force,
                        1.15
                        * measured_tangential_force
                        / max(object_friction, 1e-6),
                    )
                    force_slip_detected = bool(
                        downward_slip_speed > cfg.hold_slip_speed_threshold
                        or contact_slip_speed
                        > cfg.maximum_contact_slip_speed
                    )
                    # Absent a genuine detected slip, keep the requested hold
                    # force near the computed minimum no-slip force instead of
                    # creeping toward the global safety ceiling. A real,
                    # physics-derived slip demand (friction_force) is still
                    # allowed to exceed this working ceiling, up to the hard
                    # maximum_hold_normal_force limit.
                    hold_force_ceiling = min(
                        cfg.maximum_hold_normal_force,
                        minimum_hold_force * cfg.hold_force_margin_ratio,
                    )
                    if force_slip_detected:
                        adaptive_hold_force = max(
                            adaptive_hold_force, friction_force
                        )
                        adaptive_hold_force = min(
                            max(hold_force_ceiling, friction_force),
                            adaptive_hold_force
                            + cfg.hold_force_increase_rate * dt,
                        )
                    else:
                        adaptive_hold_force = max(
                            minimum_hold_force,
                            adaptive_hold_force
                            - cfg.hold_force_release_rate * dt,
                        )
                        adaptive_hold_force = min(
                            adaptive_hold_force, hold_force_ceiling
                        )
                    adaptive_hold_force = min(
                        adaptive_hold_force, cfg.maximum_hold_normal_force
                    )
                if squeeze_now:
                    if phase is DynamicSqueezePhase.HOLD:
                        requested_force = adaptive_hold_force
                    elif impact.in_first_contact_window:
                        requested_force: float | tuple[float, float] = (
                            impact_command.desired_force
                        )
                    elif allocation is not None:
                        left_inward = -box_rotation[:, 1]
                        right_inward = box_rotation[:, 1]
                        blend = rotating_cfg.wrench_normal_force_blend
                        requested_force = (
                            float(
                                np.clip(
                                    (1.0 - blend) * cfg.desired_normal_force
                                    + blend
                                    * np.dot(left_inward, allocation.left_wrench[:3]),
                                    cfg.minimum_contact_force,
                                    rotating_cfg.maximum_qp_normal_force,
                                )
                            ),
                            float(
                                np.clip(
                                    (1.0 - blend) * cfg.desired_normal_force
                                    + blend
                                    * np.dot(right_inward, allocation.right_wrench[:3]),
                                    cfg.minimum_contact_force,
                                    rotating_cfg.maximum_qp_normal_force,
                                )
                            ),
                        )
                    else:
                        requested_force = cfg.desired_normal_force
                    command = force_controller.update(
                        box_center=target_center,
                        contact=contact,
                        dt=dt,
                        desired_force=requested_force,
                    )
                    if rotating_cfg is None:
                        left_target, right_target = limiter.relieve(
                            command, contact, impact
                        )
                    else:
                        y_axis = (
                            capture_rotation[:, 1]
                            if not tracking_live_prediction
                            and capture_rotation is not None
                            else box_rotation[:, 1]
                        )
                        nominal = cfg.box_half_y + cfg.pad_half_thickness
                        left_target = target_center + y_axis * (
                            nominal - command.left_compression
                        )
                        right_target = target_center - y_axis * (
                            nominal - command.right_compression
                        )
                        if impact.in_first_contact_window:
                            left_relief, right_relief = limiter.relief_distances(
                                contact, impact
                            )
                            left_target += y_axis * left_relief
                            right_target -= y_axis * right_relief
                else:
                    left_target = plan_left_position
                    right_target = plan_right_position

                if rotating_cfg is not None:
                    if not tracking_live_prediction and capture_rotation is not None:
                        base_rotations = _ee_rotations_for_box(
                            capture_rotation,
                            pad_relative_rotations["left"],
                            pad_relative_rotations["right"],
                        )
                    elif planned_rotations is not None:
                        base_rotations = planned_rotations
                    else:
                        base_rotations = (
                            desired_transforms["left"][:3, :3],
                            desired_transforms["right"][:3, :3],
                        )
                    if allocation is not None:
                        for index, (name, wrench) in enumerate(
                            zip(("left", "right"), (allocation.left_wrench, allocation.right_wrench))
                        ):
                            inward = (
                                -box_rotation[:, 1]
                                if name == "left"
                                else box_rotation[:, 1]
                            )
                            tangential_force = wrench[:3] - inward * np.dot(
                                inward, wrench[:3]
                            )
                            position_offset = _limit_norm(
                                rotating_cfg.wrench_position_compliance
                                * tangential_force
                                / max(cfg.tangential_stiffness, 1.0),
                                rotating_cfg.maximum_wrench_position_offset,
                            )
                            rotation_offset = _limit_norm(
                                rotating_cfg.wrench_rotation_compliance
                                * wrench[3:]
                                / max(cfg.rotational_stiffness, 1.0),
                                rotating_cfg.maximum_wrench_rotation_offset,
                            )
                            if name == "left":
                                left_target += position_offset
                            else:
                                right_target += position_offset
                            desired_transforms[name][:3, :3] = (
                                rotation_exp(rotation_offset) @ base_rotations[index]
                            )
                    else:
                        desired_transforms["left"][:3, :3] = base_rotations[0]
                        desired_transforms["right"][:3, :3] = base_rotations[1]
                if impact.first_contact_time_s is None:
                    maximum_step = cfg.maximum_pad_reach_speed * dt
                    left_target = _move_toward(
                        desired_transforms["left"][:3, 3],
                        left_target,
                        maximum_step,
                    )
                    right_target = _move_toward(
                        desired_transforms["right"][:3, 3],
                        right_target,
                        maximum_step,
                    )
                desired_transforms["left"][:3, 3] = left_target
                desired_transforms["right"][:3, 3] = right_target

                if impact.first_contact_time_s is None:
                    ttc = plan_ttc
                    tangential_k = float(
                        adaptive_stiffness(
                            ttc,
                            cfg.tangential_stiffness,
                            impact_command.tangential_stiffness,
                            cfg.ttc_soften_window_s,
                        )
                    )
                    normal_k = float(
                        adaptive_stiffness(
                            ttc,
                            cfg.normal_stiffness,
                            impact_command.normal_stiffness,
                            cfg.ttc_soften_window_s,
                        )
                    )
                    rotational_k = float(
                        adaptive_stiffness(
                            ttc,
                            cfg.rotational_stiffness,
                            impact_command.rotational_stiffness,
                            cfg.ttc_soften_window_s,
                        )
                    )
                elif impact.in_first_contact_window:
                    tangential_k = impact_command.tangential_stiffness
                    normal_k = impact_command.normal_stiffness
                    rotational_k = impact_command.rotational_stiffness
                else:
                    tangential_k = cfg.tangential_stiffness
                    normal_k = cfg.normal_stiffness
                    rotational_k = cfg.rotational_stiffness
                minimum_stiffness = min(minimum_stiffness, tangential_k)
                for controller in impedance.values():
                    _set_impedance_axes(
                        controller,
                        cfg,
                        tangential_stiffness=tangential_k,
                        normal_stiffness=normal_k,
                        rotational_stiffness=rotational_k,
                    )

                if (
                    impact.first_contact_time_s is not None
                    and phase is DynamicSqueezePhase.TRACK
                ):
                    capture_center = box_position.copy()
                    capture_reference_velocity = _limit_norm(
                        box_velocity, cfg.maximum_capture_speed
                    )
                    capture_reference_angular_velocity = _limit_norm(
                        box_angular_velocity,
                        (
                            rotating_cfg.maximum_capture_angular_speed
                            if rotating_cfg is not None
                            else 0.0
                        ),
                    )
                    if rotating_cfg is not None:
                        capture_rotation = box_rotation.copy()
                    if contact.bilateral:
                        phase = DynamicSqueezePhase.IMPACT
                    else:
                        phase = DynamicSqueezePhase.SINGLE_CONTACT
                        single_contact_start_time = time_s
                if phase is DynamicSqueezePhase.SINGLE_CONTACT:
                    if contact.bilateral:
                        phase = DynamicSqueezePhase.IMPACT
                    elif (
                        single_contact_start_time is not None
                        and time_s - single_contact_start_time
                        > cfg.single_contact_timeout_s
                    ):
                        phase = DynamicSqueezePhase.FAILED
                        failure_reason = "bilateral contact not established"
                if phase is DynamicSqueezePhase.IMPACT and not impact.in_first_contact_window:
                    if impact.force_limit_exceeded:
                        phase = DynamicSqueezePhase.FAILED
                        failure_reason = "first-contact force limit exceeded"
                    else:
                        # The compliant impact phase intentionally follows the
                        # moving box.  Arrest from its *current* location rather
                        # than pulling back toward the first-touch location.
                        capture_center = box_position.copy()
                        # The pads have already followed the object through the
                        # compliant impact window.  The velocity-match phase now
                        # arrests from this pose; carrying the full measured
                        # twist into the reference can rotate a broad pad off its
                        # face before the QP has dissipated the spin.
                        capture_reference_velocity = np.zeros(3)
                        capture_reference_angular_velocity = np.zeros(3)
                        if rotating_cfg is not None:
                            capture_rotation = box_rotation.copy()
                        velocity_match_timer = 0.0
                        phase = DynamicSqueezePhase.VELOCITY_MATCH

                filtered_left_force = force_controller.left_filtered_force
                filtered_right_force = force_controller.right_filtered_force
                stable_contact = bool(
                    filtered_left_force >= cfg.minimum_contact_force
                    and filtered_right_force >= cfg.minimum_contact_force
                    and abs(filtered_left_force - filtered_right_force)
                    <= cfg.maximum_force_imbalance
                    and max(contact.left.normal_force, contact.right.normal_force)
                    <= cfg.maximum_safe_force
                )
                if bilateral_contact_time is None and contact.bilateral:
                    bilateral_contact_time = time_s
                if phase is DynamicSqueezePhase.VELOCITY_MATCH:
                    velocity_matched = bool(
                        stable_contact
                        and np.linalg.norm(box_velocity)
                        <= 1.5 * cfg.maximum_capture_speed
                        and (
                            rotating_cfg is None
                            or np.linalg.norm(box_angular_velocity)
                            <= 1.5
                            * rotating_cfg.maximum_capture_angular_speed
                        )
                    )
                    if velocity_matched:
                        capture_timer = 0.0
                        phase = DynamicSqueezePhase.CAPTURE
                if phase is DynamicSqueezePhase.CAPTURE:
                    captured = bool(
                        stable_contact
                        and np.linalg.norm(box_velocity) <= cfg.maximum_capture_speed
                        and (
                            rotating_cfg is None
                            or np.linalg.norm(box_angular_velocity)
                            <= rotating_cfg.maximum_capture_angular_speed
                        )
                    )
                    capture_timer = capture_timer + dt if captured else 0.0
                    if capture_timer >= cfg.capture_stable_s:
                        capture_center = box_position.copy()
                        if rotating_cfg is not None:
                            capture_rotation = box_rotation.copy()
                        hold_timer = 0.0
                        adaptive_hold_force = minimum_hold_force
                        phase = DynamicSqueezePhase.HOLD
                elif phase is DynamicSqueezePhase.HOLD:
                    drift = float(np.linalg.norm(box_position - capture_center))
                    drop = float(capture_center[2] - box_position[2])
                    safe = bool(
                        stable_contact
                        and np.linalg.norm(box_velocity) <= cfg.maximum_capture_speed
                        and (
                            rotating_cfg is None
                            or (
                                np.linalg.norm(box_angular_velocity)
                                <= rotating_cfg.maximum_capture_angular_speed
                                and slip_cost
                                <= 2.0 * rotating_cfg.maximum_hold_slip_ratio**2
                                and slip_speed
                                <= rotating_cfg.maximum_contact_slip_speed
                            )
                        )
                        and drift <= cfg.maximum_capture_drift
                        and drop <= cfg.maximum_capture_drop
                    )
                    if safe:
                        hold_unsafe_timer = 0.0
                        hold_timer += dt
                    else:
                        hold_unsafe_timer += dt
                        if hold_unsafe_timer > cfg.hold_instability_grace_s:
                            hold_timer = 0.0
                    if hold_timer >= cfg.required_dynamic_hold_s:
                        phase = DynamicSqueezePhase.SUCCESS

                if impact.emergency:
                    phase = DynamicSqueezePhase.FAILED
                    failure_reason = "emergency contact force exceeded"
                if phase in {
                    DynamicSqueezePhase.SINGLE_CONTACT,
                    DynamicSqueezePhase.IMPACT,
                    DynamicSqueezePhase.VELOCITY_MATCH,
                    DynamicSqueezePhase.CAPTURE,
                    DynamicSqueezePhase.HOLD,
                } and capture_center is not None and (
                    np.linalg.norm(box_position - capture_center) > 0.25
                    or box_position[2] < 0.55
                ):
                    phase = DynamicSqueezePhase.FAILED
                    failure_reason = "box escaped during capture"
                if phase in {
                    DynamicSqueezePhase.TRACK,
                    DynamicSqueezePhase.SINGLE_CONTACT,
                    DynamicSqueezePhase.IMPACT,
                    DynamicSqueezePhase.VELOCITY_MATCH,
                } and (
                    box_position[0] < cfg.catch_plane_x - 0.16
                    or (
                        box_position[0] <= cfg.catch_plane_x + 0.10
                        and box_position[2] < 0.55
                    )
                ):
                    phase = DynamicSqueezePhase.FAILED
                    failure_reason = "box passed the interception workspace"

            for name in ("left", "right"):
                impedance[name].apply(
                    model,
                    data,
                    arms[name],
                    desired_transforms[name],
                    wrench_feedforward=wrench_feedforward[name],
                )
                data.ctrl[gripper_ids[name]] = FFW_GRIPPERS[name].open_ctrl
                grippers_remained_open = grippers_remained_open and bool(
                    np.allclose(data.ctrl[gripper_ids[name]], 0.0, atol=1e-12)
                )

            if step % 5 == 0:
                log_rows.append(
                    {
                        "time_s": time_s,
                        "phase": phase.value,
                        "box_x": box_position[0],
                        "box_y": box_position[1],
                        "box_z": box_position[2],
                        "box_speed_mps": float(np.linalg.norm(box_velocity)),
                        "box_angular_speed_radps": float(
                            np.linalg.norm(box_angular_velocity)
                        ),
                        "left_force_n": contact.left.normal_force,
                        "right_force_n": contact.right.normal_force,
                        "left_pad_x": float(data.geom_xpos[pad_ids["left"]][0]),
                        "left_pad_y": float(data.geom_xpos[pad_ids["left"]][1]),
                        "left_pad_z": float(data.geom_xpos[pad_ids["left"]][2]),
                        "right_pad_x": float(data.geom_xpos[pad_ids["right"]][0]),
                        "right_pad_y": float(data.geom_xpos[pad_ids["right"]][1]),
                        "right_pad_z": float(data.geom_xpos[pad_ids["right"]][2]),
                        "left_target_x": float(desired_transforms["left"][0, 3]),
                        "left_target_y": float(desired_transforms["left"][1, 3]),
                        "left_target_z": float(desired_transforms["left"][2, 3]),
                        "right_target_x": float(desired_transforms["right"][0, 3]),
                        "right_target_y": float(desired_transforms["right"][1, 3]),
                        "right_target_z": float(desired_transforms["right"][2, 3]),
                        "plan_ttc_s": float(plan_ttc),
                        "plan_center_x": float(plan_center[0]),
                        "plan_center_y": float(plan_center[1]),
                        "plan_center_z": float(plan_center[2]),
                        "locked_interception_time_s": float(
                            locked_interception_time_s
                            if locked_interception_time_s is not None
                            else -1.0
                        ),
                        "predicted_vx": float(prediction.velocity[0]),
                        "predicted_vz": float(prediction.velocity[2]),
                        "first_contact_peak_n": impact.peak_first_contact_force,
                        "tangential_stiffness_npm": float(impedance["left"].K[3, 3]),
                        "hold_time_s": hold_timer,
                        "minimum_hold_force_n": minimum_hold_force,
                        "applied_hold_force_n": adaptive_hold_force,
                        "slip_cost": slip_cost,
                        "angular_velocity_cost": angular_velocity_cost,
                        "wrench_tracking_error": wrench_tracking_error,
                    }
                )

            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if phase in {DynamicSqueezePhase.SUCCESS, DynamicSqueezePhase.FAILED}:
                break
    finally:
        if viewer is not None:
            viewer.close()

    if phase not in {DynamicSqueezePhase.SUCCESS, DynamicSqueezePhase.FAILED}:
        phase = DynamicSqueezePhase.FAILED
        failure_reason = failure_reason or "timeout"
    final_position = data.xpos[box_body_id].copy()
    final_velocity = data.qvel[box_dof_address : box_dof_address + 3].copy()
    final_angular_velocity = data.qvel[
        box_dof_address + 3 : box_dof_address + 6
    ].copy()
    reference_center = capture_center if capture_center is not None else final_position
    capture_drop = float(reference_center[2] - final_position[2])
    capture_drift = float(np.linalg.norm(final_position - reference_center))

    if config.log_path and log_rows:
        path = Path(config.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    return DynamicSqueezeSummary(
        success=phase is DynamicSqueezePhase.SUCCESS,
        final_phase=phase.value,
        random_seed=cfg.random_seed,
        launch_position_m=tuple(float(value) for value in launch_position),
        launch_velocity_mps=tuple(float(value) for value in launch_velocity),
        launch_angular_velocity_radps=tuple(
            float(value) for value in launch_angular_velocity
        ),
        box_half_size_m=(
            tuple(float(value) for value in config.domain_parameters.half_size)
            if config.domain_parameters is not None
            else (0.055, float(cfg.box_half_y), 0.055)
        ),
        box_mass_kg=(
            float(config.domain_parameters.mass)
            if config.domain_parameters is not None
            else float(rotating_cfg.box_mass if rotating_cfg is not None else 0.50)
        ),
        box_friction=(
            float(config.domain_parameters.friction)
            if config.domain_parameters is not None
            else 1.20
        ),
        initial_ttc_s=float(launch_ttc),
        simultaneous_start=cfg.simultaneous_start,
        collision_mode=config.collision_mode,
        robot_collision_enabled=robot_collision_enabled,
        simulated_time_s=float(data.time),
        first_contact_time_s=limiter.first_contact_time_s,
        first_nonpad_contact_time_s=first_nonpad_contact_time,
        first_nonpad_contact_object=first_nonpad_contact_object,
        bilateral_contact_time_s=bilateral_contact_time,
        first_contact_peak_force_n=float(limiter.peak_first_contact_force),
        first_contact_force_limit_n=cfg.first_contact_force_limit,
        peak_total_force_n=float(peak_total_force),
        minimum_no_slip_force_per_pad_n=minimum_hold_force,
        final_hold_force_per_pad_n=float(adaptive_hold_force),
        minimum_ttc_stiffness_npm=float(minimum_stiffness),
        final_box_speed_mps=float(np.linalg.norm(final_velocity)),
        final_angular_speed_radps=float(np.linalg.norm(final_angular_velocity)),
        peak_angular_speed_radps=float(peak_angular_speed),
        prediction_velocity_error_mps=(
            float(np.mean(velocity_errors)) if velocity_errors else float("inf")
        ),
        interception_center_error_m=(
            float(np.mean(interception_errors)) if interception_errors else float("inf")
        ),
        reachable_intercept_seen=reachable_intercept_seen,
        angular_prediction_error_radps=(
            float(np.mean(angular_velocity_errors))
            if angular_velocity_errors
            else 0.0
        ),
        wrench_qp_status=wrench_qp_status,
        wrench_tracking_error=float(wrench_tracking_error),
        slip_cost=float(slip_cost),
        angular_velocity_cost=float(angular_velocity_cost),
        dynamic_hold_time_s=float(hold_timer),
        capture_drop_m=capture_drop,
        capture_drift_m=capture_drift,
        grippers_remained_open=grippers_remained_open,
        failure_reason=failure_reason,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--show-prepare",
        action="store_true",
        help="show the pre-trial motion into the ready pose",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--collision-mode",
        choices=("pad_only", "miss_backstop", "full"),
        default="miss_backstop",
    )
    parser.add_argument(
        "--log",
        default=str(ROOT / "sweep_results" / "dynamic_box_squeeze.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    squeeze = DynamicSideSqueezeConfig(random_seed=args.seed)
    summary = run_dynamic_side_squeeze(
        DynamicRunConfig(
            viewer=args.viewer,
            viewer_show_prepare=args.show_prepare,
            log_path=args.log,
            squeeze=squeeze,
            collision_mode=args.collision_mode,
        )
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    if not summary.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
