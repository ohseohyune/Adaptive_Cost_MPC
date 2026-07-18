"""Catch a ballistically-launched box with the differentiable AC-MPC pipeline.

This reuses the box scene, ballistic launch/prediction, and first-contact
impact-safety machinery already validated in ``main_dynamic_box_squeeze.py``,
but replaces its hand-engineered interception/squeeze state machine with the
same actor -> differentiable-MPC -> Cartesian-velocity pipeline used by
``main_bimanual_acmpc.py`` (now PPO+GAE trained). Box orientation is tracked
directly from the measured rotation rather than through the MPC's own state:
``DifferentiableBimanualMPC`` is a position-only model, and the curriculum's
angular velocities are gentle enough that a decoupled orientation overlay
(mirroring ``_ee_rotations_for_box`` in ``main_dynamic_box_squeeze.py``) is a
reasonable simplification for this first integration pass.

This is a fixed-nominal-box baseline (no domain randomization / curriculum
yet): the goal here is validating that the ballistic-aware MPC reference and
the ported impact-safety scheduling actually produce a physical catch before
layering randomization back on top.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from control.clik import build_serial_arm, get_ee_transform
from control.clik.catching import adaptive_stiffness
from control.clik.impedance import CartesianImpedanceConfig, CartesianImpedanceController
from control.mpc import (
    ACMPCRolloutBuffer,
    BimanualPhase,
    DifferentiableMPCConfig,
    OnlineActorCriticACMPC,
    OnlineActorCriticConfig,
    PPOUpdateSummary,
    build_bimanual_observation,
)
from control.squeeze import (
    BallisticBoxPredictor,
    DynamicSideSqueezeConfig,
    FirstContactForceLimiter,
    adaptive_impact_command,
    read_bilateral_pad_contact,
    resolve_ballistic_launch_position,
    resolve_ballistic_launch_velocity,
)
from main_box_squeeze import LEFT_HOME_Q, RIGHT_HOME_Q
from main_dynamic_box_squeeze import (
    HAND_CAMERA_COLLISION_BIT,
    _disable_duplicate_end_effector_collisions,
)
from robot.ffw_config import FFW_ARMS, FFW_GRIPPERS


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "model/robotis_ffw/scene_ffw_sg2_fixed_base_box_dynamic_squeeze.xml"

_NOMINAL_BOX_MASS = 0.50
_NOMINAL_BOX_FRICTION = 1.20
_NOMINAL_BOX_HALF_SIZE = (0.055, 0.150, 0.055)


class CatchPhase(Enum):
    INTERCEPT = "intercept"
    PRE_CONTACT = "pre_contact"
    GRASPING = "grasping"
    GRASPED = "grasped"
    SUCCESS = "success"
    FAILED = "failed"


_PHASE_TO_BIMANUAL = {
    CatchPhase.INTERCEPT: BimanualPhase.INTERCEPT,
    CatchPhase.PRE_CONTACT: BimanualPhase.PRE_CONTACT,
    CatchPhase.GRASPING: BimanualPhase.GRASPING,
    CatchPhase.GRASPED: BimanualPhase.GRASPED,
    CatchPhase.SUCCESS: BimanualPhase.GRASPED,
    CatchPhase.FAILED: BimanualPhase.GRASPED,
}

# The handle-grasp demo's grasp weight (10) gives a hand-separation gain that
# is far too weak to close a large initial separation error within a ~0.5 s
# ballistic flight (measured: ~0.03 m/s of closing velocity per 0.1 m of
# error at weight 10, vs ~0.5 m/s at weight 250) -- the home pose's natural
# hand separation does not match the box's target pad separation, and this
# scenario has no prepare/ready-pose phase to close that gap in advance the
# way the box-squeeze track's does. Object/velocity/force/smoothness priors
# are left close to the original INTERCEPT/PRE_CONTACT values.
_BOX_CATCH_PHASE_PRIORS = (
    (30.0, 250.0, 0.05, 4.0, 0.4),
    (30.0, 250.0, 1.5, 3.0, 0.5),
    (16.0, 60.0, 12.0, 1.5, 1.0),
    (18.0, 40.0, 9.0, 2.0, 1.5),
    (32.0, 18.0, 7.0, 3.0, 1.5),
)


@dataclass
class AcmpcBoxCatchConfig:
    seed: int = 7
    device: str = "auto"
    online_learning: bool = True
    exploration_std: float = 0.08
    rollout_size: int = 16
    offline_training: bool = False
    mpc_horizon: int = 6
    mpc_velocity_limit: float = 1.8
    command_lookahead_s: float = 0.02
    precontact_distance: float = 0.10
    contact_force: float = 0.05
    stable_grasp_force: float = 1.0
    stable_grasp_time: float = 0.05
    required_hold_s: float = 0.30
    timeout_s: float = 2.5
    viewer: bool = False
    log_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    squeeze: DynamicSideSqueezeConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.squeeze is None:
            self.squeeze = DynamicSideSqueezeConfig(random_seed=self.seed)


@dataclass
class BoxCatchSummary:
    success: bool
    final_phase: str
    failure_reason: str
    simulated_time_s: float
    first_contact_time_s: Optional[float]
    first_contact_peak_force_n: float
    bilateral_contact_time_s: Optional[float]
    hold_time_s: float
    minimum_endpoint_error_m: float
    final_box_speed_mps: float
    online_updates: int
    total_transitions: int
    actor_weight_change_l2: float
    device: str


def _require_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(value)


def _reward(
    *,
    previous_endpoint_error: float,
    endpoint_error: float,
    left_force: float,
    right_force: float,
    velocity: np.ndarray,
    phase: CatchPhase,
    force_limit_exceeded: bool,
    emergency: bool,
) -> float:
    progress = 25.0 * (previous_endpoint_error - endpoint_error)
    endpoint_penalty = 2.5 * endpoint_error
    force_sum = min(left_force, 20.0) + min(right_force, 20.0)
    contact_reward = 0.025 * force_sum
    bilateral_bonus = 0.8 if left_force > 0.35 and right_force > 0.35 else 0.0
    force_balance_penalty = 0.006 * abs(left_force - right_force)
    effort_penalty = 0.01 * float(np.linalg.norm(velocity))
    phase_bonus = {
        CatchPhase.INTERCEPT: 0.0,
        CatchPhase.PRE_CONTACT: 0.10,
        CatchPhase.GRASPING: 0.30,
        CatchPhase.GRASPED: 0.60,
        CatchPhase.SUCCESS: 0.60,
        CatchPhase.FAILED: 0.0,
    }[phase]
    safety_penalty = 3.0 if force_limit_exceeded else 0.0
    safety_penalty += 5.0 if emergency else 0.0
    return float(
        progress
        - endpoint_penalty
        + contact_reward
        + bilateral_bonus
        + phase_bonus
        - force_balance_penalty
        - effort_penalty
        - safety_penalty
    )


def run_box_catch(config: Optional[AcmpcBoxCatchConfig] = None) -> BoxCatchSummary:
    config = config or AcmpcBoxCatchConfig()
    cfg = config.squeeze
    rng = np.random.default_rng(cfg.random_seed)
    sampled_launch_velocity = rng.uniform(cfg.launch_velocity_low, cfg.launch_velocity_high)
    launch_position = resolve_ballistic_launch_position(cfg, sampled_launch_velocity)
    launch_velocity = resolve_ballistic_launch_velocity(
        cfg, sampled_launch_velocity, launch_position=launch_position
    )
    gravity = np.asarray(cfg.gravity, dtype=float)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    control_stride = max(1, int(round(0.01 / dt)))
    control_dt = control_stride * dt

    arms = {name: build_serial_arm(model, FFW_ARMS[name]) for name in ("left", "right")}
    homes = {"left": LEFT_HOME_Q.copy(), "right": RIGHT_HOME_Q.copy()}
    gripper_ids = {
        name: np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                for actuator_name in FFW_GRIPPERS[name].actuator_names
            ],
            dtype=int,
        )
        for name in ("left", "right")
    }
    for name, arm in arms.items():
        data.qpos[arm.qpos_indices] = homes[name]
        data.ctrl[arm.actuator_ids] = homes[name]
        data.ctrl[gripper_ids[name]] = FFW_GRIPPERS[name].open_ctrl
    mujoco.mj_forward(model, data)

    box_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_box")
    box_geom_id = _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
    model.geom_conaffinity[box_geom_id] |= HAND_CAMERA_COLLISION_BIT | 1
    box_joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "dynamic_box_joint")
    box_qpos_address = int(model.jnt_qposadr[box_joint_id])
    box_dof_address = int(model.jnt_dofadr[box_joint_id])
    fixture_id = _require_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "box_launch_fixture")
    pad_ids = {
        name: _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_squeeze_pad")
        for name in ("left", "right")
    }
    _disable_duplicate_end_effector_collisions(model, set(pad_ids.values()))

    data.qpos[box_qpos_address : box_qpos_address + 3] = launch_position
    mujoco.mj_forward(model, data)

    impedance = {
        name: CartesianImpedanceController(
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
        for name in ("left", "right")
    }
    desired_transforms = {
        name: get_ee_transform(data, arms[name]).copy() for name in ("left", "right")
    }

    predictor = BallisticBoxPredictor(
        gravity=gravity,
        velocity_alpha=cfg.predictor_velocity_alpha,
        max_speed=cfg.predictor_max_speed,
    )
    limiter = FirstContactForceLimiter(cfg)
    learner = OnlineActorCriticACMPC(
        DifferentiableMPCConfig(
            horizon=config.mpc_horizon,
            dt=control_dt,
            velocity_limit=config.mpc_velocity_limit,
            gravity=tuple(float(value) for value in gravity),
        ),
        OnlineActorCriticConfig(
            device=config.device,
            seed=config.seed,
            initial_log_std=float(np.log(max(config.exploration_std, 1e-6))),
            phase_priors=_BOX_CATCH_PHASE_PRIORS,
        ),
    )
    if config.checkpoint_path and Path(config.checkpoint_path).exists():
        learner.load(config.checkpoint_path)
    initial_actor = np.concatenate(
        [parameter.detach().cpu().numpy().ravel() for parameter in learner.actor.parameters()]
    )

    pad_offset_scalar = cfg.box_half_y + cfg.pad_half_thickness + cfg.precontact_gap
    object_mass = _NOMINAL_BOX_MASS
    object_friction = _NOMINAL_BOX_FRICTION
    half_x, _, half_z = _NOMINAL_BOX_HALF_SIZE

    data.eq_active[fixture_id] = 0
    data.qvel[box_dof_address : box_dof_address + 3] = launch_velocity
    mujoco.mj_forward(model, data)
    predictor.reset(initial_velocity=launch_velocity)

    phase = CatchPhase.INTERCEPT
    phase_started = 0.0
    hold_timer = 0.0
    bilateral_contact_time: Optional[float] = None
    failure_reason = ""
    previous_control_positions = np.concatenate(
        [desired_transforms["left"][:3, 3], desired_transforms["right"][:3, 3]]
    )
    previous_velocity = np.zeros(6)
    command_velocity = np.zeros(6)
    previous_endpoint_error = 0.0
    minimum_endpoint_error = float("inf")

    rollout = ACMPCRolloutBuffer()
    updates: list[PPOUpdateSummary] = []
    total_transitions = 0
    latest_update: Optional[PPOUpdateSummary] = None
    last_observation: Optional[np.ndarray] = None
    last_phase: Optional[BimanualPhase] = None
    last_ee_positions: Optional[np.ndarray] = None
    last_object_position: Optional[np.ndarray] = None
    last_object_velocity: Optional[np.ndarray] = None
    last_relative_reference: Optional[np.ndarray] = None
    last_previous_velocity: Optional[np.ndarray] = None
    last_action = None
    rows: list[dict[str, float | str]] = []

    viewer = None
    if config.viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)

    total_steps = int(np.ceil(config.timeout_s / dt))
    try:
        for step in range(total_steps):
            if viewer is not None and not viewer.is_running():
                failure_reason = "viewer closed"
                break
            time_s = float(data.time)
            box_position = data.xpos[box_body_id].copy()
            box_velocity = data.qvel[box_dof_address : box_dof_address + 3].copy()
            box_rotation = data.xmat[box_body_id].reshape(3, 3).copy()
            contact = read_bilateral_pad_contact(model, data, box_geom_name="dynamic_box_geom")
            impact = limiter.update(time_s, contact)

            if step % control_stride == 0:
                prediction = predictor.update(time_s, box_position)
                # The MPC's own horizon lookahead is only
                # horizon*dt (~tens of ms) -- far shorter than the ~0.5 s
                # flight time -- so handing it the box's *current* position
                # as "object_position" leaves it always reacting to a target
                # that is still ~1 m away and closing fast, never given
                # enough real time to converge. Instead, aim at the
                # predicted *interception point* (mirroring
                # BoxFaceInterceptionPlanner in control/squeeze/ballistic.py)
                # and hand that to the MPC as a near-static reference
                # (zero object_velocity) that it tracks and settles onto
                # over the whole remaining flight, re-solved fresh each step
                # as the TTC estimate refines.
                # (catch_plane_x - position_x) and velocity_x are both
                # negative while approaching, so their ratio is the correct
                # positive time-to-plane -- do not negate velocity_x here
                # (that flips the sign and was clamping every estimate to
                # the minimum_intercept_ttc floor).
                remaining_ttc = float(
                    np.clip(
                        (cfg.catch_plane_x - prediction.position[0])
                        / min(prediction.velocity[0], -1e-3),
                        cfg.minimum_intercept_ttc,
                        cfg.maximum_intercept_ttc,
                    )
                )
                intercept_center = prediction.position_after(remaining_ttc)
                y_axis = box_rotation[:, 1]
                pad_vector = pad_offset_scalar * y_axis
                left_target = intercept_center + pad_vector
                right_target = intercept_center - pad_vector

                left_ee = get_ee_transform(data, arms["left"])[:3, 3]
                right_ee = get_ee_transform(data, arms["right"])[:3, 3]
                # Feed the MPC the arm's actual measured pose, not the last
                # commanded desired_transforms -- the impedance controller
                # does not track its target instantaneously, so using the
                # commanded pose as "current state" makes the MPC believe
                # it is already closer to the target than it physically is,
                # systematically under-correcting.
                ee_positions = np.concatenate([left_ee, right_ee])
                measured_velocity = (ee_positions - previous_control_positions) / control_dt
                previous_control_positions = ee_positions.copy()

                endpoint_error = 0.5 * (
                    float(np.linalg.norm(left_ee - left_target))
                    + float(np.linalg.norm(right_ee - right_target))
                )
                minimum_endpoint_error = min(minimum_endpoint_error, endpoint_error)
                relative_normal_speed = max(
                    abs(float(np.dot(box_velocity, y_axis))), 1e-6
                )
                impact_command = adaptive_impact_command(
                    cfg,
                    object_mass=object_mass,
                    contact_face_area=4.0 * half_x * half_z,
                    relative_normal_speed=relative_normal_speed,
                )

                old_phase = phase
                both_contact = contact.left.active and contact.right.active
                stable_contact = bool(
                    contact.left.normal_force > config.stable_grasp_force
                    and contact.right.normal_force > config.stable_grasp_force
                )
                if phase is CatchPhase.INTERCEPT and endpoint_error < config.precontact_distance:
                    phase = CatchPhase.PRE_CONTACT
                elif phase is CatchPhase.PRE_CONTACT and both_contact:
                    phase = CatchPhase.GRASPING
                    bilateral_contact_time = time_s
                elif phase is CatchPhase.GRASPING:
                    hold_timer = hold_timer + control_dt if stable_contact else 0.0
                    if hold_timer >= config.stable_grasp_time:
                        phase = CatchPhase.GRASPED
                        hold_timer = 0.0
                elif phase is CatchPhase.GRASPED:
                    if stable_contact:
                        hold_timer += control_dt
                    else:
                        hold_timer = 0.0
                    if hold_timer >= config.required_hold_s:
                        phase = CatchPhase.SUCCESS
                if impact.emergency:
                    phase = CatchPhase.FAILED
                    failure_reason = "emergency contact force exceeded"
                elif (
                    phase in {CatchPhase.INTERCEPT, CatchPhase.PRE_CONTACT}
                    and box_position[0] < cfg.catch_plane_x - 0.16
                ):
                    phase = CatchPhase.FAILED
                    failure_reason = "box passed the interception workspace"
                if phase is not old_phase:
                    phase_started = time_s

                bimanual_phase = _PHASE_TO_BIMANUAL[phase]
                observation = build_bimanual_observation(
                    object_velocity=prediction.velocity,
                    left_endpoint_error=left_target - left_ee,
                    right_endpoint_error=right_target - right_ee,
                    left_ee_velocity=measured_velocity[:3],
                    right_ee_velocity=measured_velocity[3:],
                    left_force=contact.left.normal_force,
                    right_force=contact.right.normal_force,
                    time_to_contact=remaining_ttc,
                    prediction_confidence=prediction.confidence,
                    phase=bimanual_phase,
                )

                if last_observation is not None and config.online_learning:
                    reward = _reward(
                        previous_endpoint_error=previous_endpoint_error,
                        endpoint_error=endpoint_error,
                        left_force=contact.left.normal_force,
                        right_force=contact.right.normal_force,
                        velocity=command_velocity,
                        phase=phase,
                        force_limit_exceeded=impact.force_limit_exceeded,
                        emergency=impact.emergency,
                    )
                    rollout.add(
                        observation=last_observation,
                        phase=last_phase,
                        ee_positions=last_ee_positions,
                        object_position=last_object_position,
                        object_velocity=last_object_velocity,
                        relative_reference=last_relative_reference,
                        previous_velocity=last_previous_velocity,
                        action=last_action,
                        reward=reward,
                        done=phase in {CatchPhase.SUCCESS, CatchPhase.FAILED},
                    )
                    total_transitions += 1
                    if len(rollout) >= config.rollout_size:
                        bootstrap_value = learner.predict_value(observation)
                        latest_update = learner.update(
                            rollout,
                            online=not config.offline_training,
                            next_value=bootstrap_value,
                        )
                        updates.append(latest_update)
                        rollout.clear()

                relative_reference = right_target - left_target
                # The MPC's "velocity" cost term tracks object_velocity as a
                # feedforward reference. Feeding it zero (as if the target
                # were already reached) fights the "object" position cost's
                # pull toward a still-distant point. Feed the average
                # closing velocity actually required to arrive on time
                # instead, so both cost terms agree on approaching fast.
                ee_midpoint = 0.5 * (ee_positions[:3] + ee_positions[3:])
                required_velocity = (intercept_center - ee_midpoint) / max(
                    remaining_ttc, control_dt
                )
                action = learner.act(
                    observation=observation,
                    phase=bimanual_phase,
                    ee_positions=ee_positions,
                    object_position=intercept_center,
                    object_velocity=required_velocity,
                    relative_reference=relative_reference,
                    previous_velocity=previous_velocity,
                    training=config.online_learning,
                )
                last_phase = bimanual_phase
                last_ee_positions = ee_positions
                last_object_position = intercept_center.copy()
                last_object_velocity = required_velocity.copy()
                last_relative_reference = relative_reference
                last_previous_velocity = previous_velocity
                last_action = action
                command_velocity = action.velocity
                previous_velocity = action.mean_velocity
                previous_endpoint_error = endpoint_error
                last_observation = observation

                if impact.first_contact_time_s is None:
                    tangential_k = float(
                        adaptive_stiffness(
                            remaining_ttc,
                            cfg.tangential_stiffness,
                            impact_command.tangential_stiffness,
                            cfg.ttc_soften_window_s,
                        )
                    )
                    normal_k = float(
                        adaptive_stiffness(
                            remaining_ttc,
                            cfg.normal_stiffness,
                            impact_command.normal_stiffness,
                            cfg.ttc_soften_window_s,
                        )
                    )
                elif impact.in_first_contact_window:
                    tangential_k = impact_command.tangential_stiffness
                    normal_k = impact_command.normal_stiffness
                else:
                    tangential_k = cfg.tangential_stiffness
                    normal_k = cfg.normal_stiffness
                for name in ("left", "right"):
                    impedance[name].K[:] = np.diag(
                        [cfg.rotational_stiffness] * 3 + [tangential_k, normal_k, tangential_k]
                    )
                    impedance[name].D[:] = np.diag(
                        [cfg.rotational_damping] * 3
                        + [cfg.tangential_damping, cfg.normal_damping, cfg.tangential_damping]
                    )

                for index, name in enumerate(("left", "right")):
                    desired_transforms[name][:3, 3] = (
                        ee_positions[3 * index : 3 * (index + 1)]
                        + config.command_lookahead_s
                        * command_velocity[3 * index : 3 * (index + 1)]
                    )
                desired_transforms["left"][:3, :3] = np.column_stack(
                    [y_axis, -box_rotation[:, 0], box_rotation[:, 2]]
                )
                desired_transforms["right"][:3, :3] = np.column_stack(
                    [-y_axis, box_rotation[:, 0], box_rotation[:, 2]]
                )

                rows.append(
                    {
                        "time_s": time_s,
                        "phase": phase.value,
                        "box_x": float(box_position[0]),
                        "box_z": float(box_position[2]),
                        "left_force_n": contact.left.normal_force,
                        "right_force_n": contact.right.normal_force,
                        "endpoint_error_m": endpoint_error,
                        "left_err_x": float(left_ee[0] - left_target[0]),
                        "left_err_y": float(left_ee[1] - left_target[1]),
                        "left_err_z": float(left_ee[2] - left_target[2]),
                        "right_err_x": float(right_ee[0] - right_target[0]),
                        "right_err_y": float(right_ee[1] - right_target[1]),
                        "right_err_z": float(right_ee[2] - right_target[2]),
                        "actor_loss": 0.0 if latest_update is None else latest_update.actor_loss,
                        "critic_loss": 0.0 if latest_update is None else latest_update.critic_loss,
                    }
                )

            for name in ("left", "right"):
                impedance[name].apply(model, data, arms[name], desired_transforms[name])
                data.ctrl[gripper_ids[name]] = FFW_GRIPPERS[name].open_ctrl

            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if phase in {CatchPhase.SUCCESS, CatchPhase.FAILED}:
                break
    finally:
        if viewer is not None:
            viewer.close()

    if phase not in {CatchPhase.SUCCESS, CatchPhase.FAILED}:
        failure_reason = failure_reason or "timeout"
        phase = CatchPhase.FAILED

    if last_observation is not None and config.online_learning:
        rollout.add(
            observation=last_observation,
            phase=last_phase,
            ee_positions=last_ee_positions,
            object_position=last_object_position,
            object_velocity=last_object_velocity,
            relative_reference=last_relative_reference,
            previous_velocity=last_previous_velocity,
            action=last_action,
            reward=0.0,
            done=True,
        )
        total_transitions += 1
    if len(rollout) and config.online_learning:
        latest_update = learner.update(rollout, online=not config.offline_training, next_value=0.0)
        updates.append(latest_update)
        rollout.clear()

    if config.checkpoint_path:
        learner.save(config.checkpoint_path)
    if config.log_path and rows:
        log_path = Path(config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    final_actor = np.concatenate(
        [parameter.detach().cpu().numpy().ravel() for parameter in learner.actor.parameters()]
    )
    final_box_speed = float(np.linalg.norm(data.qvel[box_dof_address : box_dof_address + 3]))
    return BoxCatchSummary(
        success=phase is CatchPhase.SUCCESS,
        final_phase=phase.value,
        failure_reason=failure_reason,
        simulated_time_s=float(data.time),
        first_contact_time_s=limiter.first_contact_time_s,
        first_contact_peak_force_n=float(limiter.peak_first_contact_force),
        bilateral_contact_time_s=bilateral_contact_time,
        hold_time_s=float(hold_timer),
        minimum_endpoint_error_m=float(minimum_endpoint_error),
        final_box_speed_mps=final_box_speed,
        online_updates=learner.update_count,
        total_transitions=total_transitions,
        actor_weight_change_l2=float(np.linalg.norm(final_actor - initial_actor)),
        device=str(learner.device),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-online-learning", action="store_true")
    parser.add_argument(
        "--log", default=str(ROOT / "sweep_results" / "acmpc_box_catch.csv")
    )
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_box_catch(
        AcmpcBoxCatchConfig(
            seed=args.seed,
            device=args.device,
            viewer=args.viewer,
            online_learning=not args.no_online_learning,
            log_path=args.log,
            checkpoint_path=args.checkpoint,
        )
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    if not summary.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
