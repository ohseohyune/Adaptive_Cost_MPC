"""Dynamic physical bimanual grasp with online actor-critic cost adaptation.

This is the end-to-end execution path missing from the earlier phase demos:

    free moving object -> state prediction -> neural horizon cost map
      -> differentiable bimanual MPC -> Cartesian impedance -> both grippers
      -> MuJoCo contact reward -> online actor/critic TD update

The default run is headless and finite.  Pass ``--viewer`` for visualization.
CUDA is optional; ``--device auto`` selects it only when PyTorch can use it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.clik import MovingTargetPredictor, build_serial_arm, get_ee_transform
from control.clik.contact import (
    body_geom_ids,
    finger_geom_ids,
    get_contacts_between,
    total_normal_force,
)
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
from robot.ffw_config import FFW_ARMS, FFW_GRIPPERS


SCENE = ROOT / "model" / "robotis_ffw" / "scene_ffw_sg2_fixed_base_bimanual_dynamic.xml"


@dataclass
class DemoConfig:
    duration_s: float = 6.0
    control_dt: float = 0.02
    command_lookahead_s: float = 0.10
    device: str = "auto"
    online_learning: bool = True
    exploration_std: float = 0.015
    rollout_size: int = 32
    offline_training: bool = False
    seed: int = 7
    viewer: bool = False
    log_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    initial_object_velocity: tuple[float, float, float] = (0.005, 0.0, 0.0)
    precontact_distance: float = 0.100
    contact_force: float = 0.005
    stable_grasp_force: float = 0.25
    stable_grasp_time: float = 0.15
    manipulation_delay: float = 0.40
    manipulation_offset: tuple[float, float, float] = (0.045, 0.0, 0.060)
    grasp_hold_fraction: float = 0.78


@dataclass
class DemoSummary:
    success: bool
    final_phase: str
    simulated_time_s: float
    grasp_time_s: Optional[float]
    manipulation_time_s: Optional[float]
    left_contact_force_n: float
    right_contact_force_n: float
    final_endpoint_error_m: float
    final_object_goal_error_m: Optional[float]
    online_updates: int
    total_transitions: int
    actor_weight_change_l2: float
    device: str
    object_displacement_m: float


@dataclass
class Snapshot:
    object_position: np.ndarray
    object_velocity: np.ndarray
    left_handle: np.ndarray
    right_handle: np.ndarray
    left_ee: np.ndarray
    right_ee: np.ndarray
    left_force: float
    right_force: float

    @property
    def endpoint_error(self) -> float:
        return 0.5 * (
            float(np.linalg.norm(self.left_ee - self.left_handle))
            + float(np.linalg.norm(self.right_ee - self.right_handle))
        )


RIGHT_HOME_Q = np.array(
    [-0.534804, -0.460945, -1.278942, 0.871459, -0.281033, -0.379103, 0.050507]
)
LEFT_HOME_Q = np.array(
    [-0.534802, 0.460945, 1.278943, 0.871459, 0.281032, -0.379103, -0.050513]
)


def _ids(model: mujoco.MjModel, object_type: mujoco.mjtObj, names: tuple[str, ...]) -> np.ndarray:
    result = [mujoco.mj_name2id(model, object_type, name) for name in names]
    if any(value < 0 for value in result):
        raise ValueError(f"MuJoCo names not found: {names}")
    return np.asarray(result, dtype=int)


def _site_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {name}")
    return data.site_xpos[site_id].copy()


def _snapshot(runtime: dict) -> Snapshot:
    model, data = runtime["model"], runtime["data"]
    object_body_id = runtime["object_body_id"]
    left_contacts = get_contacts_between(
        model, data, runtime["finger_geoms"]["left"], runtime["object_geoms"]
    )
    right_contacts = get_contacts_between(
        model, data, runtime["finger_geoms"]["right"], runtime["object_geoms"]
    )
    dof_address = runtime["object_dof_address"]
    return Snapshot(
        object_position=data.xpos[object_body_id].copy(),
        object_velocity=data.qvel[dof_address : dof_address + 3].copy(),
        left_handle=_site_position(model, data, "object_left_handle_site"),
        right_handle=_site_position(model, data, "object_right_handle_site"),
        left_ee=get_ee_transform(data, runtime["arms"]["left"])[:3, 3].copy(),
        right_ee=get_ee_transform(data, runtime["arms"]["right"])[:3, 3].copy(),
        left_force=total_normal_force(left_contacts),
        right_force=total_normal_force(right_contacts),
    )


def _build_runtime(config: DemoConfig) -> dict:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    arms = {name: build_serial_arm(model, FFW_ARMS[name]) for name in ("left", "right")}
    homes = {"left": LEFT_HOME_Q.copy(), "right": RIGHT_HOME_Q.copy()}
    for name, arm in arms.items():
        data.qpos[arm.qpos_indices] = homes[name]
        data.ctrl[arm.actuator_ids] = homes[name]

    object_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "bimanual_object_joint"
    )
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bimanual_object")
    if object_joint_id < 0 or object_body_id < 0:
        raise ValueError("dynamic bimanual object is missing from the scene")
    object_dof_address = int(model.jnt_dofadr[object_joint_id])
    data.qvel[object_dof_address : object_dof_address + 3] = np.asarray(
        config.initial_object_velocity, dtype=float
    )
    mujoco.mj_forward(model, data)

    impedance_config = CartesianImpedanceConfig(
        K_pos=360.0,
        K_rot=55.0,
        D_pos=75.0,
        D_rot=14.0,
        tau_limit=260.0,
        Kp_ns=6.0,
        Kd_ns=2.5,
    )
    impedance = {
        name: CartesianImpedanceController(
            CartesianImpedanceConfig(**asdict(impedance_config)),
            arm,
            model,
            homes[name],
        )
        for name, arm in arms.items()
    }
    initial_transforms = {name: get_ee_transform(data, arm).copy() for name, arm in arms.items()}
    gripper_actuators = {
        name: _ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, FFW_GRIPPERS[name].actuator_names)
        for name in ("left", "right")
    }
    finger_geoms = {
        name: finger_geom_ids(model, FFW_GRIPPERS[name].finger_body_names)
        for name in ("left", "right")
    }
    return {
        "model": model,
        "data": data,
        "arms": arms,
        "homes": homes,
        "impedance": impedance,
        "initial_transforms": initial_transforms,
        "gripper_actuators": gripper_actuators,
        "finger_geoms": finger_geoms,
        "object_geoms": body_geom_ids(model, "bimanual_object"),
        "object_body_id": object_body_id,
        "object_dof_address": object_dof_address,
    }


def _observation(
    snapshot: Snapshot,
    *,
    left_velocity: np.ndarray,
    right_velocity: np.ndarray,
    time_to_contact: float,
    confidence: float,
    phase: BimanualPhase,
) -> np.ndarray:
    return build_bimanual_observation(
        object_velocity=snapshot.object_velocity,
        left_endpoint_error=snapshot.left_handle - snapshot.left_ee,
        right_endpoint_error=snapshot.right_handle - snapshot.right_ee,
        left_ee_velocity=left_velocity,
        right_ee_velocity=right_velocity,
        left_force=snapshot.left_force,
        right_force=snapshot.right_force,
        time_to_contact=time_to_contact,
        prediction_confidence=confidence,
        phase=phase,
    )


def _reward(
    previous: Snapshot,
    current: Snapshot,
    phase: BimanualPhase,
    velocity: np.ndarray,
    manipulation_goal: Optional[np.ndarray],
) -> float:
    progress = 25.0 * (previous.endpoint_error - current.endpoint_error)
    endpoint_penalty = 2.5 * current.endpoint_error
    force_sum = min(current.left_force, 20.0) + min(current.right_force, 20.0)
    contact_reward = 0.025 * force_sum
    bilateral_bonus = 0.8 if current.left_force > 0.35 and current.right_force > 0.35 else 0.0
    force_balance_penalty = 0.006 * abs(current.left_force - current.right_force)
    effort_penalty = 0.015 * float(np.linalg.norm(velocity))
    phase_bonus = {
        BimanualPhase.INTERCEPT: 0.0,
        BimanualPhase.PRE_CONTACT: 0.05,
        BimanualPhase.GRASPING: 0.15,
        BimanualPhase.GRASPED: 0.35,
        BimanualPhase.MANIPULATION: 0.45,
    }[phase]
    goal_penalty = 0.0
    if manipulation_goal is not None and phase is BimanualPhase.MANIPULATION:
        goal_penalty = 1.5 * float(np.linalg.norm(current.object_position - manipulation_goal))
    drop_penalty = 2.0 if current.object_position[2] < 0.55 else 0.0
    return float(
        progress
        - endpoint_penalty
        + contact_reward
        + bilateral_bonus
        + phase_bonus
        - force_balance_penalty
        - effort_penalty
        - goal_penalty
        - drop_penalty
    )


def run_demo(config: Optional[DemoConfig] = None) -> DemoSummary:
    config = config or DemoConfig()
    runtime = _build_runtime(config)
    model, data = runtime["model"], runtime["data"]
    sim_dt = float(model.opt.timestep)
    control_stride = max(1, int(round(config.control_dt / sim_dt)))
    control_dt = control_stride * sim_dt
    learner = OnlineActorCriticACMPC(
        DifferentiableMPCConfig(horizon=8, dt=control_dt, velocity_limit=0.22),
        OnlineActorCriticConfig(
            device=config.device,
            seed=config.seed,
            initial_log_std=float(np.log(max(config.exploration_std, 1e-6))),
        ),
    )
    if config.checkpoint_path and Path(config.checkpoint_path).exists():
        learner.load(config.checkpoint_path)
    initial_actor = np.concatenate(
        [parameter.detach().cpu().numpy().ravel() for parameter in learner.actor.parameters()]
    )
    predictor = MovingTargetPredictor(
        horizon_s=0.18,
        smoothing_alpha=0.25,
        max_prediction_speed=0.5,
    )

    phase = BimanualPhase.INTERCEPT
    phase_started = 0.0
    stable_timer = 0.0
    grasp_time: Optional[float] = None
    manipulation_time: Optional[float] = None
    manipulation_goal: Optional[np.ndarray] = None
    previous_snapshot = _snapshot(runtime)
    initial_object_position = previous_snapshot.object_position.copy()
    previous_control_positions = np.concatenate(
        [previous_snapshot.left_ee, previous_snapshot.right_ee]
    )
    previous_velocity = np.zeros(6)
    command_velocity = np.zeros(6)
    desired_transforms = {
        name: runtime["initial_transforms"][name].copy() for name in ("left", "right")
    }
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
    last_reward = 0.0
    latest_weights = None
    rows: list[dict[str, float | int | str]] = []

    viewer = None
    if config.viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)

    total_steps = int(np.ceil(config.duration_s / sim_dt))
    try:
        for step in range(total_steps):
            if viewer is not None and not viewer.is_running():
                break
            if step % control_stride == 0:
                current = _snapshot(runtime)
                ee_positions = np.concatenate([current.left_ee, current.right_ee])
                measured_velocity = (ee_positions - previous_control_positions) / control_dt
                previous_control_positions = ee_positions.copy()
                prediction = predictor.update(float(data.time), current.object_position)
                ttc = current.endpoint_error / max(0.05, 0.22 + np.linalg.norm(prediction.velocity))

                max_endpoint_error = max(
                    np.linalg.norm(current.left_ee - current.left_handle),
                    np.linalg.norm(current.right_ee - current.right_handle),
                )
                both_contact = (
                    current.left_force > config.contact_force
                    and current.right_force > config.contact_force
                )
                stable_contact = (
                    current.left_force > config.stable_grasp_force
                    and current.right_force > config.stable_grasp_force
                )
                old_phase = phase
                if phase is BimanualPhase.INTERCEPT and max_endpoint_error < config.precontact_distance:
                    phase = BimanualPhase.PRE_CONTACT
                elif (
                    phase is BimanualPhase.PRE_CONTACT
                    and both_contact
                    and float(data.time) - phase_started >= 0.12
                ):
                    phase = BimanualPhase.GRASPING
                elif phase is BimanualPhase.GRASPING:
                    stable_timer = stable_timer + control_dt if stable_contact else 0.0
                    if stable_timer >= config.stable_grasp_time:
                        phase = BimanualPhase.GRASPED
                        grasp_time = float(data.time)
                elif (
                    phase is BimanualPhase.GRASPED
                    and float(data.time) - phase_started >= config.manipulation_delay
                ):
                    phase = BimanualPhase.MANIPULATION
                    manipulation_time = float(data.time)
                    manipulation_goal = current.object_position + np.asarray(
                        config.manipulation_offset, dtype=float
                    )
                if phase is not old_phase:
                    phase_started = float(data.time)

                observation = _observation(
                    current,
                    left_velocity=measured_velocity[:3],
                    right_velocity=measured_velocity[3:],
                    time_to_contact=ttc,
                    confidence=prediction.confidence,
                    phase=phase,
                )
                if last_observation is not None and config.online_learning:
                    last_reward = _reward(
                        previous_snapshot,
                        current,
                        phase,
                        command_velocity,
                        manipulation_goal,
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
                        reward=last_reward,
                        done=False,
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

                if phase is BimanualPhase.MANIPULATION and manipulation_goal is not None:
                    reference_center = manipulation_goal
                    reference_velocity = np.zeros(3)
                else:
                    reference_center = current.object_position
                    reference_velocity = prediction.velocity
                predicted_delta = (
                    reference_center - current.object_position
                    + 0.5 * control_dt * reference_velocity
                )
                left_reference = current.left_handle + predicted_delta
                right_reference = current.right_handle + predicted_delta
                relative_reference = right_reference - left_reference

                action = learner.act(
                    observation=observation,
                    phase=phase,
                    ee_positions=ee_positions,
                    object_position=reference_center,
                    object_velocity=reference_velocity,
                    relative_reference=relative_reference,
                    previous_velocity=previous_velocity,
                    training=config.online_learning,
                )
                last_phase = phase
                last_ee_positions = ee_positions
                last_object_position = reference_center
                last_object_velocity = reference_velocity
                last_relative_reference = relative_reference
                last_previous_velocity = previous_velocity
                last_action = action
                command_velocity = action.velocity
                previous_velocity = action.mean_velocity
                latest_weights = action.weights
                for index, name in enumerate(("left", "right")):
                    desired_transforms[name][:3, 3] = (
                        ee_positions[3 * index : 3 * (index + 1)]
                        + config.command_lookahead_s
                        * command_velocity[3 * index : 3 * (index + 1)]
                    )
                    mean_force_weight = float(np.mean(action.weights["force"]))
                    stiffness = float(np.clip(430.0 - 17.0 * mean_force_weight, 150.0, 430.0))
                    controller = runtime["impedance"][name]
                    controller.K[3:, 3:] = stiffness * np.eye(3)
                    controller.config.K_pos = stiffness

                rows.append(
                    {
                        "time_s": float(data.time),
                        "phase": phase.name.lower(),
                        "endpoint_error_m": current.endpoint_error,
                        "left_force_n": current.left_force,
                        "right_force_n": current.right_force,
                        "object_x": current.object_position[0],
                        "object_y": current.object_position[1],
                        "object_z": current.object_position[2],
                        "reward": last_reward,
                        "approximate_kl": 0.0 if latest_update is None else latest_update.approximate_kl,
                        "actor_loss": 0.0 if latest_update is None else latest_update.actor_loss,
                        "critic_loss": 0.0 if latest_update is None else latest_update.critic_loss,
                        **{
                            f"weight_{name}": float(np.mean(values))
                            for name, values in action.weights.items()
                        },
                    }
                )
                previous_snapshot = current
                last_observation = observation

            elapsed_in_phase = float(data.time) - phase_started
            if phase is BimanualPhase.INTERCEPT:
                close_fraction = 0.0
            elif phase is BimanualPhase.PRE_CONTACT:
                close_fraction = np.clip(elapsed_in_phase / 0.35, 0.0, 0.45)
            elif phase is BimanualPhase.GRASPING:
                close_fraction = np.clip(0.45 + elapsed_in_phase / 0.35, 0.0, 1.0)
            else:
                close_fraction = config.grasp_hold_fraction
            for name in ("left", "right"):
                runtime["impedance"][name].apply(
                    model, data, runtime["arms"][name], desired_transforms[name]
                )
                gripper = FFW_GRIPPERS[name]
                data.ctrl[runtime["gripper_actuators"][name]] = (
                    (1.0 - close_fraction) * gripper.open_ctrl
                    + close_fraction * gripper.close_ctrl
                )
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
    finally:
        if viewer is not None:
            viewer.close()

    final = _snapshot(runtime)
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
            reward=_reward(previous_snapshot, final, phase, command_velocity, manipulation_goal),
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
    goal_error = (
        None
        if manipulation_goal is None
        else float(np.linalg.norm(final.object_position - manipulation_goal))
    )
    physical_grasp = final.left_force > config.contact_force and final.right_force > config.contact_force
    reached_grasp = grasp_time is not None
    return DemoSummary(
        success=bool(reached_grasp and physical_grasp),
        final_phase=phase.name.lower(),
        simulated_time_s=float(data.time),
        grasp_time_s=grasp_time,
        manipulation_time_s=manipulation_time,
        left_contact_force_n=float(final.left_force),
        right_contact_force_n=float(final.right_force),
        final_endpoint_error_m=final.endpoint_error,
        final_object_goal_error_m=goal_error,
        online_updates=learner.update_count,
        total_transitions=total_transitions,
        actor_weight_change_l2=float(np.linalg.norm(final_actor - initial_actor)),
        device=str(learner.device),
        object_displacement_m=float(np.linalg.norm(final.object_position - initial_object_position)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-online-learning", action="store_true")
    parser.add_argument("--exploration-std", type=float, default=0.015)
    parser.add_argument("--rollout-size", type=int, default=32)
    parser.add_argument(
        "--offline-training",
        action="store_true",
        help="use multi-epoch PPO updates instead of deployment-safe online updates",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log", default=str(ROOT / "sweep_results" / "bimanual_acmpc_online.csv"))
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_demo(
        DemoConfig(
            duration_s=args.duration,
            device=args.device,
            online_learning=not args.no_online_learning,
            exploration_std=args.exploration_std,
            rollout_size=args.rollout_size,
            offline_training=args.offline_training,
            seed=args.seed,
            viewer=args.viewer,
            log_path=args.log,
            checkpoint_path=args.checkpoint,
        )
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    if not summary.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
