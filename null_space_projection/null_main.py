"""Standalone null-space QP self-motion test in MuJoCo.

Scenario:
    Keep the right end-effector fixed (position + orientation) and hold the
    initial posture until the EE error is small enough, then switch q_nom to a
    distant posture so only null-space motion can reconfigure the arm.

Run:
    python null_main.py
"""

from __future__ import annotations

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

try:
    from .null_config import (
        MOTION_SPEED_SCALE,
        PROJECT_ROOT,
        XML_PATH,
        NullSpaceQPConfig,
        SimParams,
    )
except ImportError:
    from null_config import (
        MOTION_SPEED_SCALE,
        PROJECT_ROOT,
        XML_PATH,
        NullSpaceQPConfig,
        SimParams,
    )

sys.path.insert(0, str(PROJECT_ROOT))

from control.clik.bimanual import joint_limit_margin
from control.clik.kinematics import compute_pose_error, get_ee_transform
from control.clik.mujoco_utils import build_serial_arm
from null_space_projection.null_kinematics import (
    ee_position_jacobian,
    ee_pose_jacobian,
    set_arm_position,
    svd_pseudoinverse,
)
from null_space_projection.null_qp_controller import NullSpaceQPController
from robot.ffw_config import FFW_ARMS
from null_space_projection.null_utils import (
    limit_vector_norm,
    log_step,
    null_motion_error_scale,
    update_scene_visuals,
)


TASK_MODE = "position"
RECONFIGURE_DELAY_S = 1.0
NULL_POSTURE_OFFSET_NORM = 2.0
VISIBLE_QDOT_LIMIT = 1.2
VISIBLE_MAX_NULL_VELOCITY = 0.5


def _setup_model():
    """Load model, build arms, and set initial configuration."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    right_arm = build_serial_arm(model, FFW_ARMS["right"])
    left_arm = build_serial_arm(model, FFW_ARMS["left"])

    q_left_initial = np.zeros(left_arm.dof)
    # Keep joint 2 away from its upper limit (0) so task scaling does not
    # freeze the controller before the null-space behavior can be observed.
    q_right_initial = np.array([0.0, -1.5, 0.6, -1.2, 0.6, 0.5, 0.0])
    set_arm_position(data, right_arm, q_right_initial)
    set_arm_position(data, left_arm, q_left_initial)
    mujoco.mj_forward(model, data)

    return model, data, right_arm, left_arm, q_right_initial, q_left_initial


def _make_reconfigure_nominal(
    q_initial: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    nullspace_projector: np.ndarray,
) -> np.ndarray:
    """Build a posture target from the actual start-pose null-space direction."""

    q_nominal = np.asarray(q_initial, dtype=float).copy()
    preferred_offset = np.array([1.1, 0.0, -1.0, 0.8, 0.8, -0.8, 0.5])
    null_offset = nullspace_projector @ preferred_offset
    null_offset_norm = np.linalg.norm(null_offset)
    if null_offset_norm < 1e-8:
        _, _, vh = np.linalg.svd(nullspace_projector)
        null_offset = vh[0]
        null_offset_norm = np.linalg.norm(null_offset)
    null_offset = NULL_POSTURE_OFFSET_NORM * null_offset / null_offset_norm
    q_nominal += null_offset
    return np.clip(q_nominal, q_min + 0.05, q_max - 0.05)


def _compute_target(model, right_arm, q_right_initial):
    """Use the start-pose FK as the fixed EE target and only change q_nom."""

    tmp = mujoco.MjData(model)
    set_arm_position(tmp, right_arm, q_right_initial)
    mujoco.mj_forward(model, tmp)
    fixed_transform = get_ee_transform(tmp, right_arm)
    jacobian = _task_jacobian(model, tmp, right_arm)
    jacobian_pinv = svd_pseudoinverse(jacobian)
    nullspace_projector = np.eye(right_arm.dof) - jacobian_pinv @ jacobian
    q_nom_hold = q_right_initial.copy()
    q_nom_reconfigure = _make_reconfigure_nominal(
        q_right_initial,
        right_arm.ctrl_low,
        right_arm.ctrl_high,
        nullspace_projector,
    )
    nullspace_leak = np.linalg.norm(jacobian @ (q_nom_reconfigure - q_nom_hold))

    print(
        "Self-motion target prepared from start-pose FK.\n"
        f"  task mode         = {TASK_MODE}\n"
        f"  fixed EE position = {np.round(fixed_transform[:3, 3], 4)}\n"
        f"  q_nom_hold        = {np.round(q_nom_hold, 4)}\n"
        f"  q_nom_reconfigure = {np.round(q_nom_reconfigure, 4)}\n"
        f"  initial J*dq_nom  = {nullspace_leak:.2e}",
        flush=True,
    )

    return fixed_transform, q_nom_hold, q_nom_reconfigure


def _task_jacobian(model, data, right_arm) -> np.ndarray:
    if TASK_MODE == "position":
        return ee_position_jacobian(model, data, right_arm)
    if TASK_MODE == "pose":
        return ee_pose_jacobian(model, data, right_arm)
    raise ValueError(f"Unknown TASK_MODE: {TASK_MODE}")


def _task_error_and_command(
    current_transform: np.ndarray,
    fixed_transform: np.ndarray,
    params: SimParams,
) -> tuple[np.ndarray, np.ndarray]:
    if TASK_MODE == "position":
        task_error = fixed_transform[:3, 3] - current_transform[:3, 3]
        command = params.gain_vector[3:] * task_error
    elif TASK_MODE == "pose":
        pose_error_6d = compute_pose_error(current_transform, fixed_transform)
        task_error = pose_error_6d
        command = params.gain_vector * task_error
    else:
        raise ValueError(f"Unknown TASK_MODE: {TASK_MODE}")
    return task_error, command


def _build_controller(n_dof: int, physics_dt: float):
    """Create the QP controller and bundle all simulation parameters."""
    controller = NullSpaceQPController(
        n_dof,
        NullSpaceQPConfig(
            posture_weight=6.0,
            damping_weight=1e-4,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=200,
        ),
    )

    position_gain = 64.0 * MOTION_SPEED_SCALE
    orientation_gain = 32.0 * MOTION_SPEED_SCALE

    params = SimParams(
        physics_dt=physics_dt,
        # gain_vector: [orientation (3), position (3)] matches SE(3) log order
        gain_vector=np.concatenate([np.full(3, orientation_gain), np.full(3, position_gain)]),
        qdot_limit=np.full(n_dof, VISIBLE_QDOT_LIMIT),
        switch_threshold=0.015,
        null_scale_start=0.012,
        null_scale_stop=0.035,
        max_null_velocity=VISIBLE_MAX_NULL_VELOCITY,
    )

    return controller, params


def _run_simulation(
    model,
    data,
    right_arm,
    left_arm,
    q_left_initial,
    fixed_transform,
    q_nom_hold,
    q_nom_reconfigure,
    controller,
    params: SimParams,
) -> None:
    print("Null-space QP self-motion test")
    print(f"XML: {XML_PATH}")
    print(f"Task mode: {TASK_MODE}")
    print(f"Fixed right EE position: {fixed_transform[:3, 3]}")
    if TASK_MODE == "pose":
        print(f"Fixed right EE rotation:\n{fixed_transform[:3, :3]}")
    else:
        print("Right EE orientation is intentionally free for a visible redundancy demo.")
    print(
        f"q_nom switches after t >= {RECONFIGURE_DELAY_S:.1f} s "
        f"and position error < {params.switch_threshold:.3f} m."
    )
    print(f"q_nom reconfigure target: {q_nom_reconfigure}")
    print(
        f"control_dt={params.physics_dt:.4f} s, "
        f"motion_speed_scale={MOTION_SPEED_SCALE:.2f}"
    )
    print(
        "Columns: t | phase | ori_err | pos_err | vel_task_err | "
        "null_vel | null_scale | min_margin | alpha | status"
    )

    last_print_time = -1.0
    reconfigure_started = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            sim_time = float(data.time)
            q_current = data.qpos[right_arm.qpos_indices].copy()

            current_transform = get_ee_transform(data, right_arm)
            pose_error_6d = compute_pose_error(current_transform, fixed_transform)
            task_error, xdot_desired = _task_error_and_command(
                current_transform, fixed_transform, params
            )

            jacobian = _task_jacobian(model, data, right_arm)
            jacobian_pinv = svd_pseudoinverse(jacobian)
            qdot_task = jacobian_pinv @ xdot_desired
            nullspace_projector = np.eye(right_arm.dof) - jacobian_pinv @ jacobian

            ori_err = float(np.linalg.norm(pose_error_6d[:3]))
            if TASK_MODE == "position":
                pos_err = float(np.linalg.norm(task_error))
            else:
                pos_err = float(np.linalg.norm(pose_error_6d[3:]))

            if (
                not reconfigure_started
                and sim_time >= RECONFIGURE_DELAY_S
                and pos_err <= params.switch_threshold
            ):
                reconfigure_started = True
                print(
                    f"--- q_nom switched at t={sim_time:.3f} s "
                    f"(pos_err={pos_err:.4f} m, ori_err={ori_err:.4f} rad) ---",
                    flush=True,
                )

            q_nominal = q_nom_reconfigure if reconfigure_started else q_nom_hold
            result = controller.solve(
                q_current=q_current,
                qdot_task=qdot_task,
                nullspace_projector=nullspace_projector,
                q_nominal=q_nominal,
                dt=params.physics_dt,
                q_min=right_arm.ctrl_low,
                q_max=right_arm.ctrl_high,
                qdot_min=-params.qdot_limit,
                qdot_max=params.qdot_limit,
            )

            null_qdot = limit_vector_norm(nullspace_projector @ result.z, params.max_null_velocity)
            null_scale = (
                null_motion_error_scale(pos_err, params.null_scale_start, params.null_scale_stop)
                if reconfigure_started
                else 0.0
            )
            qdot_command = result.qdot_task + null_scale * null_qdot
            q_command = np.clip(
                q_current + params.physics_dt * qdot_command,
                right_arm.ctrl_low,
                right_arm.ctrl_high,
            )
            data.qpos[right_arm.qpos_indices] = q_command
            data.qpos[left_arm.qpos_indices] = q_left_initial
            data.ctrl[right_arm.actuator_ids] = q_command
            data.ctrl[left_arm.actuator_ids] = q_left_initial
            data.qvel[:] = 0.0
            data.time += params.physics_dt
            mujoco.mj_forward(model, data)

            with viewer.lock():
                update_scene_visuals(viewer, current_transform, fixed_transform[:3, 3])
            viewer.sync()

            if sim_time - last_print_time >= 0.1:
                min_margin = float(
                    np.min(joint_limit_margin(q_current, right_arm.ctrl_low, right_arm.ctrl_high))
                )
                log_step(
                    sim_time, reconfigure_started,
                    ori_err, pos_err,
                    jacobian @ qdot_command - xdot_desired,
                    null_scale * null_qdot,
                    null_scale, min_margin,
                    result.task_scaling, result.status,
                )
                last_print_time = sim_time

            sleep_time = params.physics_dt - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def main() -> None:
    model, data, right_arm, left_arm, q_right_initial, q_left_initial = _setup_model()
    fixed_transform, q_nom_hold, q_nom_reconfigure = _compute_target(model, right_arm, q_right_initial)
    controller, params = _build_controller(right_arm.dof, float(model.opt.timestep))
    _run_simulation(
        model, data,
        right_arm, left_arm, q_left_initial,
        fixed_transform, q_nom_hold, q_nom_reconfigure,
        controller, params,
    )


if __name__ == "__main__":
    main()
