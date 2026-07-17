"""Top-level CLIK update loop."""

import mujoco
import numpy as np

from control.clik.jacobian import compute_full_body_jacobian
from control.clik.kinematics import compute_pose_error, get_ee_transform
from control.clik.solvers import (
    damped_least_squares,
    damped_least_squares_with_nullspace,
)
from control.clik.types import SerialArm


def _apply_position_actuator_command(
    data: mujoco.MjData,
    arm: SerialArm,
    joint_delta: np.ndarray,
) -> np.ndarray:
    """Apply a joint-space increment through MuJoCo position actuators."""
    q_current = data.qpos[arm.qpos_indices].copy()
    q_command = q_current + joint_delta
    q_command = np.clip(q_command, arm.ctrl_low, arm.ctrl_high)
    data.ctrl[arm.actuator_ids] = q_command
    return q_command


def _scale_pose_error(pose_error: np.ndarray, gain: float | np.ndarray) -> np.ndarray:
    """Apply scalar or per-axis gain to a 6D pose error."""
    gain_array = np.asarray(gain, dtype=float)
    if gain_array.ndim == 0:
        return float(gain_array) * pose_error
    return gain_array.reshape(6) * pose_error


def pose_clik_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
    target_transform: np.ndarray,
    gain: float | np.ndarray = 0.5,
    damping: float = 1e-3,
    max_joint_step: float = 0.03,
    posture_reference: np.ndarray | None = None,
    posture_gain: float = 0.0,
    posture_weights: np.ndarray | None = None,
) -> dict:
    """
    Run one full SE(3) body-frame CLIK update.

    The flow is intentionally short and explicit:
    1. Read the current end-effector pose.
    2. Compute body-frame SE(3) pose error.
    3. Compute the full body Jacobian.
    4. Solve for dq with damped least squares, optionally with a null-space
       posture objective.
    5. Apply q + dq through MuJoCo position actuators.

    The closed-loop update mirrors the old OpenManipulator-Y structure, but
    uses MuJoCo to get the current EE transform and body Jacobian:

        T_err = T_cur^{-1} @ T_des
        e_b   = log(T_err)^vee              # [rot_error, pos_error]
        dq    = J_body_dls^+ @ (K * e_b)
        ctrl  = q_current + dq

    Redundant manipulators such as a 7-DOF arm have more joints than the 6D task.
    If posture_reference is provided, the extra freedom nudges the arm toward
    that posture without directly changing the end-effector task.
    """

    current_transform = get_ee_transform(data, arm)
    target_transform = np.asarray(target_transform, dtype=float).reshape(4, 4)
    pose_error = compute_pose_error(current_transform, target_transform)
    q_current = data.qpos[arm.qpos_indices].copy()

    commanded_error = _scale_pose_error(pose_error, gain)
    jacobian = compute_full_body_jacobian(model, data, arm)

    if posture_reference is None or posture_gain <= 0.0:
        joint_delta_unclipped = damped_least_squares(
            jacobian,
            commanded_error,
            damping,
        )
        task_delta = joint_delta_unclipped.copy()
        null_delta = np.zeros_like(joint_delta_unclipped)
        q_reference = None
    else:
        q_reference = np.asarray(posture_reference, dtype=float).reshape(arm.dof)
        q_reference = np.clip(q_reference, arm.ctrl_low, arm.ctrl_high)
        (
            joint_delta_unclipped,
            task_delta,
            null_delta,
        ) = damped_least_squares_with_nullspace(
            jacobian=jacobian,
            error=commanded_error,
            damping=damping,
            q_current=q_current,
            q_reference=q_reference,
            posture_gain=posture_gain,
            posture_weights=posture_weights,
        )

    joint_delta = np.clip(joint_delta_unclipped, -max_joint_step, max_joint_step)
    q_command_unclipped = q_current + joint_delta
    q_command = _apply_position_actuator_command(data, arm, joint_delta)

    rotation_error = pose_error[:3]
    position_error_body = pose_error[3:]
    position_error_world = target_transform[:3, 3] - current_transform[:3, 3]

    return {
        "current_transform": current_transform,
        "target_transform": target_transform.copy(),
        "pose_error": pose_error,
        "rotation_error": rotation_error,
        "position_error_body": position_error_body,
        "position_error_world": position_error_world,
        "rotation_error_norm": float(np.linalg.norm(rotation_error)),
        "position_error_norm": float(np.linalg.norm(position_error_world)),
        "pose_error_norm": float(np.linalg.norm(pose_error)),
        "jacobian": jacobian,
        "dq": joint_delta,
        "dq_unclipped": joint_delta_unclipped,
        "dq_was_clipped": bool(not np.allclose(joint_delta, joint_delta_unclipped)),
        "dq_task": task_delta,
        "dq_null": null_delta,
        "q_command": q_command,
        "q_command_unclipped": q_command_unclipped,
        "q_command_was_clipped": bool(not np.allclose(q_command, q_command_unclipped)),
        "q_reference": None if q_reference is None else q_reference.copy(),
    }
