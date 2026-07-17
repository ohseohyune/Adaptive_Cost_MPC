"""Kinematics and math helpers for null-space QP self-motion tests."""

from __future__ import annotations

import mujoco
import numpy as np

from control.clik.jacobian import compute_full_body_jacobian
from control.clik.kinematics import compute_pose_error, get_ee_transform
from control.clik.solvers import damped_least_squares
from control.clik.types import SerialArm


def set_arm_position(data: mujoco.MjData, arm: SerialArm, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=float).reshape(arm.dof)
    data.qpos[arm.qpos_indices] = q
    data.ctrl[arm.actuator_ids] = q


def ee_position_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    if arm.ee_site_id is not None:
        mujoco.mj_jacSite(model, data, jacp, jacr, arm.ee_site_id)
    else:
        mujoco.mj_jacBody(model, data, jacp, jacr, arm.ee_body_id)
    return jacp[:, arm.qvel_indices].copy()


def ee_pose_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
) -> np.ndarray:
    """6×n body-frame Jacobian for full pose CLIK (Modern Robotics PoE)."""
    return compute_full_body_jacobian(model, data, arm)


def svd_pseudoinverse(jacobian: np.ndarray, rcond: float = 1e-5) -> np.ndarray:
    """Return an SVD pseudo-inverse so J @ (I - J+J) is nearly zero."""

    return np.linalg.pinv(jacobian, rcond=rcond)


def solve_ik(
    model: mujoco.MjModel,
    arm: SerialArm,
    target_transform: np.ndarray,
    q_init: np.ndarray | None = None,
    max_iter: int = 300,
    gain: float = 0.5,
    damping: float = 1e-3,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-4,
) -> tuple[np.ndarray, float, float]:
    """Iterative CLIK IK. Returns (q, final_pos_err, final_rot_err).

    Uses a private MjData copy so the caller's simulation state is untouched.
    """
    ik_data = mujoco.MjData(model)
    q = (q_init.copy() if q_init is not None else np.zeros(arm.dof)).astype(float)
    q = np.clip(q, arm.ctrl_low, arm.ctrl_high)
    target = np.asarray(target_transform, dtype=float).reshape(4, 4)

    pos_err = rot_err = float("inf")
    for _ in range(max_iter):
        ik_data.qpos[arm.qpos_indices] = q
        mujoco.mj_forward(model, ik_data)
        error = compute_pose_error(get_ee_transform(ik_data, arm), target)
        rot_err = float(np.linalg.norm(error[:3]))
        pos_err = float(np.linalg.norm(error[3:]))
        if rot_err < rot_tol and pos_err < pos_tol:
            break
        jacobian = compute_full_body_jacobian(model, ik_data, arm)
        q = np.clip(q + damped_least_squares(jacobian, gain * error, damping),
                    arm.ctrl_low, arm.ctrl_high)

    return q, pos_err, rot_err


def make_distant_nominal(q_initial: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
    offset = np.array([0.9, -0.8, 1.1, -1.2, 0.9, -0.8, 0.7])
    return np.clip(q_initial + offset, q_min + 0.05, q_max - 0.05)
