"""Pure kinematic and math utility functions for null-space QP control."""

from __future__ import annotations

import mujoco
import numpy as np

from control.clik.bimanual import joint_limit_margin  # noqa: F401 — re-exported for callers
from control.clik.kinematics import get_ee_position  # noqa: F401 — re-exported for callers
from control.clik.types import SerialArm


def set_arm_position(data: mujoco.MjData, arm: SerialArm, q: np.ndarray) -> None:
    """Teleport the arm to q and set matching actuator commands;
    used for initialisation before the physics loop starts."""
    q = np.asarray(q, dtype=float).reshape(arm.dof)
    data.qpos[arm.qpos_indices] = q
    data.ctrl[arm.actuator_ids] = q


def ee_position_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
) -> np.ndarray:
    """Extract the 3×n position Jacobian for the arm's end-effector,
    slicing full-model columns down to the arm's own qvel indices."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    if arm.ee_site_id is not None:
        mujoco.mj_jacSite(model, data, jacp, jacr, arm.ee_site_id)
    else:
        mujoco.mj_jacBody(model, data, jacp, jacr, arm.ee_body_id)
    jacobian = jacp[:, arm.qvel_indices].copy()
    assert jacobian.ndim == 2, "Jacobian must be 2D"
    # m <= n required for a kinematically redundant manipulator
    assert jacobian.shape[0] <= jacobian.shape[1], (
        f"Expected fat Jacobian (m<=n), got shape {jacobian.shape}"
    )
    return jacobian


def svd_pseudoinverse(jacobian: np.ndarray, rcond: float = 1e-5) -> np.ndarray:
    """SVD-based pseudoinverse with rcond threshold to avoid amplifying
    near-singular Jacobian directions at kinematic singularities."""
    return np.linalg.pinv(jacobian, rcond=rcond)


def limit_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """Clamp a velocity vector to a maximum Euclidean norm while preserving
    direction; prevents null-space motion from dominating task motion."""
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)


def null_motion_error_scale(
    position_error_norm: float,
    start: float,
    stop: float,
) -> float:
    """Linear fade-out of null-space motion as EE position error grows;
    suppresses self-motion when the task loop has not yet converged."""
    if position_error_norm <= start:
        return 1.0
    if position_error_norm >= stop:
        return 0.0
    return float((stop - position_error_norm) / (stop - start))
