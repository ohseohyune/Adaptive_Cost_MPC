"""Rigid-body dynamics helpers extracted from MuJoCo state."""

from __future__ import annotations

import mujoco
import numpy as np

from control.clik.types import SerialArm


def arm_mass_matrix(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
) -> np.ndarray:
    """Return the (n x n) joint-space mass matrix block for the arm."""
    M_full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M_full)
    idx = arm.qvel_indices
    return M_full[np.ix_(idx, idx)]


def arm_bias_torque(
    data: mujoco.MjData,
    arm: SerialArm,
) -> np.ndarray:
    """Return Coriolis + gravity torques for the arm joints.

    data.qfrc_bias[qvel_index] is the generalized force due to Coriolis
    and gravity evaluated at the current (q, qdot) state. For slow motion
    this is dominated by gravity and serves as a feedforward compensation term.
    """
    return data.qfrc_bias[arm.qvel_indices].copy()


def arm_joint_damping(
    model: mujoco.MjModel,
    arm: SerialArm,
) -> np.ndarray:
    """Return per-joint passive damping coefficients (from XML joint damping=...)."""
    return model.dof_damping[arm.qvel_indices].copy()


def arm_joint_state(
    data: mujoco.MjData,
    arm: SerialArm,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (q, qdot) for the arm joints."""
    q = data.qpos[arm.qpos_indices].copy()
    qdot = data.qvel[arm.qvel_indices].copy()
    return q, qdot


def arm_actuator_kp(
    model: mujoco.MjModel,
    arm: SerialArm,
) -> np.ndarray:
    """Return the kp gain for each position actuator driving this arm.

    For a <position kp="..."> actuator, MuJoCo stores kp in
    model.actuator_gainprm[:, 0].
    """
    return model.actuator_gainprm[arm.actuator_ids, 0].copy()
