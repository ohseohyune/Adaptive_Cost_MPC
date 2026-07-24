"""World-frame Cartesian impedance controller with null-space stabilisation.

Controller law:
    e   = [e_rot; e_pos]                 # 6D pose error (world frame)
    dx  = J_world @ qdot                 # 6D EE velocity [omega; v]
    F   = K @ e - D @ dx                 # impedance wrench
    τ   = J_world^T @ F + τ_bias + τ_ns  # joint torques

Null-space stabilisation (prevents 1-DOF null-space drift in the 7-DOF arm):
    N   = I - J^+ @ J                   # 7×7 null-space projector
    τ_ns = N @ (Kp_ns*(q_home-q) - Kd_ns*qdot)

Position actuator inversion (same as Phase 1):
    ctrl = q + (τ + d * qdot) / kp

Stiffness profiles
------------------
K_approach  high stiffness for free-space tracking
K_contact   low stiffness for compliant contact (set externally)

Convention: world-frame Jacobian from mj_jacSite is [J_rot; J_pos] (angular first),
so the 6D error vector and impedance wrench follow the same ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from control.clik.dynamics import (
    arm_actuator_kp,
    arm_bias_torque,
    arm_joint_damping,
    arm_joint_state,
)
from control.clik.kinematics import get_ee_transform
from control.clik.types import SerialArm
from control.clik.utils import matrix_log_se3


@dataclass
class CartesianImpedanceConfig:
    """Stiffness/damping gains and limits for one arm."""

    K_pos: float = 500.0    # translational stiffness [N/m]
    K_rot: float = 50.0     # rotational stiffness [N·m/rad]
    D_pos: float = 100.0    # translational damping  [N·s/m]
    D_rot: float = 15.0     # rotational damping     [N·m·s/rad]
    tau_limit: float = 300.0
    gravity_comp: bool = True
    Kp_ns: float = 5.0      # null-space joint stiffness toward home [Nm/rad]
    Kd_ns: float = 2.0      # null-space joint damping               [Nm·s/rad]


@dataclass
class ImpedanceResult:
    """Diagnostic snapshot from one impedance control step."""

    e_pos: np.ndarray       # position error [m], world frame
    e_rot: np.ndarray       # rotation error [rad·axis], world frame
    e_pos_norm: float       # |e_pos| [m]
    e_rot_norm: float       # |e_rot| [rad]
    v_ee: np.ndarray        # EE linear velocity  [m/s]
    omega_ee: np.ndarray    # EE angular velocity [rad/s]
    F_imp: np.ndarray       # 6D impedance wrench [torque; force] world frame
    tau: np.ndarray         # joint torques after clipping
    ctrl_command: np.ndarray  # position actuator ctrl values


def _rot_log_world(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Rotation error in world frame via SO(3) log: log(R_des @ R_cur^T)."""
    T_rot = np.eye(4)
    T_rot[:3, :3] = R_des @ R_cur.T
    return matrix_log_se3(T_rot)[:3]


def ee_jacobian_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
) -> np.ndarray:
    """Return (6 x dof) world-frame Jacobian ordered [J_rot; J_pos]."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    if arm.ee_site_id is not None:
        mujoco.mj_jacSite(model, data, jacp, jacr, arm.ee_site_id)
    else:
        mujoco.mj_jacBody(model, data, jacp, jacr, arm.ee_body_id)
    J_pos = jacp[:, arm.qvel_indices]
    J_rot = jacr[:, arm.qvel_indices]
    return np.vstack([J_rot, J_pos])


class CartesianImpedanceController:
    """World-frame Cartesian impedance controller with null-space stabilisation.

    Implements arbitrary stiffness/damping through the existing position
    actuator interface (same ctrl-inversion as Phase 1 PDTorqueController).

    The 7-DOF arm has a 1-D null-space when tracking 6-DOF EE pose.  Without
    null-space control the arm drifts freely in that direction, causing the
    EE to drift by ~20 mm over 3 s (gravity-compensation imperfection amplified
    by the Jacobian).  A small Kp_ns/Kd_ns restoring term prevents this.
    """

    def __init__(
        self,
        config: CartesianImpedanceConfig,
        arm: SerialArm,
        model: mujoco.MjModel,
        q_home: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self._dof = arm.dof
        self._kp_act = arm_actuator_kp(model, arm)
        self._damp = arm_joint_damping(model, arm)
        self._ctrl_low = arm.ctrl_low.copy()
        self._ctrl_high = arm.ctrl_high.copy()
        self._q_home = q_home.copy() if q_home is not None else None

        K_diag = np.array([config.K_rot] * 3 + [config.K_pos] * 3, dtype=float)
        D_diag = np.array([config.D_rot] * 3 + [config.D_pos] * 3, dtype=float)
        self.K = np.diag(K_diag)
        self.D = np.diag(D_diag)

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm: SerialArm,
        T_des: np.ndarray,
        wrench_feedforward: np.ndarray | None = None,
    ) -> ImpedanceResult:
        """Compute impedance torques without writing to data."""
        T_cur = get_ee_transform(data, arm)
        R_cur, p_cur = T_cur[:3, :3], T_cur[:3, 3]
        R_des, p_des = T_des[:3, :3], T_des[:3, 3]

        J = ee_jacobian_world(model, data, arm)          # 6 x dof
        q, qdot = arm_joint_state(data, arm)
        dx = J @ qdot                                    # [omega_world; v_world]

        e_rot = _rot_log_world(R_cur, R_des)             # [rad·axis]
        e_pos = p_des - p_cur                            # [m]
        e = np.concatenate([e_rot, e_pos])               # 6D, matches J ordering

        F_imp = self.K @ e - self.D @ dx                 # 6D wrench
        if wrench_feedforward is not None:
            F_imp = F_imp + np.asarray(wrench_feedforward, dtype=float).reshape(6)

        tau_imp = J.T @ F_imp
        if self.config.gravity_comp:
            tau_bias = arm_bias_torque(data, arm)
        else:
            tau_bias = np.zeros(self._dof)

        # Null-space stabilisation: J^+ = J.T @ (J @ J.T)^{-1} (right pseudoinverse)
        # N = I - J^+ @ J projects into the null-space of J.
        J_pinv = np.linalg.pinv(J)                          # dof x 6
        N = np.eye(self._dof) - J_pinv @ J                  # dof x dof
        if self._q_home is not None:
            tau_ns = N @ (self.config.Kp_ns * (self._q_home - q)
                          - self.config.Kd_ns * qdot)
        else:
            tau_ns = N @ (-self.config.Kd_ns * qdot)

        tau_raw = tau_imp + tau_bias + tau_ns
        tau = np.clip(tau_raw, -self.config.tau_limit, self.config.tau_limit)

        ctrl = q + (tau + self._damp * qdot) / self._kp_act
        ctrl = np.clip(ctrl, self._ctrl_low, self._ctrl_high)

        return ImpedanceResult(
            e_pos=e_pos,
            e_rot=e_rot,
            e_pos_norm=float(np.linalg.norm(e_pos)),
            e_rot_norm=float(np.linalg.norm(e_rot)),
            v_ee=dx[3:].copy(),
            omega_ee=dx[:3].copy(),
            F_imp=F_imp,
            tau=tau,
            ctrl_command=ctrl,
        )

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm: SerialArm,
        T_des: np.ndarray,
        wrench_feedforward: np.ndarray | None = None,
    ) -> ImpedanceResult:
        """Compute impedance torques and write ctrl to data."""
        result = self.compute(
            model,
            data,
            arm,
            T_des,
            wrench_feedforward=wrench_feedforward,
        )
        data.ctrl[arm.actuator_ids] = result.ctrl_command
        return result


def cartesian_force_to_qfrc(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
    F_cart: np.ndarray,
) -> np.ndarray:
    """Map a 3D Cartesian force (world frame at EE site) to joint-space torques.

    This uses the same site Jacobian as the impedance controller, ensuring that
    the applied generalized force is consistent with the task-space stiffness
    (avoids the body-center vs site mismatch from xfrc_applied).

    Returns the 7D joint torque vector; caller should write it to
    ``data.qfrc_applied[arm.qvel_indices]`` before mj_step.
    """
    J = ee_jacobian_world(model, data, arm)
    J_pos = J[3:, :]   # 3 x dof linear part
    return J_pos.T @ np.asarray(F_cart, dtype=float)
