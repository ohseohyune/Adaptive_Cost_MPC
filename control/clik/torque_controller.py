"""Joint-space PD torque controller with gravity compensation.

Strategy (no XML change required):
    MuJoCo position actuator applies:  τ_act = kp * (ctrl - q)
    Joint passive damping applies:      τ_damp = -d * qdot   (in physics solver)
    Net joint torque:                   τ_net = τ_act + τ_damp

To achieve a desired net torque τ_desired, invert the actuator model:
    ctrl = q + (τ_desired + d * qdot) / kp

This lets us implement arbitrary torque laws (PD + gravity comp, impedance, etc.)
through the existing position actuator interface without XML modifications.

Controller law:
    τ_desired = Kp*(q_des - q) + Kd*(qdot_des - qdot) + τ_bias

where τ_bias = data.qfrc_bias[arm.qvel_indices] (Coriolis + gravity feedforward).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from control.clik.dynamics import (
    arm_actuator_kp,
    arm_bias_torque,
    arm_joint_damping,
    arm_joint_state,
)
from control.clik.types import SerialArm


@dataclass
class PDTorqueConfig:
    """Gain and limit configuration for the joint-space PD torque controller."""

    Kp: float = 400.0
    Kd: float = 20.0
    tau_limit: float = 300.0
    gravity_comp: bool = True


@dataclass
class PDTorqueResult:
    """Diagnostic snapshot from one torque control step."""

    q: np.ndarray
    qdot: np.ndarray
    q_des: np.ndarray
    q_error: np.ndarray
    qdot: np.ndarray
    tau_pd: np.ndarray
    tau_bias: np.ndarray
    tau_desired: np.ndarray
    tau_clipped: bool
    ctrl_command: np.ndarray
    q_error_norm: float
    qdot_norm: float
    tau_desired_norm: float


class PDTorqueController:
    """Joint-space PD + gravity-compensation controller.

    Uses position actuators as a torque interface via ctrl inversion,
    so no XML actuator changes are needed.

    Note: MuJoCo joint damping (d=100 N·m·s/rad) is passive and always
    active. The controller Kd adds on top of it. Effective velocity
    damping = Kd + d.
    """

    def __init__(self, config: PDTorqueConfig, arm: SerialArm, model: mujoco.MjModel) -> None:
        self.config = config
        self._dof = arm.dof
        self.Kp = config.Kp * np.ones(arm.dof)
        self.Kd = config.Kd * np.ones(arm.dof)
        self._kp_act = arm_actuator_kp(model, arm)
        self._damp = arm_joint_damping(model, arm)
        self._ctrl_low = arm.ctrl_low.copy()
        self._ctrl_high = arm.ctrl_high.copy()

    def compute_torque(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm: SerialArm,
        q_des: np.ndarray,
        qdot_des: np.ndarray | None = None,
    ) -> tuple[np.ndarray, PDTorqueResult]:
        """Compute desired joint torques and the resulting ctrl command."""
        q, qdot = arm_joint_state(data, arm)
        q_des = np.asarray(q_des, dtype=float)
        if qdot_des is None:
            qdot_des = np.zeros_like(q)

        tau_pd = self.Kp * (q_des - q) + self.Kd * (qdot_des - qdot)
        tau_bias = arm_bias_torque(data, arm) if self.config.gravity_comp else np.zeros(self._dof)

        tau_desired_unclipped = tau_pd + tau_bias
        tau_desired = np.clip(tau_desired_unclipped, -self.config.tau_limit, self.config.tau_limit)
        clipped = not np.allclose(tau_desired, tau_desired_unclipped)

        # Invert position actuator: ctrl = q + (tau_desired + d*qdot) / kp
        ctrl = q + (tau_desired + self._damp * qdot) / self._kp_act
        ctrl = np.clip(ctrl, self._ctrl_low, self._ctrl_high)

        result = PDTorqueResult(
            q=q,
            qdot=qdot,
            q_des=q_des,
            q_error=q_des - q,
            tau_pd=tau_pd,
            tau_bias=tau_bias,
            tau_desired=tau_desired,
            tau_clipped=clipped,
            ctrl_command=ctrl,
            q_error_norm=float(np.linalg.norm(q_des - q)),
            qdot_norm=float(np.linalg.norm(qdot)),
            tau_desired_norm=float(np.linalg.norm(tau_desired)),
        )
        return tau_desired, result

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm: SerialArm,
        q_des: np.ndarray,
        qdot_des: np.ndarray | None = None,
    ) -> PDTorqueResult:
        """Compute torques and write ctrl to data. Returns diagnostics."""
        _, result = self.compute_torque(model, data, arm, q_des, qdot_des)
        data.ctrl[arm.actuator_ids] = result.ctrl_command
        return result
