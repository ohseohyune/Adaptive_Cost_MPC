"""Dataclasses used by the reusable CLIK framework."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SerialArmConfig:
    """Robot-specific names needed by the generic CLIK code."""

    name: str
    joint_names: tuple[str, ...]
    end_effector_body_name: str
    actuator_names: tuple[str, ...] | None = None
    end_effector_site_name: str | None = None


@dataclass(frozen=True)
class SerialArm:
    """
    Resolved MuJoCo IDs and array addresses for one serial manipulator.

    qpos and qvel are not the same array:
    - qpos stores generalized positions. A hinge joint uses one qpos entry,
      but a free joint uses seven entries (xyz + quaternion).
    - qvel stores generalized velocities. A hinge joint uses one qvel entry,
      while a free joint uses six entries.
    - MuJoCo Jacobians are shaped against nv, so their columns match qvel/dof
      indices, not qpos indices.

    ctrl is different again:
    - ctrl stores actuator commands and has length nu.
    - Actuator order is defined by the XML <actuator> block and is not
      guaranteed to match joint order. A position actuator's ctrl value is a
      desired joint position, but its array index is still an actuator ID.
    """

    config: SerialArmConfig
    joint_ids: np.ndarray
    actuator_ids: np.ndarray
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    ctrl_low: np.ndarray
    ctrl_high: np.ndarray
    ee_body_id: int
    ee_site_id: int | None

    @property
    def dof(self) -> int:
        return len(self.joint_ids)
