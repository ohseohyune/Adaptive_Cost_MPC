"""Hybrid tangential-position / normal-force side-squeeze controller.

Normal force is regulated through an admittance offset because the FFW model
uses position actuators. Tangential position and pad orientation remain under
Cartesian impedance control in the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.squeeze.config import SideSqueezeConfig
from control.squeeze.pad_contact import BilateralPadContact


@dataclass(frozen=True)
class SqueezeCommand:
    left_position: np.ndarray
    right_position: np.ndarray
    left_compression: float
    right_compression: float
    left_filtered_force: float
    right_filtered_force: float


class HybridSqueezeController:
    """Generate pad targets with independent left/right force admittance."""

    def __init__(self, config: SideSqueezeConfig) -> None:
        self.config = config
        self.left_inward = np.array([0.0, -1.0, 0.0])
        self.right_inward = np.array([0.0, 1.0, 0.0])
        self.reset()

    def reset(self) -> None:
        self.left_compression = -self.config.precontact_gap
        self.right_compression = -self.config.precontact_gap
        self.left_filtered_force = 0.0
        self.right_filtered_force = 0.0

    def update(
        self,
        *,
        box_center: np.ndarray,
        contact: BilateralPadContact,
        dt: float,
        desired_force: float | tuple[float, float] | None = None,
    ) -> SqueezeCommand:
        """Advance force admittance and return world-frame pad positions."""

        alpha = float(np.clip(self.config.force_filter_alpha, 0.0, 1.0))
        self.left_filtered_force += alpha * (
            contact.left.normal_force - self.left_filtered_force
        )
        self.right_filtered_force += alpha * (
            contact.right.normal_force - self.right_filtered_force
        )
        if isinstance(desired_force, tuple):
            left_desired_force, right_desired_force = desired_force
        else:
            left_desired_force = right_desired_force = desired_force
        self.left_compression = self._advance(
            self.left_compression, self.left_filtered_force, dt, left_desired_force
        )
        self.right_compression = self._advance(
            self.right_compression, self.right_filtered_force, dt, right_desired_force
        )

        box_center = np.asarray(box_center, dtype=float).reshape(3)
        nominal_offset = self.config.box_half_y + self.config.pad_half_thickness
        left_nominal = box_center + np.array([0.0, nominal_offset, 0.0])
        right_nominal = box_center - np.array([0.0, nominal_offset, 0.0])
        return SqueezeCommand(
            left_position=left_nominal + self.left_inward * self.left_compression,
            right_position=right_nominal + self.right_inward * self.right_compression,
            left_compression=float(self.left_compression),
            right_compression=float(self.right_compression),
            left_filtered_force=float(self.left_filtered_force),
            right_filtered_force=float(self.right_filtered_force),
        )

    def _advance(
        self,
        compression: float,
        force: float,
        dt: float,
        desired_force: float | None = None,
    ) -> float:
        target_force = (
            self.config.desired_normal_force
            if desired_force is None
            else float(desired_force)
        )
        force_error = target_force - force
        rate = self.config.admittance_gain * force_error
        rate = float(
            np.clip(
                rate,
                -self.config.maximum_compression_rate,
                self.config.maximum_compression_rate,
            )
        )
        lower = -self.config.precontact_gap
        upper = self.config.max_compression
        return float(np.clip(compression + rate * float(dt), lower, upper))
