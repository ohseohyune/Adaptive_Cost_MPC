"""First-contact impact monitoring and compliant command relief."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.squeeze.config import DynamicSideSqueezeConfig
from control.squeeze.hybrid_controller import SqueezeCommand
from control.squeeze.pad_contact import BilateralPadContact


@dataclass(frozen=True)
class ImpactState:
    first_contact_time_s: float | None
    in_first_contact_window: bool
    peak_first_contact_force: float
    force_limit_exceeded: bool
    emergency: bool
    predicted_peak_force: float
    force_guard_active: bool


@dataclass(frozen=True)
class AdaptiveImpactCommand:
    tangential_stiffness: float
    normal_stiffness: float
    rotational_stiffness: float
    desired_force: float
    relative_normal_speed: float


def adaptive_impact_command(
    config: DynamicSideSqueezeConfig,
    *,
    object_mass: float,
    contact_face_area: float,
    relative_normal_speed: float,
) -> AdaptiveImpactCommand:
    """Schedule compliant impact gains from kinetic severity and face area."""

    nominal_area = 0.11 * 0.11
    area_ratio = max(0.25, float(contact_face_area) / nominal_area)
    speed_ratio = max(0.0, float(relative_normal_speed)) / max(
        config.relative_normal_speed_limit, 1e-6
    )
    severity = (max(float(object_mass), 1e-6) / 0.50) * speed_ratio**2 / area_ratio
    tangential = np.clip(
        config.impact_tangential_stiffness / (1.0 + 0.60 * severity),
        config.minimum_adaptive_impact_stiffness,
        config.maximum_adaptive_impact_stiffness,
    )
    normal = np.clip(
        config.impact_normal_stiffness / (1.0 + 0.80 * severity),
        config.minimum_adaptive_impact_stiffness,
        config.maximum_adaptive_impact_stiffness,
    )
    rotational = np.clip(
        config.impact_rotational_stiffness / (1.0 + 0.60 * severity),
        config.minimum_adaptive_impact_rotational_stiffness,
        config.maximum_adaptive_impact_rotational_stiffness,
    )
    desired = np.clip(
        config.impact_desired_force / (1.0 + 0.50 * severity),
        config.minimum_contact_force,
        config.maximum_adaptive_impact_force,
    )
    return AdaptiveImpactCommand(
        tangential_stiffness=float(tangential),
        normal_stiffness=float(normal),
        rotational_stiffness=float(rotational),
        desired_force=float(desired),
        relative_normal_speed=float(relative_normal_speed),
    )


class FirstContactForceLimiter:
    """Monitor initial impact and move overloaded pad targets outward."""

    def __init__(self, config: DynamicSideSqueezeConfig) -> None:
        self.config = config
        self.first_contact_time_s: float | None = None
        self.peak_first_contact_force = 0.0
        self._previous_time_s: float | None = None
        self._previous_forces = np.zeros(2)
        self._force_rates = np.zeros(2)

    def update(self, time_s: float, contact: BilateralPadContact) -> ImpactState:
        forces = np.array(
            [contact.left.normal_force, contact.right.normal_force], dtype=float
        )
        if self._previous_time_s is not None:
            dt = float(time_s) - self._previous_time_s
            if dt > 1e-9:
                measured_rate = (forces - self._previous_forces) / dt
                self._force_rates = 0.35 * measured_rate + 0.65 * self._force_rates
        self._previous_time_s = float(time_s)
        self._previous_forces = forces
        force = max(contact.left.normal_force, contact.right.normal_force)
        if self.first_contact_time_s is None and (contact.left.active or contact.right.active):
            self.first_contact_time_s = float(time_s)
        in_window = bool(
            self.first_contact_time_s is not None
            and time_s - self.first_contact_time_s <= self.config.first_contact_window_s
        )
        if in_window:
            self.peak_first_contact_force = max(self.peak_first_contact_force, force)
        predicted_forces = forces + np.maximum(self._force_rates, 0.0) * (
            self.config.force_prediction_horizon_s
        )
        predicted_peak = float(np.max(predicted_forces))
        guard_force = (
            self.config.predictive_force_guard_ratio
            * self.config.first_contact_force_limit
        )
        return ImpactState(
            first_contact_time_s=self.first_contact_time_s,
            in_first_contact_window=in_window,
            peak_first_contact_force=float(self.peak_first_contact_force),
            force_limit_exceeded=self.peak_first_contact_force > self.config.first_contact_force_limit,
            emergency=force > self.config.emergency_contact_force,
            predicted_peak_force=predicted_peak,
            force_guard_active=bool(in_window and predicted_peak > guard_force),
        )

    def relieve(
        self,
        command: SqueezeCommand,
        contact: BilateralPadContact,
        state: ImpactState,
    ) -> tuple[np.ndarray, np.ndarray]:
        left = command.left_position.copy()
        right = command.right_position.copy()
        if not state.in_first_contact_window:
            return left, right
        left_relief = self._relief(contact.left.normal_force, self._force_rates[0])
        right_relief = self._relief(contact.right.normal_force, self._force_rates[1])
        # left inward is -y, right inward is +y; relief is outward.
        left[1] += left_relief
        right[1] -= right_relief
        return left, right

    def relief_distances(
        self, contact: BilateralPadContact, state: ImpactState
    ) -> tuple[float, float]:
        if not state.in_first_contact_window:
            return 0.0, 0.0
        return (
            self._relief(contact.left.normal_force, self._force_rates[0]),
            self._relief(contact.right.normal_force, self._force_rates[1]),
        )

    def _relief(self, force: float, force_rate: float = 0.0) -> float:
        predicted_force = float(force) + max(0.0, float(force_rate)) * (
            self.config.force_prediction_horizon_s
        )
        guard_force = (
            self.config.predictive_force_guard_ratio
            * self.config.first_contact_force_limit
        )
        overload = max(0.0, predicted_force - guard_force)
        return float(
            np.clip(
                self.config.impact_relief_gain * overload,
                0.0,
                self.config.maximum_impact_relief,
            )
        )
