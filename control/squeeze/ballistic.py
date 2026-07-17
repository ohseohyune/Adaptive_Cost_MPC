"""Ballistic state estimation and two-face interception for moving boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.squeeze.config import DynamicSideSqueezeConfig


def acceleration_limited_reach_distance(
    time_s: float,
    *,
    maximum_speed: float,
    maximum_acceleration: float,
) -> float:
    """Conservative distance available to a pad starting from rest."""

    time_s = max(0.0, float(time_s))
    maximum_speed = max(0.0, float(maximum_speed))
    maximum_acceleration = max(1e-9, float(maximum_acceleration))
    acceleration_time = maximum_speed / maximum_acceleration
    if time_s <= acceleration_time:
        return 0.5 * maximum_acceleration * time_s**2
    return maximum_speed * time_s - 0.5 * maximum_speed**2 / maximum_acceleration


def resolve_ballistic_launch_velocity(
    config: DynamicSideSqueezeConfig,
    sampled_velocity: np.ndarray,
    *,
    launch_position: np.ndarray | None = None,
) -> np.ndarray:
    """Return a launch velocity whose arc reaches the configured catch point.

    Horizontal components remain randomized. In long-flight mode the vertical
    component is solved analytically from the ballistic position equation,
    avoiding an independent vz sample that would miss the robot after TTC is
    changed.
    """

    velocity = np.asarray(sampled_velocity, dtype=float).reshape(3).copy()
    if not config.derive_vertical_launch_velocity:
        return velocity
    position = np.asarray(
        config.launch_position if launch_position is None else launch_position,
        dtype=float,
    )
    gravity = np.asarray(config.gravity, dtype=float)
    if abs(float(velocity[0])) < 1e-8:
        raise ValueError("horizontal launch velocity must be nonzero")
    flight_time = (config.catch_plane_x - position[0]) / velocity[0]
    if flight_time <= 0.0:
        raise ValueError("launch velocity must point toward the catch plane")
    velocity[2] = (
        config.target_catch_height
        - position[2]
        - 0.5 * gravity[2] * flight_time**2
    ) / flight_time
    return velocity


def resolve_ballistic_launch_position(
    config: DynamicSideSqueezeConfig,
    sampled_velocity: np.ndarray,
) -> np.ndarray:
    """Place the launcher so randomized vx retains a fixed initial TTC."""

    position = np.asarray(config.launch_position, dtype=float).copy()
    if config.target_flight_time_s is None:
        return position
    flight_time = float(config.target_flight_time_s)
    if flight_time <= 0.0:
        raise ValueError("target_flight_time_s must be positive")
    velocity = np.asarray(sampled_velocity, dtype=float).reshape(3)
    if velocity[0] >= 0.0:
        raise ValueError("launch x velocity must point toward decreasing x")
    position[0] = config.catch_plane_x - velocity[0] * flight_time
    return position


@dataclass(frozen=True)
class BallisticPrediction:
    time_s: float
    position: np.ndarray
    velocity: np.ndarray
    gravity: np.ndarray
    confidence: float

    def position_after(self, lead_s: float) -> np.ndarray:
        lead_s = max(0.0, float(lead_s))
        return self.position + self.velocity * lead_s + 0.5 * self.gravity * lead_s**2


@dataclass(frozen=True)
class FaceInterceptionPlan:
    time_to_contact_s: float
    box_center: np.ndarray
    left_face_center: np.ndarray
    right_face_center: np.ndarray
    left_pad_target: np.ndarray
    right_pad_target: np.ndarray
    reachable: bool
    reachability_margin: float
    confidence: float


class BallisticBoxPredictor:
    """Estimate instantaneous velocity from positions with known gravity."""

    def __init__(
        self,
        *,
        gravity: np.ndarray,
        velocity_alpha: float = 0.35,
        max_speed: float | None = None,
    ) -> None:
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        self.velocity_alpha = float(np.clip(velocity_alpha, 0.0, 1.0))
        self.max_speed = None if max_speed is None else float(max_speed)
        self.reset()

    def reset(self, *, initial_velocity: np.ndarray | None = None) -> None:
        self._last_time: float | None = None
        self._last_position: np.ndarray | None = None
        self._velocity = (
            np.zeros(3)
            if initial_velocity is None
            else self._clip_speed(np.asarray(initial_velocity, dtype=float).reshape(3))
        )
        self._samples = 0

    def update(self, time_s: float, position: np.ndarray) -> BallisticPrediction:
        time_s = float(time_s)
        position = np.asarray(position, dtype=float).reshape(3)
        if self._last_time is not None and self._last_position is not None:
            dt = time_s - self._last_time
            if dt > 1e-9:
                # Finite difference gives interval-average velocity.  Advancing
                # by half a gravity step estimates velocity at the new sample.
                measured = (position - self._last_position) / dt + 0.5 * self.gravity * dt
                if self._samples <= 1:
                    velocity = measured
                else:
                    alpha = self.velocity_alpha
                    # Compare the measurement with the previous estimate
                    # propagated through known gravity.  Blending against the
                    # stale previous velocity biases vz upward during flight
                    # and makes a locked interception target drift vertically.
                    propagated = self._velocity + self.gravity * dt
                    velocity = alpha * measured + (1.0 - alpha) * propagated
                self._velocity = self._clip_speed(velocity)
                self._samples += 1
        else:
            self._samples = 1

        self._last_time = time_s
        self._last_position = position.copy()
        confidence = float(np.clip((self._samples - 1) / 4.0, 0.0, 1.0))
        return BallisticPrediction(
            time_s=time_s,
            position=position.copy(),
            velocity=self._velocity.copy(),
            gravity=self.gravity.copy(),
            confidence=confidence,
        )

    def _clip_speed(self, velocity: np.ndarray) -> np.ndarray:
        if self.max_speed is None:
            return velocity
        speed = float(np.linalg.norm(velocity))
        if speed <= self.max_speed or speed <= 1e-12:
            return velocity
        return velocity * (self.max_speed / speed)


class BoxFaceInterceptionPlanner:
    """Predict the box crossing of a catch plane and generate both pad targets."""

    def __init__(self, config: DynamicSideSqueezeConfig) -> None:
        self.config = config

    def plan(
        self,
        prediction: BallisticPrediction,
        *,
        left_pad_position: np.ndarray,
        right_pad_position: np.ndarray,
    ) -> FaceInterceptionPlan:
        ttc = self._plane_crossing_ttc(prediction)
        finite = np.isfinite(ttc)
        evaluation_ttc = float(np.clip(ttc if finite else 0.0, 0.0, self.config.maximum_intercept_ttc))
        center = prediction.position_after(evaluation_ttc)
        face_offset = np.array([0.0, self.config.box_half_y, 0.0])
        pad_offset = np.array(
            [
                0.0,
                self.config.box_half_y
                + self.config.pad_half_thickness
                + self.config.precontact_gap,
                0.0,
            ]
        )
        left_face = center + face_offset
        right_face = center - face_offset
        left_target = center + pad_offset
        right_target = center - pad_offset

        maneuver_time = max(0.0, evaluation_ttc - self.config.pad_tracking_delay_s)
        available = acceleration_limited_reach_distance(
            maneuver_time,
            maximum_speed=self.config.maximum_pad_reach_speed,
            maximum_acceleration=self.config.maximum_pad_reach_acceleration,
        )
        required = max(
            float(np.linalg.norm(left_target - np.asarray(left_pad_position, dtype=float))),
            float(np.linalg.norm(right_target - np.asarray(right_pad_position, dtype=float))),
        )
        margin = available - required
        reachable = bool(
            finite
            and self.config.minimum_intercept_ttc <= ttc <= self.config.maximum_intercept_ttc
            and self.config.minimum_catch_z <= center[2] <= self.config.maximum_catch_z
            and margin >= self.config.minimum_reachability_margin
        )
        return FaceInterceptionPlan(
            time_to_contact_s=float(ttc),
            box_center=center,
            left_face_center=left_face,
            right_face_center=right_face,
            left_pad_target=left_target,
            right_pad_target=right_target,
            reachable=reachable,
            reachability_margin=float(margin),
            confidence=prediction.confidence if reachable else 0.5 * prediction.confidence,
        )

    def _plane_crossing_ttc(self, prediction: BallisticPrediction) -> float:
        x0 = float(prediction.position[0])
        vx = float(prediction.velocity[0])
        gx = float(prediction.gravity[0])
        c = x0 - self.config.catch_plane_x
        if abs(gx) < 1e-10:
            if abs(vx) < 1e-8:
                return float("inf")
            root = -c / vx
            return float(root) if root >= 0.0 else float("inf")
        roots = np.roots([0.5 * gx, vx, c])
        valid = [float(root.real) for root in roots if abs(root.imag) < 1e-8 and root.real >= 0.0]
        return min(valid) if valid else float("inf")
