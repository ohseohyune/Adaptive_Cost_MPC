"""Quaternion prediction and SE(3) two-face targets for rotating boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.squeeze.ballistic import (
    BallisticBoxPredictor,
    BallisticPrediction,
    acceleration_limited_reach_distance,
)
from control.squeeze.config import RotatingSideSqueezeConfig


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_exp(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        return np.eye(3) + skew(vector)
    axis_hat = skew(vector / angle)
    return np.eye(3) + np.sin(angle) * axis_hat + (1.0 - np.cos(angle)) * (axis_hat @ axis_hat)


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
    )
    if angle < 1e-8:
        return 0.5 * vee
    return (0.5 * angle / np.sin(angle)) * vee


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    w, x, y, z = np.array([w, x, y, z]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array([(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale])
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array([(rotation[0, 2] - rotation[2, 0]) / scale, (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale])
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array([(rotation[1, 0] - rotation[0, 1]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0.0 else -quaternion


@dataclass(frozen=True)
class RotatingBoxPrediction:
    ballistic: BallisticPrediction
    quaternion: np.ndarray
    rotation: np.ndarray
    angular_velocity: np.ndarray
    angular_confidence: float

    def rotation_after(self, lead_s: float) -> np.ndarray:
        return rotation_exp(self.angular_velocity * max(0.0, float(lead_s))) @ self.rotation


@dataclass(frozen=True)
class SE3FaceInterceptionTarget:
    time_to_contact_s: float
    box_center: np.ndarray
    box_rotation: np.ndarray
    left_face_center: np.ndarray
    right_face_center: np.ndarray
    left_pad_transform: np.ndarray
    right_pad_transform: np.ndarray
    reachable: bool
    reachability_margin: float
    confidence: float


class QuaternionAngularVelocityPredictor:
    """Estimate world angular velocity from quaternion observations."""

    def __init__(self, config: RotatingSideSqueezeConfig) -> None:
        self.config = config
        self.ballistic = BallisticBoxPredictor(
            gravity=np.asarray(config.gravity),
            velocity_alpha=config.predictor_velocity_alpha,
            max_speed=config.predictor_max_speed,
        )
        self.reset()

    def reset(self, *, initial_velocity: np.ndarray | None = None) -> None:
        self.ballistic.reset(initial_velocity=initial_velocity)
        self._last_time: float | None = None
        self._last_rotation: np.ndarray | None = None
        self._angular_velocity = np.zeros(3)
        self._samples = 0

    def update(
        self,
        time_s: float,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> RotatingBoxPrediction:
        rotation = quaternion_to_rotation(quaternion)
        ballistic = self.ballistic.update(time_s, position)
        if self._last_time is not None and self._last_rotation is not None:
            dt = float(time_s) - self._last_time
            if dt > 1e-9:
                measured = rotation_log(rotation @ self._last_rotation.T) / dt
                if self._samples <= 1:
                    angular_velocity = measured
                else:
                    alpha = self.config.predictor_angular_velocity_alpha
                    angular_velocity = alpha * measured + (1.0 - alpha) * self._angular_velocity
                speed = float(np.linalg.norm(angular_velocity))
                maximum = self.config.maximum_predicted_angular_speed
                if speed > maximum:
                    angular_velocity *= maximum / speed
                self._angular_velocity = angular_velocity
                self._samples += 1
        else:
            self._samples = 1
        self._last_time = float(time_s)
        self._last_rotation = rotation.copy()
        confidence = float(np.clip((self._samples - 1) / 4.0, 0.0, 1.0))
        return RotatingBoxPrediction(
            ballistic=ballistic,
            quaternion=rotation_to_quaternion(rotation),
            rotation=rotation,
            angular_velocity=self._angular_velocity.copy(),
            angular_confidence=confidence,
        )


class SE3BoxFaceTargetPlanner:
    """Generate both EE transforms from predicted box position and rotation."""

    def __init__(
        self,
        config: RotatingSideSqueezeConfig,
        *,
        left_ee_to_pad_rotation: np.ndarray,
        right_ee_to_pad_rotation: np.ndarray,
    ) -> None:
        self.config = config
        self.left_relative = np.asarray(left_ee_to_pad_rotation).reshape(3, 3)
        self.right_relative = np.asarray(right_ee_to_pad_rotation).reshape(3, 3)

    def plan(
        self,
        prediction: RotatingBoxPrediction,
        *,
        left_pad_position: np.ndarray | None = None,
        right_pad_position: np.ndarray | None = None,
        target_ttc_s: float | None = None,
    ) -> SE3FaceInterceptionTarget:
        position = prediction.ballistic.position
        velocity = prediction.ballistic.velocity
        if abs(float(velocity[0])) < 1e-8:
            ttc = float("inf")
        else:
            root = (self.config.catch_plane_x - position[0]) / velocity[0]
            ttc = float(root) if root >= 0.0 else float("inf")
        evaluation_ttc = float(
            np.clip(
                ttc if np.isfinite(ttc) else 0.0,
                0.0,
                self.config.maximum_intercept_ttc,
            )
        )
        candidate_ttc = (
            float(
                np.clip(
                    target_ttc_s,
                    self.config.minimum_intercept_ttc,
                    self.config.maximum_intercept_ttc,
                )
            )
            if target_ttc_s is not None
            else self._select_candidate_ttc(
                prediction,
                plane_ttc=evaluation_ttc,
                left_pad_position=left_pad_position,
                right_pad_position=right_pad_position,
            )
        )
        center = prediction.ballistic.position_after(candidate_ttc)
        rotation = prediction.rotation_after(candidate_ttc)
        x_axis, y_axis, z_axis = rotation[:, 0], rotation[:, 1], rotation[:, 2]
        face_offset = self.config.box_half_y * y_axis
        pad_offset = (
            self.config.box_half_y + self.config.pad_half_thickness + self.config.precontact_gap
        ) * y_axis
        left_pad_rotation = np.column_stack([y_axis, -x_axis, z_axis])
        right_pad_rotation = np.column_stack([-y_axis, x_axis, z_axis])
        left_transform = np.eye(4)
        right_transform = np.eye(4)
        left_transform[:3, :3] = left_pad_rotation @ self.left_relative.T
        right_transform[:3, :3] = right_pad_rotation @ self.right_relative.T
        left_transform[:3, 3] = center + pad_offset
        right_transform[:3, 3] = center - pad_offset
        maneuver_time = max(0.0, candidate_ttc - self.config.pad_tracking_delay_s)
        available = acceleration_limited_reach_distance(
            maneuver_time,
            maximum_speed=self.config.maximum_pad_reach_speed,
            maximum_acceleration=self.config.maximum_pad_reach_acceleration,
        )
        if left_pad_position is None or right_pad_position is None:
            reachability_margin = float("inf")
        else:
            required = max(
                float(
                    np.linalg.norm(
                        left_transform[:3, 3]
                        - np.asarray(left_pad_position, dtype=float)
                    )
                ),
                float(
                    np.linalg.norm(
                        right_transform[:3, 3]
                        - np.asarray(right_pad_position, dtype=float)
                    )
                ),
            )
            reachability_margin = available - required
        reachable = bool(
            np.isfinite(ttc)
            and self.config.minimum_intercept_ttc
            <= candidate_ttc
            <= self.config.maximum_intercept_ttc
            and self.config.minimum_catch_z <= center[2] <= self.config.maximum_catch_z
            and reachability_margin >= self.config.minimum_reachability_margin
        )
        return SE3FaceInterceptionTarget(
            time_to_contact_s=float(candidate_ttc),
            box_center=center,
            box_rotation=rotation,
            left_face_center=center + face_offset,
            right_face_center=center - face_offset,
            left_pad_transform=left_transform,
            right_pad_transform=right_transform,
            reachable=reachable,
            reachability_margin=float(reachability_margin),
            confidence=min(prediction.ballistic.confidence, prediction.angular_confidence),
        )

    def _select_candidate_ttc(
        self,
        prediction: RotatingBoxPrediction,
        *,
        plane_ttc: float,
        left_pad_position: np.ndarray | None,
        right_pad_position: np.ndarray | None,
    ) -> float:
        """Select a future two-face target that both pads can physically reach."""

        if left_pad_position is None or right_pad_position is None:
            return plane_ttc
        plane_center = prediction.ballistic.position_after(plane_ttc)
        plane_rotation = prediction.rotation_after(plane_ttc)
        plane_offset = (
            self.config.box_half_y
            + self.config.pad_half_thickness
            + self.config.precontact_gap
        ) * plane_rotation[:, 1]
        plane_required = max(
            float(
                np.linalg.norm(
                    plane_center + plane_offset - left_pad_position
                )
            ),
            float(
                np.linalg.norm(
                    plane_center - plane_offset - right_pad_position
                )
            ),
        )
        plane_maneuver_time = max(0.0, plane_ttc - self.config.pad_tracking_delay_s)
        plane_available = acceleration_limited_reach_distance(
            plane_maneuver_time,
            maximum_speed=self.config.maximum_pad_reach_speed,
            maximum_acceleration=self.config.maximum_pad_reach_acceleration,
        )
        if (
            self.config.minimum_catch_z
            <= plane_center[2]
            <= self.config.maximum_catch_z
            and plane_available - plane_required
            >= self.config.minimum_reachability_margin
        ):
            return plane_ttc
        count = max(2, int(self.config.interception_candidate_count))
        lower = max(self.config.minimum_intercept_ttc, plane_ttc - 0.16)
        upper = min(self.config.maximum_intercept_ttc, plane_ttc + 0.16)
        candidates = np.linspace(lower, max(lower, upper), count)
        best_time = plane_ttc
        best_score = -float("inf")
        for candidate in candidates:
            center = prediction.ballistic.position_after(float(candidate))
            rotation = prediction.rotation_after(float(candidate))
            y_axis = rotation[:, 1]
            offset = (
                self.config.box_half_y
                + self.config.pad_half_thickness
                + self.config.precontact_gap
            )
            left_target = center + offset * y_axis
            right_target = center - offset * y_axis
            required = max(
                float(np.linalg.norm(left_target - left_pad_position)),
                float(np.linalg.norm(right_target - right_pad_position)),
            )
            candidate_maneuver_time = max(
                0.0, float(candidate) - self.config.pad_tracking_delay_s
            )
            available = acceleration_limited_reach_distance(
                candidate_maneuver_time,
                maximum_speed=self.config.maximum_pad_reach_speed,
                maximum_acceleration=self.config.maximum_pad_reach_acceleration,
            )
            margin = available - required
            height_penalty = abs(center[2] - self.config.preferred_catch_height)
            plane_penalty = abs(center[0] - self.config.catch_plane_x)
            invalid_height = not (
                self.config.minimum_catch_z
                <= center[2]
                <= self.config.maximum_catch_z
            )
            score = margin - 0.30 * height_penalty - 0.15 * plane_penalty
            if invalid_height:
                score -= 10.0
            if score > best_score:
                best_score = score
                best_time = float(candidate)
        return best_time
