"""Moving-target state estimation and short-horizon prediction helpers."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# 한 순간의 target 예측 결과를 담는 snapshot
@dataclass(frozen=True)
class MovingTargetPrediction:
    """One moving-target prediction snapshot."""

    time_s: float # prediction을 계산한 시간 => trajectory가 시작된 뒤 몇 초 지났는가
    position: np.ndarray # 현재 관측된 target 위치 => 현재는 실제 센서값이 아니라 analytic target
    velocity: np.ndarray # 현재 target의 추정 속도 => position의 finite difference + LPF(smoothing)
    acceleration: np.ndarray # target의 추정 가속도 => velocity의 finite difference + LPF(smoothing)
    predicted_position: np.ndarray # 미래 target 위치 예측값
    # current model 
    # predicted_position = (
    # position + velocity * horizon_s + 0.5 * acceleration * horizon_s**2 )
    horizon_s: float # 몇 초 뒤를 예측할지 [s]
    confidence: float # prediction 신뢰도 (0.0 ~ 1.0)
    dt_s: float # 이전 target sample과 현재 sample 사이의 시간 간격 (dt = time_s - last_time_s) => main loop timestep이 약 0.002s라서 0.002 근처로 나와야함. 
    speed_was_clipped: bool # 추정 velocity 또는 predicted displacement가 safety limit을 넘어서 clamp되었는지


class MovingTargetPredictor:
    """Estimate target velocity/acceleration and predict a future position."""

    def __init__(
        self,
        *,
        horizon_s: float,
        smoothing_alpha: float,
        max_prediction_speed: float | None = None,
    ) -> None:
        self.horizon_s = float(horizon_s)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.max_prediction_speed = (
            None if max_prediction_speed is None else float(max_prediction_speed)
        )
        if self.horizon_s < 0.0:
            raise ValueError("prediction horizon must be non-negative.")
        if self.max_prediction_speed is not None and self.max_prediction_speed <= 0.0:
            raise ValueError("max prediction speed must be positive when set.")
        self.reset()

    def reset(self) -> None:
        """Clear the estimator history."""

        self._last_time_s: float | None = None
        self._last_position: np.ndarray | None = None
        self._velocity = np.zeros(3)
        self._acceleration = np.zeros(3)
        self._has_velocity = False

    def update(self, time_s: float, position: np.ndarray) -> MovingTargetPrediction:
        """Update the target state estimate and return a future prediction."""

        time_s = float(time_s)
        position = np.asarray(position, dtype=float).reshape(3)
        if self._last_time_s is None or self._last_position is None:
            dt = 0.0
            velocity = self._velocity.copy()
            acceleration = self._acceleration.copy()
            confidence = 0.0
            speed_was_clipped = False
        else:
            dt = time_s - self._last_time_s
            if dt <= 1e-9:
                velocity = self._velocity.copy()
                acceleration = self._acceleration.copy()
                confidence = 0.0
                speed_was_clipped = False
            else:
                raw_velocity = (position - self._last_position) / dt
                alpha = self.smoothing_alpha
                previous_velocity = self._velocity.copy()
                if self._has_velocity:
                    velocity = alpha * raw_velocity + (1.0 - alpha) * self._velocity
                    velocity, speed_was_clipped = self._limit_speed(velocity)
                    raw_acceleration = (velocity - previous_velocity) / dt
                    acceleration = alpha * raw_acceleration + (
                        1.0 - alpha
                    ) * self._acceleration
                    confidence = 0.7 if speed_was_clipped else 1.0
                else:
                    velocity, speed_was_clipped = self._limit_speed(raw_velocity)
                    acceleration = self._acceleration.copy()
                    self._has_velocity = True
                    confidence = 0.5 if speed_was_clipped else 0.8

        predicted_position = (
            position
            + velocity * self.horizon_s
            + 0.5 * acceleration * (self.horizon_s**2)
        )
        predicted_position, displacement_was_clipped = self._limit_displacement(
            position,
            predicted_position,
        )
        if displacement_was_clipped:
            speed_was_clipped = True
            confidence = min(confidence, 0.7)

        self._last_time_s = time_s
        self._last_position = position.copy()
        self._velocity = velocity.copy()
        self._acceleration = acceleration.copy()

        return MovingTargetPrediction(
            time_s=time_s,
            position=position.copy(),
            velocity=velocity.copy(),
            acceleration=acceleration.copy(),
            predicted_position=predicted_position.copy(),
            horizon_s=self.horizon_s,
            confidence=float(confidence),
            dt_s=float(dt),
            speed_was_clipped=bool(speed_was_clipped),
        )

    def _limit_speed(self, velocity: np.ndarray) -> tuple[np.ndarray, bool]:
        if self.max_prediction_speed is None:
            return velocity, False
        speed = float(np.linalg.norm(velocity))
        if speed <= self.max_prediction_speed:
            return velocity, False
        return velocity * (self.max_prediction_speed / speed), True

    def _limit_displacement(
        self,
        position: np.ndarray,
        predicted_position: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        if self.max_prediction_speed is None or self.horizon_s <= 0.0:
            return predicted_position, False
        displacement = predicted_position - position
        distance = float(np.linalg.norm(displacement))
        max_distance = self.max_prediction_speed * self.horizon_s
        if distance <= max_distance:
            return predicted_position, False
        return position + displacement * (max_distance / distance), True
