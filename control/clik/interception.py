"""Interception planning helpers for moving bimanual targets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterceptionPlan:
    """One interception-planning diagnostic snapshot."""

    time_s: float
    current_object_position: np.ndarray
    current_target_position: np.ndarray
    interception_position: np.ndarray
    lead_time_s: float
    time_to_contact_s: float
    distance_to_intercept: float
    reachable_distance: float
    reachability_margin: float
    reachable: bool
    confidence: float


class InterceptionPlanner:
    """Choose a future target point that the grasp center can plausibly reach."""

    def __init__(
        self,
        *,
        min_lead_s: float,
        max_lead_s: float,
        lead_samples: int,
        reach_speed: float,
    ) -> None:
        self.min_lead_s = float(min_lead_s)
        self.max_lead_s = float(max_lead_s)
        self.lead_samples = int(lead_samples)
        self.reach_speed = float(reach_speed)
        if self.min_lead_s < 0.0:
            raise ValueError("min interception lead time must be non-negative.")
        if self.max_lead_s < self.min_lead_s:
            raise ValueError("max interception lead time must be >= min lead time.")
        if self.lead_samples <= 0:
            raise ValueError("interception lead_samples must be positive.")
        if self.reach_speed <= 0.0:
            raise ValueError("interception reach_speed must be positive.")

    def plan(
        self,
        *,
        time_s: float,
        current_object_position: np.ndarray,
        current_target_position: np.ndarray,
        target_position_at_time: Callable[[float], np.ndarray],
    ) -> InterceptionPlan:
        """Return the first reachable future target point, or the least-unreachable one."""

        time_s = float(time_s)
        current_object_position = np.asarray(
            current_object_position, dtype=float
        ).reshape(3)
        current_target_position = np.asarray(
            current_target_position, dtype=float
        ).reshape(3)
        lead_times = np.linspace(
            self.min_lead_s,
            self.max_lead_s,
            self.lead_samples,
        )

        candidates: list[tuple[float, float, float, float, np.ndarray]] = []
        for lead_time_s in lead_times:
            future_position = np.asarray(
                target_position_at_time(time_s + float(lead_time_s)),
                dtype=float,
            ).reshape(3)
            distance = float(np.linalg.norm(future_position - current_object_position))
            reachable_distance = self.reach_speed * float(lead_time_s)
            margin = reachable_distance - distance
            candidates.append(
                (
                    margin,
                    float(lead_time_s),
                    distance,
                    reachable_distance,
                    future_position,
                )
            )

        reachable_candidates = [candidate for candidate in candidates if candidate[0] >= 0.0]
        if reachable_candidates:
            margin, lead_time_s, distance, reachable_distance, position = min(
                reachable_candidates,
                key=lambda candidate: candidate[1],
            )
            reachable = True
            confidence = 1.0
        else:
            margin, lead_time_s, distance, reachable_distance, position = max(
                candidates,
                key=lambda candidate: candidate[0],
            )
            reachable = False
            confidence = 0.5

        return InterceptionPlan(
            time_s=time_s,
            current_object_position=current_object_position.copy(),
            current_target_position=current_target_position.copy(),
            interception_position=position.copy(),
            lead_time_s=float(lead_time_s),
            time_to_contact_s=float(lead_time_s),
            distance_to_intercept=float(distance),
            reachable_distance=float(reachable_distance),
            reachability_margin=float(margin),
            reachable=bool(reachable),
            confidence=float(confidence),
        )
