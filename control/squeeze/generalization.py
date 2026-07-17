"""Curriculum and physical domain randomization for box catching.

The randomizer changes the compiled MuJoCo model in memory.  Each episode
loads a fresh model, so randomized mass, inertia, dimensions, and contact
friction cannot leak into the next rollout.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

import mujoco
import numpy as np


NOMINAL_BOX_HALF_SIZE = np.array([0.055, 0.150, 0.055], dtype=float)
NOMINAL_BOX_MASS = 0.50
NOMINAL_BOX_FRICTION = 1.20


def _classify_shape(aspect_scale: np.ndarray) -> str:
    """Give the sampled cuboid a readable label for logs and evaluation."""

    aspect = np.asarray(aspect_scale, dtype=float)
    deviation = aspect - 1.0
    axis = int(np.argmax(np.abs(deviation)))
    if abs(deviation[axis]) < 0.04:
        return "balanced"
    if axis == 0:
        return "deep" if deviation[axis] > 0.0 else "shallow"
    if axis == 1:
        return "wide_grip" if deviation[axis] > 0.0 else "narrow_grip"
    return "tall" if deviation[axis] > 0.0 else "flat"


@dataclass(frozen=True)
class BoxDomainParameters:
    """Physical and launch parameters sampled for one episode."""

    half_size: tuple[float, float, float]
    overall_size_scale: float
    aspect_scale: tuple[float, float, float]
    shape_family: str
    mass: float
    friction: float
    launch_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    stage_name: str
    stage_index: int

    def validate(self) -> None:
        if min(self.half_size) <= 0.0:
            raise ValueError("box half sizes must be positive")
        if self.overall_size_scale <= 0.0 or min(self.aspect_scale) <= 0.0:
            raise ValueError("box size and aspect scales must be positive")
        if not self.shape_family:
            raise ValueError("box shape family must be named")
        if self.mass <= 0.0:
            raise ValueError("box mass must be positive")
        if self.friction <= 0.0:
            raise ValueError("box friction must be positive")


@dataclass(frozen=True)
class CurriculumStage:
    """Size, aspect-ratio, physics, and launch distributions for one level."""

    name: str
    axis_scale_low: tuple[float, float, float]
    axis_scale_high: tuple[float, float, float]
    mass_range: tuple[float, float]
    friction_range: tuple[float, float]
    launch_velocity_low: tuple[float, float, float]
    launch_velocity_high: tuple[float, float, float]
    angular_velocity_low: tuple[float, float, float]
    angular_velocity_high: tuple[float, float, float]

    def sample(self, rng: np.random.Generator, stage_index: int) -> BoxDomainParameters:
        axis_scale = rng.uniform(self.axis_scale_low, self.axis_scale_high)
        # Decompose the three sampled dimensions into a common size component
        # and a unit-volume aspect component. This keeps geometry randomization
        # from consuming an extra RNG sample and changing the launch sequence.
        overall_size_scale = float(np.cbrt(np.prod(axis_scale)))
        aspect_scale = axis_scale / overall_size_scale
        half_size = NOMINAL_BOX_HALF_SIZE * axis_scale
        result = BoxDomainParameters(
            half_size=tuple(float(value) for value in half_size),
            overall_size_scale=overall_size_scale,
            aspect_scale=tuple(float(value) for value in aspect_scale),
            shape_family=_classify_shape(aspect_scale),
            mass=float(rng.uniform(*self.mass_range)),
            friction=float(rng.uniform(*self.friction_range)),
            launch_velocity=tuple(
                float(value)
                for value in rng.uniform(
                    self.launch_velocity_low, self.launch_velocity_high
                )
            ),
            angular_velocity=tuple(
                float(value)
                for value in rng.uniform(
                    self.angular_velocity_low, self.angular_velocity_high
                )
            ),
            stage_name=self.name,
            stage_index=int(stage_index),
        )
        result.validate()
        return result


def default_curriculum() -> tuple[CurriculumStage, ...]:
    """Return validated warmup/intermediate and wider stress distributions."""

    return (
        CurriculumStage(
            name="warmup",
            axis_scale_low=(0.90, 0.97, 0.90),
            axis_scale_high=(1.10, 1.03, 1.10),
            mass_range=(0.48, 0.52),
            friction_range=(1.15, 1.25),
            launch_velocity_low=(-1.47, -0.010, 0.0),
            launch_velocity_high=(-1.35, 0.010, 0.0),
            angular_velocity_low=(-0.005, -0.005, -0.025),
            angular_velocity_high=(0.005, 0.005, 0.025),
        ),
        CurriculumStage(
            name="intermediate",
            axis_scale_low=(0.78, 0.91, 0.78),
            axis_scale_high=(1.22, 1.09, 1.22),
            mass_range=(0.42, 0.58),
            friction_range=(1.00, 1.40),
            launch_velocity_low=(-1.53, -0.020, 0.0),
            launch_velocity_high=(-1.29, 0.020, 0.0),
            angular_velocity_low=(-0.0075, -0.0075, -0.0325),
            angular_velocity_high=(0.0075, 0.0075, 0.0325),
        ),
        CurriculumStage(
            name="full",
            axis_scale_low=(0.65, 0.82, 0.65),
            axis_scale_high=(1.35, 1.18, 1.35),
            mass_range=(0.35, 0.70),
            friction_range=(0.75, 1.55),
            launch_velocity_low=(-1.47, -0.010, 0.0),
            launch_velocity_high=(-1.35, 0.010, 0.0),
            angular_velocity_low=(-0.005, -0.005, -0.025),
            angular_velocity_high=(0.005, 0.005, 0.025),
        ),
    )


class CurriculumScheduler:
    """Promote on sustained success and demote quickly after unsafe trials."""

    def __init__(
        self,
        stages: Optional[Iterable[CurriculumStage]] = None,
        *,
        window: int = 6,
        promote_success_rate: float = 0.80,
        demote_success_rate: float = 0.35,
        minimum_episodes_per_stage: int = 4,
    ) -> None:
        self.stages = tuple(default_curriculum() if stages is None else stages)
        if not self.stages:
            raise ValueError("at least one curriculum stage is required")
        if window <= 0 or minimum_episodes_per_stage <= 0:
            raise ValueError("curriculum window sizes must be positive")
        self.window = int(window)
        self.promote_success_rate = float(promote_success_rate)
        self.demote_success_rate = float(demote_success_rate)
        self.minimum_episodes_per_stage = int(minimum_episodes_per_stage)
        self.stage_index = 0
        self.episodes_at_stage = 0
        self._outcomes: deque[bool] = deque(maxlen=self.window)

    @property
    def stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def success_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        return float(np.mean(self._outcomes))

    def sample(self, rng: np.random.Generator) -> BoxDomainParameters:
        return self.stage.sample(rng, self.stage_index)

    def record(self, success: bool, *, safety_violation: bool = False) -> int:
        self.episodes_at_stage += 1
        self._outcomes.append(bool(success and not safety_violation))
        enough = self.episodes_at_stage >= self.minimum_episodes_per_stage
        if not enough:
            return self.stage_index

        rate = self.success_rate
        previous = self.stage_index
        if (
            not safety_violation
            and rate >= self.promote_success_rate
            and self.stage_index < len(self.stages) - 1
        ):
            self.stage_index += 1
        elif (
            (safety_violation or rate <= self.demote_success_rate)
            and self.stage_index > 0
        ):
            self.stage_index -= 1

        if self.stage_index != previous:
            self.episodes_at_stage = 0
            self._outcomes.clear()
        return self.stage_index


def apply_box_domain_randomization(
    model: mujoco.MjModel,
    parameters: BoxDomainParameters,
) -> None:
    """Apply one sampled box domain to a freshly loaded MuJoCo model."""

    parameters.validate()
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom"
    )
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_box")
    if geom_id < 0 or body_id < 0:
        raise ValueError("dynamic box geom/body is missing from the MuJoCo model")

    half_size = np.asarray(parameters.half_size, dtype=float)
    model.geom_size[geom_id, :3] = half_size
    model.geom_rbound[geom_id] = float(np.linalg.norm(half_size))
    model.geom_friction[geom_id, 0] = parameters.friction
    model.body_mass[body_id] = parameters.mass
    hx, hy, hz = half_size
    model.body_inertia[body_id] = parameters.mass / 3.0 * np.array(
        [hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy]
    )

    # Explicit contact pairs override per-geom sliding friction in MuJoCo.
    for name in ("dynamic_left_pad_box", "dynamic_right_pad_box"):
        pair_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, name)
        if pair_id >= 0:
            model.pair_friction[pair_id, :2] = parameters.friction

    for name, sign in (
        ("dynamic_box_left_face", 1.0),
        ("dynamic_box_right_face", -1.0),
    ):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            model.site_pos[site_id, 1] = sign * hy

    # mj_setConst recomputes mass/inertia-dependent constant fields (e.g.
    # body_invweight0, dof_M0) from the body_mass/body_inertia just written.
    # Without it those fields stay at their compile-time values for the
    # nominal 0.50 kg box, and free-fall acceleration comes out scaled by
    # (randomized_mass / 0.50) instead of matching configured gravity --
    # heavier boxes fall much faster than predicted, lighter ones much
    # slower, silently breaking every ballistic prediction and interception
    # plan for any non-nominal mass.
    mujoco.mj_setConst(model, mujoco.MjData(model))
