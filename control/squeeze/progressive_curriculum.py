"""Performance-adaptive progressive catch curriculum.

Extends (does not replace) `control/squeeze/generalization.py`'s legacy
3-stage `CurriculumScheduler`/`default_curriculum()`, which stay completely
untouched and still drive `curriculum_mode="adaptive"`/`"balanced"`. This
module adds:

- `BoxCatchDifficulty` / `compute_box_catch_difficulty`: a task-aligned
  difficulty descriptor built from the *resolved* ballistic quantities
  (post `resolve_ballistic_launch_position/velocity`), not raw sampled
  ranges -- reuses `control.squeeze.ballistic.plane_crossing_ttc` rather
  than re-deriving TTC.
- `sample_stage_domain`: resample-with-recheck against a stage's
  `difficulty_range`, with infeasible domains caught before any episode
  ever runs (never pollutes the scheduler's success-rate denominator).
- `progressive_catch_curriculum()`: 6 new `CurriculumStage` instances
  (reusing the existing dataclass, no new stage type needed).
- `AdaptiveCurriculumScheduler`: funnel-based promotion, anchor-vs-mixture
  statistic separation, ordinary-vs-emergency safety split, transition
  cooldown, per-stage lifetime stats, full checkpoint round-trip.

Deliberately narrower than a full reachability analysis: `FaceInterceptionPlan
.reachability_margin` needs live end-effector positions that don't exist yet
at domain-sampling time (before the episode/model has even loaded), so this
module only uses quantities fully determined by a `BoxDomainParameters` +
`DynamicSideSqueezeConfig` alone -- TTC and impact energy, not reachability.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from control.squeeze.ballistic import (
    BallisticPrediction,
    plane_crossing_ttc,
    resolve_ballistic_launch_position,
    resolve_ballistic_launch_velocity,
)
from control.squeeze.config import DynamicSideSqueezeConfig
from control.squeeze.generalization import BoxDomainParameters, CurriculumStage

# ---------------------------------------------------------------------------
# Difficulty descriptor
# ---------------------------------------------------------------------------

# First-pass reference constants, not empirically tuned -- see the
# "known limitations" note in the implementation report. Chosen so a
# nominal 0.5kg box at a moderate ~2-3 m/s impact lands near a difficulty
# score of ~1.0, with the 6 stage `difficulty_range`s below spread around it.
_TTC_REFERENCE_S = 0.4
_ENERGY_REFERENCE_J = 2.0
_GRAVITY_MPS2 = 9.81
_REFERENCE_NORMAL_FORCE_N = 6.0


@dataclass(frozen=True)
class BoxCatchDifficulty:
    """Task-aligned difficulty for one resolved (post-ballistic) domain sample."""

    ttc_s: float
    impact_speed_mps: float
    impact_energy_j: float
    hold_difficulty: float
    total_difficulty: float


def _hold_difficulty(domain: BoxDomainParameters) -> float:
    """Proportional to the grip force needed to hold this box against gravity.

    hold_difficulty ~ mass * g / (2 * friction * reference_normal_force):
    heavier and/or lower-friction boxes need more grip force per pad to avoid
    slipping, which is exactly what makes HOLD harder independent of TTC.
    """

    return float(
        domain.mass * _GRAVITY_MPS2 / (2.0 * domain.friction * _REFERENCE_NORMAL_FORCE_N)
    )


def compute_box_catch_difficulty(
    domain: BoxDomainParameters,
    squeeze_config: DynamicSideSqueezeConfig,
) -> BoxCatchDifficulty:
    """Difficulty from resolved ballistic quantities, not raw sample ranges.

    Raises ValueError (propagated from `resolve_ballistic_launch_position/
    velocity`) when the sampled launch velocity can't reach the catch plane
    at all -- callers (see `sample_stage_domain`) must catch this and tag
    the domain `environment_infeasible`, not `policy_failure`.
    """

    launch_velocity = np.asarray(domain.launch_velocity, dtype=float)
    gravity = np.asarray(squeeze_config.gravity, dtype=float)
    launch_position = resolve_ballistic_launch_position(squeeze_config, launch_velocity)
    resolved_velocity = resolve_ballistic_launch_velocity(
        squeeze_config, launch_velocity, launch_position=launch_position
    )
    prediction = BallisticPrediction(
        time_s=0.0,
        position=launch_position,
        velocity=resolved_velocity,
        gravity=gravity,
        confidence=1.0,
    )
    ttc = plane_crossing_ttc(prediction, squeeze_config.catch_plane_x)
    hold_difficulty = _hold_difficulty(domain)
    if not np.isfinite(ttc):
        impact_speed = float(np.linalg.norm(resolved_velocity))
        impact_energy = 0.5 * domain.mass * impact_speed**2
        return BoxCatchDifficulty(
            ttc_s=float("inf"),
            impact_speed_mps=impact_speed,
            impact_energy_j=float(impact_energy),
            hold_difficulty=hold_difficulty,
            total_difficulty=float("inf"),
        )
    impact_velocity = resolved_velocity + gravity * ttc
    impact_speed = float(np.linalg.norm(impact_velocity))
    impact_energy = 0.5 * domain.mass * impact_speed**2
    total = (
        max(0.0, 1.0 - ttc / _TTC_REFERENCE_S)
        + impact_energy / _ENERGY_REFERENCE_J
        + hold_difficulty
    )
    return BoxCatchDifficulty(
        ttc_s=float(ttc),
        impact_speed_mps=impact_speed,
        impact_energy_j=float(impact_energy),
        hold_difficulty=hold_difficulty,
        total_difficulty=float(total),
    )


# ---------------------------------------------------------------------------
# Resample-with-ballistic-recheck
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedDomainSample:
    """One accepted (or exhausted) domain draw, with everything a caller
    needs to build the episode config and log the sampling process."""

    domain: BoxDomainParameters
    squeeze_config: DynamicSideSqueezeConfig
    difficulty: Optional[BoxCatchDifficulty]
    resample_count: int
    resample_exhausted: bool
    infeasible: bool


def sample_stage_domain(
    stage: CurriculumStage,
    rng: np.random.Generator,
    stage_index: int,
    base_squeeze_config: DynamicSideSqueezeConfig,
    *,
    max_resamples: int = 5,
) -> ResolvedDomainSample:
    """Sample -> resolve ballistic velocity/position -> check difficulty ->
    resample up to `max_resamples` -> on exhaustion, return the last attempt
    flagged `resample_exhausted=True` rather than silently substituting a
    clamped value that was never actually sampled."""

    last_domain: Optional[BoxDomainParameters] = None
    last_squeeze: Optional[DynamicSideSqueezeConfig] = None
    last_difficulty: Optional[BoxCatchDifficulty] = None
    last_infeasible = False

    if stage.bypass_ballistic_resolution:
        # No ballistic arc to solve or score here at all: the real static-
        # grasp mechanism is main_acmpc_box_catch.py's AcmpcBoxCatchConfig.
        # use_launch_fixture (a MuJoCo weld holding the box at a fixed,
        # already-reachable pose until bilateral contact dwell releases it)
        # -- see main_acmpc_box_catch_curriculum.py's Stage-0-only kwargs.
        # An earlier version of this branch tried to fake "stationary" via
        # launch_position/derive_vertical_launch_velocity overrides without
        # the fixture; that box still free-fell from the first physics
        # step (the fixture is what actually holds it in place), so those
        # overrides never mattered and are removed here. mass/friction/size
        # from the normal sample still matter (contact dynamics, grip
        # force); launch_velocity does not, since qvel gets explicitly
        # zeroed at fixture release regardless of what's sampled here.
        domain = stage.sample(rng, stage_index)
        squeeze_config = replace(base_squeeze_config, box_half_y=domain.half_size[1])
        return ResolvedDomainSample(domain, squeeze_config, None, 0, False, False)

    for attempt in range(max_resamples + 1):
        domain = stage.sample(rng, stage_index)
        if stage.target_flight_time_s_range is not None:
            flight_time = float(rng.uniform(*stage.target_flight_time_s_range))
            squeeze_config = replace(
                base_squeeze_config,
                target_flight_time_s=flight_time,
                box_half_y=domain.half_size[1],
            )
        else:
            squeeze_config = replace(base_squeeze_config, box_half_y=domain.half_size[1])

        try:
            launch_position = resolve_ballistic_launch_position(
                squeeze_config, np.asarray(domain.launch_velocity, dtype=float)
            )
            resolved_velocity = resolve_ballistic_launch_velocity(
                squeeze_config,
                np.asarray(domain.launch_velocity, dtype=float),
                launch_position=launch_position,
            )
        except ValueError:
            last_domain, last_squeeze, last_difficulty = domain, squeeze_config, None
            last_infeasible = True
            continue

        # Bake the resolved velocity into the squeeze config the episode will
        # actually run with -- compute_box_catch_difficulty re-resolves the
        # same (cheap, pure) math internally rather than taking it as an arg,
        # so both call sites always agree by construction, not by convention.
        resolved_velocity_tuple = tuple(float(v) for v in resolved_velocity)
        squeeze_config = replace(
            squeeze_config,
            launch_velocity_low=resolved_velocity_tuple,
            launch_velocity_high=resolved_velocity_tuple,
        )
        difficulty = compute_box_catch_difficulty(domain, squeeze_config)

        last_domain, last_squeeze, last_difficulty = domain, squeeze_config, difficulty
        if not np.isfinite(difficulty.total_difficulty):
            last_infeasible = True
            continue
        last_infeasible = False
        low, high = stage.difficulty_range
        if low <= difficulty.total_difficulty <= high:
            return ResolvedDomainSample(domain, squeeze_config, difficulty, attempt, False, False)

    return ResolvedDomainSample(
        last_domain, last_squeeze, last_difficulty, max_resamples, True, last_infeasible
    )


# ---------------------------------------------------------------------------
# The 6 new stages
# ---------------------------------------------------------------------------


def progressive_catch_curriculum() -> tuple[CurriculumStage, ...]:
    """6-stage progressive curriculum: static grasp bootstrap -> full domain.

    Numeric ranges are a first-pass calibration (see the implementation
    report's "known limitations"), not empirically tuned. Angular velocity
    is kept at the same small legacy warmup-level range across ALL stages,
    including full_domain -- the controller's orientation-tracking capability
    hasn't been confirmed sufficient for larger values (per the spec's own
    caveat), so this stays conservative everywhere rather than assuming it.
    """

    _small_angular_low = (-0.005, -0.005, -0.025)
    _small_angular_high = (0.005, 0.005, 0.025)

    return (
        CurriculumStage(
            name="static_grasp_bootstrap",
            axis_scale_low=(0.97, 0.97, 0.97),
            axis_scale_high=(1.03, 1.03, 1.03),
            mass_range=(0.48, 0.52),
            friction_range=(1.40, 1.60),
            # Near-zero horizontal speed. bypass_ballistic_resolution=True
            # (below) means the ballistic velocity/position solve never runs
            # for this stage at all -- see sample_stage_domain and the field
            # docstring on CurriculumStage. target_flight_time_s_range is
            # unused in that path; kept populated for schema consistency.
            launch_velocity_low=(-0.05, -0.001, 0.0),
            launch_velocity_high=(-0.02, 0.001, 0.0),
            angular_velocity_low=(0.0, 0.0, 0.0),
            angular_velocity_high=(0.0, 0.0, 0.0),
            target_flight_time_s_range=(0.05, 0.15),
            # Promotion for this stage uses funnel metrics (bilateral/impact
            # -safe/hold rates), not ballistic-intercept difficulty -- accept
            # the first sample unconditionally.
            difficulty_range=(0.0, float("inf")),
            bypass_ballistic_resolution=True,
        ),
        CurriculumStage(
            name="near_low_speed_catch",
            axis_scale_low=(0.93, 0.97, 0.93),
            axis_scale_high=(1.07, 1.03, 1.07),
            mass_range=(0.46, 0.54),
            friction_range=(1.20, 1.40),
            launch_velocity_low=(-0.80, -0.01, 0.0),
            launch_velocity_high=(-0.60, 0.01, 0.0),
            angular_velocity_low=_small_angular_low,
            angular_velocity_high=_small_angular_high,
            target_flight_time_s_range=(0.45, 0.55),
            difficulty_range=(0.0, 1.2),
        ),
        CurriculumStage(
            name="long_flight_low_speed",
            # Same physical range as near_low_speed_catch -- the difficulty
            # here is prediction/tracking over a longer horizon, not a
            # heavier/faster box (per spec: "핵심 난이도는 충돌 속도가 아니라
            # prediction과 reference tracking").
            axis_scale_low=(0.93, 0.97, 0.93),
            axis_scale_high=(1.07, 1.03, 1.07),
            mass_range=(0.46, 0.54),
            friction_range=(1.20, 1.40),
            launch_velocity_low=(-0.80, -0.02, 0.0),
            launch_velocity_high=(-0.60, 0.02, 0.0),
            angular_velocity_low=_small_angular_low,
            angular_velocity_high=_small_angular_high,
            target_flight_time_s_range=(0.65, 0.85),
            difficulty_range=(0.0, 1.2),
        ),
        CurriculumStage(
            name="medium_dynamic_catch",
            axis_scale_low=(0.85, 0.92, 0.85),
            axis_scale_high=(1.15, 1.08, 1.15),
            mass_range=(0.40, 0.62),
            friction_range=(1.00, 1.50),
            launch_velocity_low=(-1.20, -0.02, 0.0),
            launch_velocity_high=(-1.00, 0.02, 0.0),
            angular_velocity_low=_small_angular_low,
            angular_velocity_high=_small_angular_high,
            target_flight_time_s_range=(0.35, 0.50),
            difficulty_range=(0.6, 2.2),
        ),
        CurriculumStage(
            name="high_speed_catch",
            axis_scale_low=(0.75, 0.88, 0.75),
            axis_scale_high=(1.25, 1.12, 1.25),
            mass_range=(0.35, 0.75),
            friction_range=(0.85, 1.50),
            launch_velocity_low=(-1.80, -0.03, 0.0),
            launch_velocity_high=(-1.50, 0.03, 0.0),
            angular_velocity_low=_small_angular_low,
            angular_velocity_high=_small_angular_high,
            target_flight_time_s_range=(0.20, 0.35),
            # Excludes samples so extreme they're near-certain emergency
            # violations rather than "hard but learnable" -- upper bound left
            # open (full_domain covers the true tail).
            difficulty_range=(1.4, 3.5),
        ),
        CurriculumStage(
            name="full_domain",
            axis_scale_low=(0.65, 0.82, 0.65),
            axis_scale_high=(1.35, 1.18, 1.35),
            mass_range=(0.30, 0.85),
            friction_range=(0.70, 1.55),
            launch_velocity_low=(-1.80, -0.02, 0.0),
            launch_velocity_high=(-1.35, 0.02, 0.0),
            angular_velocity_low=_small_angular_low,
            angular_velocity_high=_small_angular_high,
            target_flight_time_s_range=(0.15, 0.85),
            # Final target distribution -- not a union of the earlier
            # stages' ranges, evaluated against the true full physical range.
            difficulty_range=(0.0, float("inf")),
        ),
    )


def check_stage_difficulty_ordering(
    stages: tuple[CurriculumStage, ...],
    base_squeeze_config: DynamicSideSqueezeConfig,
    *,
    samples_per_stage: int = 200,
    seed: int = 0,
) -> list[str]:
    """Sample each stage and warn (return message strings, don't raise) if
    mean difficulty isn't roughly non-decreasing along the stage sequence.
    Perfect monotonicity isn't required or expected -- see the spec's own
    "완벽한 단조 증가가 항상 보장될 필요는 없지만... 역전되면 warning" allowance."""

    rng = np.random.default_rng(seed)
    warnings: list[str] = []
    means: list[Optional[float]] = []
    for index, stage in enumerate(stages):
        scores = []
        for _ in range(samples_per_stage):
            resolved = sample_stage_domain(stage, rng, index, base_squeeze_config)
            if resolved.difficulty is not None and np.isfinite(resolved.difficulty.total_difficulty):
                scores.append(resolved.difficulty.total_difficulty)
        means.append(float(np.mean(scores)) if scores else None)

    for i in range(len(means) - 1):
        current, following = means[i], means[i + 1]
        if current is None or following is None:
            continue
        if following < current:
            warnings.append(
                f"stage difficulty order reversed: {stages[i].name} (mean={current:.3f}) "
                f"> {stages[i + 1].name} (mean={following:.3f})"
            )
    return warnings


# ---------------------------------------------------------------------------
# Funnel-based, safety-split, cooldown-aware scheduler
# ---------------------------------------------------------------------------


@dataclass
class StageFunnelRequirement:
    """Promotion gate for one stage. All rates are first-pass initial
    experiment values, not derived theoretical bounds (see spec)."""

    min_bilateral_rate: float = 0.0
    min_impact_safe_rate: float = 0.0
    min_hold_rate: float = 0.0
    min_success_rate: float = 0.0
    min_catch_region_rate: float = 0.0


DEFAULT_FUNNEL_REQUIREMENTS: dict[str, StageFunnelRequirement] = {
    "static_grasp_bootstrap": StageFunnelRequirement(
        min_bilateral_rate=0.90, min_impact_safe_rate=0.95, min_hold_rate=0.80
    ),
    "near_low_speed_catch": StageFunnelRequirement(
        min_bilateral_rate=0.85,
        min_impact_safe_rate=0.90,
        min_hold_rate=0.75,
        min_success_rate=0.75,
    ),
    "long_flight_low_speed": StageFunnelRequirement(
        min_catch_region_rate=0.85,
        min_bilateral_rate=0.80,
        min_impact_safe_rate=0.90,
        min_success_rate=0.70,
    ),
    "medium_dynamic_catch": StageFunnelRequirement(
        min_catch_region_rate=0.85,
        min_bilateral_rate=0.75,
        min_impact_safe_rate=0.85,
        min_success_rate=0.65,
    ),
    "high_speed_catch": StageFunnelRequirement(
        min_catch_region_rate=0.80,
        min_bilateral_rate=0.70,
        min_impact_safe_rate=0.80,
        min_success_rate=0.55,
    ),
    "full_domain": StageFunnelRequirement(
        min_catch_region_rate=0.80,
        min_bilateral_rate=0.70,
        min_impact_safe_rate=0.80,
        min_success_rate=0.55,
    ),
}


@dataclass
class _EpisodeOutcome:
    success: bool
    reached_pre_impact: bool
    bilateral_contact_achieved: bool
    impact_safe: bool
    stable_hold_completed: bool


@dataclass
class StageLifetimeStats:
    """Per-stage stats that accumulate for the whole run, never reset on
    stage transitions (only the anchor decision window resets)."""

    total_episodes: int = 0
    anchor_episodes: int = 0
    mixture_episodes: int = 0
    success_count: int = 0
    bilateral_contact_count: int = 0
    impact_safe_count: int = 0
    stable_hold_count: int = 0
    ordinary_unsafe_count: int = 0
    emergency_violation_count: int = 0
    policy_failure_count: int = 0
    environment_infeasible_count: int = 0
    simulation_error_count: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, values: dict) -> "StageLifetimeStats":
        return cls(**values)


class AdaptiveCurriculumScheduler:
    """Funnel-based scheduler for `progressive_catch_curriculum()`.

    Anchor vs mixture: `sample()` returns `(domain, is_anchor)`. Only anchor
    episodes (sampled strictly from the current stage) feed the promotion
    decision window -- mixture episodes (adjacent-stage or failure-replay)
    still update lifetime stats but never affect cooldown/promotion, so an
    easy mixture success can't inflate the anchor stage's apparent mastery.

    Safety split: ordinary unsafe episodes accumulate into a rolling rate;
    emergency violations demote immediately and reset cooldown to 0 (bypass),
    regardless of the current success rate.
    """

    def __init__(
        self,
        stages: Optional[tuple[CurriculumStage, ...]] = None,
        *,
        funnel_requirements: Optional[dict[str, StageFunnelRequirement]] = None,
        decision_window: int = 20,
        minimum_anchor_episodes: int = 12,
        transition_cooldown_episodes: int = 5,
        demote_safe_success_rate: float = 0.40,
        demote_ordinary_unsafe_rate: float = 0.15,
        mixture_adjacent_probability: float = 0.20,
        failure_replay_probability: float = 0.10,
        failure_replay_capacity: int = 32,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.stages = tuple(stages) if stages is not None else progressive_catch_curriculum()
        if not self.stages:
            raise ValueError("at least one curriculum stage is required")
        self.funnel_requirements = dict(funnel_requirements or DEFAULT_FUNNEL_REQUIREMENTS)
        self.decision_window = int(decision_window)
        self.minimum_anchor_episodes = int(minimum_anchor_episodes)
        self.transition_cooldown_episodes = int(transition_cooldown_episodes)
        self.demote_safe_success_rate = float(demote_safe_success_rate)
        self.mixture_adjacent_probability = float(mixture_adjacent_probability)
        self.failure_replay_probability = float(failure_replay_probability)
        self.demote_ordinary_unsafe_rate = float(demote_ordinary_unsafe_rate)

        self.stage_index = 0
        self._episodes_at_stage = 0
        self._cooldown_remaining = 0
        self._anchor_outcomes: deque[_EpisodeOutcome] = deque(maxlen=self.decision_window)
        self._ordinary_unsafe_outcomes: deque[bool] = deque(maxlen=self.decision_window)
        self._lifetime_stats: dict[str, StageLifetimeStats] = {
            stage.name: StageLifetimeStats() for stage in self.stages
        }
        self._failure_replay_domains: deque[BoxDomainParameters] = deque(
            maxlen=int(failure_replay_capacity)
        )
        self._rng = np.random.default_rng(rng_seed)

    @property
    def stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def cooldown_remaining(self) -> int:
        return self._cooldown_remaining

    @property
    def anchor_success_rate(self) -> float:
        if not self._anchor_outcomes:
            return 0.0
        return float(np.mean([outcome.success for outcome in self._anchor_outcomes]))

    def lifetime_stats(self, stage_name: str) -> StageLifetimeStats:
        return self._lifetime_stats[stage_name]

    # -- sampling -----------------------------------------------------

    def sample(self, squeeze_config: DynamicSideSqueezeConfig, *, max_resamples: int = 5):
        """Return `(ResolvedDomainSample, is_anchor)`."""

        draw = self._rng.random()
        if draw < self.failure_replay_probability and self._failure_replay_domains:
            index = int(self._rng.integers(0, len(self._failure_replay_domains)))
            replay_domain = self._failure_replay_domains[index]
            squeeze = replace(squeeze_config, box_half_y=replay_domain.half_size[1])
            try:
                difficulty = compute_box_catch_difficulty(replay_domain, squeeze)
                infeasible = not np.isfinite(difficulty.total_difficulty)
            except ValueError:
                difficulty, infeasible = None, True
            return (
                ResolvedDomainSample(replay_domain, squeeze, difficulty, 0, False, infeasible),
                False,
            )
        if draw < self.failure_replay_probability + self.mixture_adjacent_probability:
            offset = -1 if self._rng.random() < 0.5 else 1
            adjacent_index = int(np.clip(self.stage_index + offset, 0, len(self.stages) - 1))
            resolved = sample_stage_domain(
                self.stages[adjacent_index],
                self._rng,
                adjacent_index,
                squeeze_config,
                max_resamples=max_resamples,
            )
            return resolved, False
        resolved = sample_stage_domain(
            self.stage, self._rng, self.stage_index, squeeze_config, max_resamples=max_resamples
        )
        return resolved, True

    # -- recording ------------------------------------------------------

    def record(
        self,
        *,
        is_anchor: bool,
        stage_name: str,
        domain: Optional[BoxDomainParameters],
        funnel,  # EpisodeFunnel from main_acmpc_box_catch.py (duck-typed to avoid an import cycle)
        ordinary_safety_violation: bool,
        emergency_safety_violation: bool,
        environment_infeasible: bool = False,
        simulation_error: bool = False,
    ) -> int:
        stats = self._lifetime_stats.setdefault(stage_name, StageLifetimeStats())
        stats.total_episodes += 1
        if is_anchor:
            stats.anchor_episodes += 1
        else:
            stats.mixture_episodes += 1

        if environment_infeasible:
            stats.environment_infeasible_count += 1
            return self.stage_index
        if simulation_error:
            stats.simulation_error_count += 1
            return self.stage_index

        if funnel.episode_success:
            stats.success_count += 1
        if funnel.bilateral_contact_achieved:
            stats.bilateral_contact_count += 1
        if funnel.impact_safe:
            stats.impact_safe_count += 1
        if funnel.stable_hold_completed:
            stats.stable_hold_count += 1
        if emergency_safety_violation:
            stats.emergency_violation_count += 1
        elif ordinary_safety_violation:
            stats.ordinary_unsafe_count += 1
        if not funnel.episode_success and not emergency_safety_violation:
            stats.policy_failure_count += 1
        if not funnel.episode_success and domain is not None:
            self._failure_replay_domains.append(domain)

        if not is_anchor:
            return self.stage_index

        self._episodes_at_stage += 1
        outcome = _EpisodeOutcome(
            success=bool(funnel.episode_success and not emergency_safety_violation),
            reached_pre_impact=funnel.reached_pre_impact,
            bilateral_contact_achieved=funnel.bilateral_contact_achieved,
            impact_safe=funnel.impact_safe,
            stable_hold_completed=funnel.stable_hold_completed,
        )

        if emergency_safety_violation:
            self._demote()
            self._cooldown_remaining = 0  # emergency bypasses cooldown, always demotes
            return self.stage_index

        self._anchor_outcomes.append(outcome)
        self._ordinary_unsafe_outcomes.append(bool(ordinary_safety_violation))

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return self.stage_index

        if self._episodes_at_stage < self.minimum_anchor_episodes:
            return self.stage_index

        n = len(self._anchor_outcomes)
        rates = {
            "min_catch_region_rate": np.mean([o.reached_pre_impact for o in self._anchor_outcomes]),
            "min_bilateral_rate": np.mean(
                [o.bilateral_contact_achieved for o in self._anchor_outcomes]
            ),
            "min_impact_safe_rate": np.mean([o.impact_safe for o in self._anchor_outcomes]),
            "min_hold_rate": np.mean([o.stable_hold_completed for o in self._anchor_outcomes]),
            "min_success_rate": np.mean([o.success for o in self._anchor_outcomes]),
        }
        safe_success_rate = float(rates["min_success_rate"])
        ordinary_unsafe_rate = (
            float(np.mean(self._ordinary_unsafe_outcomes)) if self._ordinary_unsafe_outcomes else 0.0
        )

        requirement = self.funnel_requirements.get(stage_name, StageFunnelRequirement())
        meets_requirement = all(
            rates[key] >= getattr(requirement, key) for key in rates if getattr(requirement, key) > 0.0
        )

        if meets_requirement and n >= self.decision_window:
            self._promote()
        elif safe_success_rate <= self.demote_safe_success_rate or (
            ordinary_unsafe_rate >= self.demote_ordinary_unsafe_rate and n >= self.minimum_anchor_episodes
        ):
            self._demote()
        return self.stage_index

    def _promote(self) -> None:
        self.stage_index = min(len(self.stages) - 1, self.stage_index + 1)
        self._reset_decision_window()

    def _demote(self) -> None:
        self.stage_index = max(0, self.stage_index - 1)
        self._reset_decision_window()

    def _reset_decision_window(self) -> None:
        self._episodes_at_stage = 0
        self._anchor_outcomes.clear()
        self._ordinary_unsafe_outcomes.clear()
        self._cooldown_remaining = self.transition_cooldown_episodes

    # -- checkpoint round-trip -------------------------------------------

    def state_dict(self) -> dict:
        return {
            "stage_index": self.stage_index,
            "episodes_at_stage": self._episodes_at_stage,
            "cooldown_remaining": self._cooldown_remaining,
            "anchor_outcomes": [outcome.__dict__ for outcome in self._anchor_outcomes],
            "ordinary_unsafe_outcomes": list(self._ordinary_unsafe_outcomes),
            "lifetime_stats": {
                name: stats.as_dict() for name, stats in self._lifetime_stats.items()
            },
            "failure_replay_domains": [
                dict(domain.__dict__) for domain in self._failure_replay_domains
            ],
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.stage_index = int(state["stage_index"])
        self._episodes_at_stage = int(state["episodes_at_stage"])
        self._cooldown_remaining = int(state["cooldown_remaining"])
        self._anchor_outcomes = deque(
            (_EpisodeOutcome(**outcome) for outcome in state["anchor_outcomes"]),
            maxlen=self.decision_window,
        )
        self._ordinary_unsafe_outcomes = deque(
            state["ordinary_unsafe_outcomes"], maxlen=self.decision_window
        )
        self._lifetime_stats = {
            name: StageLifetimeStats.from_dict(values)
            for name, values in state["lifetime_stats"].items()
        }
        self._failure_replay_domains = deque(
            (BoxDomainParameters(**values) for values in state["failure_replay_domains"]),
            maxlen=self._failure_replay_domains.maxlen,
        )
        self._rng.bit_generator.state = state["rng_state"]
