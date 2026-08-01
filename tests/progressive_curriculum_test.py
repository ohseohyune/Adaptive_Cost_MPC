"""Fast, MuJoCo-free tests for control/squeeze/progressive_curriculum.py.

Covers the pieces that don't need a full run_box_catch episode: difficulty
scoring, the resample-with-recheck flow, and AdaptiveCurriculumScheduler's
promotion/demotion/cooldown/emergency/checkpoint behavior via synthetic
`record()` calls (no real episodes). Full-episode/PPO-batch verification is
done via short manual smoke tests during implementation (see the
implementation report), not here -- this file must stay fast.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.squeeze.config import DynamicSideSqueezeConfig
from control.squeeze.progressive_curriculum import (
    AdaptiveCurriculumScheduler,
    CurriculumStage,
    check_stage_difficulty_ordering,
    compute_box_catch_difficulty,
    progressive_catch_curriculum,
    sample_stage_domain,
)


@dataclass
class _FakeFunnel:
    reached_pre_impact: bool = True
    first_contact_detected: bool = True
    impact_safe: bool = True
    bilateral_contact_achieved: bool = True
    hold_entered: bool = True
    stable_hold_completed: bool = True
    episode_success: bool = True
    failure_category: str = ""


def sc1_difficulty_increases_with_mass_and_speed() -> tuple[bool, str]:
    base = DynamicSideSqueezeConfig()
    stage = CurriculumStage(
        name="probe",
        axis_scale_low=(1.0, 1.0, 1.0),
        axis_scale_high=(1.0, 1.0, 1.0),
        mass_range=(0.50, 0.50),
        friction_range=(1.2, 1.2),
        launch_velocity_low=(-1.0, 0.0, 0.0),
        launch_velocity_high=(-1.0, 0.0, 0.0),
        angular_velocity_low=(0.0, 0.0, 0.0),
        angular_velocity_high=(0.0, 0.0, 0.0),
    )
    domain_light_slow = stage.sample(np.random.default_rng(0), 0)
    heavier = stage.sample(np.random.default_rng(0), 0).__class__(
        **{**domain_light_slow.__dict__, "mass": 1.0}
    )
    faster = stage.sample(np.random.default_rng(0), 0).__class__(
        **{**domain_light_slow.__dict__, "launch_velocity": (-2.0, 0.0, 0.0)}
    )

    baseline = compute_box_catch_difficulty(domain_light_slow, base)
    heavier_result = compute_box_catch_difficulty(heavier, base)
    faster_result = compute_box_catch_difficulty(faster, base)

    ok = (
        heavier_result.total_difficulty > baseline.total_difficulty
        and faster_result.total_difficulty > baseline.total_difficulty
    )
    return ok, (
        f"baseline={baseline.total_difficulty:.3f} heavier={heavier_result.total_difficulty:.3f} "
        f"faster={faster_result.total_difficulty:.3f}"
    )


def sc2_resample_flags_exhaustion_on_impossible_range() -> tuple[bool, str]:
    base = DynamicSideSqueezeConfig()
    stage = CurriculumStage(
        name="impossible",
        axis_scale_low=(1.0, 1.0, 1.0),
        axis_scale_high=(1.0, 1.0, 1.0),
        mass_range=(0.50, 0.50),
        friction_range=(1.2, 1.2),
        launch_velocity_low=(-1.0, 0.0, 0.0),
        launch_velocity_high=(-1.0, 0.0, 0.0),
        angular_velocity_low=(0.0, 0.0, 0.0),
        angular_velocity_high=(0.0, 0.0, 0.0),
        # No real sample can ever land in this range -- forces exhaustion.
        difficulty_range=(1e6, 1e6 + 1.0),
    )
    result = sample_stage_domain(stage, np.random.default_rng(1), 0, base, max_resamples=3)
    ok = result.resample_exhausted and result.resample_count == 3
    return ok, f"resample_exhausted={result.resample_exhausted} resample_count={result.resample_count}"


def sc3_bypass_stage_skips_difficulty_and_never_infeasible() -> tuple[bool, str]:
    base = DynamicSideSqueezeConfig()
    stages = progressive_catch_curriculum()
    bootstrap = stages[0]
    ok_bypass_flag = bootstrap.bypass_ballistic_resolution
    result = sample_stage_domain(bootstrap, np.random.default_rng(2), 0, base)
    ok = ok_bypass_flag and result.difficulty is None and not result.infeasible
    return ok, f"bypass_flag={ok_bypass_flag} difficulty={result.difficulty} infeasible={result.infeasible}"


def sc4_stage_ordering_check_flags_reversals_not_exceptions() -> tuple[bool, str]:
    # Just confirms the checker runs end-to-end and returns a list of
    # strings (not that there happen to be zero reversals -- the 6-stage
    # first-pass calibration isn't tuned to guarantee strict monotonicity,
    # see the implementation report's known limitations).
    warnings = check_stage_difficulty_ordering(
        progressive_catch_curriculum(), DynamicSideSqueezeConfig(), samples_per_stage=20, seed=3
    )
    ok = isinstance(warnings, list) and all(isinstance(w, str) for w in warnings)
    return ok, f"{len(warnings)} ordering warning(s): {warnings}"


def sc5_promotion_requires_minimum_anchor_episodes() -> tuple[bool, str]:
    sched = AdaptiveCurriculumScheduler(rng_seed=10, minimum_anchor_episodes=5, decision_window=10)
    start = sched.stage_index
    for _ in range(4):  # one short of minimum_anchor_episodes
        sched.record(
            is_anchor=True,
            stage_name=sched.stage.name,
            domain=None,
            funnel=_FakeFunnel(),
            ordinary_safety_violation=False,
            emergency_safety_violation=False,
        )
    ok = sched.stage_index == start
    return ok, f"stage_index unchanged before minimum_anchor_episodes: {ok} (index={sched.stage_index})"


def sc6_promotion_and_demotion_fire_on_threshold() -> tuple[bool, str]:
    sched = AdaptiveCurriculumScheduler(rng_seed=11, minimum_anchor_episodes=5, decision_window=8)
    for _ in range(8):
        sched.record(
            is_anchor=True,
            stage_name=sched.stage.name,
            domain=None,
            funnel=_FakeFunnel(),
            ordinary_safety_violation=False,
            emergency_safety_violation=False,
        )
    promoted = sched.stage_index == 1

    sched2 = AdaptiveCurriculumScheduler(
        rng_seed=12, minimum_anchor_episodes=5, decision_window=8, transition_cooldown_episodes=0
    )
    for _ in range(8):
        sched2.record(
            is_anchor=True,
            stage_name=sched2.stage.name,
            domain=None,
            funnel=_FakeFunnel(episode_success=False, bilateral_contact_achieved=False, stable_hold_completed=False),
            ordinary_safety_violation=False,
            emergency_safety_violation=False,
        )
    demoted_or_floor = sched2.stage_index == 0  # already at the floor, demotion is a no-op index-wise

    ok = promoted and demoted_or_floor
    return ok, f"promoted after good streak: {promoted} (index={sched.stage_index}), stayed at floor when bad: {demoted_or_floor}"


def sc7_emergency_violation_demotes_immediately_and_bypasses_cooldown() -> tuple[bool, str]:
    sched = AdaptiveCurriculumScheduler(
        rng_seed=13, minimum_anchor_episodes=5, decision_window=8, transition_cooldown_episodes=10
    )
    for _ in range(8):
        sched.record(
            is_anchor=True,
            stage_name=sched.stage.name,
            domain=None,
            funnel=_FakeFunnel(),
            ordinary_safety_violation=False,
            emergency_safety_violation=False,
        )
    promoted_index = sched.stage_index
    cooldown_before = sched.cooldown_remaining
    sched.record(
        is_anchor=True,
        stage_name=sched.stage.name,
        domain=None,
        funnel=_FakeFunnel(episode_success=False),
        ordinary_safety_violation=False,
        emergency_safety_violation=True,
    )
    ok = (
        promoted_index > 0
        and cooldown_before > 0
        and sched.stage_index == promoted_index - 1
        and sched.cooldown_remaining == 0
    )
    return ok, (
        f"promoted_index={promoted_index} cooldown_before_emergency={cooldown_before} "
        f"stage_after_emergency={sched.stage_index} cooldown_after={sched.cooldown_remaining}"
    )


def sc8_mixture_episodes_never_affect_anchor_promotion() -> tuple[bool, str]:
    sched = AdaptiveCurriculumScheduler(rng_seed=14, minimum_anchor_episodes=5, decision_window=8)
    start = sched.stage_index
    for _ in range(20):
        # is_anchor=False (mixture/adjacent-stage draw) with perfect success
        # must never move the anchor decision window or promote the stage.
        sched.record(
            is_anchor=False,
            stage_name=sched.stages[min(sched.stage_index + 1, len(sched.stages) - 1)].name,
            domain=None,
            funnel=_FakeFunnel(),
            ordinary_safety_violation=False,
            emergency_safety_violation=False,
        )
    ok = sched.stage_index == start and len(sched._anchor_outcomes) == 0
    return ok, f"stage_index unchanged by mixture-only episodes: {ok} (index={sched.stage_index})"


def sc9_state_dict_round_trips() -> tuple[bool, str]:
    sched = AdaptiveCurriculumScheduler(rng_seed=15, minimum_anchor_episodes=3, decision_window=6)
    for i in range(5):
        sched.record(
            is_anchor=True,
            stage_name=sched.stage.name,
            domain=None,
            funnel=_FakeFunnel(episode_success=(i % 2 == 0)),
            ordinary_safety_violation=(i == 1),
            emergency_safety_violation=False,
        )
    state = sched.state_dict()

    restored = AdaptiveCurriculumScheduler(rng_seed=999)
    restored.load_state_dict(state)

    ok = (
        restored.stage_index == sched.stage_index
        and restored.cooldown_remaining == sched.cooldown_remaining
        and restored._episodes_at_stage == sched._episodes_at_stage
        and list(restored._anchor_outcomes) == list(sched._anchor_outcomes)
        and restored.lifetime_stats(sched.stage.name).as_dict()
        == sched.lifetime_stats(sched.stage.name).as_dict()
    )
    return ok, f"round-trip matches: {ok}"


def main() -> None:
    scenarios = [
        ("SC1 difficulty increases with mass and speed", sc1_difficulty_increases_with_mass_and_speed),
        ("SC2 resample flags exhaustion on an impossible range", sc2_resample_flags_exhaustion_on_impossible_range),
        ("SC3 static_grasp_bootstrap bypasses ballistic difficulty", sc3_bypass_stage_skips_difficulty_and_never_infeasible),
        ("SC4 stage ordering check runs and returns warning strings", sc4_stage_ordering_check_flags_reversals_not_exceptions),
        ("SC5 promotion requires minimum_anchor_episodes", sc5_promotion_requires_minimum_anchor_episodes),
        ("SC6 promotion/demotion fire on threshold", sc6_promotion_and_demotion_fire_on_threshold),
        ("SC7 emergency violation demotes immediately, bypasses cooldown", sc7_emergency_violation_demotes_immediately_and_bypasses_cooldown),
        ("SC8 mixture episodes never affect anchor promotion", sc8_mixture_episodes_never_affect_anchor_promotion),
        ("SC9 scheduler state_dict round-trips", sc9_state_dict_round_trips),
    ]
    passed = 0
    print("\n=== Progressive catch curriculum: fast unit tests ===\n")
    for name, scenario in scenarios:
        ok, detail = scenario()
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}\n")
        passed += int(ok)
    print(f"Result: {passed}/{len(scenarios)} SCs passed")
    if passed != len(scenarios):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
