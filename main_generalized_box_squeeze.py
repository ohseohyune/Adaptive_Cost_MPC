"""Milestone 4: curriculum PPO adaptation across randomized box domains."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from control.mpc import (
    PPOCostAdapter,
    PPOCostConfig,
    PPORolloutBuffer,
    PPOUpdateSummary,
    apply_cost_action,
    build_generalization_observation,
)
from control.squeeze import (
    BoxDomainParameters,
    CurriculumScheduler,
    RotatingSideSqueezeConfig,
    resolve_ballistic_launch_position,
    resolve_ballistic_launch_velocity,
)
from main_dynamic_box_squeeze import (
    DynamicRunConfig,
    DynamicSqueezeSummary,
    run_dynamic_side_squeeze,
)


ROOT = Path(__file__).resolve().parent


@dataclass
class GeneralizationRunConfig:
    episodes: int = 8
    rollout_size: int = 4
    random_seed: int = 7
    device: str = "auto"
    online_adaptation: bool = True
    offline_training: bool = False
    viewer: bool = False
    viewer_show_prepare: bool = False
    checkpoint_path: Optional[str] = None
    load_checkpoint: bool = False
    collision_mode: str = "miss_backstop"
    curriculum_mode: str = "adaptive"


@dataclass(frozen=True)
class GeneralizationEpisode:
    episode: int
    curriculum_stage: str
    curriculum_stage_index: int
    domain: BoxDomainParameters
    cost_multipliers: dict[str, float]
    reward: float
    success: bool
    safety_violation: bool
    first_contact_peak_force_n: float
    minimum_no_slip_force_per_pad_n: float
    final_hold_force_per_pad_n: float
    first_nonpad_contact_time_s: Optional[float]
    first_nonpad_contact_object: str
    final_box_speed_mps: float
    final_angular_speed_radps: float
    slip_cost: float
    failure_reason: str


@dataclass(frozen=True)
class GeneralizationSummary:
    episodes: int
    successes: int
    success_rate: float
    safety_violations: int
    mean_reward: float
    stages_visited: tuple[str, ...]
    final_stage: str
    ppo_updates_applied: int
    ppo_updates_skipped: int
    actor_parameter_delta: float
    device: str
    online_adaptation: bool
    offline_training: bool
    update_summaries: tuple[PPOUpdateSummary, ...]
    episode_summaries: tuple[GeneralizationEpisode, ...]
    stage_summaries: tuple["StagePerformance", ...]


@dataclass(frozen=True)
class StagePerformance:
    stage: str
    episodes: int
    successes: int
    success_rate: float
    safety_violations: int
    mean_reward: float


def episode_reward(
    result: DynamicSqueezeSummary,
    config: RotatingSideSqueezeConfig,
) -> float:
    """Reward safe capture, low residual motion, and low contact effort."""

    reward = 3.0 if result.success else -2.0
    if not result.success:
        reason = result.failure_reason.lower()
        if "emergency" in reason:
            reward -= 3.0
        elif "force limit" in reason:
            reward -= 2.0
        elif "escaped" in reason:
            reward -= 1.5
        elif "interception workspace" in reason:
            reward -= 1.0
        elif "bilateral contact" in reason:
            reward -= 0.75
        if result.first_contact_peak_force_n <= 1e-6:
            # No pad ever touched the box. This is a distinct, worse failure
            # mode than a late or unstable catch: force/stiffness residuals
            # cannot fix it, only the interception timing/reach can. Penalize
            # it separately so the actor sees a clear gradient away from
            # never reaching the box at all.
            reward -= 1.5
    reward += 0.5 * np.clip(
        result.dynamic_hold_time_s / max(config.required_dynamic_hold_s, 1e-6),
        0.0,
        1.0,
    )
    reward -= 0.45 * np.clip(
        result.final_box_speed_mps / max(config.maximum_capture_speed, 1e-6),
        0.0,
        3.0,
    )
    reward -= 0.35 * np.clip(
        result.final_angular_speed_radps
        / max(config.maximum_capture_angular_speed, 1e-6),
        0.0,
        3.0,
    )
    reward -= 0.35 * max(
        0.0,
        result.first_contact_peak_force_n
        / max(config.first_contact_force_limit, 1e-6)
        - 0.75,
    )
    reward -= 0.08 * min(result.slip_cost, 5.0)
    reward -= 0.01 * min(result.wrench_tracking_error, 20.0)
    return float(np.clip(reward, -5.0, 5.0))


def is_safety_violation(
    result: DynamicSqueezeSummary,
    config: RotatingSideSqueezeConfig,
) -> bool:
    return bool(
        not np.isfinite(result.first_contact_peak_force_n)
        or result.first_contact_peak_force_n > config.first_contact_force_limit + 1e-6
        or "emergency" in result.failure_reason.lower()
    )


def _episode_config(
    base: RotatingSideSqueezeConfig,
    domain: BoxDomainParameters,
    *,
    seed: int,
) -> RotatingSideSqueezeConfig:
    # The QP friction pyramid remains slightly inside the physical Coulomb
    # cone.  This avoids teaching the actor to rely on unrealizable friction.
    qp_friction = min(base.wrench_friction_coefficient, 0.90 * domain.friction)
    # A low-height/deep ("flat") cuboid has less rotational leverage at the
    # broad pad and needs a firmer tangential pose constraint, not more normal
    # squeeze force. Other randomized shapes benefit from stronger rotational
    # damping during the five-second hold.
    flat_shape = domain.shape_family == "flat"
    tangential_stiffness = (
        max(base.tangential_stiffness, 1200.0)
        if flat_shape
        else base.tangential_stiffness
    )
    rotational_damping = 20.0 if flat_shape else 30.0
    return replace(
        base,
        random_seed=seed,
        box_half_y=domain.half_size[1],
        box_mass=domain.mass,
        wrench_friction_coefficient=qp_friction,
        tangential_stiffness=tangential_stiffness,
        rotational_damping=rotational_damping,
        launch_velocity_low=domain.launch_velocity,
        launch_velocity_high=domain.launch_velocity,
        random_angular_velocity_low=domain.angular_velocity,
        random_angular_velocity_high=domain.angular_velocity,
    )


def run_generalized_box_squeeze(
    config: Optional[GeneralizationRunConfig] = None,
    *,
    adapter: Optional[PPOCostAdapter] = None,
    curriculum: Optional[CurriculumScheduler] = None,
) -> GeneralizationSummary:
    config = config or GeneralizationRunConfig()
    if config.episodes <= 0 or config.rollout_size <= 0:
        raise ValueError("episodes and rollout_size must be positive")
    if config.curriculum_mode not in {"adaptive", "balanced"}:
        raise ValueError("curriculum_mode must be adaptive or balanced")
    adaptation_enabled = config.online_adaptation or config.offline_training

    rng = np.random.default_rng(config.random_seed)
    curriculum = curriculum or CurriculumScheduler()
    adapter = adapter or PPOCostAdapter(
        PPOCostConfig(
            device=config.device,
            seed=config.random_seed,
            minimum_online_rollout=config.rollout_size,
        )
    )
    if config.load_checkpoint:
        if not config.checkpoint_path:
            raise ValueError("load_checkpoint requires checkpoint_path")
        adapter.load(config.checkpoint_path)

    initial_actor = adapter.actor_parameter_vector().copy()
    base = RotatingSideSqueezeConfig(random_seed=config.random_seed)
    rollout = PPORolloutBuffer()
    episodes: list[GeneralizationEpisode] = []
    updates: list[PPOUpdateSummary] = []
    stages_visited: list[str] = []

    for episode_index in range(config.episodes):
        if config.curriculum_mode == "balanced":
            stage_index = episode_index % len(curriculum.stages)
            domain = curriculum.stages[stage_index].sample(rng, stage_index)
        else:
            domain = curriculum.sample(rng)
        sampled_launch_velocity = np.asarray(domain.launch_velocity, dtype=float)
        episode_launch_position = resolve_ballistic_launch_position(
            base, sampled_launch_velocity
        )
        domain = replace(
            domain,
            launch_velocity=tuple(
                float(value)
                for value in resolve_ballistic_launch_velocity(
                    base,
                    sampled_launch_velocity,
                    launch_position=episode_launch_position,
                )
            ),
        )
        if domain.stage_name not in stages_visited:
            stages_visited.append(domain.stage_name)
        observation = build_generalization_observation(
            domain,
            curriculum_stage_count=len(curriculum.stages),
            catch_plane_x=base.catch_plane_x,
            launch_position_x=float(episode_launch_position[0]),
        )
        action = adapter.act(observation, training=adaptation_enabled)
        episode_config = _episode_config(
            base, domain, seed=config.random_seed + episode_index
        )
        episode_config = apply_cost_action(
            episode_config, action.normalized_action
        )
        result = run_dynamic_side_squeeze(
            DynamicRunConfig(
                viewer=config.viewer,
                viewer_show_prepare=config.viewer_show_prepare,
                squeeze=episode_config,
                domain_parameters=domain,
                collision_mode=config.collision_mode,
            )
        )
        safety_violation = is_safety_violation(result, episode_config)
        reward = episode_reward(result, episode_config)
        rollout.add(
            observation=observation,
            action=action,
            reward=reward,
            done=True,
            safety_violation=safety_violation,
        )
        episodes.append(
            GeneralizationEpisode(
                episode=episode_index,
                curriculum_stage=domain.stage_name,
                curriculum_stage_index=domain.stage_index,
                domain=domain,
                cost_multipliers=action.multipliers,
                reward=reward,
                success=result.success,
                safety_violation=safety_violation,
                first_contact_peak_force_n=result.first_contact_peak_force_n,
                minimum_no_slip_force_per_pad_n=(
                    result.minimum_no_slip_force_per_pad_n
                ),
                final_hold_force_per_pad_n=result.final_hold_force_per_pad_n,
                first_nonpad_contact_time_s=result.first_nonpad_contact_time_s,
                first_nonpad_contact_object=result.first_nonpad_contact_object,
                final_box_speed_mps=result.final_box_speed_mps,
                final_angular_speed_radps=result.final_angular_speed_radps,
                slip_cost=result.slip_cost,
                failure_reason=result.failure_reason,
            )
        )
        if config.curriculum_mode == "adaptive":
            curriculum.record(result.success, safety_violation=safety_violation)

        if len(rollout) >= config.rollout_size:
            if adaptation_enabled:
                updates.append(
                    adapter.update(rollout, online=not config.offline_training)
                )
            rollout.clear()

    if len(rollout) and adaptation_enabled:
        updates.append(
            adapter.update(rollout, online=not config.offline_training)
        )

    if config.checkpoint_path:
        checkpoint = Path(config.checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        adapter.save(str(checkpoint))

    successes = sum(episode.success for episode in episodes)
    safety_violations = sum(episode.safety_violation for episode in episodes)
    final_actor = adapter.actor_parameter_vector()
    stage_summaries: list[StagePerformance] = []
    for stage_name in stages_visited:
        stage_episodes = [
            episode
            for episode in episodes
            if episode.curriculum_stage == stage_name
        ]
        stage_successes = sum(episode.success for episode in stage_episodes)
        stage_summaries.append(
            StagePerformance(
                stage=stage_name,
                episodes=len(stage_episodes),
                successes=stage_successes,
                success_rate=stage_successes / len(stage_episodes),
                safety_violations=sum(
                    episode.safety_violation for episode in stage_episodes
                ),
                mean_reward=float(
                    np.mean([episode.reward for episode in stage_episodes])
                ),
            )
        )
    return GeneralizationSummary(
        episodes=len(episodes),
        successes=successes,
        success_rate=successes / len(episodes),
        safety_violations=safety_violations,
        mean_reward=float(np.mean([episode.reward for episode in episodes])),
        stages_visited=tuple(stages_visited),
        final_stage=(
            episodes[-1].curriculum_stage
            if config.curriculum_mode == "balanced"
            else curriculum.stage.name
        ),
        ppo_updates_applied=sum(update.applied for update in updates),
        ppo_updates_skipped=sum(not update.applied for update in updates),
        actor_parameter_delta=float(np.linalg.norm(final_actor - initial_actor)),
        device=str(adapter.device),
        online_adaptation=(
            config.online_adaptation and not config.offline_training
        ),
        offline_training=config.offline_training,
        update_summaries=tuple(updates),
        episode_summaries=tuple(episodes),
        stage_summaries=tuple(stage_summaries),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--rollout-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--show-prepare",
        action="store_true",
        help="show the pre-trial ready-pose initialization",
    )
    parser.add_argument(
        "--collision-mode",
        choices=("pad_only", "miss_backstop", "full"),
        default="miss_backstop",
    )
    parser.add_argument("--no-online-adaptation", action="store_true")
    parser.add_argument(
        "--offline-training",
        action="store_true",
        help="use multi-epoch PPO updates instead of deployment-safe online updates",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--load-checkpoint", action="store_true")
    parser.add_argument(
        "--curriculum-mode",
        choices=("adaptive", "balanced"),
        default="adaptive",
        help="adaptive promotion or equal round-robin sampling of all stages",
    )
    parser.add_argument(
        "--evaluation-suite",
        action="store_true",
        help="fixed-policy, balanced-stage evaluation on the requested seed",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_generalized_box_squeeze(
        GeneralizationRunConfig(
            episodes=args.episodes,
            rollout_size=args.rollout_size,
            random_seed=args.seed,
            device=args.device,
            online_adaptation=(
                not args.no_online_adaptation
                and not args.offline_training
                and not args.evaluation_suite
            ),
            offline_training=(
                args.offline_training and not args.evaluation_suite
            ),
            viewer=args.viewer,
            viewer_show_prepare=args.show_prepare,
            checkpoint_path=args.checkpoint,
            load_checkpoint=args.load_checkpoint,
            collision_mode=args.collision_mode,
            curriculum_mode=(
                "balanced" if args.evaluation_suite else args.curriculum_mode
            ),
        )
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
