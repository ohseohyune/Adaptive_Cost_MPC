"""Curriculum training loop for the AC-MPC ballistic box-catch pipeline.

Mirrors main_generalized_box_squeeze.py's CurriculumScheduler-driven training
across randomized box domains (see control/squeeze/generalization.py), but
wraps main_acmpc_box_catch.run_box_catch (the differentiable-MPC AC-MPC
track) instead of main_dynamic_box_squeeze.run_dynamic_side_squeeze (the
engineered box-squeeze track).

Unlike the box-squeeze track's outer PPOCostAdapter (one action per episode,
selecting engineered-gain multipliers), the AC-MPC track's learning already
happens *inside* run_box_catch, at the control-step level, via
OnlineActorCriticACMPC. So this loop's job is simpler: sample a domain from
the curriculum each episode, run one catch attempt with online_learning=True
and a checkpoint path shared across episodes (so the actor/critic keep
improving between calls instead of resetting), and feed the outcome back
into the scheduler to promote/demote curriculum stages.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.mpc.wandb_logger import build_eval_log, init_wandb
from control.squeeze import (
    CurriculumScheduler,
    DynamicSideSqueezeConfig,
    resolve_ballistic_launch_position,
    resolve_ballistic_launch_velocity,
)
from acmpc.main_acmpc_box_catch import AcmpcBoxCatchConfig, BoxCatchSummary, run_box_catch


@dataclass
class CurriculumBoxCatchConfig:
    # Effectively "run until stopped" -- Ctrl+C (foreground) or killing the
    # background task -- rather than a fixed budget. Set a smaller number
    # for a quick smoke test.
    episodes: int = 1_000_000
    random_seed: int = 7
    device: str = "cpu"
    online_learning: bool = True
    checkpoint_path: Optional[str] = str(
        ROOT / "sweep_results" / "acmpc_box_catch_curriculum.pt"
    )
    # A fresh run should not silently resume a stale checkpoint left over
    # from a previous experiment -- set True to intentionally continue one.
    load_checkpoint: bool = False
    curriculum_mode: str = "adaptive"  # or "balanced" (round-robins stages)
    # None (default) uses main_acmpc_box_catch.py's own validated table --
    # override for ablations (e.g. a deliberately degraded prior, to test
    # whether the learned residual actually compensates for a real gap).
    phase_priors: Optional[tuple] = None
    # 0.65 (default, matches AcmpcBoxCatchConfig) allows the residual actor
    # to move each weight up to +-65% from its phase prior; 0.0 forces pure
    # prior-only behavior (no learned deviation at all) -- for the
    # prior-only baseline arm of an ablation.
    weight_delta_fraction: float = 0.65
    log_path: Optional[str] = str(
        ROOT / "sweep_results" / "acmpc_box_catch_curriculum.json"
    )
    # How often (in episodes) to print a rolling-window progress line and
    # flush the JSON log -- so a long run that gets interrupted still has a
    # recent, readable record on disk, not just the checkpoint.
    progress_every: int = 20
    # W&B (see control/mpc/wandb_logger.py). This loop is the only place
    # that owns a run spanning multiple episodes -- each run_box_catch call
    # is handed this same logger instead of creating/finishing its own.
    use_wandb: bool = False
    wandb_run_name: Optional[str] = None
    wandb_log_interval: int = 10


@dataclass(frozen=True)
class CurriculumBoxCatchEpisode:
    episode: int
    stage_name: str
    stage_index: int
    mass: float
    friction: float
    half_size: tuple[float, float, float]
    success: bool
    safety_violation: bool
    # Physics-grounded "stable hold" criterion (force/speed/angular-speed
    # all within margin for required_hold_s) -- see BoxCatchSummary.
    # strict_success in main_acmpc_box_catch.py; independent of the official
    # success/hold_timer used for training and curriculum promotion.
    strict_success: bool
    hold_time_s: float
    final_box_speed_mps: float
    online_updates: int
    actor_weight_change_l2: float
    failure_reason: str
    # Standard RL training diagnostics (see BoxCatchSummary) -- what you'd
    # normally plot against episode/step count to judge whether training is
    # actually progressing, not just whether the latest episode succeeded.
    total_reward: float
    mean_reward_per_step: float
    actor_loss: float
    critic_loss: float
    entropy: float
    approximate_kl: float
    # Phase-structure/cost-contribution diagnostics (see BoxCatchSummary in
    # main_acmpc_box_catch.py) -- rolled up per-episode here so a curriculum
    # run's JSON log can be aggregated across episodes (e.g. mean dwell time
    # per stage) the same way the reward/loss diagnostics above already are.
    phase_transition_count: int
    intercept_dwell_s: float
    pre_impact_dwell_s: float
    capture_dwell_s: float
    hold_dwell_s: float
    blend_active_fraction: float
    velocity_saturation_fraction: float
    mean_mpc_solve_time_s: float
    weight_lower_bound_hit_fraction: float
    weight_upper_bound_hit_fraction: float
    mean_actor_output_variation: float


@dataclass(frozen=True)
class StagePerformance:
    stage: str
    episodes: int
    successes: int
    success_rate: float
    safety_violations: int


@dataclass(frozen=True)
class CurriculumBoxCatchSummary:
    episodes: int
    successes: int
    success_rate: float
    safety_violations: int
    stages_visited: tuple[str, ...]
    final_stage: str
    stage_summaries: tuple[StagePerformance, ...]
    episode_summaries: tuple[CurriculumBoxCatchEpisode, ...]


def _is_safety_violation(
    summary: BoxCatchSummary, squeeze_cfg: DynamicSideSqueezeConfig
) -> bool:
    return bool(
        not np.isfinite(summary.first_contact_peak_force_n)
        or summary.first_contact_peak_force_n
        > squeeze_cfg.first_contact_force_limit + 1e-6
        or "emergency" in summary.failure_reason.lower()
    )


def run_curriculum_box_catch(
    config: Optional[CurriculumBoxCatchConfig] = None,
    *,
    curriculum: Optional[CurriculumScheduler] = None,
) -> CurriculumBoxCatchSummary:
    config = config or CurriculumBoxCatchConfig()
    if config.episodes <= 0:
        raise ValueError("episodes must be positive")
    if config.curriculum_mode not in {"adaptive", "balanced"}:
        raise ValueError("curriculum_mode must be adaptive or balanced")

    rng = np.random.default_rng(config.random_seed)
    curriculum = curriculum or CurriculumScheduler()

    checkpoint_path = config.checkpoint_path
    if (
        checkpoint_path
        and not config.load_checkpoint
        and Path(checkpoint_path).exists()
    ):
        Path(checkpoint_path).unlink()

    base_squeeze = DynamicSideSqueezeConfig(random_seed=config.random_seed)
    episodes: list[CurriculumBoxCatchEpisode] = []
    stages_visited: list[str] = []

    run_name = config.wandb_run_name or f"phase-fixed-residual-seed-{config.random_seed}"
    wandb_logger = init_wandb(
        enabled=config.use_wandb,
        run_name=run_name,
        config={"seed": config.random_seed, "weight_mode": "phase-fixed-residual"},
    )
    global_step = 0

    try:
        for episode_index in range(config.episodes):
            if config.curriculum_mode == "balanced":
                stage_index = episode_index % len(curriculum.stages)
                domain = curriculum.stages[stage_index].sample(rng, stage_index)
            else:
                domain = curriculum.sample(rng)

            # Route the curriculum's sampled launch velocity through the same
            # reachability-aware resolver main_dynamic_box_squeeze.py uses,
            # rather than the raw sample -- resolve_ballistic_launch_velocity
            # can adjust the vertical component so the arc actually reaches the
            # configured catch height (see resolve_ballistic_launch_velocity in
            # control/squeeze/ballistic.py).
            sampled_launch_velocity = np.asarray(domain.launch_velocity, dtype=float)
            launch_position = resolve_ballistic_launch_position(
                base_squeeze, sampled_launch_velocity
            )
            launch_velocity = resolve_ballistic_launch_velocity(
                base_squeeze, sampled_launch_velocity, launch_position=launch_position
            )

            if domain.stage_name not in stages_visited:
                stages_visited.append(domain.stage_name)

            episode_seed = config.random_seed + episode_index
            squeeze = DynamicSideSqueezeConfig(
                random_seed=episode_seed,
                box_half_y=domain.half_size[1],
                launch_velocity_low=tuple(float(v) for v in launch_velocity),
                launch_velocity_high=tuple(float(v) for v in launch_velocity),
            )
            episode_config = AcmpcBoxCatchConfig(
                seed=episode_seed,
                device=config.device,
                online_learning=config.online_learning,
                squeeze=squeeze,
                domain_parameters=domain,
                checkpoint_path=checkpoint_path,
                weight_delta_fraction=config.weight_delta_fraction,
                wandb_log_interval=config.wandb_log_interval,
                **({"phase_priors": config.phase_priors} if config.phase_priors else {}),
            )
            summary = run_box_catch(
                episode_config,
                wandb_logger=wandb_logger,
                global_step_start=global_step,
            )
            global_step += summary.total_transitions
            safety_violation = _is_safety_violation(summary, squeeze)
            curriculum.record(summary.success, safety_violation=safety_violation)

            episodes.append(
                CurriculumBoxCatchEpisode(
                    episode=episode_index,
                    stage_name=domain.stage_name,
                    stage_index=domain.stage_index,
                    mass=domain.mass,
                    friction=domain.friction,
                    half_size=domain.half_size,
                    success=summary.success,
                    safety_violation=safety_violation,
                    strict_success=summary.strict_success,
                    hold_time_s=summary.hold_time_s,
                    final_box_speed_mps=summary.final_box_speed_mps,
                    online_updates=summary.online_updates,
                    actor_weight_change_l2=summary.actor_weight_change_l2,
                    failure_reason=summary.failure_reason,
                    total_reward=summary.total_reward,
                    mean_reward_per_step=summary.mean_reward_per_step,
                    actor_loss=summary.mean_actor_loss,
                    critic_loss=summary.mean_critic_loss,
                    entropy=summary.mean_entropy,
                    approximate_kl=summary.mean_approximate_kl,
                    phase_transition_count=summary.phase_transition_count,
                    intercept_dwell_s=summary.intercept_dwell_s,
                    pre_impact_dwell_s=summary.pre_impact_dwell_s,
                    capture_dwell_s=summary.capture_dwell_s,
                    hold_dwell_s=summary.hold_dwell_s,
                    blend_active_fraction=summary.blend_active_fraction,
                    velocity_saturation_fraction=summary.velocity_saturation_fraction,
                    mean_mpc_solve_time_s=summary.mean_mpc_solve_time_s,
                    weight_lower_bound_hit_fraction=summary.weight_lower_bound_hit_fraction,
                    weight_upper_bound_hit_fraction=summary.weight_upper_bound_hit_fraction,
                    mean_actor_output_variation=summary.mean_actor_output_variation,
                )
            )

            if (episode_index + 1) % config.progress_every == 0:
                window = episodes[max(0, len(episodes) - config.progress_every):]
                rate = sum(e.success for e in window) / len(window)
                mean_reward = float(np.mean([e.total_reward for e in window]))
                mean_kl = float(np.mean([e.approximate_kl for e in window]))
                mean_actor_loss = float(np.mean([e.actor_loss for e in window]))
                # eval/* -- this loop has no separate frozen-policy eval
                # phase (every episode both trains and is scored), so the
                # rolling window just completed is treated as one evaluation
                # batch. Reuses each episode's already-computed
                # success/safety_violation/strict_success/hold_time_s --
                # no new judgment logic.
                impact_safe_rate = sum(not e.safety_violation for e in window) / len(window)
                stable_hold_rate = sum(e.strict_success for e in window) / len(window)
                mean_hold_time = float(np.mean([e.hold_time_s for e in window]))
                wandb_logger.log(
                    build_eval_log(
                        success_rate=rate,
                        impact_safe_rate=impact_safe_rate,
                        stable_hold_rate=stable_hold_rate,
                        mean_hold_time=mean_hold_time,
                    ),
                    step=global_step,
                )
                print(
                    f"episodes {window[0].episode}-{window[-1].episode}: "
                    f"success_rate={rate:.3f} mean_reward={mean_reward:.2f} "
                    f"mean_actor_loss={mean_actor_loss:.4f} mean_kl={mean_kl:.4f} "
                    f"stage={curriculum.stage.name} scheduler_success_rate={curriculum.success_rate:.3f}",
                    flush=True,
                )
                if config.log_path:
                    log_path = Path(config.log_path)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("w") as stream:
                        json.dump(
                            {
                                "episodes": len(episodes),
                                "successes": sum(e.success for e in episodes),
                                "success_rate": sum(e.success for e in episodes) / len(episodes),
                                "stages_visited": stages_visited,
                                "final_stage": curriculum.stage.name,
                                "episode_summaries": [asdict(e) for e in episodes],
                            },
                            stream,
                            indent=2,
                        )
    finally:
        wandb_logger.finish()

    successes = sum(1 for episode in episodes if episode.success)
    stage_summaries: list[StagePerformance] = []
    for stage in curriculum.stages:
        stage_episodes = [e for e in episodes if e.stage_name == stage.name]
        if not stage_episodes:
            continue
        stage_successes = sum(1 for e in stage_episodes if e.success)
        stage_summaries.append(
            StagePerformance(
                stage=stage.name,
                episodes=len(stage_episodes),
                successes=stage_successes,
                success_rate=stage_successes / len(stage_episodes),
                safety_violations=sum(1 for e in stage_episodes if e.safety_violation),
            )
        )

    result = CurriculumBoxCatchSummary(
        episodes=len(episodes),
        successes=successes,
        success_rate=successes / len(episodes) if episodes else 0.0,
        safety_violations=sum(1 for e in episodes if e.safety_violation),
        stages_visited=tuple(stages_visited),
        final_stage=curriculum.stage.name,
        stage_summaries=tuple(stage_summaries),
        episode_summaries=tuple(episodes),
    )

    if config.log_path:
        log_path = Path(config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as stream:
            json.dump(
                {
                    "episodes": result.episodes,
                    "successes": result.successes,
                    "success_rate": result.success_rate,
                    "safety_violations": result.safety_violations,
                    "stages_visited": result.stages_visited,
                    "final_stage": result.final_stage,
                    "stage_summaries": [asdict(s) for s in result.stage_summaries],
                    "episode_summaries": [asdict(e) for e in result.episode_summaries],
                },
                stream,
                indent=2,
            )

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--curriculum-mode", default="adaptive", choices=["adaptive", "balanced"])
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "sweep_results" / "acmpc_box_catch_curriculum.pt"),
    )
    parser.add_argument("--load-checkpoint", action="store_true")
    parser.add_argument(
        "--log", default=str(ROOT / "sweep_results" / "acmpc_box_catch_curriculum.json")
    )
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_curriculum_box_catch(
        CurriculumBoxCatchConfig(
            episodes=args.episodes,
            random_seed=args.seed,
            device=args.device,
            curriculum_mode=args.curriculum_mode,
            checkpoint_path=args.checkpoint,
            load_checkpoint=args.load_checkpoint,
            log_path=args.log,
            progress_every=args.progress_every,
            use_wandb=args.use_wandb,
            wandb_run_name=args.wandb_run_name,
        )
    )
    print(
        f"episodes={summary.episodes} successes={summary.successes} "
        f"success_rate={summary.success_rate:.3f} "
        f"safety_violations={summary.safety_violations} "
        f"stages_visited={summary.stages_visited} final_stage={summary.final_stage}"
    )
    for stage in summary.stage_summaries:
        print(
            f"  {stage.stage}: {stage.successes}/{stage.episodes} "
            f"({stage.success_rate:.3f}), safety_violations={stage.safety_violations}"
        )


if __name__ == "__main__":
    main()
