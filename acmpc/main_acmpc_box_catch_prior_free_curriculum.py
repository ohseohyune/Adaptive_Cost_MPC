"""Curriculum training for the prior-free (condition 3) cost actor.

Straight online training of PriorFreeCostActor on the full-difficulty
"warmup" ballistic catch (see acmpc_box_catch_integration_status.md) found
zero successes across 2,199+ episodes / 1.2M+ transitions: the box always
arrives too fast, from too far away, for an unstructured cost-weight
network to ever stumble into a stable catch, so the reward signal for
*holding* contact (which only fires after contact happens) is essentially
never experienced. Two changes address this directly, both scoped to this
training script only (main_acmpc_box_catch.py's defaults are untouched, so
conditions 1/2 are unaffected):

1. A difficulty ramp: episodes start with the box launched much closer
   (short target_flight_time_s) and slower (scaled-down launch velocity)
   than the nominal "warmup" curriculum stage, linearly increasing to the
   full nominal difficulty over `ramp_episodes`. This isolates "learn to
   hold contact once touching" from "also solve interception from a hard
   ballistic arc" early in training, then gradually reintroduces the harder
   approach.
2. `hold_reward_scale` (see main_acmpc_box_catch._reward) amplifies only
   the contact/hold-specific reward terms, so on the rare early episodes
   where contact does happen, the gradient toward "keep it" is much
   stronger relative to the dense per-step tracking/effort terms that fire
   every step regardless of phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.mpc import ACMPCRolloutBuffer
from control.mpc.wandb_logger import init_wandb
from control.squeeze import DynamicSideSqueezeConfig, default_curriculum
from acmpc.main_acmpc_box_catch import AcmpcBoxCatchConfig, BoxCatchSummary, run_box_catch

_NOMINAL_TARGET_FLIGHT_TIME_S = 0.56  # DynamicSideSqueezeConfig's own default


@dataclass
class PriorFreeCurriculumConfig:
    episodes: int = 5000
    ramp_episodes: int = 3000
    base_seed: int = 7
    device: str = "cuda"
    hold_reward_scale: float = 8.0
    # Stop after this many wall-clock hours regardless of `episodes` --
    # episode length varies a lot (a quick early failure vs. a full 5s hold
    # attempt), so a fixed episode count is a poor way to target a fixed
    # time budget. None (default) means episodes is the only stopping
    # condition, unchanged from before this field existed.
    max_hours: Optional[float] = None
    easy_flight_time_s: float = 0.15
    easy_speed_scale: float = 0.35
    # Box mass sweep (2026-07-31) found the trained policy collapses for
    # box mass above ~0.70kg (50% -> 5% -> 0% by 1.00kg) because mass was
    # pinned at the nominal 0.50kg the whole time training (domain_parameters
    # was never passed to AcmpcBoxCatchConfig here, so apply_box_domain_
    # randomization never ran). Ramping the sampled mass range alongside the
    # existing speed/distance ramp closes that gap without touching
    # conditions 1/2. hard_mass_low/high are the endpoints at difficulty=1.0;
    # at difficulty=0.0 the range collapses to the nominal 0.50kg point.
    hard_mass_low: float = 0.35
    hard_mass_high: float = 1.00
    # When continuing training from a checkpoint that already handles full
    # speed/flight-time (e.g. expl_v2), ramping mass on the SAME difficulty
    # schedule as speed compounds: at difficulty~0.7 "near-max speed" and
    # "near-max mass" coincide, which is harder than either alone and
    # collapsed bilateral contact to 0% for 2000+ episodes (see
    # acmpc_paper_reproduction_reframe.md's mass-sweep follow-up, 2026-07-31).
    # Setting this holds speed/flight-time pinned at their full-difficulty
    # values throughout, so only mass ramps -- appropriate for a warm-started
    # continuation run, not for training from scratch.
    fix_speed_at_full: bool = False
    actor_lr: float = 2e-4
    critic_lr: float = 5e-4
    checkpoint_path: Optional[str] = str(
        ROOT / "sweep_results" / "acmpc_box_catch_prior_free_curriculum.pt"
    )
    load_checkpoint: bool = False
    log_path: Optional[str] = str(
        ROOT / "sweep_results" / "acmpc_box_catch_prior_free_curriculum.json"
    )
    progress_every: int = 50
    # See main_acmpc_box_catch_curriculum.py's identical fields -- the
    # batch-structure diagnostic (2026-07-27) found the default
    # rollout_size=16 leaves the critic's explained_variance near/below 0
    # (useless), while a larger accumulated-across-episodes batch reached
    # ~0.82. Condition 3 (this script) starts from scratch with no prior at
    # all, so a healthier critic signal plausibly matters even more here
    # than for the residual actor.
    rollout_size: int = 16
    accumulate_rollout_across_episodes: bool = False
    # See AcmpcBoxCatchConfig's identical fields -- raised to keep
    # exploration alive longer after the batchfix-v1 run showed log_std
    # collapsing to its floor by ~episode 800, with cost weights still
    # essentially identical to cold-start init across all 4 phases even at
    # episode 2700+ (critic well-converged, explained_variance ~0.99, but no
    # phase differentiation ever discovered). Defaults match condition-2's
    # validated behavior (exact no-op unless overridden).
    entropy_coef: float = 1e-3
    log_std_min: float = -5.0
    log_std_max: float = -1.8
    use_wandb: bool = False
    wandb_run_name: Optional[str] = None
    wandb_log_interval: int = 10


def _difficulty(episode_index: int, ramp_episodes: int) -> float:
    return float(np.clip(episode_index / max(ramp_episodes, 1), 0.0, 1.0))


def _curriculum_squeeze_config(
    config: PriorFreeCurriculumConfig, episode_index: int, seed: int
):
    stage = default_curriculum()[0]  # "warmup" -- only used for its launch_velocity distribution
    domain = stage.sample(np.random.default_rng(seed), 0)
    # Pin size/aspect/friction to the MJCF's exact nominal values (see
    # scene_ffw_sg2_fixed_base_box_dynamic_squeeze.xml's dynamic_box_geom:
    # size 0.055/0.150/0.055, friction 1.2) instead of "warmup"'s sampled
    # values. The first mass-curriculum attempt (2026-07-31) let stage.sample
    # jitter size/friction from episode 0 on top of a checkpoint that had
    # only ever seen the exact fixed nominal geometry, and collapsed to 0%
    # hold success by episode ~200 -- isolating mass as the only new
    # variable avoids stacking an extra, unintended distribution shift.
    domain = replace(
        domain,
        half_size=(0.055, 0.150, 0.055),
        overall_size_scale=1.0,
        aspect_scale=(1.0, 1.0, 1.0),
        friction=1.2,
    )
    difficulty = _difficulty(episode_index, config.ramp_episodes)
    speed_difficulty = 1.0 if config.fix_speed_at_full else difficulty
    flight_time = config.easy_flight_time_s + speed_difficulty * (
        _NOMINAL_TARGET_FLIGHT_TIME_S - config.easy_flight_time_s
    )
    speed_scale = config.easy_speed_scale + speed_difficulty * (1.0 - config.easy_speed_scale)
    launch_velocity = tuple(float(v) * speed_scale for v in domain.launch_velocity)
    mass_low = 0.50 - difficulty * (0.50 - config.hard_mass_low)
    mass_high = 0.50 + difficulty * (config.hard_mass_high - 0.50)
    domain = replace(domain, mass=float(np.random.default_rng(seed + 1).uniform(mass_low, mass_high)))
    domain.validate()
    squeeze = DynamicSideSqueezeConfig(
        random_seed=seed,
        box_half_y=domain.half_size[1],
        launch_velocity_low=launch_velocity,
        launch_velocity_high=launch_velocity,
        target_flight_time_s=flight_time,
    )
    return squeeze, domain, difficulty


def run_prior_free_curriculum(
    config: Optional[PriorFreeCurriculumConfig] = None,
) -> list[dict]:
    config = config or PriorFreeCurriculumConfig()
    checkpoint_path = config.checkpoint_path
    if (
        checkpoint_path
        and not config.load_checkpoint
        and Path(checkpoint_path).exists()
    ):
        Path(checkpoint_path).unlink()

    import time

    t_start = time.time()
    max_seconds = (
        config.max_hours * 3600.0 if config.max_hours is not None else float("inf")
    )

    wandb_logger = init_wandb(
        enabled=config.use_wandb,
        run_name=config.wandb_run_name or "prior-free-curriculum",
        config={"seed": config.base_seed, "weight_mode": "prior-free"},
    )
    global_step = 0
    shared_rollout_buffer = (
        ACMPCRolloutBuffer() if config.accumulate_rollout_across_episodes else None
    )

    results: list[dict] = []
    episode_index = 0
    while episode_index < config.episodes and (time.time() - t_start) < max_seconds:
        seed = config.base_seed + episode_index
        squeeze, domain, difficulty = _curriculum_squeeze_config(config, episode_index, seed)
        cfg = AcmpcBoxCatchConfig(
            seed=seed,
            device=config.device,
            online_learning=True,
            squeeze=squeeze,
            domain_parameters=domain,
            use_prior_free_actor=True,
            maximum_cumulative_actor_delta=None,
            hold_reward_scale=config.hold_reward_scale,
            checkpoint_path=checkpoint_path,
            rollout_size=config.rollout_size,
            wandb_log_interval=config.wandb_log_interval,
            entropy_coef=config.entropy_coef,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
        )
        summary = run_box_catch(
            cfg,
            wandb_logger=wandb_logger,
            global_step_start=global_step,
            rollout_buffer=shared_rollout_buffer,
        )
        global_step += summary.control_step_count
        results.append(
            dict(
                episode=episode_index,
                difficulty=difficulty,
                success=summary.success,
                hold=summary.hold_time_s,
                bilateral_contact_time_s=summary.bilateral_contact_time_s,
                final_speed=summary.final_box_speed_mps,
                failure_reason=summary.failure_reason,
                transitions=summary.total_transitions,
            )
        )
        if (episode_index + 1) % config.progress_every == 0:
            window = results[max(0, episode_index + 1 - config.progress_every) :]
            rate = sum(r["success"] for r in window) / len(window)
            bilateral_rate = sum(
                r["bilateral_contact_time_s"] is not None for r in window
            ) / len(window)
            elapsed_hours = (time.time() - t_start) / 3600.0
            print(
                f"episodes {window[0]['episode']}-{window[-1]['episode']}: "
                f"difficulty={difficulty:.2f} success_rate={rate:.3f} "
                f"bilateral_contact_rate={bilateral_rate:.3f} "
                f"elapsed={elapsed_hours:.2f}h",
                flush=True,
            )
            if config.log_path:
                log_path = Path(config.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w") as stream:
                    json.dump(results, stream, indent=2)
        episode_index += 1

    if config.log_path:
        log_path = Path(config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as stream:
            json.dump(results, stream, indent=2)

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--ramp-episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hold-reward-scale", type=float, default=8.0)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--easy-flight-time", type=float, default=0.15)
    parser.add_argument("--easy-speed-scale", type=float, default=0.35)
    parser.add_argument("--hard-mass-low", type=float, default=0.35)
    parser.add_argument("--hard-mass-high", type=float, default=1.00)
    parser.add_argument("--fix-speed-at-full", action="store_true")
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "sweep_results" / "acmpc_box_catch_prior_free_curriculum.pt"),
    )
    parser.add_argument("--load-checkpoint", action="store_true")
    parser.add_argument(
        "--log",
        default=str(ROOT / "sweep_results" / "acmpc_box_catch_prior_free_curriculum.json"),
    )
    parser.add_argument("--rollout-size", type=int, default=16)
    parser.add_argument("--accumulate-rollout-across-episodes", action="store_true")
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--log-std-min", type=float, default=-5.0)
    parser.add_argument("--log-std-max", type=float, default=-1.8)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_prior_free_curriculum(
        PriorFreeCurriculumConfig(
            episodes=args.episodes,
            ramp_episodes=args.ramp_episodes,
            base_seed=args.seed,
            device=args.device,
            hold_reward_scale=args.hold_reward_scale,
            easy_flight_time_s=args.easy_flight_time,
            easy_speed_scale=args.easy_speed_scale,
            hard_mass_low=args.hard_mass_low,
            hard_mass_high=args.hard_mass_high,
            fix_speed_at_full=args.fix_speed_at_full,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            progress_every=args.progress_every,
            checkpoint_path=args.checkpoint,
            load_checkpoint=args.load_checkpoint,
            log_path=args.log,
            max_hours=args.max_hours,
            rollout_size=args.rollout_size,
            accumulate_rollout_across_episodes=args.accumulate_rollout_across_episodes,
            entropy_coef=args.entropy_coef,
            log_std_min=args.log_std_min,
            log_std_max=args.log_std_max,
            use_wandb=args.use_wandb,
            wandb_run_name=args.wandb_run_name,
        )
    )
    overall = sum(r["success"] for r in results) / len(results)
    print(f"overall: {sum(r['success'] for r in results)}/{len(results)} = {overall:.3f}")


if __name__ == "__main__":
    main()
