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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    checkpoint_path: Optional[str] = str(
        ROOT / "sweep_results" / "acmpc_box_catch_prior_free_curriculum.pt"
    )
    load_checkpoint: bool = False
    log_path: Optional[str] = str(
        ROOT / "sweep_results" / "acmpc_box_catch_prior_free_curriculum.json"
    )
    progress_every: int = 50


def _difficulty(episode_index: int, ramp_episodes: int) -> float:
    return float(np.clip(episode_index / max(ramp_episodes, 1), 0.0, 1.0))


def _curriculum_squeeze_config(
    config: PriorFreeCurriculumConfig, episode_index: int, seed: int
) -> DynamicSideSqueezeConfig:
    stage = default_curriculum()[0]  # "warmup" -- fixed mass/friction/size, only speed/distance ramp here
    domain = stage.sample(np.random.default_rng(seed), 0)
    difficulty = _difficulty(episode_index, config.ramp_episodes)
    flight_time = config.easy_flight_time_s + difficulty * (
        _NOMINAL_TARGET_FLIGHT_TIME_S - config.easy_flight_time_s
    )
    speed_scale = config.easy_speed_scale + difficulty * (1.0 - config.easy_speed_scale)
    launch_velocity = tuple(float(v) * speed_scale for v in domain.launch_velocity)
    return DynamicSideSqueezeConfig(
        random_seed=seed,
        box_half_y=domain.half_size[1],
        launch_velocity_low=launch_velocity,
        launch_velocity_high=launch_velocity,
        target_flight_time_s=flight_time,
    ), difficulty


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

    results: list[dict] = []
    episode_index = 0
    while episode_index < config.episodes and (time.time() - t_start) < max_seconds:
        seed = config.base_seed + episode_index
        squeeze, difficulty = _curriculum_squeeze_config(config, episode_index, seed)
        cfg = AcmpcBoxCatchConfig(
            seed=seed,
            device=config.device,
            online_learning=True,
            squeeze=squeeze,
            use_prior_free_actor=True,
            maximum_cumulative_actor_delta=None,
            hold_reward_scale=config.hold_reward_scale,
            checkpoint_path=checkpoint_path,
        )
        summary = run_box_catch(cfg)
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
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "sweep_results" / "acmpc_box_catch_prior_free_curriculum.pt"),
    )
    parser.add_argument("--load-checkpoint", action="store_true")
    parser.add_argument(
        "--log",
        default=str(ROOT / "sweep_results" / "acmpc_box_catch_prior_free_curriculum.json"),
    )
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
            checkpoint_path=args.checkpoint,
            load_checkpoint=args.load_checkpoint,
            log_path=args.log,
            max_hours=args.max_hours,
        )
    )
    overall = sum(r["success"] for r in results) / len(results)
    print(f"overall: {sum(r['success'] for r in results)}/{len(results)} = {overall:.3f}")


if __name__ == "__main__":
    main()
