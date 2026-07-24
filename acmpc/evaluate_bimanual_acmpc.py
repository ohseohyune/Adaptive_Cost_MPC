"""Multi-seed statistical evaluation of the PPO+GAE bimanual AC-MPC demo.

Runs ``run_demo`` across several seeds (varying actor/critic initialization
and PPO exploration noise; the environment's initial conditions are held
fixed) and reports aggregate success rate and mean +/- std for the
continuous metrics in ``DemoSummary``. Intended for reporting a reproducible
results table rather than eyeballing a single run.

    python acmpc/evaluate_bimanual_acmpc.py --seeds 5
    python acmpc/evaluate_bimanual_acmpc.py --seed-list 7,11,23,42,101 --duration 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.main_bimanual_acmpc import DemoConfig, run_demo

# Fields from DemoSummary to aggregate as mean +/- std. Boolean/optional
# fields (success, phase, timings) are summarized separately.
_CONTINUOUS_FIELDS = (
    "left_contact_force_n",
    "right_contact_force_n",
    "final_endpoint_error_m",
    "online_updates",
    "total_transitions",
    "actor_weight_change_l2",
    "object_displacement_m",
)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def run_multiseed_evaluation(
    seeds: list[int],
    *,
    duration_s: float = 6.0,
    device: str = "cpu",
    exploration_std: float = 0.015,
    rollout_size: int = 32,
    offline_training: bool = False,
) -> dict:
    """Run one demo per seed and return per-seed and aggregate results."""

    per_seed: list[dict] = []
    for seed in seeds:
        summary = run_demo(
            DemoConfig(
                duration_s=duration_s,
                device=device,
                online_learning=True,
                exploration_std=exploration_std,
                rollout_size=rollout_size,
                offline_training=offline_training,
                seed=seed,
                viewer=False,
                log_path=None,
                checkpoint_path=None,
            )
        )
        per_seed.append({"seed": seed, **asdict(summary)})

    successes = sum(1 for row in per_seed if row["success"])
    aggregate: dict[str, dict[str, float]] = {}
    for field in _CONTINUOUS_FIELDS:
        mean, std = _mean_std([row[field] for row in per_seed])
        aggregate[field] = {"mean": mean, "std": std}

    grasp_times = [row["grasp_time_s"] for row in per_seed if row["grasp_time_s"] is not None]
    grasp_mean, grasp_std = _mean_std(grasp_times)

    return {
        "seeds": seeds,
        "episodes": len(per_seed),
        "successes": successes,
        "success_rate": successes / len(per_seed),
        "grasp_time_s_mean": grasp_mean,
        "grasp_time_s_std": grasp_std,
        "grasp_reached_count": len(grasp_times),
        "aggregate": aggregate,
        "per_seed": per_seed,
    }


def _print_report(result: dict) -> None:
    print(f"\n=== Bimanual AC-MPC multi-seed evaluation ({result['episodes']} seeds) ===\n")
    print(f"Success rate: {result['successes']}/{result['episodes']} "
          f"({result['success_rate'] * 100:.1f}%)")
    if result["grasp_reached_count"]:
        print(
            f"Grasp time (s), reached in {result['grasp_reached_count']}/"
            f"{result['episodes']}: {result['grasp_time_s_mean']:.3f} +/- "
            f"{result['grasp_time_s_std']:.3f}"
        )
    else:
        print("Grasp time (s): no seed reached a grasp")
    print()
    header = f"{'metric':<28}{'mean':>12}{'std':>12}"
    print(header)
    print("-" * len(header))
    for field, stats in result["aggregate"].items():
        print(f"{field:<28}{stats['mean']:>12.4f}{stats['std']:>12.4f}")
    print()
    print(f"{'seed':>6}  {'success':<8}{'phase':<14}{'grasp_t':>9}{'actor_delta':>13}")
    for row in result["per_seed"]:
        grasp_t = "-" if row["grasp_time_s"] is None else f"{row['grasp_time_s']:.2f}"
        print(
            f"{row['seed']:>6}  {str(row['success']):<8}{row['final_phase']:<14}"
            f"{grasp_t:>9}{row['actor_weight_change_l2']:>13.4f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=5, help="number of seeds, starting at --base-seed"
    )
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument(
        "--seed-list",
        type=str,
        default=None,
        help="comma-separated explicit seed list, overrides --seeds/--base-seed",
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--device", default="cpu", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--exploration-std", type=float, default=0.015)
    parser.add_argument("--rollout-size", type=int, default=32)
    parser.add_argument("--offline-training", action="store_true")
    parser.add_argument(
        "--out", default=str(ROOT / "sweep_results" / "bimanual_acmpc_multiseed.json")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seed_list:
        seeds = [int(value.strip()) for value in args.seed_list.split(",") if value.strip()]
    else:
        seeds = [args.base_seed + index for index in range(args.seeds)]

    result = run_multiseed_evaluation(
        seeds,
        duration_s=args.duration,
        device=args.device,
        exploration_std=args.exploration_std,
        rollout_size=args.rollout_size,
        offline_training=args.offline_training,
    )
    _print_report(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as stream:
        json.dump(result, stream, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
