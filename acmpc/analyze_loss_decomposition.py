"""Decompose the constraint ablation's already-written actor losses.

Nothing is re-run. The historical ``ppo_update_diagnostics`` recorded a single
``policy_loss`` field that is in fact the *total* actor loss:

    total_actor_loss = policy_surrogate_loss - entropy_coef * entropy

so both other terms follow from values that were logged alongside it:

    entropy_bonus         = -entropy_coef * entropy
    policy_surrogate_loss = total_actor_loss + entropy_coef * entropy

With entropy_coef=1e-3 and an entropy near -12.4, the entropy bonus alone is
~0.0124 -- the entire magnitude that was previously read as a surrogate loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ("D0", "D1", "D2", "D3", "M0")
SEEDS = (7, 17, 27)
EPISODE_MARKS = (0, 100, 200, 300, 399)
ENTROPY_COEF = 1e-3
TERMS = ("total_actor_loss", "entropy_bonus", "policy_surrogate_loss")


def _decompose(update: dict, entropy_coef: float) -> dict[str, float]:
    # New-format runs record the split directly; older ones only have the
    # contaminated total plus the entropy it was contaminated with.
    total = float(update.get("total_actor_loss", update["policy_loss"]))
    entropy = float(update["entropy"])
    coef = float(update.get("entropy_coef") or entropy_coef)
    return {
        "total_actor_loss": total,
        "entropy_bonus": -coef * entropy,
        "policy_surrogate_loss": total + coef * entropy,
    }


def _episode_statistics(
    episode: dict, entropy_coef: float
) -> dict[str, dict[str, float]]:
    updates = episode.get("ppo_update_diagnostics") or ()
    rows = [_decompose(update, entropy_coef) for update in updates]
    out: dict[str, dict[str, float]] = {}
    for term in TERMS:
        values = [row[term] for row in rows]
        out[term] = {
            "mean": statistics.mean(values) if values else float("nan"),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
            "updates": len(values),
        }
    return out


def _load(root: Path, condition: str, seed: int) -> list[dict]:
    path = root / condition / f"seed_{seed}" / "result.json"
    return json.loads(path.read_text())["episode_summaries"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "sweep_results" / "constraint_ablation_20260804" / "pilot",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "sweep_results" / "constraint_ablation_20260804",
    )
    parser.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF)
    args = parser.parse_args()

    series: dict[tuple[str, int], list[dict[str, dict[str, float]]]] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            episodes = _load(args.root, condition, seed)
            series[(condition, seed)] = [
                _episode_statistics(episode, args.entropy_coef) for episode in episodes
            ]

    csv_path = args.output_root / "constraint_ablation_loss_decomposition.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["condition", "seed", "episode", "term", "mean", "std", "min", "max", "updates"]
        )
        for (condition, seed), episodes in series.items():
            for mark in EPISODE_MARKS:
                if mark >= len(episodes):
                    continue
                for term in TERMS:
                    stats = episodes[mark][term]
                    writer.writerow(
                        [
                            condition,
                            seed,
                            mark,
                            term,
                            f"{stats['mean']:.12g}",
                            f"{stats['std']:.12g}",
                            f"{stats['min']:.12g}",
                            f"{stats['max']:.12g}",
                            stats["updates"],
                        ]
                    )

    summary_path = args.output_root / "constraint_ablation_loss_decomposition_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "episode", *[f"{t}_{s}" for t in TERMS for s in ("mean", "std")]])
        for condition in CONDITIONS:
            for mark in EPISODE_MARKS:
                row: list[object] = [condition, mark]
                for term in TERMS:
                    values = [
                        series[(condition, seed)][mark][term]["mean"]
                        for seed in SEEDS
                        if mark < len(series[(condition, seed)])
                    ]
                    row.append(f"{statistics.mean(values):.12g}")
                    row.append(
                        f"{statistics.stdev(values):.12g}" if len(values) > 1 else "0"
                    )
                writer.writerow(row)

    def _band(axis, condition: str, term: str, color: str, label: str) -> None:
        length = min(len(series[(condition, seed)]) for seed in SEEDS)
        xs = list(range(length))
        per_seed = [
            [series[(condition, seed)][i][term]["mean"] for i in xs] for seed in SEEDS
        ]
        means = [statistics.mean(v[i] for v in per_seed) for i in xs]
        lows = [min(v[i] for v in per_seed) for i in xs]
        highs = [max(v[i] for v in per_seed) for i in xs]
        axis.plot(xs, means, color=color, linewidth=1.2, label=label)
        axis.fill_between(xs, lows, highs, color=color, alpha=0.18, linewidth=0)

    colors = dict(zip(CONDITIONS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    for condition in CONDITIONS:
        _band(axes[0], condition, "total_actor_loss", colors[condition], condition)
        _band(axes[1], condition, "entropy_bonus", colors[condition], condition)
    axes[0].set_title("total_actor_loss (historically logged as policy_loss)")
    axes[1].set_title("entropy_bonus = -entropy_coef x entropy")
    for axis in axes:
        axis.set_xlabel("episode")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("The two curves overlap: the logged loss is almost entirely entropy bonus")
    figure.tight_layout()
    figure.savefig(args.output_root / "constraint_ablation_loss_decomposition.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 4.6))
    for condition in CONDITIONS:
        _band(axis, condition, "policy_surrogate_loss", colors[condition], condition)
    axis.set_title("policy_surrogate_loss = total - entropy_bonus (own scale)")
    axis.set_xlabel("episode")
    axis.set_ylabel("surrogate loss")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_root / "constraint_ablation_surrogate_loss.png", dpi=140)
    plt.close(figure)

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    for condition in CONDITIONS:
        stats = series[(condition, 7)][EPISODE_MARKS[-1]]
        print(
            f"{condition} seed=7 ep399: total={stats['total_actor_loss']['mean']:.6g} "
            f"entropy_bonus={stats['entropy_bonus']['mean']:.6g} "
            f"surrogate={stats['policy_surrogate_loss']['mean']:.6g}"
        )


if __name__ == "__main__":
    main()
