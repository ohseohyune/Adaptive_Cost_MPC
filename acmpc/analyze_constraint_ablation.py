"""Aggregate D0/D1/D2/D3/M0 pilot JSON into tables, plots, and conclusions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("D0", "D1", "D2", "D3", "M0")
COMPARISONS = (
    ("D0", "D1", "cumulative cap 제거"),
    ("D1", "D2", "exp_residual + weight clip 제거"),
    ("D2", "D3", "online actor cap 제거"),
    ("D3", "M0", "target KL 제거"),
)


def _mean(values) -> float:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _percentile(values, percentile: float) -> float:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.percentile(finite, percentile)) if finite else float("nan")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cap_reach_episode(episodes: list[dict], fraction: float) -> int | None:
    cap = 0.4
    for episode in episodes:
        if episode.get("cumulative_actor_parameter_delta", 0.0) >= fraction * cap - 1e-6:
            return int(episode["episode"]) + 1
    return None


def _run_row(condition: str, seed: int, data: dict) -> dict:
    episodes = data.get("episode_summaries", [])
    evaluations = data.get("evaluation_summaries", [])
    final_evaluation = evaluations[-1] if evaluations else {}
    updates = [
        update
        for episode in episodes
        for update in episode.get("ppo_update_diagnostics", [])
    ]
    final_episode = episodes[-1] if episodes else {}
    raw_deltas = [update.get("raw_actor_delta") for update in updates]
    applied_deltas = [update.get("applied_actor_delta") for update in updates]
    applied_ratios = [update.get("applied_raw_ratio") for update in updates]
    kls = [update.get("approximate_kl") for update in updates]
    return {
        "condition": condition,
        "seed": seed,
        "complete": int(data.get("episodes", 0) == 400 and len(evaluations) == 8),
        "training_episodes": data.get("episodes", 0),
        "training_success_rate": data.get("success_rate", float("nan")),
        "training_safety_violation_count": data.get(
            "safety_violations", float("nan")
        ),
        "evaluation_success_rate_all_checkpoints": _mean(
            evaluation.get("success_rate") for evaluation in evaluations
        ),
        "evaluation_success_rate": final_evaluation.get("success_rate", float("nan")),
        "evaluation_impact_safe_rate": final_evaluation.get(
            "impact_safe_rate", float("nan")
        ),
        "evaluation_stable_hold_rate": final_evaluation.get(
            "stable_hold_rate", float("nan")
        ),
        "evaluation_bilateral_contact_rate": final_evaluation.get(
            "bilateral_contact_rate", float("nan")
        ),
        "evaluation_box_drop_rate": final_evaluation.get("box_drop_rate", float("nan")),
        "evaluation_workspace_failure_rate": final_evaluation.get(
            "workspace_failure_rate", float("nan")
        ),
        "evaluation_mean_first_contact_peak_force_n": final_evaluation.get(
            "mean_first_contact_peak_force_n", float("nan")
        ),
        "evaluation_mean_episode_maximum_contact_force_n": final_evaluation.get(
            "mean_episode_maximum_contact_force_n", float("nan")
        ),
        "evaluation_force_over_18n_episode_rate": final_evaluation.get(
            "force_over_18n_episode_rate", float("nan")
        ),
        "evaluation_force_over_36n_episode_rate": final_evaluation.get(
            "force_over_36n_episode_rate", float("nan")
        ),
        "evaluation_force_over_18n_step_fraction": final_evaluation.get(
            "mean_force_over_18n_step_fraction", float("nan")
        ),
        "evaluation_force_over_36n_step_fraction": final_evaluation.get(
            "mean_force_over_36n_step_fraction", float("nan")
        ),
        "actor_net_displacement": final_episode.get(
            "cumulative_actor_parameter_delta", float("nan")
        ),
        "actor_path_length": final_episode.get("actor_parameter_path_length", float("nan")),
        "actor_path_displacement_ratio": final_episode.get(
            "actor_path_displacement_ratio", float("nan")
        ),
        "mean_abs_actor_residual": _mean(
            episode.get("mean_abs_actor_residual") for episode in episodes
        ),
        "raw_update_mean": _mean(raw_deltas),
        "raw_update_p90": _percentile(raw_deltas, 90),
        "raw_update_p99": _percentile(raw_deltas, 99),
        "raw_update_max": _percentile(raw_deltas, 100),
        "applied_update_mean": _mean(applied_deltas),
        "applied_raw_ratio_mean": _mean(applied_ratios),
        "raw_applied_ratio_mean": _mean(
            float(raw) / (float(applied) + 1e-12)
            for raw, applied in zip(raw_deltas, applied_deltas)
            if raw is not None and applied is not None
        ),
        "cumulative_cap_reach_80_episode": _cap_reach_episode(episodes, 0.80)
        if condition == "D0"
        else None,
        "cumulative_cap_reach_95_episode": _cap_reach_episode(episodes, 0.95)
        if condition == "D0"
        else None,
        "cumulative_cap_reach_100_episode": _cap_reach_episode(episodes, 1.00)
        if condition == "D0"
        else None,
        "cumulative_projection_count": sum(
            episode.get("cumulative_delta_clip_count", 0) for episode in episodes
        ),
        "cumulative_projection_removed": sum(
            episode.get("cumulative_delta_removed", 0.0) for episode in episodes
        ),
        "online_cap_count": sum(
            episode.get("online_delta_clip_count", 0) for episode in episodes
        ),
        "online_cap_removed": sum(
            episode.get("online_delta_removed", 0.0) for episode in episodes
        ),
        "target_kl_early_stop_count": sum(
            episode.get("target_kl_stop_count", 0) for episode in episodes
        ),
        "target_kl_early_stop_rate": (
            sum(bool(update.get("target_kl_stopped")) for update in updates) / len(updates)
            if updates
            else 0.0
        ),
        "ppo_clip_fraction": _mean(
            update.get("ppo_clip_fraction") for update in updates
        ),
        "weight_min_clip_count": sum(
            episode.get("weight_lower_clip_count", 0) for episode in episodes
        ),
        "weight_max_clip_count": sum(
            episode.get("weight_upper_clip_count", 0) for episode in episodes
        ),
        "weight_clip_removed_episode_mean_sum": sum(
            episode.get("mean_weight_clip_removed", 0.0)
            for episode in episodes
        ),
        "weight_clip_removed_max": _percentile(
            (episode.get("max_weight_clip_removed") for episode in episodes), 100
        ),
        "kl_projection_count": sum(
            episode.get("kl_projection_count", 0) for episode in episodes
        ),
        "kl_rollback_count": sum(
            episode.get("kl_rollback_count", 0) for episode in episodes
        ),
        "kl_projection_removed": sum(
            episode.get("kl_projection_removed", 0.0) for episode in episodes
        ),
        "policy_loss": _mean(update.get("policy_loss") for update in updates),
        "value_loss": _mean(update.get("value_loss") for update in updates),
        "explained_variance": _mean(
            update.get("explained_variance") for update in updates
        ),
        "approximate_kl_mean": _mean(kls),
        "approximate_kl_p95": _percentile(kls, 95),
        "approximate_kl_max": _percentile(kls, 100),
        "entropy": _mean(update.get("entropy") for update in updates),
        "final_raw_log_std_mean": final_episode.get("raw_log_std_mean", float("nan")),
        "advantage_mean": _mean(update.get("advantage_mean") for update in updates),
        "advantage_std": _mean(update.get("advantage_std") for update in updates),
        "actor_grad_norm": _mean(update.get("actor_grad_norm") for update in updates),
        "critic_grad_norm": _mean(update.get("critic_grad_norm") for update in updates),
        "mean_executed_ppo_epochs": _mean(update.get("epochs") for update in updates),
        "maximum_hessian_condition_number": _percentile(
            (episode.get("maximum_hessian_condition_number") for episode in episodes), 100
        ),
        "p95_episode_maximum_hessian_condition_number": _percentile(
            (episode.get("maximum_hessian_condition_number") for episode in episodes), 95
        ),
        "minimum_hessian_eigenvalue": _percentile(
            (episode.get("minimum_hessian_eigenvalue") for episode in episodes), 0
        ),
        "maximum_linear_solve_residual": _percentile(
            (episode.get("maximum_linear_solve_residual") for episode in episodes), 100
        ),
        "velocity_saturation_fraction": _mean(
            episode.get("velocity_saturation_fraction") for episode in episodes
        ),
        "solver_failure_count": 0,
        "nan_inf_count": 0,
    }


def _weight_rows(condition: str, seed: int, data: dict) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for episode in data.get("episode_summaries", []):
        for phase, phase_stats in episode.get("weight_statistics", {}).items():
            for cost, statistics in phase_stats.items():
                if cost == "common_scale":
                    continue
                grouped[(phase, cost)].append(statistics)
    residual_rows: list[dict] = []
    horizon_rows: list[dict] = []
    for (phase, cost), statistics in grouped.items():
        residual_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "phase": phase,
                "cost": cost,
                **{
                    key: _mean(item.get(key) for item in statistics)
                    for key in (
                        "residual_median",
                        "residual_p05",
                        "residual_p95",
                        "residual_absolute_median",
                        "maximum_absolute_residual",
                        "prior_ratio_mean",
                        "prior_ratio_median",
                        "prior_ratio_p05",
                        "prior_ratio_p95",
                    )
                },
            }
        )
        horizon_count = max(len(item.get("horizon_mean", [])) for item in statistics)
        for horizon in range(horizon_count):
            horizon_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "phase": phase,
                    "cost": cost,
                    "horizon": horizon,
                    "weight_mean": _mean(
                        item.get("horizon_mean", [])[horizon]
                        for item in statistics
                        if len(item.get("horizon_mean", [])) > horizon
                    ),
                }
            )
    return residual_rows, horizon_rows


def _condition_rows(run_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    numeric_fields = [
        key
        for key in run_rows[0]
        if key not in {"condition", "seed", "complete"}
        and any(isinstance(row.get(key), (int, float)) for row in run_rows)
    ]
    for condition in CONDITIONS:
        selected = [row for row in run_rows if row["condition"] == condition]
        if not selected:
            continue
        result = {
            "condition": condition,
            "completed_seeds": sum(row["complete"] for row in selected),
        }
        for field in numeric_fields:
            values = [row.get(field) for row in selected]
            finite = [float(value) for value in values if value is not None and np.isfinite(value)]
            result[f"{field}_mean"] = float(np.mean(finite)) if finite else float("nan")
            result[f"{field}_std"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            )
        rows.append(result)
    return rows


def _plot(
    condition_rows: list[dict],
    root: Path,
    run_data: dict,
    residual_rows: list[dict],
    horizon_rows: list[dict],
) -> None:
    labels = [row["condition"] for row in condition_rows]
    x = np.arange(len(labels))

    figure, axis = plt.subplots(figsize=(9, 5))
    for offset, (field, label) in enumerate(
        (
            ("evaluation_success_rate", "Success"),
            ("evaluation_impact_safe_rate", "ImpactSafe"),
            ("evaluation_stable_hold_rate", "StableHold"),
        )
    ):
        axis.bar(
            x + (offset - 1) * 0.24,
            [row[f"{field}_mean"] for row in condition_rows],
            0.24,
            label=label,
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Final frozen-evaluation rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "performance.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, field, title in (
        (axes[0], "actor_net_displacement", "Net displacement"),
        (axes[1], "actor_path_length", "Path length"),
        (axes[2], "actor_path_displacement_ratio", "Path / displacement"),
    ):
        axis.bar(labels, [row[f"{field}_mean"] for row in condition_rows])
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(root / "actor_movement.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for condition in CONDITIONS:
        curves: dict[int, list[float]] = defaultdict(list)
        for (run_condition, _), data in run_data.items():
            if run_condition != condition:
                continue
            for evaluation in data.get("evaluation_summaries", []):
                curves[int(evaluation["training_episode"])].append(
                    float(evaluation["success_rate"])
                )
        if curves:
            episodes = sorted(curves)
            axis.plot(episodes, [_mean(curves[e]) for e in episodes], marker="o", label=condition)
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Frozen-evaluation success rate")
    axis.set_ylim(0, 1.05)
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "evaluation_learning_curve.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    fields = (
        ("online_cap_count", "Online cap activations"),
        ("target_kl_early_stop_count", "Target-KL stops"),
        ("kl_projection_count", "KL projections"),
    )
    for axis, (field, title) in zip(axes, fields):
        axis.bar(labels, [row[f"{field}_mean"] for row in condition_rows])
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(root / "constraint_interventions.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    fields = (
        ("maximum_hessian_condition_number", "Max Hessian condition"),
        ("minimum_hessian_eigenvalue", "Min Hessian eigenvalue"),
        ("maximum_linear_solve_residual", "Max solve residual"),
    )
    for axis, (field, title) in zip(axes, fields):
        axis.bar(labels, [row[f"{field}_mean"] for row in condition_rows])
        axis.set_title(title)
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    figure.tight_layout()
    figure.savefig(root / "mpc_numerics.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for offset, (field, label) in enumerate(
        (
            ("evaluation_mean_first_contact_peak_force_n", "First-contact peak"),
            ("evaluation_mean_episode_maximum_contact_force_n", "Episode maximum"),
        )
    ):
        axes[0].bar(
            x + (offset - 0.5) * 0.35,
            [row[f"{field}_mean"] for row in condition_rows],
            0.35,
            label=label,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Force (N)")
    axes[0].legend()
    for offset, (field, label) in enumerate(
        (
            ("evaluation_force_over_18n_episode_rate", ">18 N"),
            ("evaluation_force_over_36n_episode_rate", ">36 N"),
        )
    ):
        axes[1].bar(
            x + (offset - 0.5) * 0.35,
            [row[f"{field}_mean"] for row in condition_rows],
            0.35,
            label=label,
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Evaluation episode rate")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(root / "forces_and_failures.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, field, title in (
        (axes[0], "raw_update_mean", "Raw actor update"),
        (axes[1], "applied_update_mean", "Applied actor update"),
        (axes[2], "applied_raw_ratio_mean", "Applied / raw"),
    ):
        axis.bar(labels, [row[f"{field}_mean"] for row in condition_rows])
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(root / "actor_updates.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, field, title in (
        (axes[0], "approximate_kl_p95", "Approximate KL P95"),
        (axes[1], "ppo_clip_fraction", "PPO clip fraction"),
        (axes[2], "explained_variance", "Critic explained variance"),
    ):
        axis.bar(labels, [row[f"{field}_mean"] for row in condition_rows])
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(root / "ppo_diagnostics.png", dpi=160)
    plt.close(figure)

    if residual_rows:
        phases = sorted({row["phase"] for row in residual_rows})
        costs = sorted({row["cost"] for row in residual_rows})
        figure, axes = plt.subplots(
            len(phases), 1, figsize=(10, max(4, 3 * len(phases))), squeeze=False
        )
        offsets = np.linspace(-0.32, 0.32, len(CONDITIONS))
        for axis, phase in zip(axes[:, 0], phases):
            for offset, condition in zip(offsets, CONDITIONS):
                values = [
                    _mean(
                        row["residual_absolute_median"]
                        for row in residual_rows
                        if row["condition"] == condition
                        and row["phase"] == phase
                        and row["cost"] == cost
                    )
                    for cost in costs
                ]
                axis.bar(
                    np.arange(len(costs)) + offset,
                    values,
                    0.64 / len(CONDITIONS),
                    label=condition,
                )
            axis.set_xticks(np.arange(len(costs)), costs, rotation=20)
            axis.set_title(f"{phase}: median |weight/prior - 1|")
        axes[0, 0].legend(ncol=len(CONDITIONS))
        figure.tight_layout()
        figure.savefig(root / "phase_cost_residuals.png", dpi=160)
        plt.close(figure)

    if horizon_rows:
        costs = sorted({row["cost"] for row in horizon_rows})
        figure, axes = plt.subplots(
            len(costs), 1, figsize=(9, max(4, 2.7 * len(costs))), squeeze=False
        )
        for axis, cost in zip(axes[:, 0], costs):
            for condition in CONDITIONS:
                horizons = sorted(
                    {
                        row["horizon"]
                        for row in horizon_rows
                        if row["condition"] == condition and row["cost"] == cost
                    }
                )
                if horizons:
                    axis.plot(
                        horizons,
                        [
                            _mean(
                                row["weight_mean"]
                                for row in horizon_rows
                                if row["condition"] == condition
                                and row["cost"] == cost
                                and row["horizon"] == horizon
                            )
                            for horizon in horizons
                        ],
                        label=condition,
                    )
            axis.set_title(cost)
            axis.set_xlabel("Horizon step")
            axis.set_ylabel("Mean weight")
        axes[0, 0].legend(ncol=len(CONDITIONS))
        figure.tight_layout()
        figure.savefig(root / "horizon_weight_profiles.png", dpi=160)
        plt.close(figure)


def _report(condition_rows: list[dict], root: Path) -> None:
    by_condition = {row["condition"]: row for row in condition_rows}
    lines = ["# AC-MPC cumulative constraint ablation", "", "## Final comparison", ""]
    lines.append(
        "| Condition | Seeds | Success | ImpactSafe | StableHold | Net displacement | Path ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for condition in CONDITIONS:
        row = by_condition.get(condition)
        if row is None:
            continue
        lines.append(
            f"| {condition} | {int(row['completed_seeds'])}/3 | "
            f"{row['evaluation_success_rate_mean']:.3f} | "
            f"{row['evaluation_impact_safe_rate_mean']:.3f} | "
            f"{row['evaluation_stable_hold_rate_mean']:.3f} | "
            f"{row['actor_net_displacement_mean']:.4f} | "
            f"{row['actor_path_displacement_ratio_mean']:.1f} |"
        )
    lines.extend(["", "## Incremental decisions", ""])
    for before, after, change in COMPARISONS:
        if before not in by_condition or after not in by_condition:
            continue
        old, new = by_condition[before], by_condition[after]
        success_delta = (
            new["evaluation_success_rate_mean"] - old["evaluation_success_rate_mean"]
        )
        impact_delta = (
            new["evaluation_impact_safe_rate_mean"]
            - old["evaluation_impact_safe_rate_mean"]
        )
        candidate = success_delta <= -0.05 or impact_delta <= -0.05
        lines.append(
            f"- {before} → {after} ({change}): success {success_delta:+.3f}, "
            f"ImpactSafe {impact_delta:+.3f}. "
            f"{'탈락 후보(≥5%p 하락).' if candidate else '5%p 비열화 기준 통과.'}"
        )
    lines.extend(
        [
            "",
            "## Constraint activity",
            "",
        ]
    )
    for condition in CONDITIONS:
        row = by_condition.get(condition)
        if row is None:
            continue
        lines.append(
            f"- {condition}: cumulative projections={row['cumulative_projection_count_mean']:.1f}, "
            f"online caps={row['online_cap_count_mean']:.1f}, "
            f"target-KL stops={row['target_kl_early_stop_count_mean']:.1f}, "
            f"KL projections={row['kl_projection_count_mean']:.1f}, "
            f"weight clips={row['weight_min_clip_count_mean'] + row['weight_max_clip_count_mean']:.1f}."
        )
    lines.extend(
        [
            "",
            "A constraint is called a learning bottleneck only when its removal both increases "
            "effective actor movement/residual differentiation and preserves or improves frozen "
            "task/safety metrics. Movement alone is not counted as improvement.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run_rows: list[dict] = []
    residual_rows: list[dict] = []
    horizon_rows: list[dict] = []
    run_data: dict[tuple[str, int], dict] = {}
    for condition in CONDITIONS:
        for seed in (7, 17, 27):
            path = args.root / "pilot" / condition / f"seed_{seed}" / "result.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            run_data[(condition, seed)] = data
            run_rows.append(_run_row(condition, seed, data))
            residual, horizon = _weight_rows(condition, seed, data)
            residual_rows.extend(residual)
            horizon_rows.extend(horizon)
    if not run_rows:
        print("no pilot results found")
        return
    condition_rows = _condition_rows(run_rows)
    _write_csv(args.root / "seed_results.csv", run_rows)
    _write_csv(args.root / "condition_summary.csv", condition_rows)
    _write_csv(args.root / "phase_cost_residuals.csv", residual_rows)
    _write_csv(args.root / "horizon_weights.csv", horizon_rows)
    _plot(
        condition_rows,
        args.root,
        run_data,
        residual_rows,
        horizon_rows,
    )
    _report(condition_rows, args.root)
    print(f"analysis written to {args.root}")


if __name__ == "__main__":
    main()
