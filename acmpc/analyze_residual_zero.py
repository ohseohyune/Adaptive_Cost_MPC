"""Paired learned-vs-residual-zero analysis of the deterministic evaluations.

Both modes replay the same fixed scenario sequence, so episode i of the
learned run and episode i of the zero run are the same catch attempt with and
without the actor's cost-weight residual. That pairing is verified here (the
domain parameters must match episode-for-episode) before any comparison, and
the bootstrap resamples episode *pairs* rather than modes independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ("D0", "D1", "D2", "D3", "M0")
SEEDS = (7, 17, 27)
MODES = ("learned", "zero")
RESAMPLES = 20_000
# Nominal exploration std in normalized action units, times the MPC velocity
# limit -- the per-dimension sigma of the noise added to the *command*.
EXPLORATION_STD = 0.03
VELOCITY_LIMIT = 1.8
ACTION_DIMENSIONS = 6


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100.0
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def _episode_row(episode: dict) -> dict:
    peak = episode.get("first_contact_peak_force_n")
    return {
        "episode": episode["episode"],
        "success": bool(episode["success"]),
        "impact_safe": bool(episode.get("impact_safe", False)),
        "stable_hold": bool(episode.get("stable_hold_completed", False)),
        "first_contact_peak_force_n": (
            float(peak) if peak is not None and episode.get("first_contact_detected") else None
        ),
        "force_exceeded_18n": float(episode.get("force_over_18n_fraction") or 0.0) > 0.0,
        "force_exceeded_36n": float(episode.get("force_over_36n_fraction") or 0.0) > 0.0,
        "emergency_failure": bool(episode.get("emergency_violation", False)),
        "safety_violation": bool(episode.get("safety_violation", False)),
        "failure_category": episode.get("failure_category") or "",
        "failure_reason": episode.get("failure_reason") or "",
        "hold_time_s": float(episode.get("hold_time_s") or 0.0),
        "episode_return": float(episode.get("total_reward") or 0.0),
        "final_phase": episode.get("stage_name") or "",
        "mass": float(episode.get("mass") or 0.0),
        "friction": float(episode.get("friction") or 0.0),
        "final_box_speed_mps": float(episode.get("final_box_speed_mps") or 0.0),
        "actor_residual_abs_mean": float(episode.get("mean_abs_actor_residual") or 0.0),
        "policy_command_delta": episode.get("mean_policy_command_delta"),
        "max_policy_command_delta": episode.get("max_policy_command_delta"),
    }


def _load(root: Path, condition: str, seed: int, mode: str) -> list[dict]:
    path = root / condition / f"seed_{seed}" / mode / "result.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [_episode_row(episode) for episode in data["episode_summaries"]]


def _paired_bootstrap(
    pairs: list[tuple[float, float]], resamples: int, seed: int = 0
) -> tuple[float, float, float]:
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    generator = random.Random(seed)
    observed = statistics.mean(a - b for a, b in pairs)
    draws = []
    count = len(pairs)
    for _ in range(resamples):
        sample = [pairs[generator.randrange(count)] for _ in range(count)]
        draws.append(statistics.mean(a - b for a, b in sample))
    draws.sort()
    low = draws[int(0.025 * resamples)]
    high = draws[min(int(0.975 * resamples), resamples - 1)]
    return observed, low, high


def _verdict(
    delta_success: float,
    ci: tuple[float, float],
    delta_impact_safe: float,
    peak_learned: float,
    peak_zero: float,
    delta_18n: float,
    delta_36n: float,
) -> str:
    force_drop = (
        (peak_zero - peak_learned) / peak_zero if peak_zero and peak_zero == peak_zero else 0.0
    )
    if delta_success <= -0.03 or delta_18n >= 0.05 or delta_36n > 0.0:
        return "3: Actor가 성능을 악화"
    if delta_success >= 0.03 and ci[0] > 0.0:
        return "1: Actor가 유의미하게 도움"
    if abs(delta_success) < 0.02 and force_drop >= 0.10:
        return "1: Actor가 유의미하게 도움 (force)"
    if (
        abs(delta_success) < 0.02
        and abs(delta_impact_safe) < 0.02
        and abs(force_drop) < 0.05
        and ci[0] <= 0.0 <= ci[1]
    ):
        return "2: Actor 기여가 거의 없음"
    return "판정 불가 (기준 사이)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=ROOT / "sweep_results" / "residual_zero_20260805"
    )
    args = parser.parse_args()
    out = args.root

    data: dict[tuple[str, int, str], list[dict]] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            for mode in MODES:
                data[(condition, seed, mode)] = _load(out, condition, seed, mode)

    episode_csv = out / "residual_zero_episodes.csv"
    sample = next((v for v in data.values() if v), None)
    if sample is None:
        raise SystemExit(f"no results under {out}")
    fields = ["condition", "training_seed", "evaluation_mode", *sample[0].keys()]
    with episode_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (condition, seed, mode), episodes in data.items():
            for row in episodes:
                writer.writerow({"condition": condition, "training_seed": seed, "evaluation_mode": mode, **row})

    pairing_problems: list[str] = []
    seed_rows: list[dict] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            learned = data[(condition, seed, "learned")]
            zero = data[(condition, seed, "zero")]
            if not learned or not zero:
                continue
            count = min(len(learned), len(zero))
            for index in range(count):
                if (
                    abs(learned[index]["mass"] - zero[index]["mass"]) > 1e-9
                    or abs(learned[index]["friction"] - zero[index]["friction"]) > 1e-9
                ):
                    pairing_problems.append(f"{condition} seed={seed} episode={index}")
                    break
            learned, zero = learned[:count], zero[:count]

            success_pairs = [
                (float(a["success"]), float(b["success"])) for a, b in zip(learned, zero)
            ]
            delta_success, low, high = _paired_bootstrap(success_pairs, RESAMPLES)
            peaks_learned = [
                r["first_contact_peak_force_n"]
                for r in learned
                if r["first_contact_peak_force_n"] is not None
            ]
            peaks_zero = [
                r["first_contact_peak_force_n"]
                for r in zero
                if r["first_contact_peak_force_n"] is not None
            ]
            deltas = [
                r["policy_command_delta"]
                for r in learned
                if r["policy_command_delta"] is not None
            ]
            seed_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "episodes": count,
                    "learned_success": statistics.mean(r["success"] for r in learned),
                    "zero_success": statistics.mean(r["success"] for r in zero),
                    "delta_success": delta_success,
                    "ci_low": low,
                    "ci_high": high,
                    "learned_impact_safe": statistics.mean(r["impact_safe"] for r in learned),
                    "zero_impact_safe": statistics.mean(r["impact_safe"] for r in zero),
                    "learned_stable_hold": statistics.mean(r["stable_hold"] for r in learned),
                    "zero_stable_hold": statistics.mean(r["stable_hold"] for r in zero),
                    "learned_peak_p95": _percentile(peaks_learned, 95),
                    "zero_peak_p95": _percentile(peaks_zero, 95),
                    "learned_over_18n": statistics.mean(r["force_exceeded_18n"] for r in learned),
                    "zero_over_18n": statistics.mean(r["force_exceeded_18n"] for r in zero),
                    "learned_over_36n": statistics.mean(r["force_exceeded_36n"] for r in learned),
                    "zero_over_36n": statistics.mean(r["force_exceeded_36n"] for r in zero),
                    "learned_emergency": statistics.mean(r["emergency_failure"] for r in learned),
                    "zero_emergency": statistics.mean(r["emergency_failure"] for r in zero),
                    "learned_return": statistics.mean(r["episode_return"] for r in learned),
                    "zero_return": statistics.mean(r["episode_return"] for r in zero),
                    "actor_residual_abs_mean": statistics.mean(
                        r["actor_residual_abs_mean"] for r in learned
                    ),
                    "policy_command_delta": statistics.mean(deltas) if deltas else float("nan"),
                }
            )

    seed_csv = out / "residual_zero_seed_summary.csv"
    with seed_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)

    condition_rows: list[dict] = []
    for condition in CONDITIONS:
        rows = [r for r in seed_rows if r["condition"] == condition]
        if not rows:
            continue
        pooled_success: list[tuple[float, float]] = []
        for seed in SEEDS:
            learned = data[(condition, seed, "learned")]
            zero = data[(condition, seed, "zero")]
            count = min(len(learned), len(zero))
            pooled_success.extend(
                (float(a["success"]), float(b["success"]))
                for a, b in zip(learned[:count], zero[:count])
            )
        delta, low, high = _paired_bootstrap(pooled_success, RESAMPLES)
        mean = lambda key: statistics.mean(r[key] for r in rows)  # noqa: E731
        verdict = _verdict(
            delta,
            (low, high),
            mean("learned_impact_safe") - mean("zero_impact_safe"),
            mean("learned_peak_p95"),
            mean("zero_peak_p95"),
            mean("learned_over_18n") - mean("zero_over_18n"),
            mean("learned_over_36n") - mean("zero_over_36n"),
        )
        condition_rows.append(
            {
                "condition": condition,
                "learned_success": mean("learned_success"),
                "zero_success": mean("zero_success"),
                "paired_delta": delta,
                "ci95_low": low,
                "ci95_high": high,
                "learned_impact_safe": mean("learned_impact_safe"),
                "zero_impact_safe": mean("zero_impact_safe"),
                "learned_peak_p95": mean("learned_peak_p95"),
                "zero_peak_p95": mean("zero_peak_p95"),
                "learned_over_18n": mean("learned_over_18n"),
                "zero_over_18n": mean("zero_over_18n"),
                "learned_over_36n": mean("learned_over_36n"),
                "zero_over_36n": mean("zero_over_36n"),
                "actor_residual_abs_mean": mean("actor_residual_abs_mean"),
                "policy_command_delta": mean("policy_command_delta"),
                "verdict": verdict,
            }
        )

    condition_csv = out / "residual_zero_condition_summary.csv"
    with condition_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0].keys()))
        writer.writeheader()
        writer.writerows(condition_rows)

    exploration_norm = EXPLORATION_STD * VELOCITY_LIMIT * (ACTION_DIMENSIONS ** 0.5)

    lines = ["# Residual-zero ablation", ""]
    if pairing_problems:
        lines += ["**PAIRING MISMATCH**: " + ", ".join(pairing_problems[:10]), ""]
    else:
        lines += ["Scenario pairing verified: identical mass/friction per episode index.", ""]
    lines += [
        "| Condition | Learned success | Zero success | Paired diff | CI95 | Actor 판정 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in condition_rows:
        lines.append(
            f"| {row['condition']} | {row['learned_success']:.3f} | {row['zero_success']:.3f} | "
            f"{row['paired_delta']:+.3f} | [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] | "
            f"{row['verdict']} |"
        )
    lines += [
        "",
        "| Condition | Actor residual abs mean | Policy command delta (m/s) | Noise/signal | Δ success | Δ peak P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in condition_rows:
        delta_cmd = row["policy_command_delta"]
        ratio = exploration_norm / delta_cmd if delta_cmd and delta_cmd == delta_cmd else float("nan")
        lines.append(
            f"| {row['condition']} | {row['actor_residual_abs_mean']:.4f} | {delta_cmd:.5f} | "
            f"{ratio:.1f} | {row['paired_delta']:+.3f} | "
            f"{row['learned_peak_p95'] - row['zero_peak_p95']:+.2f} |"
        )
    lines += [
        "",
        f"Noise/signal is E||u_sample - u_mean|| / ||u_mean - u_zero|| with "
        f"E||u_sample - u_mean|| ~ {exploration_norm:.4f} m/s "
        f"(std {EXPLORATION_STD} x velocity limit {VELOCITY_LIMIT} over "
        f"{ACTION_DIMENSIONS} dims). Higher means the exploration noise dominates "
        "the learned residual's effect on the command.",
    ]
    (out / "residual_zero_report.md").write_text("\n".join(lines) + "\n")

    colors = {"learned": "#1f77b4", "zero": "#ff7f0e"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    xs = range(len(condition_rows))
    for offset, mode in enumerate(MODES):
        key = f"{mode}_success"
        values = [r[key] for r in condition_rows]
        errors = [
            statistics.stdev([s[key] for s in seed_rows if s["condition"] == r["condition"]])
            if len([s for s in seed_rows if s["condition"] == r["condition"]]) > 1
            else 0.0
            for r in condition_rows
        ]
        axes[0].bar(
            [x + offset * 0.38 for x in xs], values, 0.36, yerr=errors, capsize=3,
            label=mode, color=colors[mode],
        )
    axes[0].set_xticks([x + 0.19 for x in xs])
    axes[0].set_xticklabels([r["condition"] for r in condition_rows])
    axes[0].set_ylabel("deterministic success rate")
    axes[0].set_title("learned vs residual-zero (seed std bars)")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    for row in seed_rows:
        axes[1].plot(
            [0, 1],
            [row["zero_success"], row["learned_success"]],
            marker="o",
            alpha=0.8,
            label=f"{row['condition']} s{row['seed']}",
        )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["residual-zero", "learned"])
    axes[1].set_ylabel("success rate")
    axes[1].set_title("per seed")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(out / "residual_zero_success.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for index, condition in enumerate(CONDITIONS):
        rows = [r for r in seed_rows if r["condition"] == condition]
        if not rows:
            continue
        axes[0].scatter(
            [r["actor_residual_abs_mean"] for r in rows],
            [r["delta_success"] for r in rows],
            label=condition,
        )
        axes[1].scatter(
            [r["policy_command_delta"] for r in rows],
            [r["delta_success"] for r in rows],
            label=condition,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("actor residual abs mean")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("policy command delta (m/s)")
    for axis in axes[:2]:
        axis.set_ylabel("paired delta success")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

    learned_peaks: list[float] = []
    zero_peaks: list[float] = []
    for (condition, seed, mode), episodes in data.items():
        target = learned_peaks if mode == "learned" else zero_peaks
        target.extend(
            r["first_contact_peak_force_n"]
            for r in episodes
            if r["first_contact_peak_force_n"] is not None
        )
    if learned_peaks and zero_peaks:
        axes[2].hist(
            [learned_peaks, zero_peaks], bins=30, label=["learned", "zero"],
            color=[colors["learned"], colors["zero"]],
        )
    axes[2].set_xlabel("first-contact peak force (N)")
    axes[2].set_title("all conditions pooled")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / "residual_zero_diagnostics.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for index, condition in enumerate(CONDITIONS):
        diffs = []
        for seed in SEEDS:
            learned = data[(condition, seed, "learned")]
            zero = data[(condition, seed, "zero")]
            count = min(len(learned), len(zero))
            diffs.extend(
                float(a["success"]) - float(b["success"])
                for a, b in zip(learned[:count], zero[:count])
            )
        if not diffs:
            continue
        axes[0].bar(
            [index - 0.25, index, index + 0.25],
            [
                sum(1 for d in diffs if d > 0) / len(diffs),
                sum(1 for d in diffs if d == 0) / len(diffs),
                sum(1 for d in diffs if d < 0) / len(diffs),
            ],
            0.22,
            color=["#2ca02c", "#999999", "#d62728"],
        )
    axes[0].set_xticks(range(len(CONDITIONS)))
    axes[0].set_xticklabels(CONDITIONS)
    axes[0].set_ylabel("fraction of paired episodes")
    axes[0].set_title("paired outcome: learned only wins / tie / zero only wins")
    axes[0].grid(alpha=0.3, axis="y")

    width = 0.2
    for offset, (key, color, label) in enumerate(
        [
            ("learned_over_18n", "#1f77b4", "learned >18N"),
            ("zero_over_18n", "#aec7e8", "zero >18N"),
            ("learned_over_36n", "#d62728", "learned >36N"),
            ("zero_over_36n", "#ff9896", "zero >36N"),
        ]
    ):
        axes[1].bar(
            [i + offset * width for i in range(len(condition_rows))],
            [r[key] for r in condition_rows],
            width,
            color=color,
            label=label,
        )
    axes[1].set_xticks([i + 1.5 * width for i in range(len(condition_rows))])
    axes[1].set_xticklabels([r["condition"] for r in condition_rows])
    axes[1].set_ylabel("episode rate")
    axes[1].set_title("force exceedance")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(out / "residual_zero_paired_and_forces.png", dpi=140)
    plt.close(figure)

    print(f"wrote {episode_csv}")
    print(f"wrote {seed_csv}")
    print(f"wrote {condition_csv}")
    print((out / "residual_zero_report.md").read_text())


if __name__ == "__main__":
    main()
