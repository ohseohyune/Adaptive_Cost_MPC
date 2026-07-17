"""Headless gain-sweep utilities for SE(3) CLIK controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import csv
import math

import mujoco
import numpy as np

from control.clik.controller import pose_clik_step
from control.clik.kinematics import make_transform
from control.clik.types import SerialArm


@dataclass(frozen=True)
class GainSweepConfig:
    """Configuration for brute-force CLIK gain experiments."""

    translation_gains: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5)
    rotation_gains: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)
    damping_values: tuple[float, ...] = (0.003, 0.005, 0.01, 0.02, 0.03)
    horizon_s: float = 10.0
    max_joint_step: float = 0.03
    position_tolerance: float = 0.015
    rotation_tolerance: float = 0.08
    settling_window_s: float = 0.5
    divergence_position_error: float = 1.5
    divergence_rotation_error: float = 3.2
    divergence_joint_velocity_norm: float = 20.0
    divergence_error_growth_factor: float = 8.0
    min_growth_check_s: float = 0.5
    joint_limit_margin: float = 0.02
    settling_score_weight: float = 0.0


@dataclass(frozen=True)
class TrialResult:
    """One CSV-friendly result row from a gain-sweep trial."""

    translation_gain: float
    rotation_gain: float
    damping: float
    horizon_s: float
    simulated_time_s: float
    steps: int
    final_position_error_norm: float
    final_rotation_error_norm: float
    position_rms_error: float
    rotation_rms_error: float
    peak_position_error_norm: float
    peak_rotation_error_norm: float
    settling_time_s: float
    target_reached: bool
    target_reached_time_s: float
    mean_joint_velocity_norm: float
    max_joint_velocity_norm: float
    control_smoothness: float
    oscillation_metric: float
    mean_min_singular_value: float
    min_min_singular_value: float
    max_condition_number: float
    mean_manipulability: float
    diverged: bool
    failure_reason: str
    score: float


@dataclass(frozen=True)
class TrialTrace:
    """Time-series data for one CLIK trial."""

    times: np.ndarray
    position_errors: np.ndarray
    rotation_errors: np.ndarray
    joint_velocity_norms: np.ndarray
    min_singular_values: np.ndarray
    condition_numbers: np.ndarray
    manipulability_values: np.ndarray


def make_pose_gain(rotation_gain: float, translation_gain: float) -> np.ndarray:
    """Build the controller gain vector for pose_error=[rot_x,y,z,pos_x,y,z]."""

    return np.array(
        [
            rotation_gain,
            rotation_gain,
            rotation_gain,
            translation_gain,
            translation_gain,
            translation_gain,
        ],
        dtype=float,
    )


def _first_time_within_tolerance(
    times: np.ndarray,
    position_errors: np.ndarray,
    rotation_errors: np.ndarray,
    position_tolerance: float,
    rotation_tolerance: float,
) -> float:
    within = (position_errors <= position_tolerance) & (
        rotation_errors <= rotation_tolerance
    )
    if not np.any(within):
        return math.nan
    return float(times[int(np.argmax(within))])


def _settling_time(
    times: np.ndarray,
    position_errors: np.ndarray,
    rotation_errors: np.ndarray,
    position_tolerance: float,
    rotation_tolerance: float,
    settling_window_s: float,
) -> float:
    """Return first time that stays inside tolerance through the horizon."""

    if len(times) == 0:
        return math.nan

    within = (position_errors <= position_tolerance) & (
        rotation_errors <= rotation_tolerance
    )
    if not np.any(within):
        return math.nan

    for start_index, start_time in enumerate(times):
        if times[-1] - start_time < settling_window_s:
            continue
        if np.all(within[start_index:]):
            return float(start_time)

    return math.nan


def _oscillation_metric(total_error: np.ndarray) -> float:
    """
    Measure excess error variation beyond the net start-to-finish change.

    A monotonic error curve is close to 0. Larger values mean the error moved
    back and forth instead of converging smoothly.
    """

    if len(total_error) < 3 or not np.all(np.isfinite(total_error)):
        return math.inf

    total_variation = float(np.sum(np.abs(np.diff(total_error))))
    net_change = float(abs(total_error[0] - total_error[-1]))
    excess_variation = max(0.0, total_variation - net_change)
    return excess_variation / max(net_change, 1e-9)


def _root_mean_square(values: np.ndarray) -> float:
    if len(values) == 0 or not np.all(np.isfinite(values)):
        return math.inf
    return float(np.sqrt(np.mean(values**2)))


def _peak(values: np.ndarray) -> float:
    if len(values) == 0 or not np.any(np.isfinite(values)):
        return math.inf
    return float(np.nanmax(values))


def _mean_finite(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return math.inf
    return float(np.mean(finite_values))


def _control_smoothness(joint_velocities: np.ndarray) -> float:
    if len(joint_velocities) < 2 or not np.all(np.isfinite(joint_velocities)):
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(joint_velocities, axis=0), axis=1)))


def _jacobian_quality(jacobian: np.ndarray) -> tuple[float, float, float]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if len(singular_values) == 0:
        return 0.0, math.inf, 0.0

    min_singular_value = float(np.min(singular_values))
    max_singular_value = float(np.max(singular_values))
    condition_number = (
        math.inf
        if min_singular_value < 1e-12
        else max_singular_value / min_singular_value
    )
    manipulability = float(np.sqrt(np.prod(singular_values**2)))
    return min_singular_value, condition_number, manipulability


def _score_trial(
    position_rms_error: float,
    rotation_rms_error: float,
    peak_position_error_norm: float,
    peak_rotation_error_norm: float,
    settling_time_s: float,
    horizon_s: float,
    max_joint_velocity_norm: float,
    control_smoothness: float,
    oscillation_metric: float,
    config: GainSweepConfig,
    diverged: bool,
) -> float:
    if diverged:
        return math.inf

    settling_fraction = (
        1.0 if math.isnan(settling_time_s) else settling_time_s / max(horizon_s, 1e-9)
    )
    return float(
        position_rms_error / max(config.position_tolerance, 1e-9)
        + rotation_rms_error / max(config.rotation_tolerance, 1e-9)
        + 0.25 * peak_position_error_norm / max(config.position_tolerance, 1e-9)
        + 0.25 * peak_rotation_error_norm / max(config.rotation_tolerance, 1e-9)
        + config.settling_score_weight * settling_fraction
        + 0.1 * max_joint_velocity_norm / max(config.divergence_joint_velocity_norm, 1e-9)
        + 0.05 * control_smoothness
        + 0.2 * oscillation_metric
    )


def _divergence_reason(
    model: mujoco.MjModel,
    info: dict,
    data: mujoco.MjData,
    arm: SerialArm,
    joint_velocity_norm: float,
    total_error: float,
    initial_total_error: float,
    t: float,
    config: GainSweepConfig,
) -> str | None:
    arrays_to_check = (
        info["pose_error"],
        info["q_command"],
        info["dq_unclipped"],
        data.qpos,
        data.qvel,
        data.ctrl,
    )
    if any(not np.all(np.isfinite(value)) for value in arrays_to_check):
        return "nonfinite_state"

    if info["position_error_norm"] > config.divergence_position_error:
        return "position_error_threshold"

    if info["rotation_error_norm"] > config.divergence_rotation_error:
        return "rotation_error_threshold"

    if joint_velocity_norm > config.divergence_joint_velocity_norm:
        return "joint_velocity_threshold"

    if (
        t >= config.min_growth_check_s
        and total_error
        > max(initial_total_error, 1e-9) * config.divergence_error_growth_factor
    ):
        return "error_growth_threshold"

    q = data.qpos[arm.qpos_indices]
    joint_ranges = model.jnt_range[arm.joint_ids]
    limited = model.jnt_limited[arm.joint_ids].astype(bool)
    below = q < joint_ranges[:, 0] - config.joint_limit_margin
    above = q > joint_ranges[:, 1] + config.joint_limit_margin
    if np.any(limited & (below | above)):
        return "joint_limit_threshold"

    return None


def _run_pose_clik_trial(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
    target_position_fn,
    target_rotation: np.ndarray,
    translation_gain: float,
    rotation_gain: float,
    damping: float,
    config: GainSweepConfig,
) -> TrialResult:
    """Run one headless MuJoCo trial and compute performance metrics."""

    gain = make_pose_gain(rotation_gain, translation_gain)
    times: list[float] = []
    position_errors: list[float] = []
    rotation_errors: list[float] = []
    total_errors: list[float] = []
    joint_velocities: list[np.ndarray] = []
    joint_velocity_norms: list[float] = []
    min_singular_values: list[float] = []
    condition_numbers: list[float] = []
    manipulability_values: list[float] = []
    max_joint_velocity_norm = 0.0
    failure_reason = ""
    steps = 0
    initial_total_error = math.nan

    while data.time < config.horizon_s:
        target_position = np.asarray(target_position_fn(float(data.time)), dtype=float)
        target_transform = make_transform(target_position, target_rotation)
        info = pose_clik_step(
            model=model,
            data=data,
            arm=arm,
            target_transform=target_transform,
            gain=gain,
            damping=damping,
            max_joint_step=config.max_joint_step,
            posture_reference=None,
            posture_gain=0.0,
            posture_weights=None,
        )

        joint_velocity_norm = float(np.linalg.norm(data.qvel[arm.qvel_indices]))
        min_singular_value, condition_number, manipulability = _jacobian_quality(
            info["jacobian"]
        )
        max_joint_velocity_norm = max(max_joint_velocity_norm, joint_velocity_norm)
        total_error = float(
            math.hypot(info["position_error_norm"], info["rotation_error_norm"])
        )
        if math.isnan(initial_total_error):
            initial_total_error = total_error

        times.append(float(data.time))
        position_errors.append(float(info["position_error_norm"]))
        rotation_errors.append(float(info["rotation_error_norm"]))
        total_errors.append(total_error)
        joint_velocities.append(data.qvel[arm.qvel_indices].copy())
        joint_velocity_norms.append(joint_velocity_norm)
        min_singular_values.append(min_singular_value)
        condition_numbers.append(condition_number)
        manipulability_values.append(manipulability)

        failure_reason = _divergence_reason(
            model=model,
            info=info,
            data=data,
            arm=arm,
            joint_velocity_norm=joint_velocity_norm,
            total_error=total_error,
            initial_total_error=initial_total_error,
            t=float(data.time),
            config=config,
        ) or ""
        if failure_reason:
            break

        mujoco.mj_step(model, data)
        steps += 1

    time_array = np.asarray(times, dtype=float)
    position_array = np.asarray(position_errors, dtype=float)
    rotation_array = np.asarray(rotation_errors, dtype=float)
    total_array = np.asarray(total_errors, dtype=float)
    joint_velocity_array = np.asarray(joint_velocities, dtype=float)
    joint_velocity_norm_array = np.asarray(joint_velocity_norms, dtype=float)
    min_singular_array = np.asarray(min_singular_values, dtype=float)
    condition_array = np.asarray(condition_numbers, dtype=float)
    manipulability_array = np.asarray(manipulability_values, dtype=float)

    final_position_error_norm = (
        float(position_array[-1]) if len(position_array) else math.inf
    )
    final_rotation_error_norm = (
        float(rotation_array[-1]) if len(rotation_array) else math.inf
    )
    position_rms_error = _root_mean_square(position_array)
    rotation_rms_error = _root_mean_square(rotation_array)
    peak_position_error_norm = _peak(position_array)
    peak_rotation_error_norm = _peak(rotation_array)
    target_reached_time_s = _first_time_within_tolerance(
        time_array,
        position_array,
        rotation_array,
        config.position_tolerance,
        config.rotation_tolerance,
    )
    settling_time_s = _settling_time(
        time_array,
        position_array,
        rotation_array,
        config.position_tolerance,
        config.rotation_tolerance,
        config.settling_window_s,
    )
    oscillation_metric = _oscillation_metric(total_array)
    mean_joint_velocity_norm = _mean_finite(joint_velocity_norm_array)
    control_smoothness = _control_smoothness(joint_velocity_array)
    mean_min_singular_value = _mean_finite(min_singular_array)
    min_min_singular_value = (
        float(np.nanmin(min_singular_array)) if len(min_singular_array) else math.inf
    )
    max_condition_number = _peak(condition_array)
    mean_manipulability = _mean_finite(manipulability_array)
    diverged = bool(failure_reason)
    score = _score_trial(
        position_rms_error=position_rms_error,
        rotation_rms_error=rotation_rms_error,
        peak_position_error_norm=peak_position_error_norm,
        peak_rotation_error_norm=peak_rotation_error_norm,
        settling_time_s=settling_time_s,
        horizon_s=config.horizon_s,
        max_joint_velocity_norm=max_joint_velocity_norm,
        control_smoothness=control_smoothness,
        oscillation_metric=oscillation_metric,
        config=config,
        diverged=diverged,
    )

    result = TrialResult(
        translation_gain=float(translation_gain),
        rotation_gain=float(rotation_gain),
        damping=float(damping),
        horizon_s=float(config.horizon_s),
        simulated_time_s=float(data.time),
        steps=int(steps),
        final_position_error_norm=final_position_error_norm,
        final_rotation_error_norm=final_rotation_error_norm,
        position_rms_error=position_rms_error,
        rotation_rms_error=rotation_rms_error,
        peak_position_error_norm=peak_position_error_norm,
        peak_rotation_error_norm=peak_rotation_error_norm,
        settling_time_s=settling_time_s,
        target_reached=not math.isnan(target_reached_time_s),
        target_reached_time_s=target_reached_time_s,
        mean_joint_velocity_norm=float(mean_joint_velocity_norm),
        max_joint_velocity_norm=float(max_joint_velocity_norm),
        control_smoothness=float(control_smoothness),
        oscillation_metric=float(oscillation_metric),
        mean_min_singular_value=float(mean_min_singular_value),
        min_min_singular_value=float(min_min_singular_value),
        max_condition_number=float(max_condition_number),
        mean_manipulability=float(mean_manipulability),
        diverged=diverged,
        failure_reason=failure_reason,
        score=score,
    )
    trace = TrialTrace(
        times=time_array,
        position_errors=position_array,
        rotation_errors=rotation_array,
        joint_velocity_norms=joint_velocity_norm_array,
        min_singular_values=min_singular_array,
        condition_numbers=condition_array,
        manipulability_values=manipulability_array,
    )
    return result, trace


def run_pose_clik_trial(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
    target_position_fn,
    target_rotation: np.ndarray,
    translation_gain: float,
    rotation_gain: float,
    damping: float,
    config: GainSweepConfig,
) -> TrialResult:
    """Run one headless MuJoCo trial and compute performance metrics."""

    result, _ = _run_pose_clik_trial(
        model=model,
        data=data,
        arm=arm,
        target_position_fn=target_position_fn,
        target_rotation=target_rotation,
        translation_gain=translation_gain,
        rotation_gain=rotation_gain,
        damping=damping,
        config=config,
    )
    return result


def run_pose_clik_trial_with_trace(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
    target_position_fn,
    target_rotation: np.ndarray,
    translation_gain: float,
    rotation_gain: float,
    damping: float,
    config: GainSweepConfig,
) -> tuple[TrialResult, TrialTrace]:
    """Run one trial and return aggregate metrics plus time-series data."""

    return _run_pose_clik_trial(
        model=model,
        data=data,
        arm=arm,
        target_position_fn=target_position_fn,
        target_rotation=target_rotation,
        translation_gain=translation_gain,
        rotation_gain=rotation_gain,
        damping=damping,
        config=config,
    )


def run_gain_sweep(make_trial_context, target_position_fn, target_rotation, config):
    """
    Run the full Cartesian product of gains.

    make_trial_context must return a fresh (model, data, arm) tuple for each
    parameter set.
    """

    results: list[TrialResult] = []
    combinations = list(
        product(config.translation_gains, config.rotation_gains, config.damping_values)
    )
    for trial_index, (translation_gain, rotation_gain, damping) in enumerate(
        combinations,
        start=1,
    ):
        model, data, arm = make_trial_context()
        result = run_pose_clik_trial(
            model=model,
            data=data,
            arm=arm,
            target_position_fn=target_position_fn,
            target_rotation=target_rotation,
            translation_gain=translation_gain,
            rotation_gain=rotation_gain,
            damping=damping,
            config=config,
        )
        print(
            f"[{trial_index:03d}/{len(combinations):03d}] "
            f"kt={translation_gain:g} kr={rotation_gain:g} d={damping:g} | "
            f"pos_rms={result.position_rms_error:.4f} "
            f"rot_rms={result.rotation_rms_error:.4f} "
            f"peak_pos={result.peak_position_error_norm:.4f} "
            f"settle={result.settling_time_s:.3f} "
            f"qvel={result.max_joint_velocity_norm:.3f} "
            f"osc={result.oscillation_metric:.3f} "
            f"{'FAIL ' + result.failure_reason if result.diverged else 'OK'}",
            flush=True,
        )
        results.append(result)
    return results


def write_results_csv(results: list[TrialResult], csv_path: Path) -> None:
    """Write gain-sweep results to a CSV file."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        fieldnames = list(asdict(results[0]).keys()) if results else []
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def best_results(results: list[TrialResult], top_n: int = 5) -> list[TrialResult]:
    """Return stable trials ordered by the default performance score."""

    stable = [result for result in results if not result.diverged]
    return sorted(stable, key=lambda result: result.score)[:top_n]


def print_best_results(results: list[TrialResult], top_n: int = 5) -> None:
    """Print the best stable parameter combinations."""

    best = best_results(results, top_n=top_n)
    if not best:
        print("No stable parameter combinations found.")
        return

    print("\nBest stable parameter combinations:")
    print(
        "rank  kt       kr       damping    pos_rms  rot_rms  "
        "peak_pos peak_rot qvel     smooth   score"
    )
    for rank, result in enumerate(best, start=1):
        print(
            f"{rank:>4}  "
            f"{result.translation_gain:<8.4g} "
            f"{result.rotation_gain:<8.4g} "
            f"{result.damping:<10.4g} "
            f"{result.position_rms_error:<8.4f} "
            f"{result.rotation_rms_error:<8.4f} "
            f"{result.peak_position_error_norm:<8.4f} "
            f"{result.peak_rotation_error_norm:<8.4f} "
            f"{result.max_joint_velocity_norm:<8.3f} "
            f"{result.control_smoothness:<8.3f} "
            f"{result.score:<8.3f}"
        )


def plot_metric_trends(results: list[TrialResult], output_dir: Path) -> None:
    """Create scatter trend plots grouped by damping value."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("position_rms_error", "Position RMS error [m]"),
        ("rotation_rms_error", "Rotation RMS error [rad]"),
        ("peak_position_error_norm", "Peak position error [m]"),
        ("peak_rotation_error_norm", "Peak rotation error [rad]"),
        ("max_joint_velocity_norm", "Max joint velocity norm"),
        ("control_smoothness", "Control smoothness"),
        ("oscillation_metric", "Error oscillation metric"),
        ("min_min_singular_value", "Minimum singular value"),
        ("max_condition_number", "Max condition number"),
        ("score", "Score"),
    )

    damping_values = sorted({result.damping for result in results})
    for damping in damping_values:
        group = [result for result in results if result.damping == damping]
        translation = np.array([result.translation_gain for result in group])
        rotation = np.array([result.rotation_gain for result in group])

        fig, axes = plt.subplots(2, 5, figsize=(19, 8), constrained_layout=True)
        for axis, (field_name, title) in zip(axes.flat, metrics):
            values = np.array([getattr(result, field_name) for result in group], dtype=float)
            values[~np.isfinite(values)] = np.nan
            scatter = axis.scatter(
                translation,
                rotation,
                c=values,
                cmap="viridis",
                edgecolors="black",
                linewidths=0.4,
            )
            axis.set_title(title)
            axis.set_xlabel("Translation gain")
            axis.set_ylabel("Rotation gain")
            axis.grid(True, alpha=0.25)
            fig.colorbar(scatter, ax=axis)

        fig.suptitle(f"CLIK gain sweep metrics, damping={damping:g}")
        figure_path = output_dir / f"metric_trends_damping_{damping:g}.png"
        fig.savefig(figure_path, dpi=160)
        plt.close(fig)


def plot_trial_trace(
    trace: TrialTrace,
    result: TrialResult,
    output_path: Path,
) -> None:
    """Plot time-series diagnostics for one trial."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping time-series plots.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)

    axes[0].plot(trace.times, trace.position_errors, color="tab:blue")
    axes[0].set_ylabel("pos err [m]")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(trace.times, trace.rotation_errors, color="tab:orange")
    axes[1].set_ylabel("rot err [rad]")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(trace.times, trace.joint_velocity_norms, color="tab:green")
    axes[2].set_ylabel("||qdot||")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(
        trace.times,
        trace.min_singular_values,
        label="min singular value",
        color="tab:red",
    )
    condition_axis = axes[3].twinx()
    condition_axis.plot(
        trace.times,
        trace.condition_numbers,
        label="condition number",
        color="tab:purple",
        alpha=0.65,
    )
    axes[3].set_ylabel("sigma min")
    condition_axis.set_ylabel("condition")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(True, alpha=0.25)

    fig.suptitle(
        "CLIK trace | "
        f"kt={result.translation_gain:g}, kr={result.rotation_gain:g}, "
        f"damping={result.damping:g}, score={result.score:.3f}"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_best_time_series(
    make_trial_context,
    target_position_fn,
    target_rotation,
    config: GainSweepConfig,
    results: list[TrialResult],
    output_dir: Path,
    top_n: int = 3,
) -> None:
    """Re-run the best stable trials and save diagnostic time-series plots."""

    for rank, result in enumerate(best_results(results, top_n=top_n), start=1):
        model, data, arm = make_trial_context()
        rerun_result, trace = run_pose_clik_trial_with_trace(
            model=model,
            data=data,
            arm=arm,
            target_position_fn=target_position_fn,
            target_rotation=target_rotation,
            translation_gain=result.translation_gain,
            rotation_gain=result.rotation_gain,
            damping=result.damping,
            config=config,
        )
        output_path = output_dir / (
            f"trace_rank_{rank}_kt_{result.translation_gain:g}_"
            f"kr_{result.rotation_gain:g}_d_{result.damping:g}.png"
        )
        plot_trial_trace(trace, rerun_result, output_path)
