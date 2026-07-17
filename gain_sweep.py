"""Run automatic gain sweeps for the MuJoCo SE(3) CLIK demo."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as demo
from control.clik import build_serial_arm, get_ee_transform
from control.clik.gain_sweep import (
    GainSweepConfig,
    plot_best_time_series,
    plot_metric_trends,
    print_best_results,
    run_gain_sweep,
    write_results_csv,
)
from robot.ffw_config import FFW_ARMS


XML_PATH = PROJECT_ROOT / "model" / "robotis_ffw" / "scene_ffw_sg2.xml"


def parse_float_tuple(text: str) -> tuple[float, ...]:
    """Parse comma-separated floats or inclusive start:stop:step ranges."""

    if ":" in text:
        parts = [item.strip() for item in text.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "range syntax must be start:stop:step, e.g. 0:20:1"
            )

        start, stop, step = (float(item) for item in parts)
        if step == 0.0:
            raise argparse.ArgumentTypeError("range step must not be zero")
        if (stop - start) * step < 0.0:
            raise argparse.ArgumentTypeError("range step points away from stop")

        values = []
        current = start
        tolerance = abs(step) * 1e-9
        if step > 0.0:
            while current <= stop + tolerance:
                values.append(current)
                current += step
        else:
            while current >= stop - tolerance:
                values.append(current)
                current += step
        return tuple(values)

    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float value")
    return values


def target_position_factory(
    arm_name: str,
    trajectory: str,
    frequency_hz: float,
    circle_radius: float,
    sine_x_amplitude: float,
    sine_z_amplitude: float,
):
    """Build a time-indexed target position function."""

    center = np.asarray(demo.TARGET_CENTERS[arm_name], dtype=float)

    def target_position_at_time(t: float) -> np.ndarray:
        phase = 2.0 * np.pi * frequency_hz * t

        if trajectory == "fixed":
            offset = np.zeros(3)
        elif trajectory == "circle":
            offset = circle_radius * np.array([np.cos(phase), 0.0, np.sin(phase)])
        elif trajectory == "sine":
            offset = np.array(
                [
                    sine_x_amplitude * np.sin(phase),
                    0.0,
                    sine_z_amplitude * np.sin(2.0 * phase),
                ]
            )
        else:
            raise ValueError(f"Unknown trajectory: {trajectory}")

        return center + offset

    return target_position_at_time


def make_trial_context(arm_name: str):
    """Create a fresh MuJoCo model/data/arm tuple for one trial."""

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    right_arm = build_serial_arm(model, FFW_ARMS["right"])
    left_arm = build_serial_arm(model, FFW_ARMS["left"])
    arms = {"right": right_arm, "left": left_arm}

    demo.set_arm_joint_positions(model, data, right_arm, demo.ZERO_ARM_HOME)
    demo.set_arm_joint_positions(model, data, left_arm, demo.ZERO_ARM_HOME)
    mujoco.mj_forward(model, data)
    return model, data, arms[arm_name]


def make_parser() -> argparse.ArgumentParser:
    defaults = GainSweepConfig()
    parser = argparse.ArgumentParser(
        description="Brute-force translation/rotation/damping sweep for SE(3) CLIK."
    )
    parser.add_argument("--arm", choices=("right", "left"), default=demo.ARM_NAME)
    parser.add_argument(
        "--translation-gains",
        type=parse_float_tuple,
        default=defaults.translation_gains,
        help="Comma-separated gains or start:stop:step, e.g. 0.8,1.0,1.2 or 0:20:1",
    )
    parser.add_argument(
        "--rotation-gains",
        type=parse_float_tuple,
        default=defaults.rotation_gains,
        help="Comma-separated gains or start:stop:step, e.g. 0.2,0.4,0.6 or 0:20:1",
    )
    parser.add_argument(
        "--damping-values",
        type=parse_float_tuple,
        default=defaults.damping_values,
        help="Comma-separated damping values or start:stop:step, e.g. 0.003,0.01,0.03",
    )
    parser.add_argument("--horizon", type=float, default=defaults.horizon_s)
    parser.add_argument("--max-joint-step", type=float, default=defaults.max_joint_step)
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=defaults.position_tolerance,
    )
    parser.add_argument(
        "--rotation-tolerance",
        type=float,
        default=defaults.rotation_tolerance,
    )
    parser.add_argument(
        "--settling-window",
        type=float,
        default=defaults.settling_window_s,
    )
    parser.add_argument(
        "--divergence-position-error",
        type=float,
        default=defaults.divergence_position_error,
    )
    parser.add_argument(
        "--divergence-rotation-error",
        type=float,
        default=defaults.divergence_rotation_error,
    )
    parser.add_argument(
        "--divergence-joint-velocity-norm",
        type=float,
        default=defaults.divergence_joint_velocity_norm,
    )
    parser.add_argument(
        "--divergence-error-growth-factor",
        type=float,
        default=defaults.divergence_error_growth_factor,
    )
    parser.add_argument(
        "--min-growth-check",
        type=float,
        default=defaults.min_growth_check_s,
    )
    parser.add_argument(
        "--joint-limit-margin",
        type=float,
        default=defaults.joint_limit_margin,
        help="Allowed qpos overshoot beyond MuJoCo joint range before failure.",
    )
    parser.add_argument(
        "--trajectory",
        choices=("fixed", "circle", "sine"),
        default=demo.TARGET_TRAJECTORY,
    )
    parser.add_argument(
        "--score-mode",
        choices=("auto", "tracking", "regulation"),
        default="auto",
        help="Use tracking score for moving targets or regulation score for fixed targets.",
    )
    parser.add_argument("--frequency-hz", type=float, default=demo.TARGET_FREQUENCY_HZ)
    parser.add_argument(
        "--circle-radius",
        type=float,
        default=demo.TARGET_CIRCLE_RADIUS,
    )
    parser.add_argument(
        "--sine-x-amplitude",
        type=float,
        default=demo.TARGET_SINE_X_AMPLITUDE,
    )
    parser.add_argument(
        "--sine-z-amplitude",
        type=float,
        default=demo.TARGET_SINE_Z_AMPLITUDE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "sweep_results"
        / datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--trace-top-n",
        type=int,
        default=3,
        help="Number of best stable trials to re-run for time-series plots.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-trace-plots", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    score_mode = args.score_mode
    if score_mode == "auto":
        score_mode = "regulation" if args.trajectory == "fixed" else "tracking"

    config = GainSweepConfig(
        translation_gains=args.translation_gains,
        rotation_gains=args.rotation_gains,
        damping_values=args.damping_values,
        horizon_s=args.horizon,
        max_joint_step=args.max_joint_step,
        position_tolerance=args.position_tolerance,
        rotation_tolerance=args.rotation_tolerance,
        settling_window_s=args.settling_window,
        divergence_position_error=args.divergence_position_error,
        divergence_rotation_error=args.divergence_rotation_error,
        divergence_joint_velocity_norm=args.divergence_joint_velocity_norm,
        divergence_error_growth_factor=args.divergence_error_growth_factor,
        min_growth_check_s=args.min_growth_check,
        joint_limit_margin=args.joint_limit_margin,
        settling_score_weight=1.0 if score_mode == "regulation" else 0.0,
    )

    model, data, arm = make_trial_context(args.arm)
    initial_ee_rotation = get_ee_transform(data, arm)[:3, :3]
    target_rotation = demo.target_rotation_for_grasp(initial_ee_rotation)
    target_position_fn = target_position_factory(
        arm_name=args.arm,
        trajectory=args.trajectory,
        frequency_hz=args.frequency_hz,
        circle_radius=args.circle_radius,
        sine_x_amplitude=args.sine_x_amplitude,
        sine_z_amplitude=args.sine_z_amplitude,
    )

    total_trials = (
        len(config.translation_gains)
        * len(config.rotation_gains)
        * len(config.damping_values)
    )
    print(
        f"Running {total_trials} CLIK gain trials for {args.arm} arm | "
        f"trajectory={args.trajectory} score_mode={score_mode} "
        f"horizon={config.horizon_s:g}s"
    )
    print(f"Results directory: {args.output_dir}")

    results = run_gain_sweep(
        make_trial_context=lambda: make_trial_context(args.arm),
        target_position_fn=target_position_fn,
        target_rotation=target_rotation,
        config=config,
    )

    csv_path = args.output_dir / "gain_sweep_results.csv"
    write_results_csv(results, csv_path)
    print(f"\nSaved CSV: {csv_path}")
    print_best_results(results, top_n=args.top_n)

    if not args.no_plots:
        plot_metric_trends(results, args.output_dir)
        print(f"Saved metric plots in: {args.output_dir}")

    if not args.no_plots and not args.no_trace_plots and args.trace_top_n > 0:
        plot_best_time_series(
            make_trial_context=lambda: make_trial_context(args.arm),
            target_position_fn=target_position_fn,
            target_rotation=target_rotation,
            config=config,
            results=results,
            output_dir=args.output_dir,
            top_n=args.trace_top_n,
        )
        print(f"Saved best-trial time-series plots in: {args.output_dir}")


if __name__ == "__main__":
    main()
