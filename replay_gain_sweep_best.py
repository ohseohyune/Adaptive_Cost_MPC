"""Replay the best gain-sweep trials in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gain_sweep as sweep_cli
import main as demo
from control.clik import get_ee_transform, make_transform, pose_clik_step
from control.clik.gain_sweep import GainSweepConfig, make_pose_gain


def _parse_bool(text: str) -> bool:
    return text.lower() in {"1", "true", "yes"}


def _load_best_rows(csv_path: Path, top_n: int) -> list[dict]:
    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    stable_rows = [row for row in rows if not _parse_bool(row["diverged"])]
    stable_rows.sort(key=lambda row: float(row["score"]))
    return stable_rows[:top_n]


def make_parser() -> argparse.ArgumentParser:
    defaults = GainSweepConfig()
    parser = argparse.ArgumentParser(
        description="Replay best gain-sweep CSV rows in the MuJoCo viewer."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--arm", choices=("right", "left"), default=demo.ARM_NAME)
    parser.add_argument(
        "--trajectory",
        choices=("fixed", "circle", "sine"),
        default=demo.TARGET_TRAJECTORY,
    )
    parser.add_argument("--frequency-hz", type=float, default=demo.TARGET_FREQUENCY_HZ)
    parser.add_argument("--circle-radius", type=float, default=demo.TARGET_CIRCLE_RADIUS)
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
    parser.add_argument("--horizon", type=float, default=None)
    parser.add_argument("--max-joint-step", type=float, default=defaults.max_joint_step)
    return parser


def replay_row(row: dict, args: argparse.Namespace, rank: int) -> None:
    model, data, arm = sweep_cli.make_trial_context(args.arm)
    initial_ee_rotation = get_ee_transform(data, arm)[:3, :3]
    target_rotation = demo.target_rotation_for_grasp(initial_ee_rotation)
    target_position_fn = sweep_cli.target_position_factory(
        arm_name=args.arm,
        trajectory=args.trajectory,
        frequency_hz=args.frequency_hz,
        circle_radius=args.circle_radius,
        sine_x_amplitude=args.sine_x_amplitude,
        sine_z_amplitude=args.sine_z_amplitude,
    )

    translation_gain = float(row["translation_gain"])
    rotation_gain = float(row["rotation_gain"])
    damping = float(row["damping"])
    horizon = args.horizon if args.horizon is not None else float(row["horizon_s"])
    gain = make_pose_gain(rotation_gain, translation_gain)

    print(
        f"Replay rank {rank}: kt={translation_gain:g}, kr={rotation_gain:g}, "
        f"damping={damping:g}, horizon={horizon:g}s"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < horizon:
            step_start = time.time()
            target_position = target_position_fn(float(data.time))
            target_transform = make_transform(target_position, target_rotation)

            with viewer.lock():
                demo.update_target_markers(viewer, args.arm, target_position)

            info = pose_clik_step(
                model=model,
                data=data,
                arm=arm,
                target_transform=target_transform,
                gain=gain,
                damping=damping,
                max_joint_step=args.max_joint_step,
            )
            mujoco.mj_step(model, data)
            viewer.sync()

            if int(data.time / model.opt.timestep) % 250 == 0:
                print(
                    f"t={data.time:6.3f}s | "
                    f"pos={info['position_error_norm']:.4f} | "
                    f"rot={info['rotation_error_norm']:.4f}",
                    flush=True,
                )

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0.0:
                time.sleep(time_until_next_step)


def main() -> None:
    args = make_parser().parse_args()
    rows = _load_best_rows(args.csv_path, args.top_n)
    if not rows:
        print("No stable rows found in CSV.")
        return

    for rank, row in enumerate(rows, start=1):
        replay_row(row, args, rank)


if __name__ == "__main__":
    main()
