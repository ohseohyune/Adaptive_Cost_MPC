"""Evaluation reproducibility audit for the AC-MPC box-catch pipeline.

Background (measured, 2026-08-05): the same D0 seed-7 checkpoint evaluated on
the same 50 scenarios scored 1.00 inside the training process and 0.92 in a
separate cold-start replay. The two runs were executed by two different
interpreters -- mujoco 3.4.0 and mujoco 3.3.7 -- whose contact solves differ
by ~1e-13 N. That is invisible before contact and decisive during the 5 s
HOLD, where it flipped 4/50 success labels. Nothing in the pipeline recorded
which library produced which number.

The tests below pin that down permanently:

A  paired replay -- the in-process evaluation path and a cold-start replay of
   the same checkpoint must agree on every episode's success, failure reason
   and phase sequence.
B  step-level trace -- the same episode run in a fresh process and inside a
   process that already ran episodes must agree step-by-step (and stay
   finite).
C  repeatability -- three independent cold-start processes must produce
   identical success vectors and first-contact forces.
E  environment stamp -- every evaluation log must carry the library identity
   it was produced with, and it must match the interpreter reading it.

Run:  python tests/acmpc_evaluation_reproducibility_test.py [--episodes N]
Test D (the pre-existing suites) is `--regression`; it is not run by default
because those suites are slow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.main_acmpc_box_catch import CatchPhase  # noqa: E402
from acmpc.main_acmpc_box_catch_curriculum import (  # noqa: E402
    CurriculumBoxCatchConfig,
    environment_stamp,
    evaluation_replay_config,
    run_curriculum_box_catch,
)

_PHASE_ORDER = list(CatchPhase)
TRAINER = ROOT / "acmpc" / "main_acmpc_box_catch_curriculum.py"
DEFAULT_CHECKPOINT = (
    ROOT / "sweep_results/constraint_ablation_20260804/pilot/D0/seed_7/checkpoint.pt"
)
EVALUATION_SEED = 100_000
# Compared per episode. hold_time_s and the forces are exact-equality
# comparisons on purpose: two runs of one deterministic simulation either
# produce the same doubles or the run is not reproducible, and a tolerance
# here would hide exactly the ~1e-13 drift that caused the original problem.
EPISODE_FIELDS = (
    "success",
    "failure_reason",
    "failure_category",
    "phase_transition_count",
    "hold_to_capture_demotion_count",
    "hold_time_s",
    "first_contact_peak_force_n",
    "episode_maximum_contact_force_n",
    "final_box_speed_mps",
)
TRACE_TOLERANCES = {
    "box_position": 1e-8,
    "box_velocity": 1e-8,
    "command_velocity": 1e-8,
    "first_contact_peak_force_n": 1e-5,
    "hold_timer": 0.011,  # one control step (control_dt = 0.01 s)
    "strict_hold_timer": 0.011,
}


def _base_config(checkpoint: Path, episodes: int) -> CurriculumBoxCatchConfig:
    return CurriculumBoxCatchConfig(
        episodes=1,
        random_seed=7,
        device="cpu",
        checkpoint_path=str(checkpoint),
        load_checkpoint=True,
        online_learning=False,
        log_path=None,
        use_wandb=False,
        weight_delta_fraction=0.65,
        evaluation_every=0,
        evaluation_episodes=episodes,
        evaluation_seed=EVALUATION_SEED,
    )


def _in_process_evaluation(checkpoint: Path, episodes: int, **kwargs) -> list[dict]:
    """The evaluation exactly as the training loop runs it, in this process."""

    config = evaluation_replay_config(_base_config(checkpoint, episodes))
    summary = run_curriculum_box_catch(config, **kwargs)
    return [vars(e) for e in summary.episode_summaries]


def _cold_start_evaluation(checkpoint: Path, episodes: int, directory: Path) -> list[dict]:
    """The same evaluation, launched as its own process from the CLI."""

    log = directory / "result.json"
    command = [
        sys.executable,
        str(TRAINER),
        "--evaluation-replay",
        "--evaluation-episodes",
        str(episodes),
        "--evaluation-seed",
        str(EVALUATION_SEED),
        "--device",
        "cpu",
        "--checkpoint",
        str(checkpoint),
        "--log",
        str(log),
        "--weight-delta-fraction",
        "0.65",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    return json.loads(log.read_text())["episode_summaries"]


def _compare_episodes(left: list[dict], right: list[dict], label: str) -> None:
    assert len(left) == len(right), f"{label}: {len(left)} vs {len(right)} episodes"
    mismatches = [
        (index, field, a[field], b[field])
        for index, (a, b) in enumerate(zip(left, right))
        for field in EPISODE_FIELDS
        if a[field] != b[field]
    ]
    matched = len(left) - len({index for index, *_ in mismatches})
    print(f"  {label}: {matched}/{len(left)} episodes identical on {len(EPISODE_FIELDS)} fields")
    assert not mismatches, "\n".join(
        f"    episode {index} {field}: {a!r} != {b!r}" for index, field, a, b in mismatches[:20]
    )


class _TraceCollector:
    """Record the decision-relevant state of every physics substep."""

    def __init__(self) -> None:
        self.rows: list[list[float]] = []

    def __call__(self, state: dict) -> None:
        data = state["data"]
        address = state["box_dof_address"]
        contact = state["contact"]
        self.rows.append(
            [
                float(state["time_s"]),
                float(_PHASE_ORDER.index(state["phase"])),
                *(float(v) for v in state["box_position"]),
                *(float(v) for v in data.qvel[address : address + 3]),
                *(float(v) for v in state["command_velocity"]),
                float(contact.left.active),
                float(contact.right.active),
                float(contact.left.normal_force),
                float(contact.right.normal_force),
                float(state["first_contact_peak_force_n"]),
                float(state["hold_timer"]),
                float(state["strict_hold_timer"]),
                float(bool(state["failure_reason"])),
            ]
        )

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.rows, dtype=float)


_TRACE_COLUMNS = {
    "time_s": (0, 1),
    "phase": (1, 2),
    "box_position": (2, 5),
    "box_velocity": (5, 8),
    "command_velocity": (8, 14),
    "contact_flags": (14, 16),
    "contact_forces": (16, 18),
    "first_contact_peak_force_n": (18, 19),
    "hold_timer": (19, 20),
    "strict_hold_timer": (20, 21),
    "failure_latched": (21, 22),
}


def _write_trace(checkpoint: Path, episodes: int, destination: Path) -> None:
    collector = _TraceCollector()
    _in_process_evaluation(checkpoint, episodes, step_callback=collector)
    np.save(destination, collector.array)


def test_a_paired_replay(checkpoint: Path, episodes: int) -> None:
    print("Test A: in-process evaluation vs cold-start replay")
    with tempfile.TemporaryDirectory() as directory:
        cold = _cold_start_evaluation(checkpoint, episodes, Path(directory))
    warm = _in_process_evaluation(checkpoint, episodes)
    _compare_episodes(warm, cold, "paired replay")


def test_b_step_trace(checkpoint: Path, episodes: int) -> None:
    print("Test B: step-level trace, fresh process vs warm process")
    with tempfile.TemporaryDirectory() as directory:
        cold_path = Path(directory) / "cold.npy"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__)), "--emit-trace", str(cold_path),
             "--checkpoint", str(checkpoint), "--episodes", str(episodes)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
        cold = np.load(cold_path)

    collector = _TraceCollector()
    # A "warm" process: one evaluation has already run here, so any state that
    # leaks between runs (caches, global RNG, module-level buffers) is present.
    _in_process_evaluation(checkpoint, episodes)
    _in_process_evaluation(checkpoint, episodes, step_callback=collector)
    warm = collector.array

    assert np.isfinite(cold).all(), "cold-start trace contains NaN/Inf"
    assert np.isfinite(warm).all(), "warm-process trace contains NaN/Inf"
    assert cold.shape == warm.shape, f"trace length {cold.shape} vs {warm.shape}"
    for name, (start, stop) in _TRACE_COLUMNS.items():
        difference = float(np.max(np.abs(cold[:, start:stop] - warm[:, start:stop])))
        tolerance = TRACE_TOLERANCES.get(name, 0.0)
        print(f"  {name}: max|diff| = {difference:.3e} (tolerance {tolerance:.0e})")
        assert difference <= tolerance, f"{name} drifted by {difference:.3e}"


def test_c_repeatability(checkpoint: Path, episodes: int, repeats: int = 3) -> None:
    print(f"Test C: {repeats} independent cold-start processes")
    with tempfile.TemporaryDirectory() as directory:
        runs = [
            _cold_start_evaluation(checkpoint, episodes, Path(directory) / f"run{index}")
            for index in range(repeats)
        ]
    reference = runs[0]
    successes = [tuple(e["success"] for e in run) for run in runs]
    assert len(set(successes)) == 1, f"success vectors differ: {successes}"
    worst = max(
        abs(a["first_contact_peak_force_n"] - b["first_contact_peak_force_n"])
        for run in runs[1:]
        for a, b in zip(reference, run)
    )
    print(f"  success vectors identical; max|first-contact force diff| = {worst:.3e} N")
    assert worst <= 1e-5, f"first-contact force drifted by {worst:.3e} N"


def test_e_environment_stamp(checkpoint: Path, episodes: int) -> None:
    print("Test E: evaluation logs carry a matching environment stamp")
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "result.json"
        _cold_start_evaluation(checkpoint, 1, Path(directory))
        stored = json.loads((Path(directory) / "result.json").read_text())["environment"]
    current = environment_stamp()
    for key in (
        "mujoco_python_version",
        "mujoco_native_version",
        "torch_version",
        "numpy_version",
        "scene_xml_sha256",
        "robot_xml_sha256",
    ):
        assert stored[key] == current[key], (
            f"{key}: log says {stored[key]!r}, this interpreter is {current[key]!r} -- "
            "results from these two are not comparable"
        )
    print(
        f"  mujoco={current['mujoco_python_version']}/"
        f"{current['mujoco_native_version']} torch={current['torch_version']} "
        f"numpy={current['numpy_version']}"
    )


def test_d_regression() -> None:
    print("Test D: pre-existing suites")
    for suite in (
        "tests/acmpc_residual_zero_test.py",
        "tests/acmpc_constraint_switch_test.py",
    ):
        completed = subprocess.run([sys.executable, suite], cwd=ROOT, capture_output=True, text=True)
        print(f"  {suite}: {'PASS' if completed.returncode == 0 else 'FAIL'}")
        assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--regression", action="store_true", help="also run Test D")
    parser.add_argument("--emit-trace", type=Path, default=None, help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if arguments.emit_trace is not None:
        _write_trace(arguments.checkpoint, arguments.episodes, arguments.emit_trace)
        return

    assert arguments.checkpoint.exists(), f"missing checkpoint: {arguments.checkpoint}"
    print(f"checkpoint={arguments.checkpoint} episodes={arguments.episodes}")
    test_a_paired_replay(arguments.checkpoint, arguments.episodes)
    test_b_step_trace(arguments.checkpoint, arguments.episodes)
    test_c_repeatability(arguments.checkpoint, arguments.episodes)
    test_e_environment_stamp(arguments.checkpoint, arguments.episodes)
    if arguments.regression:
        test_d_regression()
    print("all evaluation reproducibility tests passed")


if __name__ == "__main__":
    main()
