"""Deterministic learned-vs-residual-zero evaluation of existing checkpoints.

No training happens here. Every job loads one already-trained checkpoint from
the constraint ablation and replays the same fixed, balanced evaluation set
twice: once with the learned mean residual (mode ``learned``) and once with
the actor's residual forced to zero so the MPC runs on the engineered phase
prior alone (mode ``zero``). Both modes use ``online_learning=False``, i.e.
the deterministic policy mean -- no exploration noise, no updates.

The checkpoint is copied into each job's output directory before the run so a
bug in the evaluation path can never touch the trained artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.runtime_environment import (  # noqa: E402
    log_runtime_environment,
    validate_runtime_environment,
)
TRAINER = ROOT / "acmpc" / "main_acmpc_box_catch_curriculum.py"
AUDITS = ROOT / "tests" / "acmpc_residual_zero_test.py"


@dataclass(frozen=True)
class Condition:
    name: str
    weight_parameterization: str = "bounded_residual"
    weight_clip_min: str = "0.001"
    weight_clip_max: str = "500"


# Only the switches that change how a checkpoint's actor output maps to cost
# weights matter for replay; the learning-side caps (cumulative/online/KL) are
# inert with online_learning=False and are deliberately omitted.
CONDITIONS = (
    Condition("D0"),
    Condition("D1"),
    Condition("D2", "exp_residual", "off", "off"),
    Condition("D3", "exp_residual", "off", "off"),
    Condition("M0", "exp_residual", "off", "off"),
)
SEEDS = (7, 17, 27)
MODES = ("learned", "zero")


@dataclass(frozen=True)
class Job:
    condition: Condition
    seed: int
    mode: str
    source_checkpoint: Path
    output_dir: Path
    episodes: int
    evaluation_seed: int

    @property
    def result_path(self) -> Path:
        return self.output_dir / "result.json"

    @property
    def stdout_path(self) -> Path:
        return self.output_dir / "stdout.log"


def _complete(job: Job) -> bool:
    if not job.result_path.exists():
        return False
    try:
        data = json.loads(job.result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("episodes") == job.episodes


def _command(job: Job, device: str) -> list[str]:
    command = [
        sys.executable,
        str(TRAINER),
        # --evaluation-replay routes through evaluation_replay_config, the
        # same constructor the in-process evaluation uses, so a replay and a
        # training-time evaluation of one checkpoint cannot differ in which
        # switches they carry (episodes/seed/mode/online-learning are all set
        # there, not here).
        "--evaluation-replay",
        "--evaluation-episodes",
        str(job.episodes),
        # Scenario sampling is driven by this seed, so both modes see an
        # identical episode-by-episode domain sequence -- the pairing the
        # bootstrap in analyze_residual_zero.py depends on.
        "--evaluation-seed",
        str(job.evaluation_seed),
        "--device",
        device,
        "--checkpoint",
        str(job.output_dir / "checkpoint.pt"),
        "--log",
        str(job.result_path),
        "--progress-every",
        str(job.episodes + 1),
        "--weight-delta-fraction",
        "0.65",
        "--weight-parameterization",
        job.condition.weight_parameterization,
        "--weight-clip-min",
        job.condition.weight_clip_min,
        "--weight-clip-max",
        job.condition.weight_clip_max,
        "--evaluation-every",
        "0",
    ]
    if job.mode == "zero":
        command.append("--residual-zero")
    else:
        # ||u_learned_mean - u_zero_residual|| at every learned-mode control
        # step; costs a second MPC solve per step, so only mode A pays it.
        command.append("--log-command-delta")
    return command


def _run(jobs: list[Job], *, device: str, concurrency: int, resume: bool) -> bool:
    pending = list(jobs)
    running: dict[subprocess.Popen, tuple[Job, object, float]] = {}
    all_ok = True
    # Same cap the constraint-ablation runner uses: without it each job takes
    # every core and `concurrency` jobs oversubscribe the machine badly.
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    while pending or running:
        while pending and len(running) < concurrency:
            job = pending.pop(0)
            if resume and _complete(job):
                print(f"[skip] {job.condition.name} seed={job.seed} {job.mode}", flush=True)
                continue
            job.output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(job.source_checkpoint, job.output_dir / "checkpoint.pt")
            command = _command(job, device)
            # The interpreter and the exact argv that produced result.json --
            # the one fact missing when a 1.00 and a 0.92 evaluation of the
            # same checkpoint had to be told apart after the fact.
            (job.output_dir / "job_metadata.json").write_text(
                json.dumps(
                    {
                        "interpreter": sys.executable,
                        "command": command,
                        "environment": validate_runtime_environment(
                            context="residual-zero job"
                        ),
                    },
                    indent=2,
                )
            )
            stream = job.stdout_path.open("w")
            print(
                f"[start] {job.condition.name} seed={job.seed} {job.mode} "
                f"episodes={job.episodes}",
                flush=True,
            )
            running[subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )] = (job, stream, time.monotonic())
        if not running:
            continue
        time.sleep(1.0)
        for process, (job, stream, started) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            complete = return_code == 0 and _complete(job)
            all_ok &= complete
            print(
                f"[{'done' if complete else 'FAIL'}] {job.condition.name} "
                f"seed={job.seed} {job.mode} elapsed="
                f"{(time.monotonic() - started) / 60:.1f}m",
                flush=True,
            )
            del running[process]
    return all_ok


def main() -> None:
    # Parent-side gate: refuse to spawn a fleet of hour-long jobs from a
    # non-canonical interpreter. Each child re-validates independently, since
    # the parent cannot vouch for an interpreter it does not run in.
    log_runtime_environment(
        "residual-zero ablation", validate_runtime_environment(context="residual-zero ablation runner")
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "sweep_results" / "constraint_ablation_20260804" / "pilot",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "sweep_results" / "residual_zero_20260805",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=100_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-audits", action="store_true")
    args = parser.parse_args()

    if not args.skip_audits:
        audit = subprocess.run([sys.executable, str(AUDITS)], cwd=ROOT)
        if audit.returncode:
            raise SystemExit("residual-zero audits failed; evaluation not started")

    jobs: list[Job] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            source = args.source_root / condition.name / f"seed_{seed}" / "checkpoint.pt"
            if not source.exists():
                raise SystemExit(f"missing checkpoint: {source}")
            for mode in MODES:
                jobs.append(
                    Job(
                        condition=condition,
                        seed=seed,
                        mode=mode,
                        source_checkpoint=source,
                        output_dir=args.output_root
                        / condition.name
                        / f"seed_{seed}"
                        / mode,
                        episodes=args.episodes,
                        evaluation_seed=args.evaluation_seed,
                    )
                )

    ok = _run(jobs, device=args.device, concurrency=args.jobs, resume=args.resume)
    print(f"results: {args.output_root}", flush=True)
    if not ok:
        raise SystemExit("one or more evaluation jobs failed")


if __name__ == "__main__":
    main()
