"""Run the cumulative D0 -> D1 -> D2 -> D3 -> M0 ablation.

Each job owns an independent checkpoint and starts cold unless ``--resume``
finds a completed JSON result. Smoke jobs run sequentially; pilot jobs may run
concurrently, but every condition/seed still receives the same training and
evaluation seed sequences.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.runtime_environment import (  # noqa: E402
    log_runtime_environment,
    validate_runtime_environment,
)
TRAINER = ROOT / "acmpc" / "main_acmpc_box_catch_curriculum.py"
VERIFY = ROOT / "tests" / "acmpc_constraint_switch_test.py"
ANALYZE = ROOT / "acmpc" / "analyze_constraint_ablation.py"


@dataclass(frozen=True)
class Condition:
    name: str
    weight_parameterization: str = "bounded_residual"
    weight_clip_min: str = "0.001"
    weight_clip_max: str = "500"
    cumulative_cap: str = "0.4"
    online_cap: str = "0.02"
    target_kl: str = "0.02"


CONDITIONS = (
    Condition("D0"),
    Condition("D1", cumulative_cap="off"),
    Condition(
        "D2",
        weight_parameterization="exp_residual",
        weight_clip_min="off",
        weight_clip_max="off",
        cumulative_cap="off",
    ),
    Condition(
        "D3",
        weight_parameterization="exp_residual",
        weight_clip_min="off",
        weight_clip_max="off",
        cumulative_cap="off",
        online_cap="off",
    ),
    Condition(
        "M0",
        weight_parameterization="exp_residual",
        weight_clip_min="off",
        weight_clip_max="off",
        cumulative_cap="off",
        online_cap="off",
        target_kl="off",
    ),
)


@dataclass(frozen=True)
class Job:
    phase: str
    condition: Condition
    seed: int
    episodes: int
    output_dir: Path
    evaluation_every: int = 0
    evaluation_episodes: int = 0

    @property
    def log_path(self) -> Path:
        return self.output_dir / "result.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoint.pt"

    @property
    def stdout_path(self) -> Path:
        return self.output_dir / "stdout.log"


def _complete(job: Job) -> bool:
    if not job.log_path.exists():
        return False
    try:
        data = json.loads(job.log_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("episodes") != job.episodes:
        return False
    expected_evaluations = (
        job.episodes // job.evaluation_every if job.evaluation_every else 0
    )
    return len(data.get("evaluation_summaries", [])) == expected_evaluations


def _command(job: Job, device: str, evaluation_seed: int) -> list[str]:
    condition = job.condition
    command = [
        sys.executable,
        str(TRAINER),
        "--episodes",
        str(job.episodes),
        "--seed",
        str(job.seed),
        "--device",
        device,
        "--curriculum-mode",
        "adaptive",
        "--checkpoint",
        str(job.checkpoint_path),
        "--log",
        str(job.log_path),
        "--progress-every",
        "1" if job.phase == "smoke" else "20",
        "--weight-delta-fraction",
        "0.65",
        "--weight-parameterization",
        condition.weight_parameterization,
        "--weight-clip-min",
        condition.weight_clip_min,
        "--weight-clip-max",
        condition.weight_clip_max,
        "--maximum-cumulative-actor-delta",
        condition.cumulative_cap,
        "--maximum-online-actor-delta",
        condition.online_cap,
        "--target-kl",
        condition.target_kl,
        "--clip-ratio",
        "0.15",
        "--evaluation-seed",
        str(evaluation_seed),
    ]
    if job.evaluation_every:
        command.extend(
            [
                "--evaluation-every",
                str(job.evaluation_every),
                "--evaluation-episodes",
                str(job.evaluation_episodes),
            ]
        )
    return command


def _run_jobs(
    jobs: list[Job],
    *,
    device: str,
    evaluation_seed: int,
    concurrency: int,
    resume: bool,
) -> bool:
    pending = list(jobs)
    running: dict[subprocess.Popen, tuple[Job, object, float]] = {}
    all_ok = True
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    while pending or running:
        while pending and len(running) < concurrency:
            job = pending.pop(0)
            if resume and _complete(job):
                print(f"[skip] {job.phase} {job.condition.name} seed={job.seed}", flush=True)
                continue
            job.output_dir.mkdir(parents=True, exist_ok=True)
            stream = job.stdout_path.open("w")
            command = _command(job, device, evaluation_seed)
            print(
                f"[start] {job.phase} {job.condition.name} seed={job.seed} "
                f"episodes={job.episodes}",
                flush=True,
            )
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            running[process] = (job, stream, time.monotonic())
        if not running:
            continue
        time.sleep(1.0)
        for process, (job, stream, started) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            elapsed = time.monotonic() - started
            complete = return_code == 0 and _complete(job)
            all_ok &= complete
            try:
                output_lines = job.stdout_path.read_text(errors="replace").splitlines()
            except OSError:
                output_lines = []
            (job.output_dir / "status.json").write_text(
                json.dumps(
                    {
                        "phase": job.phase,
                        "condition": job.condition.name,
                        "seed": job.seed,
                        "return_code": return_code,
                        "complete": complete,
                        "elapsed_seconds": elapsed,
                        "last_log_lines": output_lines[-100:],
                    },
                    indent=2,
                )
            )
            print(
                f"[{'done' if complete else 'failed'}] {job.phase} "
                f"{job.condition.name} seed={job.seed} elapsed={elapsed / 60:.1f}m "
                f"log={job.stdout_path}",
                flush=True,
            )
            del running[process]
    return all_ok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "pilot", "all"], default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-seed", type=int, default=100_000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "sweep_results"
        / f"constraint_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


def main() -> None:
    # Parent-side gate: refuse to spawn a fleet of hour-long jobs from a
    # non-canonical interpreter. Each child re-validates independently, since
    # the parent cannot vouch for an interpreter it does not run in.
    log_runtime_environment(
        "constraint ablation", validate_runtime_environment(context="constraint ablation runner")
    )
    args = _parse_args()
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    verification = subprocess.run([sys.executable, str(VERIFY)], cwd=ROOT)
    if verification.returncode:
        raise SystemExit("implementation verification failed")

    smoke_ok = True
    if args.phase in {"smoke", "all"}:
        smoke_jobs = [
            Job("smoke", condition, 7, 20, args.output_root / "smoke" / condition.name)
            for condition in CONDITIONS
        ]
        smoke_ok = _run_jobs(
            smoke_jobs,
            device=args.device,
            evaluation_seed=args.evaluation_seed,
            concurrency=1,
            resume=args.resume,
        )
    if not smoke_ok:
        raise SystemExit("one or more smoke conditions failed; pilot was not started")

    pilot_ok = True
    if args.phase in {"pilot", "all"}:
        pilot_jobs = [
            Job(
                "pilot",
                condition,
                seed,
                400,
                args.output_root / "pilot" / condition.name / f"seed_{seed}",
                evaluation_every=50,
                evaluation_episodes=50,
            )
            for condition in CONDITIONS
            for seed in (7, 17, 27)
        ]
        pilot_ok = _run_jobs(
            pilot_jobs,
            device=args.device,
            evaluation_seed=args.evaluation_seed,
            concurrency=args.jobs,
            resume=args.resume,
        )
        subprocess.run(
            [sys.executable, str(ANALYZE), "--root", str(args.output_root)],
            cwd=ROOT,
            check=False,
        )
    if not pilot_ok:
        raise SystemExit("one or more pilot conditions failed; partial analysis was generated")
    print(f"results: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
