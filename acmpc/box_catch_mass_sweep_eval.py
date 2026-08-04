"""Box-mass robustness sweep: plain fixed-cost MPC vs learned condition-3.

Reproduces the paper's Fig.7-style OOD comparison (there: sweep the
quadrotor's own mass/inertia/rate-limit against a fixed-cost MPC vs
AC-MPC). Our MPC has no mass/inertia model of the *arm* (it's a pure
Jacobian + integrator QP, see adaptive_cost_mpc.py), so the arm side has no
analogous mismatch to sweep. The box's mass is the parameter that actually
enters the learned cost's grasp/force weighting, and both policies here
were trained with box mass pinned at the nominal 0.50 kg (never
randomized -- see main_acmpc_box_catch_prior_free_curriculum.py's
"warmup" stage, which fixes mass/friction/size and only ramps launch
speed/distance), so a mass sweep is a genuine train/test distribution
shift for both.

"Plain" baseline = an untrained PriorFreeCostActor with uniform initial
weights (prior_free_initial_weights=(5.0,)*5), frozen (online_learning=
False). This is deliberately the same failure-mode baseline documented in
acmpc_3way_ablation_condition3_status.md's "first attempt" (0% success,
uniform weights, no phase table) -- the closest analogue this codebase has
to the paper's fixed, unscheduled tracking-MPC cost.

Usage:
    python3 acmpc/box_catch_mass_sweep_eval.py --learned-checkpoint \
        sweep_results/acmpc_box_catch_prior_free_expl_v2.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.squeeze import BoxDomainParameters, DynamicSideSqueezeConfig, default_curriculum
from acmpc.main_acmpc_box_catch import AcmpcBoxCatchConfig, run_box_catch

PLAIN_BASELINE_CHECKPOINT = ROOT / "sweep_results" / "box_catch_plain_baseline.pt"
MASS_VALUES = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00)
N_SEEDS = 20
BASE_SEED = 1000


def _episode_config(
    *, seed: int, mass: float, checkpoint_path: str, device: str
) -> AcmpcBoxCatchConfig:
    stage = default_curriculum()[0]  # "warmup": near-nominal size/friction/launch
    domain = stage.sample(np.random.default_rng(seed), 0)
    domain = replace(domain, mass=mass)
    domain.validate()
    squeeze = DynamicSideSqueezeConfig(
        random_seed=seed,
        box_half_y=domain.half_size[1],
        launch_velocity_low=domain.launch_velocity,
        launch_velocity_high=domain.launch_velocity,
    )
    return AcmpcBoxCatchConfig(
        seed=seed,
        device=device,
        online_learning=False,
        use_prior_free_actor=True,
        prior_free_initial_weights=(5.0, 5.0, 5.0, 5.0, 5.0),
        squeeze=squeeze,
        domain_parameters=domain,
        checkpoint_path=checkpoint_path,
    )


def _ensure_plain_baseline_checkpoint(device: str) -> None:
    if PLAIN_BASELINE_CHECKPOINT.exists():
        return
    cfg = _episode_config(
        seed=0, mass=0.50, checkpoint_path=str(PLAIN_BASELINE_CHECKPOINT), device=device
    )
    run_box_catch(cfg)


def run_sweep(learned_checkpoint: str, device: str = "cpu") -> dict:
    _ensure_plain_baseline_checkpoint(device)
    results: dict[str, dict[float, float]] = {"plain": {}, "learned": {}}
    for mass in MASS_VALUES:
        for label, checkpoint in (
            ("plain", str(PLAIN_BASELINE_CHECKPOINT)),
            ("learned", learned_checkpoint),
        ):
            successes = 0
            for i in range(N_SEEDS):
                cfg = _episode_config(
                    seed=BASE_SEED + i, mass=mass, checkpoint_path=checkpoint, device=device
                )
                summary = run_box_catch(cfg)
                successes += int(summary.success)
            results[label][mass] = successes / N_SEEDS
            print(f"mass={mass:.2f} {label:8s} success_rate={results[label][mass]:.2f}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learned-checkpoint",
        default=str(ROOT / "sweep_results" / "acmpc_box_catch_prior_free_expl_v2.pt"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out", default=str(ROOT / "sweep_results" / "box_catch_mass_sweep.json")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_sweep(args.learned_checkpoint, device=args.device)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
