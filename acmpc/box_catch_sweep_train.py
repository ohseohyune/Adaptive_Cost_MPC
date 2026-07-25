"""W&B Sweep entrypoint for fixed-baseline box-catch phase_prior search.

Task 10 of the "Fixed MPC baseline" tuning program (see
fixed_mpc_baseline_tuning_program_status.md). Fixed-baseline means
online_learning=False + weight_delta_fraction=0.0 (Task 1's finding: this is
the only robust way to get a true zero-residual run regardless of a stale
checkpoint_path) -- run_one_config below always sets both.

Search space is intentionally just the two dims Task 9's manual 50-seed
sweep already found to help: HOLD's grasp weight and PRE_IMPACT's velocity
weight. Add a dim by adding one entry to SWEEP_PARAMETERS and one block to
sweep_config.yaml -- no other code changes needed.

Objective (sweep_config.yaml's metric): W&B sweep needs a single scalar, so
the requested safety-first priority (unsafe impact > phase reach > success
rate > command variation/solve time) is approximated as a weighted sum whose
tier weights (_TIER_WEIGHTS) are spread >=10x per tier -- a change in a
higher tier always outweighs any combination of changes in every lower tier,
within each metric's real [0, 1]-ish range. See compute_safety_first_objective.

Two disjoint seed sets: TRAIN_SEEDS (search + candidate reevaluation, via
--n-train-seeds) and HOLDOUT_SEEDS (only for the final chosen candidate --
call run_holdout_eval manually once a winner is picked, never during the
sweep itself).

Staging (the spec's steps 1/3/4) is one constant swap, not separate scripts:
STAGE_LOW_SPEED (default_curriculum's "warmup": narrow speed+mass+friction)
-> STAGE_WIDE_SPEED (same, only launch speed widened) -> STAGE_FULL
(default_curriculum's "full": wide mass/friction/size). Change ACTIVE_STAGE/
ACTIVE_STAGE_INDEX and re-run to move stages. Steps 2/5 (multi-seed
reevaluation / holdout) are the --n-train-seeds / run_holdout_eval knobs, not
separate stages.

Usage:
    wandb sweep sweep_config.yaml       # prints a SWEEP_ID
    wandb agent <entity>/<project>/<SWEEP_ID>

Local smoke test (no wandb account/network needed):
    python acmpc/box_catch_sweep_train.py --smoke-test
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.mpc.wandb_logger import build_sweep_summary_log
from control.squeeze import (
    CurriculumStage,
    DynamicSideSqueezeConfig,
    default_curriculum,
    resolve_ballistic_launch_position,
    resolve_ballistic_launch_velocity,
)
from acmpc.main_acmpc_box_catch import (
    AcmpcBoxCatchConfig,
    BoxCatchSummary,
    CatchControlPhase,
    EpisodeFunnel,
    _BOX_CATCH_PHASE_PRIORS,
    compute_episode_funnel,
    run_box_catch,
)

try:
    import wandb
except ImportError:  # pragma: no cover - exercised by --smoke-test without wandb
    wandb = None


# ---- search space -----------------------------------------------------------

SWEEP_PARAMETERS: dict[str, tuple[CatchControlPhase, int]] = {
    "hold_grasp_weight": (CatchControlPhase.HOLD, 1),
    "pre_impact_velocity_weight": (CatchControlPhase.PRE_IMPACT, 3),
}


def apply_overrides(overrides: dict[str, float]) -> tuple:
    rows = [list(row) for row in _BOX_CATCH_PHASE_PRIORS]
    for name, value in overrides.items():
        phase, cost_index = SWEEP_PARAMETERS[name]
        rows[int(phase)][cost_index] = float(value)
    return tuple(tuple(row) for row in rows)


# ---- seeds --------------------------------------------------------------------

BASE_SEED = 1000  # matches the tuning-program memory's Task 9 convention
DEFAULT_N_TRAIN_SEEDS = 20  # fast per-iteration default; --n-train-seeds 50 for stage 2
HOLDOUT_BASE_SEED = 9000  # disjoint from TRAIN_SEEDS at any n_train_seeds size used here
HOLDOUT_N_SEEDS = 30


def train_seeds(n: int = DEFAULT_N_TRAIN_SEEDS) -> range:
    return range(BASE_SEED, BASE_SEED + n)


def holdout_seeds() -> range:
    return range(HOLDOUT_BASE_SEED, HOLDOUT_BASE_SEED + HOLDOUT_N_SEEDS)


# ---- box domain stages (spec's steps 1/3/4) ------------------------------------

STAGE_LOW_SPEED = default_curriculum()[0]  # "warmup": narrow speed+mass+friction
STAGE_WIDE_SPEED = replace(
    STAGE_LOW_SPEED,
    name="wide_speed",
    launch_velocity_low=(-2.0, -0.03, 0.0),
    launch_velocity_high=(-1.0, 0.03, 0.0),
)
STAGE_FULL = default_curriculum()[2]  # "full": wide mass/friction/size
ACTIVE_STAGE = STAGE_WIDE_SPEED  # bump to STAGE_FULL for stage 4; was STAGE_LOW_SPEED for stage 1
ACTIVE_STAGE_INDEX = 0


# ---- objective ------------------------------------------------------------------

_TIER_WEIGHTS = (1000.0, 10.0, 1.0, 0.01)  # unsafe, phase-reach, success, command/solve


def compute_safety_first_objective(aggregate: dict) -> float:
    phase_reach = np.nanmean(
        [
            aggregate["p_pre_impact"],
            aggregate["p_impact_safe_given_pre_impact"],
            aggregate["p_bilateral_given_impact_safe"],
            aggregate["p_hold_given_bilateral"],
        ]
    )
    phase_reach = 0.0 if np.isnan(phase_reach) else float(phase_reach)
    command_cost = aggregate["mean_actor_output_variation"] + aggregate["mean_mpc_solve_time_s"]
    w_unsafe, w_phase, w_success, w_command = _TIER_WEIGHTS
    return float(
        -w_unsafe * aggregate["unsafe_rate"]
        + w_phase * phase_reach
        + w_success * aggregate["success_rate"]
        - w_command * command_cost
    )


# ---- episode rollout + aggregation -----------------------------------------------

def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def run_one_config(
    phase_priors: tuple,
    seeds: Iterable[int],
    stage: CurriculumStage,
    stage_index: int,
    *,
    device: str = "cpu",
) -> dict:
    """Run one fixed-baseline episode per seed under ``stage``'s domain and
    roll the results into the metrics the objective/logging need. Mirrors
    main_acmpc_box_catch_curriculum.py's per-episode domain/squeeze/config
    construction (same resolvers, same field names)."""

    base_squeeze = DynamicSideSqueezeConfig(random_seed=BASE_SEED)
    summaries: list[BoxCatchSummary] = []
    funnels: list[EpisodeFunnel] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        domain = stage.sample(rng, stage_index)
        sampled_launch_velocity = np.asarray(domain.launch_velocity, dtype=float)
        launch_position = resolve_ballistic_launch_position(base_squeeze, sampled_launch_velocity)
        launch_velocity = resolve_ballistic_launch_velocity(
            base_squeeze, sampled_launch_velocity, launch_position=launch_position
        )
        squeeze = DynamicSideSqueezeConfig(
            random_seed=seed,
            box_half_y=domain.half_size[1],
            launch_velocity_low=tuple(float(v) for v in launch_velocity),
            launch_velocity_high=tuple(float(v) for v in launch_velocity),
        )
        episode_config = AcmpcBoxCatchConfig(
            seed=seed,
            device=device,
            online_learning=False,
            weight_delta_fraction=0.0,
            squeeze=squeeze,
            domain_parameters=domain,
            phase_priors=phase_priors,
        )
        summary = run_box_catch(episode_config)
        summaries.append(summary)
        funnels.append(compute_episode_funnel(summary, episode_config))
    return _aggregate(summaries, funnels)


def _aggregate(summaries: list[BoxCatchSummary], funnels: list[EpisodeFunnel]) -> dict:
    n = len(summaries)
    pre_impact_population = [f for f in funnels if f.reached_pre_impact]
    passed_impact_stage = [f for f in funnels if f.first_contact_detected and f.impact_safe]
    bilateral_population = [f for f in funnels if f.bilateral_contact_achieved]
    hold_population = [f for f in funnels if f.hold_entered]
    contact_times = [
        s.bilateral_contact_time_s for s in summaries if s.bilateral_contact_time_s is not None
    ]
    # "emergency force와 unsafe impact" -- an episode that made contact but
    # exceeded the ImpactSafe force limit, or that ended via the hard
    # emergency-force termination outright (which can fire before any
    # bilateral contact is even detected).
    unsafe_count = sum(
        1
        for f in funnels
        if (f.first_contact_detected and not f.impact_safe) or f.failure_category == "emergency force"
    )

    return {
        "n": n,
        "success_rate": sum(1 for s in summaries if s.success) / n,
        "unsafe_rate": unsafe_count / n,
        "p_pre_impact": _rate(len(pre_impact_population), n),
        "p_impact_safe_given_pre_impact": _rate(
            sum(1 for f in pre_impact_population if f.first_contact_detected and f.impact_safe),
            len(pre_impact_population),
        ),
        "p_bilateral_given_impact_safe": _rate(
            sum(1 for f in passed_impact_stage if f.bilateral_contact_achieved), len(passed_impact_stage)
        ),
        "p_hold_given_bilateral": _rate(
            sum(1 for f in bilateral_population if f.hold_entered), len(bilateral_population)
        ),
        "p_success_given_hold": _rate(
            sum(1 for f in hold_population if f.episode_success), len(hold_population)
        ),
        "failure_category_counts": dict(
            Counter(f.failure_category for f in funnels if not f.episode_success)
        ),
        "mean_intercept_dwell_s": float(np.mean([s.intercept_dwell_s for s in summaries])),
        "mean_pre_impact_dwell_s": float(np.mean([s.pre_impact_dwell_s for s in summaries])),
        "mean_capture_dwell_s": float(np.mean([s.capture_dwell_s for s in summaries])),
        "mean_hold_dwell_s": float(np.mean([s.hold_dwell_s for s in summaries])),
        "mean_first_contact_peak_force_n": float(np.mean([s.first_contact_peak_force_n for s in summaries])),
        "max_first_contact_peak_force_n": float(np.max([s.first_contact_peak_force_n for s in summaries])),
        "mean_bilateral_contact_time_s": float(np.mean(contact_times)) if contact_times else float("nan"),
        "mean_hold_to_capture_demotion_count": float(
            np.mean([s.hold_to_capture_demotion_count for s in summaries])
        ),
        "mean_velocity_saturation_fraction": float(
            np.mean([s.velocity_saturation_fraction for s in summaries])
        ),
        "mean_mpc_solve_time_s": float(np.mean([s.mean_mpc_solve_time_s for s in summaries])),
        "mean_actor_output_variation": float(np.mean([s.mean_actor_output_variation for s in summaries])),
        "mean_residual_object": float(np.mean([s.mean_residual_object for s in summaries])),
        "mean_residual_grasp": float(np.mean([s.mean_residual_grasp for s in summaries])),
        "mean_residual_compression": float(np.mean([s.mean_residual_compression for s in summaries])),
        "mean_residual_velocity": float(np.mean([s.mean_residual_velocity for s in summaries])),
        "mean_residual_smoothness": float(np.mean([s.mean_residual_smoothness for s in summaries])),
    }


# ---- holdout final eval (call manually once a winner is picked) -----------------

def run_holdout_eval(overrides: dict[str, float]) -> dict:
    phase_priors = apply_overrides(overrides)
    return run_one_config(phase_priors, holdout_seeds(), ACTIVE_STAGE, ACTIVE_STAGE_INDEX)


# ---- wandb agent entrypoint ------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a few manual configs locally, no wandb agent/network needed.",
    )
    parser.add_argument("--n-train-seeds", type=int, default=DEFAULT_N_TRAIN_SEEDS)
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test(args.n_train_seeds)
        return

    if wandb is None:
        raise RuntimeError("wandb is not installed; pip install wandb to run as a sweep agent.")
    run = wandb.init()
    overrides = {name: float(run.config[name]) for name in SWEEP_PARAMETERS if name in run.config}
    phase_priors = apply_overrides(overrides)
    aggregate = run_one_config(phase_priors, train_seeds(args.n_train_seeds), ACTIVE_STAGE, ACTIVE_STAGE_INDEX)
    objective = compute_safety_first_objective(aggregate)
    wandb.log(build_sweep_summary_log(aggregate=aggregate, objective=objective, overrides=overrides))
    wandb.finish()


def _run_smoke_test(n_train_seeds: int) -> None:
    candidates = {
        "baseline": {},
        "hold_grasp_0": {"hold_grasp_weight": 0.0},
        "pre_impact_velocity_6": {"pre_impact_velocity_weight": 6.0},
        "combined": {"hold_grasp_weight": 0.0, "pre_impact_velocity_weight": 6.0},
    }
    seeds = train_seeds(n_train_seeds)
    print(f"smoke test: {len(seeds)} seeds, stage={ACTIVE_STAGE.name}")
    for label, overrides in candidates.items():
        phase_priors = apply_overrides(overrides)
        aggregate = run_one_config(phase_priors, seeds, ACTIVE_STAGE, ACTIVE_STAGE_INDEX)
        objective = compute_safety_first_objective(aggregate)
        print(
            f"{label:24s} objective={objective:8.3f} success_rate={aggregate['success_rate']:.2f} "
            f"unsafe_rate={aggregate['unsafe_rate']:.2f} n={aggregate['n']}"
        )


if __name__ == "__main__":
    main()
