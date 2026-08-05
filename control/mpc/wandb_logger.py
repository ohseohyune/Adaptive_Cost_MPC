"""Optional Weights & Biases logging for the AC-MPC box-catch pipeline.

Every value logged here is read from an already-computed quantity passed in
by the caller (see acmpc/main_acmpc_box_catch.py and
acmpc/main_acmpc_box_catch_curriculum.py for where each one comes from) --
this module only names/groups/derives (final - prior, ratios, unit
conversion) them for wandb.log, it never re-runs physics, the MPC solve, or
a network forward pass.

If ``wandb`` is not installed, or a caller never enables it, every method
here is a no-op -- callers do not need to branch on availability themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import wandb as _wandb
except ImportError:  # pragma: no cover - exercised by the "not installed" path
    _wandb = None

from control.mpc.ppo_common import PPOUpdateSummary
from control.squeeze.pad_contact import BilateralPadContact

COST_NAMES = ("object", "grasp", "compression", "velocity", "smoothness")
_RESIDUAL_EPSILON = 1e-6


class WandbLogger:
    """Thin wrapper: a disabled/unavailable run makes every call a no-op."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        run_name: str,
        config: dict[str, Any],
    ) -> None:
        self.enabled = bool(enabled and _wandb is not None)
        if enabled and _wandb is None:
            print("wandb is not installed; continuing without W&B logging.")
        self._run = None
        if self.enabled:
            self._run = _wandb.init(project=project, name=run_name, config=config)

    def update_config(self, values: dict[str, Any]) -> None:
        if self.enabled and self._run is not None:
            self._run.config.update(values, allow_val_change=True)

    def log(self, data: dict[str, Any], *, step: int) -> None:
        if self.enabled:
            _wandb.log(data, step=int(step))

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            _wandb.finish()
            self._run = None


def build_mpc_weight_log(
    *,
    final_weights: dict[str, np.ndarray],
    preclip_weights: dict[str, np.ndarray] | None = None,
    phase_prior: np.ndarray,
    phase_id: int,
    solver_time_s: float,
    hessian_condition_number: float = float("nan"),
    hessian_min_eigenvalue: float = float("nan"),
    linear_solve_residual: float = float("nan"),
) -> dict[str, Any]:
    """mpc/*, mpc/phase_prior/*, mpc/final_weight/*, mpc/residual_*/*, state/phase_id.

    ``final_weights``: ACMPCAction.weights (per-cost-dim, per-horizon-step,
    already residual-applied and clamped -- see AdaptiveCostActor.forward).
    ``phase_prior``: the blended prior row actually fed to the actor for
    this step (ACMPCAction.phase_prior / current_blended_prior), i.e.
    post-blend, pre-residual. Step-0 (the value that actually reaches this
    control step's MPC solve) is used for every *_weight entry.
    """

    phase_prior = np.asarray(phase_prior, dtype=float).reshape(len(COST_NAMES))
    data: dict[str, Any] = {
        "mpc/solver_time": float(solver_time_s) * 1000.0,  # ms
        "mpc/hessian_condition_number": float(hessian_condition_number),
        "mpc/hessian_min_eigenvalue": float(hessian_min_eigenvalue),
        "mpc/linear_solve_residual": float(linear_solve_residual),
        "state/phase_id": int(phase_id),
    }
    absolute_relative_mean: list[float] = []
    for index, name in enumerate(COST_NAMES):
        label = name
        final_value = float(final_weights[name][0])
        preclip_value = float(
            final_value if preclip_weights is None else preclip_weights[name][0]
        )
        prior_value = float(phase_prior[index])
        absolute_residual = final_value - prior_value
        relative_residual = (
            float("nan")
            if abs(prior_value) < _RESIDUAL_EPSILON
            else absolute_residual / (abs(prior_value) + _RESIDUAL_EPSILON)
        )
        data[f"mpc/{label}_weight"] = final_value
        data[f"mpc/phase_prior/{label}_weight"] = prior_value
        data[f"mpc/final_weight/{label}_weight"] = final_value
        data[f"mpc/preclip_weight/{label}_weight"] = preclip_value
        data[f"mpc/clip_removed/{label}_weight"] = abs(preclip_value - final_value)
        data[f"mpc/residual_absolute/{label}_weight"] = absolute_residual
        data[f"mpc/residual_relative/{label}_weight"] = relative_residual
        if not np.isnan(relative_residual):
            absolute_relative_mean.append(abs(relative_residual))
    data["actor/mean_abs_residual_relative"] = (
        float(np.mean(absolute_relative_mean)) if absolute_relative_mean else float("nan")
    )
    return data


def build_contact_log(
    *,
    contact: BilateralPadContact,
    box_velocity: np.ndarray,
) -> dict[str, Any]:
    """catch/*.

    ``slip_velocity`` has no source in the current pad_contact/impact
    pipeline (ContactInfo carries no velocity field, and impact.py's
    relative_normal_speed is the normal approach speed, not tangential
    slip) -- logged as NaN rather than invented. See the box-catch W&B
    integration report for what would need to be added (contact-frame
    relative surface velocity from data.contact[i].frame + body cvel).
    """

    return {
        "catch/left_normal_force": float(contact.left.normal_force),
        "catch/right_normal_force": float(contact.right.normal_force),
        "catch/slip_velocity": float("nan"),  # TODO: no source yet, see docstring
        "catch/box_velocity": float(np.linalg.norm(np.asarray(box_velocity, dtype=float))),
        "catch/bilateral_contact": 1.0 if contact.bilateral else 0.0,
    }


def build_ppo_update_log(summary: PPOUpdateSummary) -> dict[str, Any]:
    """train/actor_loss, train/critic_loss, train/entropy, train/explained_variance."""

    return {
        "train/actor_loss": float(summary.actor_loss),
        "train/critic_loss": float(summary.critic_loss),
        "train/entropy": float(summary.entropy),
        "train/explained_variance": (
            float("nan")
            if summary.explained_variance is None
            else float(summary.explained_variance)
        ),
        "train/approximate_kl": float(summary.approximate_kl),
        "train/actor_parameter_delta": float(summary.actor_parameter_delta),
        "train/raw_actor_parameter_delta": float(summary.raw_actor_parameter_delta),
        "train/actor_update_applied_fraction": float(
            summary.actor_update_applied_fraction
        ),
        "train/online_delta_clipped": int(summary.online_delta_clipped),
        "train/pre_projection_kl": float(summary.pre_projection_kl),
        "train/target_kl_stopped": int(summary.target_kl_stopped),
        "train/skipped_epochs": int(summary.skipped_epochs),
        "train/kl_projection_count": int(summary.kl_projection_count),
        "train/kl_rollback_count": int(summary.kl_rollback_count),
        "train/projection_removed_delta": float(summary.projection_removed_delta),
        "train/cumulative_actor_parameter_delta": float(
            summary.cumulative_actor_parameter_delta
        ),
        "train/cumulative_delta_clipped": int(summary.cumulative_delta_clipped),
        "train/actor_parameter_path_length": float(
            summary.actor_parameter_path_length
        ),
        "train/actor_path_displacement_ratio": float(
            summary.actor_path_displacement_ratio
        ),
        "train/ppo_clip_fraction": float(summary.ppo_clip_fraction),
        "train/advantage_std": float(summary.advantage_std),
        "train/actor_grad_norm": float(summary.actor_grad_norm),
        "train/critic_grad_norm": float(summary.critic_grad_norm),
        "train/advantage_mean": float(summary.advantage_mean),
        "train/online_delta_removed": float(summary.online_delta_removed),
        "train/cumulative_projection_removed_delta": float(
            summary.cumulative_projection_removed_delta
        ),
        "train/transitions": int(summary.transitions),
    }


def build_episode_reward_log(episode_reward: float) -> dict[str, Any]:
    """train/episode_reward."""

    return {"train/episode_reward": float(episode_reward)}


def build_eval_log(
    *,
    success_rate: float,
    impact_safe_rate: float,
    stable_hold_rate: float,
    mean_hold_time: float,
) -> dict[str, Any]:
    """eval/success_rate, eval/impact_safe_rate, eval/stable_hold_rate, eval/mean_hold_time."""

    return {
        "eval/success_rate": float(success_rate),
        "eval/impact_safe_rate": float(impact_safe_rate),
        "eval/stable_hold_rate": float(stable_hold_rate),
        "eval/mean_hold_time": float(mean_hold_time),
    }


def _failure_count_log(failure_category_counts: dict[str, int]) -> dict[str, Any]:
    """failure/<category> counts, category names sanitized for a wandb key."""

    data: dict[str, Any] = {}
    for category, count in failure_category_counts.items():
        key = category.replace(" ", "_").replace("/", "_").replace("-", "_")
        data[f"failure/{key}"] = int(count)
    return data


def build_funnel_log(
    *,
    p_pre_impact: float,
    p_impact_safe_given_pre_impact: float,
    p_bilateral_given_impact_safe: float,
    p_hold_given_bilateral: float,
    p_success_given_hold: float,
    failure_category_counts: dict[str, int],
) -> dict[str, Any]:
    """funnel/p_* conditional success rates + failure/<category> counts.

    Conditional probabilities are all NaN-safe (0/0 -> NaN, not 0.0) so an
    empty stage population reads as "no data" rather than a fabricated zero.
    failure_category_counts should come from EpisodeFunnel.failure_category
    (main_acmpc_box_catch.py) -- one mutually-exclusive category per failed
    episode, so the counts sum to exactly the window's failure count.
    """

    return {
        "funnel/p_pre_impact": float(p_pre_impact),
        "funnel/p_impact_safe_given_pre_impact": float(p_impact_safe_given_pre_impact),
        "funnel/p_bilateral_given_impact_safe": float(p_bilateral_given_impact_safe),
        "funnel/p_hold_given_bilateral": float(p_hold_given_bilateral),
        "funnel/p_success_given_hold": float(p_success_given_hold),
        **_failure_count_log(failure_category_counts),
    }


def build_curriculum_log(
    *,
    anchor_stage: str,
    stage_index: int,
    cooldown_remaining: int,
    anchor_success_rate: float,
) -> dict[str, Any]:
    """curriculum/* -- progressive-mode-only scheduler state (see
    control/squeeze/progressive_curriculum.py's AdaptiveCurriculumScheduler).
    Legacy adaptive/balanced curriculum_mode never calls this."""

    return {
        "curriculum/anchor_stage": anchor_stage,
        "curriculum/stage_index": int(stage_index),
        "curriculum/cooldown_remaining": int(cooldown_remaining),
        "curriculum/anchor_success_rate": float(anchor_success_rate),
    }


def build_difficulty_log(*, difficulty) -> dict[str, Any]:
    """difficulty/* from a control.squeeze.progressive_curriculum.
    BoxCatchDifficulty. Takes the dataclass itself (duck-typed, avoids an
    import cycle: control/squeeze depends on nothing in control/mpc)."""

    return {
        "difficulty/ttc": float(difficulty.ttc_s),
        "difficulty/impact_speed": float(difficulty.impact_speed_mps),
        "difficulty/impact_energy": float(difficulty.impact_energy_j),
        "difficulty/hold": float(difficulty.hold_difficulty),
        "difficulty/total": float(difficulty.total_difficulty),
    }


def build_domain_sample_log(
    *,
    mass: float,
    friction: float,
    half_size: tuple[float, float, float],
    angular_velocity: tuple[float, float, float],
    sampled_launch_velocity: tuple[float, float, float],
    resolved_launch_velocity: tuple[float, float, float],
    resample_count: int,
) -> dict[str, Any]:
    """domain/* -- the sampled + resolved BoxDomainParameters for one episode
    (progressive mode; see sample_stage_domain's resample-with-recheck)."""

    return {
        "domain/sample_mass": float(mass),
        "domain/sample_friction": float(friction),
        "domain/sample_size": list(float(v) for v in half_size),
        "domain/sample_angular_velocity": list(float(v) for v in angular_velocity),
        "domain/sample_launch_velocity": list(float(v) for v in sampled_launch_velocity),
        "domain/resolved_launch_velocity": list(float(v) for v in resolved_launch_velocity),
        "domain/resample_count": int(resample_count),
    }


def build_fixture_log(
    *,
    fixture_released: bool,
    fixture_release_time_s: float | None,
    bilateral_contact_duration_at_release_s: float,
    pre_release_peak_contact_force_n: float,
    post_release_peak_contact_force_n: float,
    post_release_hold_duration_s: float,
    fixture_release_force_threshold_n: float = 0.0,
    fixture_release_left_force_n: float = float("nan"),
    fixture_release_right_force_n: float = float("nan"),
    fixture_release_force_dwell_s: float = 0.0,
    fixture_release_force_safety_factor: float = 0.0,
) -> dict[str, Any]:
    """stage0/* -- static_grasp_bootstrap fixture lifecycle (see
    AcmpcBoxCatchConfig.use_launch_fixture). Only called when that flag is
    set; no other caller touches this."""

    return {
        "stage0/fixture_release_rate": float(fixture_released),
        "stage0/fixture_release_time": (
            float(fixture_release_time_s) if fixture_release_time_s is not None else float("nan")
        ),
        "stage0/bilateral_contact_dwell": float(bilateral_contact_duration_at_release_s),
        "stage0/pre_release_peak_force": float(pre_release_peak_contact_force_n),
        "stage0/post_release_peak_force": float(post_release_peak_contact_force_n),
        "stage0/post_release_hold_duration": float(post_release_hold_duration_s),
        "stage0/fixture_release_force_threshold": float(fixture_release_force_threshold_n),
        "stage0/fixture_release_left_force": float(fixture_release_left_force_n),
        "stage0/fixture_release_right_force": float(fixture_release_right_force_n),
        "stage0/fixture_release_force_dwell": float(fixture_release_force_dwell_s),
        "stage0/fixture_release_force_safety_factor": float(fixture_release_force_safety_factor),
    }


def build_sweep_summary_log(
    *,
    aggregate: dict[str, Any],
    objective: float,
    overrides: dict[str, float],
) -> dict[str, Any]:
    """sweep/*, funnel/*, catch/*, mpc/*, phase/*, cost_residual/*, failure/*.

    One call per completed sweep run (an N-seed batch for one candidate
    phase_priors override, from acmpc/box_catch_sweep_train.py's
    run_one_config/_aggregate) -- not per episode. Only renames/flattens an
    already-computed aggregate dict; no new metric is derived here.
    """

    data: dict[str, Any] = {
        "sweep/safety_first_objective": float(objective),
        "sweep/success_rate": float(aggregate["success_rate"]),
        "sweep/unsafe_rate": float(aggregate["unsafe_rate"]),
        "sweep/n_seeds": int(aggregate["n"]),
        "funnel/p_pre_impact": float(aggregate["p_pre_impact"]),
        "funnel/p_impact_safe_given_pre_impact": float(aggregate["p_impact_safe_given_pre_impact"]),
        "funnel/p_bilateral_given_impact_safe": float(aggregate["p_bilateral_given_impact_safe"]),
        "funnel/p_hold_given_bilateral": float(aggregate["p_hold_given_bilateral"]),
        "funnel/p_success_given_hold": float(aggregate["p_success_given_hold"]),
        "catch/mean_first_contact_peak_force": float(aggregate["mean_first_contact_peak_force_n"]),
        "catch/max_first_contact_peak_force": float(aggregate["max_first_contact_peak_force_n"]),
        "catch/mean_bilateral_contact_time_s": float(aggregate["mean_bilateral_contact_time_s"]),
        "catch/mean_hold_to_capture_demotion_count": float(
            aggregate["mean_hold_to_capture_demotion_count"]
        ),
        "mpc/mean_solve_time_s": float(aggregate["mean_mpc_solve_time_s"]),
        "mpc/mean_velocity_saturation_fraction": float(aggregate["mean_velocity_saturation_fraction"]),
        "actor/mean_output_variation": float(aggregate["mean_actor_output_variation"]),
        "phase/mean_intercept_dwell_s": float(aggregate["mean_intercept_dwell_s"]),
        "phase/mean_pre_impact_dwell_s": float(aggregate["mean_pre_impact_dwell_s"]),
        "phase/mean_capture_dwell_s": float(aggregate["mean_capture_dwell_s"]),
        "phase/mean_hold_dwell_s": float(aggregate["mean_hold_dwell_s"]),
        "cost_residual/object": float(aggregate["mean_residual_object"]),
        "cost_residual/grasp": float(aggregate["mean_residual_grasp"]),
        "cost_residual/compression": float(aggregate["mean_residual_compression"]),
        "cost_residual/velocity": float(aggregate["mean_residual_velocity"]),
        "cost_residual/smoothness": float(aggregate["mean_residual_smoothness"]),
        **_failure_count_log(aggregate["failure_category_counts"]),
    }
    for name, value in overrides.items():
        data[f"sweep/param/{name}"] = float(value)
    return data


def init_wandb(
    *,
    enabled: bool,
    run_name: str,
    config: dict[str, Any],
    project: str = "adaptive-cost-mpc-box-catch",
) -> WandbLogger:
    return WandbLogger(enabled=enabled, project=project, run_name=run_name, config=config)
