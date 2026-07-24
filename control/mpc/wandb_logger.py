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

COST_NAMES = ("object", "grasp", "force", "velocity", "smoothness")
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
    phase_prior: np.ndarray,
    phase_id: int,
    solver_time_s: float,
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
        "state/phase_id": int(phase_id),
    }
    absolute_relative_mean: list[float] = []
    for index, name in enumerate(COST_NAMES):
        final_value = float(final_weights[name][0])
        prior_value = float(phase_prior[index])
        absolute_residual = final_value - prior_value
        relative_residual = (
            float("nan")
            if abs(prior_value) < _RESIDUAL_EPSILON
            else absolute_residual / (abs(prior_value) + _RESIDUAL_EPSILON)
        )
        data[f"mpc/{name}_weight"] = final_value
        data[f"mpc/phase_prior/{name}_weight"] = prior_value
        data[f"mpc/final_weight/{name}_weight"] = final_value
        data[f"mpc/residual_absolute/{name}_weight"] = absolute_residual
        data[f"mpc/residual_relative/{name}_weight"] = relative_residual
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


def init_wandb(
    *,
    enabled: bool,
    run_name: str,
    config: dict[str, Any],
    project: str = "adaptive-cost-mpc-box-catch",
) -> WandbLogger:
    return WandbLogger(enabled=enabled, project=project, run_name=run_name, config=config)
