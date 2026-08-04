"""Shared PPO/GAE primitives used by both cost-adapter learners.

Kept in its own module so ``online_actor_critic.py`` and
``ppo_cost_adapter.py`` can both depend on this without importing from each
other (they would otherwise form an import cycle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``/CUDA requests with a safe CPU fallback."""

    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


@dataclass(frozen=True)
class PPOUpdateSummary:
    applied: bool
    reason: str
    transitions: int
    epochs: int
    actor_loss: float
    critic_loss: float
    entropy: float
    approximate_kl: float
    actor_parameter_delta: float
    # 1 - Var(return - value_prediction) / Var(return), using the *old*
    # (pre-update) critic predictions stored in the rollout against the GAE
    # returns -- the standard PPO diagnostic, not a fresh critic forward
    # pass. None when the caller never computes it (e.g. ppo_cost_adapter's
    # update()); NaN when return variance is too small for a stable ratio.
    explained_variance: Optional[float] = None
    # Pre-normalization GAE advantage std and mean actor gradient norm
    # (pre-clip) -- batch-size/diversity diagnostics for comparing
    # rollout_size configurations. 0.0 when the caller never computes them.
    advantage_std: float = 0.0
    actor_grad_norm: float = 0.0
    # Constraint-ablation diagnostics. Defaults keep the legacy PPO cost
    # adapter and old log readers source-compatible.
    epochs_requested: int = 0
    target_kl_stopped: bool = False
    skipped_epochs: int = 0
    raw_actor_parameter_delta: float = 0.0
    online_delta_clipped: bool = False
    actor_update_applied_fraction: float = 1.0
    pre_projection_kl: float = 0.0
    kl_projection_count: int = 0
    kl_rollback_count: int = 0
    projection_removed_delta: float = 0.0
    cumulative_actor_parameter_delta: float = 0.0
    cumulative_delta_clipped: bool = False
    actor_parameter_path_length: float = 0.0
    actor_path_displacement_ratio: float = 0.0
    ppo_clip_fraction: float = 0.0


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and bootstrapped returns."""

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have equal shapes")
    advantages = np.zeros_like(rewards)
    last_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        following_value = next_value if index == len(rewards) - 1 else values[index + 1]
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * following_value * nonterminal - values[index]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[index] = last_advantage
    return advantages, advantages + values


def explained_variance(returns: np.ndarray, value_predictions: np.ndarray) -> float:
    """1 - Var(returns - value_predictions) / Var(returns).

    NaN when Var(returns) is too small for the ratio to be numerically
    meaningful, rather than an arbitrary substitute value.
    """

    returns = np.asarray(returns, dtype=np.float64)
    value_predictions = np.asarray(value_predictions, dtype=np.float64)
    return_variance = float(np.var(returns))
    if return_variance < 1e-8:
        return float("nan")
    return float(1.0 - np.var(returns - value_predictions) / return_variance)
