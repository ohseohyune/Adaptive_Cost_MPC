"""Shared PPO/GAE primitives used by both cost-adapter learners.

Kept in its own module so ``online_actor_critic.py`` and
``ppo_cost_adapter.py`` can both depend on this without importing from each
other (they would otherwise form an import cycle).
"""

from __future__ import annotations

from dataclasses import dataclass

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
