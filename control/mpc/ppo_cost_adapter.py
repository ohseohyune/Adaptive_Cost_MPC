"""PPO/GAE actor-critic for safe episode-level MPC cost adaptation.

The policy does not command robot torques.  It selects bounded residuals around
the engineered Phase-3 costs, while the existing impedance controller and
wrench QP continue to enforce the physical constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from control.mpc.online_actor_critic import resolve_device
from control.squeeze.config import RotatingSideSqueezeConfig
from control.squeeze.generalization import (
    BoxDomainParameters,
    NOMINAL_BOX_FRICTION,
    NOMINAL_BOX_HALF_SIZE,
    NOMINAL_BOX_MASS,
)


GENERALIZATION_OBS_DIM = 13
COST_PARAMETER_NAMES = (
    "normal_force",
    "tangential_stiffness",
    "angular_damping",
    "slip_cost",
    "angular_velocity_cost",
    "wrench_tracking",
    "catch_ttc_offset",
    "squeeze_lead",
)
N_ENGINEERED_COST_PARAMETERS = 6
N_COST_PARAMETERS = len(COST_PARAMETER_NAMES)

# Multipliers are deliberately narrow: online learning cannot disable contact
# regularization, request zero grip force, or make the robot arbitrarily stiff.
COST_MULTIPLIER_LOW = np.array([0.75, 0.75, 0.65, 0.50, 0.50, 0.70])
COST_MULTIPLIER_HIGH = np.array([1.25, 1.25, 1.45, 1.80, 1.80, 1.40])

# Catch-timing actions are bounded separately: a small additive bias on the
# locked interception time, and a multiplicative window on how early the
# squeeze motion begins relative to predicted time-to-contact. The policy
# cannot disable interception replanning or squeeze before any prediction.
CATCH_TTC_OFFSET_S_MAX = 0.05
SQUEEZE_LEAD_MULTIPLIER_LOW = 0.60
SQUEEZE_LEAD_MULTIPLIER_HIGH = 1.60


def build_generalization_observation(
    parameters: BoxDomainParameters,
    *,
    curriculum_stage_count: int,
    catch_plane_x: float = 0.30,
    launch_position_x: float = 0.50,
) -> np.ndarray:
    """Encode known episode parameters for the cost actor."""

    size = np.asarray(parameters.half_size) / NOMINAL_BOX_HALF_SIZE - 1.0
    mass = parameters.mass / NOMINAL_BOX_MASS - 1.0
    friction = parameters.friction / NOMINAL_BOX_FRICTION - 1.0
    velocity = np.asarray(parameters.launch_velocity) / np.array([1.5, 0.10, 4.0])
    angular = np.asarray(parameters.angular_velocity) / 0.10
    vx = float(parameters.launch_velocity[0])
    ttc = abs((catch_plane_x - launch_position_x) / vx) if abs(vx) > 1e-6 else 1.0
    denominator = max(1, int(curriculum_stage_count) - 1)
    stage_progress = parameters.stage_index / denominator
    observation = np.concatenate(
        [
            size,
            np.array([mass, friction]),
            velocity,
            angular,
            np.array([ttc, stage_progress]),
        ]
    )
    if observation.shape != (GENERALIZATION_OBS_DIM,):
        raise RuntimeError(f"unexpected generalization observation shape: {observation.shape}")
    return np.clip(observation, -3.0, 3.0).astype(np.float32)


def _bounded_multipliers(normalized_action: np.ndarray) -> np.ndarray:
    action = np.clip(np.asarray(normalized_action, dtype=float), -1.0, 1.0)
    return np.where(
        action >= 0.0,
        1.0 + action * (COST_MULTIPLIER_HIGH - 1.0),
        1.0 + action * (1.0 - COST_MULTIPLIER_LOW),
    )


def _bounded_multiplier_scalar(action_value: float, low: float, high: float) -> float:
    """Zero-centered scalar analogue of ``_bounded_multipliers``.

    Action 0.0 always maps to multiplier 1.0 (the engineered default), so an
    untrained or deterministic-mean policy reproduces baseline behavior
    exactly, matching the convention used for the other cost multipliers.
    """

    action_value = float(np.clip(action_value, -1.0, 1.0))
    if action_value >= 0.0:
        return 1.0 + action_value * (high - 1.0)
    return 1.0 + action_value * (1.0 - low)


def apply_cost_action(
    config: RotatingSideSqueezeConfig,
    normalized_action: np.ndarray,
) -> RotatingSideSqueezeConfig:
    """Return a new controller config with bounded learned cost/timing residuals."""

    action = np.clip(np.asarray(normalized_action, dtype=float), -1.0, 1.0)
    multiplier = _bounded_multipliers(action[:N_ENGINEERED_COST_PARAMETERS])
    catch_ttc_offset_s = float(action[N_ENGINEERED_COST_PARAMETERS]) * CATCH_TTC_OFFSET_S_MAX
    squeeze_lead_multiplier = _bounded_multiplier_scalar(
        action[N_ENGINEERED_COST_PARAMETERS + 1],
        SQUEEZE_LEAD_MULTIPLIER_LOW,
        SQUEEZE_LEAD_MULTIPLIER_HIGH,
    )
    return replace(
        config,
        desired_normal_force=config.desired_normal_force * multiplier[0],
        tangential_stiffness=config.tangential_stiffness * multiplier[1],
        angular_wrench_damping=config.angular_wrench_damping * multiplier[2],
        slip_cost_weight=config.slip_cost_weight * multiplier[3],
        angular_velocity_cost_weight=(
            config.angular_velocity_cost_weight * multiplier[4]
        ),
        wrench_tracking_weight=config.wrench_tracking_weight * multiplier[5],
        catch_ttc_offset_s=catch_ttc_offset_s,
        squeeze_lead_s=config.squeeze_lead_s * squeeze_lead_multiplier,
    )


@dataclass
class PPOCostConfig:
    hidden_dim: int = 96
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.15
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 1e-3
    maximum_gradient_norm: float = 0.5
    training_epochs: int = 4
    online_epochs: int = 1
    minibatch_size: int = 16
    target_kl: float = 0.015
    minimum_online_rollout: int = 4
    maximum_online_actor_delta: float = 0.012
    maximum_safety_violation_rate: float = 0.20
    initial_log_std: float = -1.6
    device: str = "auto"
    seed: int = 7


@dataclass(frozen=True)
class CostAdaptationAction:
    raw_action: np.ndarray
    normalized_action: np.ndarray
    multipliers: dict[str, float]
    log_probability: float
    value: float
    entropy: float


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


class _CostActor(nn.Module):
    def __init__(self, hidden_dim: int, initial_log_std: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(GENERALIZATION_OBS_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, N_COST_PARAMETERS)
        self.log_std = nn.Parameter(
            torch.full((N_COST_PARAMETERS,), float(initial_log_std))
        )
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean(self.body(observation))
        log_std = torch.clamp(self.log_std, -3.0, -0.5)
        return torch.distributions.Normal(mean, log_std.exp().expand_as(mean))


class _CostCritic(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GENERALIZATION_OBS_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation).squeeze(-1)


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


class PPORolloutBuffer:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.raw_actions: list[np.ndarray] = []
        self.log_probabilities: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []
        self.safety_violations: list[bool] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        *,
        observation: np.ndarray,
        action: CostAdaptationAction,
        reward: float,
        done: bool,
        safety_violation: bool,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.raw_actions.append(np.asarray(action.raw_action, dtype=np.float32).copy())
        self.log_probabilities.append(float(action.log_probability))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(action.value))
        self.safety_violations.append(bool(safety_violation))

    def clear(self) -> None:
        self.__init__()


class PPOCostAdapter:
    """Bounded cost actor, value critic, and guarded PPO update owner."""

    def __init__(self, config: Optional[PPOCostConfig] = None) -> None:
        self.config = config or PPOCostConfig()
        self.device = resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.actor = _CostActor(
            self.config.hidden_dim, self.config.initial_log_std
        ).to(self.device)
        self.critic = _CostCritic(self.config.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )
        self.update_count = 0

    def act(
        self, observation: np.ndarray, *, training: bool = True
    ) -> CostAdaptationAction:
        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).reshape(1, GENERALIZATION_OBS_DIM)
        with torch.no_grad():
            distribution = self.actor.distribution(obs)
            raw = distribution.sample() if training else distribution.mean
            log_probability = distribution.log_prob(raw).sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1)
            value = self.critic(obs)
        raw_numpy = raw.cpu().numpy()[0]
        normalized = np.tanh(raw_numpy)
        engineered_multipliers = _bounded_multipliers(
            normalized[:N_ENGINEERED_COST_PARAMETERS]
        )
        multipliers = {
            name: float(value)
            for name, value in zip(
                COST_PARAMETER_NAMES[:N_ENGINEERED_COST_PARAMETERS],
                engineered_multipliers,
            )
        }
        multipliers[COST_PARAMETER_NAMES[N_ENGINEERED_COST_PARAMETERS]] = float(
            normalized[N_ENGINEERED_COST_PARAMETERS] * CATCH_TTC_OFFSET_S_MAX
        )
        multipliers[COST_PARAMETER_NAMES[N_ENGINEERED_COST_PARAMETERS + 1]] = (
            _bounded_multiplier_scalar(
                normalized[N_ENGINEERED_COST_PARAMETERS + 1],
                SQUEEZE_LEAD_MULTIPLIER_LOW,
                SQUEEZE_LEAD_MULTIPLIER_HIGH,
            )
        )
        return CostAdaptationAction(
            raw_action=raw_numpy,
            normalized_action=normalized,
            multipliers=multipliers,
            log_probability=float(log_probability.cpu().item()),
            value=float(value.cpu().item()),
            entropy=float(entropy.cpu().item()),
        )

    def update(
        self,
        rollout: PPORolloutBuffer,
        *,
        online: bool,
        next_value: float = 0.0,
    ) -> PPOUpdateSummary:
        transitions = len(rollout)
        if transitions == 0:
            return self._skipped("empty rollout", transitions)
        if online and transitions < self.config.minimum_online_rollout:
            return self._skipped("online rollout is below safety minimum", transitions)
        safety_rate = float(np.mean(rollout.safety_violations))
        if online and safety_rate > self.config.maximum_safety_violation_rate:
            return self._skipped("safety violation rate is too high", transitions)

        advantages, returns = generalized_advantage_estimate(
            np.asarray(rollout.rewards),
            np.asarray(rollout.values),
            np.asarray(rollout.dones),
            next_value=next_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        if transitions > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        observations = self._tensor(np.stack(rollout.observations))
        actions = self._tensor(np.stack(rollout.raw_actions))
        old_log_probabilities = self._tensor(np.asarray(rollout.log_probabilities))
        advantages_tensor = self._tensor(advantages)
        returns_tensor = self._tensor(returns)

        actor_before = [parameter.detach().clone() for parameter in self.actor.parameters()]
        epochs = self.config.online_epochs if online else self.config.training_epochs
        epochs_completed = 0
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        stop_for_kl = False

        for _ in range(epochs):
            permutation = torch.randperm(transitions, device=self.device)
            for start in range(0, transitions, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                distribution = self.actor.distribution(observations[indices])
                new_log_probability = distribution.log_prob(actions[indices]).sum(dim=-1)
                log_ratio = new_log_probability - old_log_probabilities[indices]
                ratio = log_ratio.exp()
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                if online and approximate_kl.item() > self.config.target_kl:
                    stop_for_kl = True
                    break

                unclipped = ratio * advantages_tensor[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages_tensor[indices]
                entropy = distribution.entropy().sum(dim=-1).mean()
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                actor_loss = actor_loss - self.config.entropy_coefficient * entropy

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.maximum_gradient_norm
                )
                self.actor_optimizer.step()
                with torch.no_grad():
                    self.actor.log_std.clamp_(-3.0, -0.5)

                predicted_value = self.critic(observations[indices])
                critic_loss = self.config.value_loss_coefficient * torch.mean(
                    (predicted_value - returns_tensor[indices]) ** 2
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.maximum_gradient_norm
                )
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
            epochs_completed += 1
            if stop_for_kl:
                break

        actor_delta = self._actor_delta(actor_before)
        if online and actor_delta > self.config.maximum_online_actor_delta:
            scale = self.config.maximum_online_actor_delta / max(actor_delta, 1e-12)
            with torch.no_grad():
                for parameter, previous in zip(self.actor.parameters(), actor_before):
                    parameter.copy_(previous + scale * (parameter - previous))
            actor_delta = self._actor_delta(actor_before)

        final_kl = self._policy_kl(
            observations, actions, old_log_probabilities
        )
        if online and final_kl > self.config.target_kl:
            # Parameter projection is the last line of defence if a single
            # small PPO step crosses the KL threshold before the next check.
            for _ in range(8):
                scale = min(
                    0.90,
                    0.90
                    * np.sqrt(
                        self.config.target_kl / max(final_kl, 1e-12)
                    ),
                )
                with torch.no_grad():
                    for parameter, previous in zip(
                        self.actor.parameters(), actor_before
                    ):
                        parameter.copy_(previous + scale * (parameter - previous))
                final_kl = self._policy_kl(
                    observations, actions, old_log_probabilities
                )
                if final_kl <= self.config.target_kl:
                    break
            if final_kl > self.config.target_kl:
                with torch.no_grad():
                    for parameter, previous in zip(
                        self.actor.parameters(), actor_before
                    ):
                        parameter.copy_(previous)
                final_kl = self._policy_kl(
                    observations, actions, old_log_probabilities
                )
            actor_delta = self._actor_delta(actor_before)

        self.update_count += 1
        return PPOUpdateSummary(
            applied=True,
            reason="target KL reached" if stop_for_kl else "updated",
            transitions=transitions,
            epochs=epochs_completed,
            actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.0,
            critic_loss=float(np.mean(critic_losses)) if critic_losses else 0.0,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
            approximate_kl=final_kl,
            actor_parameter_delta=actor_delta,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "config": vars(self.config),
                "update_count": self.update_count,
            },
            path,
        )

    def load(self, path: str, *, load_optimizers: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.update_count = int(checkpoint.get("update_count", 0))

    def actor_parameter_vector(self) -> np.ndarray:
        return torch.cat(
            [parameter.detach().flatten().cpu() for parameter in self.actor.parameters()]
        ).numpy()

    def _actor_delta(self, before: list[torch.Tensor]) -> float:
        with torch.no_grad():
            squared = sum(
                torch.sum((parameter - previous) ** 2)
                for parameter, previous in zip(self.actor.parameters(), before)
            )
        return float(torch.sqrt(squared).cpu())

    def _policy_kl(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            distribution = self.actor.distribution(observations)
            new_log_probability = distribution.log_prob(actions).sum(dim=-1)
            log_ratio = new_log_probability - old_log_probabilities
            ratio = log_ratio.exp()
            return float((((ratio - 1.0) - log_ratio).mean()).cpu())

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @staticmethod
    def _skipped(reason: str, transitions: int) -> PPOUpdateSummary:
        return PPOUpdateSummary(
            applied=False,
            reason=reason,
            transitions=transitions,
            epochs=0,
            actor_loss=0.0,
            critic_loss=0.0,
            entropy=0.0,
            approximate_kl=0.0,
            actor_parameter_delta=0.0,
        )
