"""Online actor-critic adaptive-cost MPC for bimanual interception and grasping.

The implementation follows the AC-MPC architecture without depending on the
CUDA-only solver from the reference project:

    observation -> neural cost map -> differentiable MPC -> Gaussian action
                                      ^                     |
                                      +-- actor gradient ----+

The MPC model is a six-dimensional Cartesian single integrator containing the
left and right end-effector positions.  A dense, differentiable quadratic
solve is fast enough for this small robot problem on CPU; PyTorch moves it to
CUDA automatically when requested and available.

Online learning uses a one-step actor-critic update.  The MPC output is the
mean of a bounded Gaussian Cartesian-velocity policy.  Consequently the
policy-gradient loss propagates through the MPC solve into the cost map, while
the critic is trained from the temporal-difference target.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from control.mpc.ppo_common import (
    PPOUpdateSummary,
    generalized_advantage_estimate,
    resolve_device,
)


class BimanualPhase(IntEnum):
    """Runtime phase encoded in the actor observation."""

    INTERCEPT = 0
    PRE_CONTACT = 1
    GRASPING = 2
    GRASPED = 3
    MANIPULATION = 4


COST_NAMES = ("object", "grasp", "force", "velocity", "smoothness")
N_COSTS = len(COST_NAMES)
N_PHASES = len(BimanualPhase)

# Observation layout:
#   object velocity (3), endpoint errors left/right (6), EE velocities (6),
#   normal forces (2), TTC/confidence (2), phase one-hot (5) = 24.
OBS_DIM = 24


def build_bimanual_observation(
    *,
    object_velocity: np.ndarray,
    left_endpoint_error: np.ndarray,
    right_endpoint_error: np.ndarray,
    left_ee_velocity: np.ndarray,
    right_ee_velocity: np.ndarray,
    left_force: float,
    right_force: float,
    time_to_contact: float,
    prediction_confidence: float,
    phase: BimanualPhase,
) -> np.ndarray:
    """Build the normalized observation consumed by the cost actor."""

    velocity = np.clip(np.asarray(object_velocity, float).reshape(3) / 0.5, -2.0, 2.0)
    endpoint_errors = np.concatenate(
        [
            np.asarray(left_endpoint_error, float).reshape(3),
            np.asarray(right_endpoint_error, float).reshape(3),
        ]
    ) / 0.20
    ee_velocities = np.concatenate(
        [
            np.asarray(left_ee_velocity, float).reshape(3),
            np.asarray(right_ee_velocity, float).reshape(3),
        ]
    ) / 0.5
    forces = np.clip(np.array([left_force, right_force], float) / 20.0, 0.0, 2.0)
    ttc = 1.0 if not np.isfinite(time_to_contact) else np.clip(time_to_contact / 2.0, 0.0, 1.0)
    prediction = np.array([ttc, np.clip(prediction_confidence, 0.0, 1.0)])
    one_hot = np.zeros(N_PHASES, dtype=float)
    one_hot[int(phase)] = 1.0
    observation = np.concatenate(
        [velocity, endpoint_errors, ee_velocities, forces, prediction, one_hot]
    )
    if observation.shape != (OBS_DIM,):
        raise RuntimeError(f"internal observation shape {observation.shape} != {(OBS_DIM,)}")
    return np.clip(observation, -5.0, 5.0).astype(np.float32)


@dataclass
class DifferentiableMPCConfig:
    horizon: int = 8
    dt: float = 0.02
    velocity_limit: float = 0.25
    regularization: float = 1e-4
    grasp_compression: float = 0.008
    # Constant-velocity center_ref is exact for a slowly drifting handled
    # object but wrong for a falling/thrown one. Zero is a no-op (existing
    # behavior); a ballistic scenario sets this to real gravity so the
    # horizon reference actually follows the parabolic arc instead of a
    # straight line tangent to the current velocity.
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class OnlineActorCriticConfig:
    hidden_dim: int = 128
    actor_lr: float = 2e-4
    critic_lr: float = 5e-4
    gamma: float = 0.985
    gae_lambda: float = 0.95
    clip_ratio: float = 0.15
    value_loss_coefficient: float = 0.5
    entropy_coef: float = 1e-3
    max_grad_norm: float = 1.0
    weight_delta_fraction: float = 0.65
    training_epochs: int = 4
    online_epochs: int = 1
    minibatch_size: int = 32
    target_kl: float = 0.02
    minimum_online_rollout: int = 8
    maximum_online_actor_delta: float = 0.02
    initial_log_std: float = -3.2
    # Overrides AdaptiveCostActor._PHASE_PRIORS when set. Leave None for the
    # default handle-grasp priors; scenarios with different geometry/timing
    # (e.g. a fast ballistic catch needing much stronger grasp-separation
    # tracking) can supply their own without touching the shared default.
    phase_priors: Optional[tuple[tuple[float, ...], ...]] = None
    device: str = "auto"
    seed: int = 7


@dataclass
class ACMPCAction:
    velocity: np.ndarray
    mean_velocity: np.ndarray
    normalized_action: np.ndarray
    weights: dict[str, np.ndarray]
    value: float
    log_prob: float
    entropy: float


class AdaptiveCostActor(nn.Module):
    """Predict bounded horizon-wise residuals around safe phase priors."""

    # object, grasp geometry, force/compression, velocity feed-forward,
    # command smoothness -- indexed by BimanualPhase value.
    _PHASE_PRIORS = (
        (30.0, 10.0, 0.05, 4.0, 0.4),
        (30.0, 16.0, 1.5, 3.0, 0.5),
        (16.0, 22.0, 12.0, 1.5, 1.0),
        (18.0, 20.0, 9.0, 2.0, 1.5),
        (32.0, 18.0, 7.0, 3.0, 1.5),
    )

    def __init__(
        self,
        horizon: int,
        hidden_dim: int,
        delta_fraction: float,
        initial_log_std: float = -3.2,
        phase_priors: Optional[tuple[tuple[float, ...], ...]] = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.delta_fraction = float(delta_fraction)
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.horizon * N_COSTS),
        )
        # Start from the engineered safe priors; online experience moves away
        # from them only when the TD advantage supports the change.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer(
            "phase_priors",
            torch.tensor(phase_priors or self._PHASE_PRIORS, dtype=torch.float32),
        )
        # Learnable exploration noise on the resulting Cartesian velocity
        # action, in the same spirit as ppo_cost_adapter's _CostActor.log_std:
        # PPO's entropy bonus and clipped ratio need a real distribution
        # parameter to act on, not a fixed schedule.
        self.log_std = nn.Parameter(torch.full((6,), float(initial_log_std)))

    def forward(self, observation: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Predict horizon-wise cost weights for a batch of observations.

        ``phase`` is a ``(batch,)`` long tensor of ``BimanualPhase`` values so
        a single call can cover a PPO minibatch mixing several phases.
        """

        residual = torch.tanh(self.net(observation)).reshape(-1, self.horizon, N_COSTS)
        phase_index = phase.to(device=observation.device, dtype=torch.long).reshape(-1)
        base = self.phase_priors.to(device=observation.device, dtype=observation.dtype)[
            phase_index
        ]
        weights = base.unsqueeze(1) * (1.0 + self.delta_fraction * residual)
        # 500 (not the previous 50) so a scenario needing a much stiffer
        # relative/grasp-tracking prior -- e.g. closing a large initial
        # hand-separation gap within a fast ballistic catch's short window --
        # is not silently capped below what its prior actually requests.
        return torch.clamp(weights, min=1e-3, max=500.0)


class ValueCritic(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation).squeeze(-1)


class DifferentiableBimanualMPC(nn.Module):
    """Dense differentiable Cartesian MPC with receding-horizon output."""

    def __init__(self, config: Optional[DifferentiableMPCConfig] = None) -> None:
        super().__init__()
        self.config = config or DifferentiableMPCConfig()
        if self.config.horizon <= 0 or self.config.dt <= 0.0:
            raise ValueError("MPC horizon and dt must be positive")
        n = self.config.horizon
        dt = self.config.dt
        gamma = dt * torch.kron(torch.tril(torch.ones(n, n)), torch.eye(6))
        phi = torch.kron(torch.ones(n, 1), torch.eye(6))
        center = torch.cat([0.5 * torch.eye(3), 0.5 * torch.eye(3)], dim=1)
        relative = torch.cat([-torch.eye(3), torch.eye(3)], dim=1)
        center_stack = torch.kron(torch.eye(n), center)
        relative_stack = torch.kron(torch.eye(n), relative)
        difference = torch.eye(6 * n)
        for k in range(1, n):
            difference[6 * k : 6 * (k + 1), 6 * (k - 1) : 6 * k] = -torch.eye(6)
        self.register_buffer("gamma", gamma)
        self.register_buffer("phi", phi)
        self.register_buffer("center_stack", center_stack)
        self.register_buffer("relative_stack", relative_stack)
        self.register_buffer("difference", difference)
        self.register_buffer(
            "gravity", torch.tensor(self.config.gravity, dtype=torch.float32)
        )

    def forward(
        self,
        *,
        ee_positions: torch.Tensor,
        object_positions: torch.Tensor,
        object_velocities: torch.Tensor,
        relative_reference: torch.Tensor,
        weights: torch.Tensor,
        previous_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return first Cartesian velocity and the full horizon sequence."""

        batch = ee_positions.shape[0]
        n = self.config.horizon
        dtype, device = ee_positions.dtype, ee_positions.device
        gamma = self.gamma.to(device=device, dtype=dtype)
        phi = self.phi.to(device=device, dtype=dtype)
        center_stack = self.center_stack.to(device=device, dtype=dtype)
        relative_stack = self.relative_stack.to(device=device, dtype=dtype)
        difference = self.difference.to(device=device, dtype=dtype)

        gravity = self.gravity.to(device=device, dtype=dtype)
        steps = torch.arange(1, n + 1, device=device, dtype=dtype).view(1, n, 1)
        lead_time = steps * self.config.dt
        center_ref = (
            object_positions[:, None, :]
            + lead_time * object_velocities[:, None, :]
            + 0.5 * lead_time**2 * gravity
        )
        center_ref = center_ref.reshape(batch, 3 * n)
        relative_ref = relative_reference[:, None, :].expand(-1, n, -1).reshape(batch, 3 * n)
        relative_force_ref = relative_reference.clone()
        rel_norm = torch.linalg.vector_norm(relative_force_ref, dim=1, keepdim=True).clamp_min(1e-6)
        relative_force_ref = relative_force_ref * (
            1.0 - self.config.grasp_compression / rel_norm
        )
        relative_force_ref = relative_force_ref[:, None, :].expand(-1, n, -1).reshape(batch, 3 * n)
        velocity_ref = object_velocities[:, None, :].expand(-1, n, -1)
        velocity_ref = torch.cat([velocity_ref, velocity_ref], dim=2).reshape(batch, 6 * n)

        base_state = torch.matmul(phi, ee_positions.unsqueeze(-1)).squeeze(-1)
        center_base = torch.matmul(center_stack, base_state.unsqueeze(-1)).squeeze(-1)
        relative_base = torch.matmul(relative_stack, base_state.unsqueeze(-1)).squeeze(-1)
        a_center = center_stack @ gamma
        a_relative = relative_stack @ gamma

        solutions: list[torch.Tensor] = []
        eye = torch.eye(6 * n, device=device, dtype=dtype)
        for b in range(batch):
            w_object = torch.repeat_interleave(weights[b, :, 0], 3)
            w_grasp = torch.repeat_interleave(weights[b, :, 1], 3)
            w_force = torch.repeat_interleave(weights[b, :, 2], 3)
            w_velocity = torch.repeat_interleave(weights[b, :, 3], 6)
            w_smooth = torch.repeat_interleave(weights[b, :, 4], 6)

            h = a_center.T @ (w_object[:, None] * a_center)
            rhs = a_center.T @ (w_object * (center_ref[b] - center_base[b]))
            h = h + a_relative.T @ ((w_grasp + w_force)[:, None] * a_relative)
            rhs = rhs + a_relative.T @ (
                w_grasp * (relative_ref[b] - relative_base[b])
                + w_force * (relative_force_ref[b] - relative_base[b])
            )
            h = h + torch.diag(w_velocity)
            rhs = rhs + w_velocity * velocity_ref[b]

            smooth_target = torch.zeros(6 * n, device=device, dtype=dtype)
            smooth_target[:6] = previous_velocity[b]
            h = h + difference.T @ (w_smooth[:, None] * difference)
            rhs = rhs + difference.T @ (w_smooth * smooth_target)
            h = 0.5 * (h + h.T) + self.config.regularization * eye
            raw = torch.linalg.solve(h, rhs)
            bounded = self.config.velocity_limit * torch.tanh(
                raw / self.config.velocity_limit
            )
            solutions.append(bounded)

        sequence = torch.stack(solutions, dim=0).reshape(batch, n, 6)
        return sequence[:, 0, :], sequence


class ACMPCRolloutBuffer:
    """Store per-control-step transitions for a batched PPO update.

    Unlike a typical PPO rollout, replaying a transition to get a fresh
    log-prob under updated actor parameters requires re-solving the
    differentiable MPC (the actor only predicts cost weights; the action
    distribution's mean comes out of the MPC solve). So every MPC input the
    actor's forward pass depends on has to be stored alongside the usual
    reward/value/log-prob bookkeeping.
    """

    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.phases: list[int] = []
        self.ee_positions: list[np.ndarray] = []
        self.object_positions: list[np.ndarray] = []
        self.object_velocities: list[np.ndarray] = []
        self.relative_references: list[np.ndarray] = []
        self.previous_velocities: list[np.ndarray] = []
        self.normalized_actions: list[np.ndarray] = []
        self.log_probabilities: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        *,
        observation: np.ndarray,
        phase: BimanualPhase,
        ee_positions: np.ndarray,
        object_position: np.ndarray,
        object_velocity: np.ndarray,
        relative_reference: np.ndarray,
        previous_velocity: np.ndarray,
        action: ACMPCAction,
        reward: float,
        done: bool,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.phases.append(int(phase))
        self.ee_positions.append(np.asarray(ee_positions, dtype=np.float32).copy())
        self.object_positions.append(np.asarray(object_position, dtype=np.float32).copy())
        self.object_velocities.append(np.asarray(object_velocity, dtype=np.float32).copy())
        self.relative_references.append(
            np.asarray(relative_reference, dtype=np.float32).copy()
        )
        self.previous_velocities.append(
            np.asarray(previous_velocity, dtype=np.float32).copy()
        )
        self.normalized_actions.append(
            np.asarray(action.normalized_action, dtype=np.float32).copy()
        )
        self.log_probabilities.append(float(action.log_prob))
        self.values.append(float(action.value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def clear(self) -> None:
        self.__init__()


@dataclass
class _MPCBatch:
    """One tensor bundle of everything the actor+MPC forward pass needs."""

    observations: torch.Tensor
    phases: torch.Tensor
    ee_positions: torch.Tensor
    object_positions: torch.Tensor
    object_velocities: torch.Tensor
    relative_references: torch.Tensor
    previous_velocities: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor

    def index(self, indices: torch.Tensor) -> "_MPCBatch":
        return _MPCBatch(
            observations=self.observations[indices],
            phases=self.phases[indices],
            ee_positions=self.ee_positions[indices],
            object_positions=self.object_positions[indices],
            object_velocities=self.object_velocities[indices],
            relative_references=self.relative_references[indices],
            previous_velocities=self.previous_velocities[indices],
            actions=self.actions[indices],
            old_log_probabilities=self.old_log_probabilities[indices],
        )


class OnlineActorCriticACMPC:
    """Own the neural cost actor, critic, differentiable MPC, and PPO updates."""

    _LOG_STD_MIN = -5.0
    _LOG_STD_MAX = -1.8

    def __init__(
        self,
        mpc_config: Optional[DifferentiableMPCConfig] = None,
        learning_config: Optional[OnlineActorCriticConfig] = None,
    ) -> None:
        self.mpc_config = mpc_config or DifferentiableMPCConfig()
        self.config = learning_config or OnlineActorCriticConfig()
        self.device = resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.actor = AdaptiveCostActor(
            self.mpc_config.horizon,
            self.config.hidden_dim,
            self.config.weight_delta_fraction,
            self.config.initial_log_std,
            self.config.phase_priors,
        ).to(self.device)
        self.critic = ValueCritic(self.config.hidden_dim).to(self.device)
        self.mpc = DifferentiableBimanualMPC(self.mpc_config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.update_count = 0

    def act(
        self,
        *,
        observation: np.ndarray,
        phase: BimanualPhase,
        ee_positions: np.ndarray,
        object_position: np.ndarray,
        object_velocity: np.ndarray,
        relative_reference: np.ndarray,
        previous_velocity: np.ndarray,
        training: bool = True,
    ) -> ACMPCAction:
        obs = self._tensor(observation).reshape(1, OBS_DIM)
        phase_tensor = torch.tensor([int(phase)], device=self.device, dtype=torch.long)
        state = self._tensor(ee_positions).reshape(1, 6)
        obj = self._tensor(object_position).reshape(1, 3)
        obj_vel = self._tensor(object_velocity).reshape(1, 3)
        rel = self._tensor(relative_reference).reshape(1, 3)
        previous = self._tensor(previous_velocity).reshape(1, 6)

        with torch.no_grad():
            weights = self.actor(obs, phase_tensor)
            mean_velocity, _ = self.mpc(
                ee_positions=state,
                object_positions=obj,
                object_velocities=obj_vel,
                relative_reference=rel,
                weights=weights,
                previous_velocity=previous,
            )
            value = self.critic(obs)
            normalized_mean = mean_velocity / self.mpc_config.velocity_limit
            log_std = torch.clamp(self.actor.log_std, self._LOG_STD_MIN, self._LOG_STD_MAX)
            std = log_std.exp().expand_as(normalized_mean)
            distribution = torch.distributions.Normal(normalized_mean, std)
            normalized_action = distribution.sample() if training else normalized_mean
            normalized_action = torch.clamp(normalized_action, -1.0, 1.0)
            velocity = normalized_action * self.mpc_config.velocity_limit
            log_prob = distribution.log_prob(normalized_action).sum(dim=1)
            entropy = distribution.entropy().sum(dim=1)

        weights_np = weights.detach().cpu().numpy()[0]
        return ACMPCAction(
            velocity=velocity.detach().cpu().numpy()[0],
            mean_velocity=mean_velocity.detach().cpu().numpy()[0],
            normalized_action=normalized_action.detach().cpu().numpy()[0],
            weights={name: weights_np[:, i].copy() for i, name in enumerate(COST_NAMES)},
            value=float(value.detach().cpu().item()),
            log_prob=float(log_prob.detach().cpu().item()),
            entropy=float(entropy.detach().cpu().item()),
        )

    def predict_value(self, observation: np.ndarray) -> float:
        """Critic value of a single observation, for a rollout's GAE bootstrap."""

        with torch.no_grad():
            obs = self._tensor(observation).reshape(1, OBS_DIM)
            return float(self.critic(obs).detach().cpu().item())

    def update(
        self,
        rollout: ACMPCRolloutBuffer,
        *,
        online: bool,
        next_value: float = 0.0,
    ) -> PPOUpdateSummary:
        transitions = len(rollout)
        if transitions == 0:
            return self._skipped("empty rollout", transitions)
        if online and transitions < self.config.minimum_online_rollout:
            return self._skipped("online rollout is below safety minimum", transitions)

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

        batch = _MPCBatch(
            observations=self._tensor(np.stack(rollout.observations)),
            phases=torch.as_tensor(rollout.phases, dtype=torch.long, device=self.device),
            ee_positions=self._tensor(np.stack(rollout.ee_positions)),
            object_positions=self._tensor(np.stack(rollout.object_positions)),
            object_velocities=self._tensor(np.stack(rollout.object_velocities)),
            relative_references=self._tensor(np.stack(rollout.relative_references)),
            previous_velocities=self._tensor(np.stack(rollout.previous_velocities)),
            actions=self._tensor(np.stack(rollout.normalized_actions)),
            old_log_probabilities=self._tensor(np.asarray(rollout.log_probabilities)),
        )
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
                mini = batch.index(indices)

                distribution, new_log_probability = self._distribution(mini)
                log_ratio = new_log_probability - mini.old_log_probabilities
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
                actor_loss = actor_loss - self.config.entropy_coef * entropy

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                self.actor_optimizer.step()
                with torch.no_grad():
                    self.actor.log_std.clamp_(self._LOG_STD_MIN, self._LOG_STD_MAX)

                predicted_value = self.critic(mini.observations)
                critic_loss = self.config.value_loss_coefficient * torch.mean(
                    (predicted_value - returns_tensor[indices]) ** 2
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
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

        final_kl = self._policy_kl(batch)
        if online and final_kl > self.config.target_kl:
            # Parameter projection is the last line of defence if a single
            # small PPO step crosses the KL threshold before the next check.
            for _ in range(8):
                scale = min(
                    0.90,
                    0.90 * np.sqrt(self.config.target_kl / max(final_kl, 1e-12)),
                )
                with torch.no_grad():
                    for parameter, previous in zip(self.actor.parameters(), actor_before):
                        parameter.copy_(previous + scale * (parameter - previous))
                final_kl = self._policy_kl(batch)
                if final_kl <= self.config.target_kl:
                    break
            if final_kl > self.config.target_kl:
                with torch.no_grad():
                    for parameter, previous in zip(self.actor.parameters(), actor_before):
                        parameter.copy_(previous)
                final_kl = self._policy_kl(batch)
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

    def _distribution(
        self, mini: _MPCBatch
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        weights = self.actor(mini.observations, mini.phases)
        mean_velocity, _ = self.mpc(
            ee_positions=mini.ee_positions,
            object_positions=mini.object_positions,
            object_velocities=mini.object_velocities,
            relative_reference=mini.relative_references,
            weights=weights,
            previous_velocity=mini.previous_velocities,
        )
        normalized_mean = mean_velocity / self.mpc_config.velocity_limit
        log_std = torch.clamp(self.actor.log_std, self._LOG_STD_MIN, self._LOG_STD_MAX)
        std = log_std.exp().expand_as(normalized_mean)
        distribution = torch.distributions.Normal(normalized_mean, std)
        new_log_probability = distribution.log_prob(mini.actions).sum(dim=-1)
        return distribution, new_log_probability

    def _policy_kl(self, batch: _MPCBatch) -> float:
        with torch.no_grad():
            _, new_log_probability = self._distribution(batch)
            log_ratio = new_log_probability - batch.old_log_probabilities
            ratio = log_ratio.exp()
            return float((((ratio - 1.0) - log_ratio).mean()).cpu())

    def _actor_delta(self, before: list[torch.Tensor]) -> float:
        with torch.no_grad():
            squared = sum(
                torch.sum((parameter - previous) ** 2)
                for parameter, previous in zip(self.actor.parameters(), before)
            )
        return float(torch.sqrt(squared).cpu())

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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "update_count": self.update_count,
                "mpc_config": vars(self.mpc_config),
                "learning_config": vars(self.config),
            },
            path,
        )

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.update_count = int(checkpoint.get("update_count", 0))

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)
