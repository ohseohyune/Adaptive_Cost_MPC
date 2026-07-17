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
import torch.nn.functional as F


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


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``/CUDA requests with a safe CPU fallback."""

    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


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


@dataclass
class OnlineActorCriticConfig:
    hidden_dim: int = 128
    actor_lr: float = 2e-4
    critic_lr: float = 5e-4
    gamma: float = 0.985
    entropy_coef: float = 1e-3
    max_grad_norm: float = 1.0
    exploration_std: float = 0.04
    min_exploration_std: float = 0.008
    weight_delta_fraction: float = 0.65
    device: str = "auto"
    seed: int = 7


@dataclass
class ACMPCAction:
    velocity: np.ndarray
    mean_velocity: np.ndarray
    weights: dict[str, np.ndarray]
    value: float
    log_prob: float
    entropy: float


@dataclass
class OnlineUpdate:
    reward: float
    td_error: float
    actor_loss: float
    critic_loss: float
    grad_norm: float


class AdaptiveCostActor(nn.Module):
    """Predict bounded horizon-wise residuals around safe phase priors."""

    def __init__(self, horizon: int, hidden_dim: int, delta_fraction: float) -> None:
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

    def forward(self, observation: torch.Tensor, phase: BimanualPhase) -> torch.Tensor:
        residual = torch.tanh(self.net(observation)).reshape(-1, self.horizon, N_COSTS)
        base = self._phase_prior(phase, observation.device, observation.dtype)
        weights = base.unsqueeze(0) * (1.0 + self.delta_fraction * residual)
        return torch.clamp(weights, min=1e-3, max=50.0)

    @staticmethod
    def _phase_prior(
        phase: BimanualPhase, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # object, grasp geometry, force/compression, velocity feed-forward,
        # command smoothness
        priors = {
            BimanualPhase.INTERCEPT: (30.0, 10.0, 0.05, 4.0, 0.4),
            BimanualPhase.PRE_CONTACT: (30.0, 16.0, 1.5, 3.0, 0.5),
            BimanualPhase.GRASPING: (16.0, 22.0, 12.0, 1.5, 1.0),
            BimanualPhase.GRASPED: (18.0, 20.0, 9.0, 2.0, 1.5),
            BimanualPhase.MANIPULATION: (32.0, 18.0, 7.0, 3.0, 1.5),
        }
        return torch.tensor(priors[BimanualPhase(phase)], device=device, dtype=dtype)


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

        steps = torch.arange(1, n + 1, device=device, dtype=dtype).view(1, n, 1)
        center_ref = object_positions[:, None, :] + steps * self.config.dt * object_velocities[:, None, :]
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


class OnlineActorCriticACMPC:
    """Own the neural cost actor, critic, differentiable MPC, and TD updates."""

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
        ).to(self.device)
        self.critic = ValueCritic(self.config.hidden_dim).to(self.device)
        self.mpc = DifferentiableBimanualMPC(self.mpc_config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self._pending_log_prob: Optional[torch.Tensor] = None
        self._pending_entropy: Optional[torch.Tensor] = None
        self._pending_value: Optional[torch.Tensor] = None
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
        state = self._tensor(ee_positions).reshape(1, 6)
        obj = self._tensor(object_position).reshape(1, 3)
        obj_vel = self._tensor(object_velocity).reshape(1, 3)
        rel = self._tensor(relative_reference).reshape(1, 3)
        previous = self._tensor(previous_velocity).reshape(1, 6)

        weights = self.actor(obs, phase)
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
        std_value = max(
            self.config.min_exploration_std,
            self.config.exploration_std if training else self.config.min_exploration_std,
        )
        std = torch.full_like(normalized_mean, std_value)
        distribution = torch.distributions.Normal(normalized_mean, std)
        if training:
            # The sampled control is treated as an environment action.  Using
            # a detached sample gives the score-function gradient expected by
            # actor-critic; an rsample left in the graph would cancel the mean
            # derivative inside Normal.log_prob.
            normalized_action = distribution.sample()
        else:
            normalized_action = normalized_mean
        normalized_action = torch.clamp(normalized_action, -1.0, 1.0)
        velocity = normalized_action * self.mpc_config.velocity_limit
        log_prob = distribution.log_prob(normalized_action).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        if training:
            self._pending_log_prob = log_prob
            self._pending_entropy = entropy
            self._pending_value = value

        weights_np = weights.detach().cpu().numpy()[0]
        return ACMPCAction(
            velocity=velocity.detach().cpu().numpy()[0],
            mean_velocity=mean_velocity.detach().cpu().numpy()[0],
            weights={name: weights_np[:, i].copy() for i, name in enumerate(COST_NAMES)},
            value=float(value.detach().cpu().item()),
            log_prob=float(log_prob.detach().cpu().item()),
            entropy=float(entropy.detach().cpu().item()),
        )

    def observe(
        self,
        *,
        reward: float,
        next_observation: np.ndarray,
        done: bool = False,
    ) -> Optional[OnlineUpdate]:
        """Apply one online actor-critic update for the pending action."""

        if self._pending_value is None or self._pending_log_prob is None:
            return None
        next_obs = self._tensor(next_observation).reshape(1, OBS_DIM)
        with torch.no_grad():
            next_value = torch.zeros(1, device=self.device) if done else self.critic(next_obs)
            target = torch.tensor([float(reward)], device=self.device) + self.config.gamma * next_value
        td_error = target - self._pending_value
        actor_loss = -self._pending_log_prob * td_error.detach()
        if self._pending_entropy is not None:
            actor_loss = actor_loss - self.config.entropy_coef * self._pending_entropy
        actor_loss = actor_loss.mean()
        critic_loss = F.smooth_l1_loss(self._pending_value, target)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.max_grad_norm
        )
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()

        self.update_count += 1
        update = OnlineUpdate(
            reward=float(reward),
            td_error=float(td_error.detach().cpu().item()),
            actor_loss=float(actor_loss.detach().cpu().item()),
            critic_loss=float(critic_loss.detach().cpu().item()),
            grad_norm=float(max(float(actor_grad), float(critic_grad))),
        )
        self._clear_pending()
        return update

    def reset_episode(self) -> None:
        self._clear_pending()

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

    def _clear_pending(self) -> None:
        self._pending_log_prob = None
        self._pending_entropy = None
        self._pending_value = None
