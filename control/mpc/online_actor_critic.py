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
    explained_variance,
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


COST_NAMES = ("object", "grasp", "compression", "velocity", "smoothness")
N_COSTS = len(COST_NAMES)
N_PHASES = len(BimanualPhase)

# Observation layout:
#   object velocity (3), endpoint errors left/right (6), EE velocities (6),
#   normal forces (2), TTC (1), phase one-hot (N_PHASES) = 18 + N_PHASES.
# prediction_confidence (BallisticBoxPredictor's sample-count warmup signal)
# was dropped after an A/B comparison on the box-catch "full" curriculum
# stage (N=50, paired) showed no measurable effect: 86% with the real value
# vs 84% fixed at a constant 1.0 -- within noise, so the actor was not
# meaningfully using this input.
_NON_PHASE_OBS_DIM = 18


def observation_dim(n_phases: int) -> int:
    """Total observation width for a one-hot (or soft one-hot) of ``n_phases``.

    Kept as a function (not just the ``OBS_DIM`` constant below) so a caller
    with a different phase count than the shared ``BimanualPhase`` -- e.g. a
    scenario with no manipulation phase -- can size its own actor/critic
    input layer and rollout buffer consistently, without every caller of
    this module being forced to share one global phase count.
    """

    return _NON_PHASE_OBS_DIM + int(n_phases)


# Default observation width for the shared 5-member BimanualPhase enum.
# Existing callers (the handle-grasp demo, its tests) rely on this exact
# value and on ``phase=`` (below) producing a plain hard one-hot -- both stay
# untouched no-ops for them.
OBS_DIM = observation_dim(N_PHASES)


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
    phase: Optional[BimanualPhase] = None,
    phase_encoding: Optional[np.ndarray] = None,
    object_velocity_scale: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Build the normalized observation consumed by the cost actor.

    ``object_velocity_scale`` defaults to a uniform 0.5 m/s on all three axes
    -- correct for the near-static handle-grasp demo this was designed for
    (object speed ~0.005 m/s) but not for a fast ballistic catch: measured
    across box-catch rollouts, the object's x-velocity is saturated at the
    clip bound 100% of the time during INTERCEPT/PRE_CONTACT (real speeds of
    -1.35 to -1.47 m/s all collapse to the same clipped -2.0), and z-velocity
    saturates 65% of the time (real fall speed reaches ~3.9 m/s by first
    contact). The actor's cost-weight network cannot distinguish different
    closing speeds through this channel as a result. Pass a scenario-specific
    scale (e.g. (1.5, 0.2, 4.0), validated against measured box-catch value
    ranges) instead of changing this default, which stays an exact no-op for
    the existing handle-grasp demo.

    Exactly one of ``phase``/``phase_encoding`` must be given. ``phase`` (a
    single ``BimanualPhase``-like int) produces a plain hard one-hot over
    ``N_PHASES`` -- the original, unchanged behavior. ``phase_encoding``
    accepts a precomputed vector of any length instead (e.g. a smoothly
    blended soft one-hot over a scenario-specific, smaller phase count) --
    the observation width is derived from its length rather than the shared
    ``N_PHASES``/``OBS_DIM`` constants, so a caller with fewer phases gets a
    correspondingly narrower observation.
    """

    velocity = np.clip(
        np.asarray(object_velocity, float).reshape(3) / np.asarray(object_velocity_scale, float),
        -2.0,
        2.0,
    )
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
    prediction = np.array([ttc])
    if phase_encoding is None:
        if phase is None:
            raise ValueError(
                "build_bimanual_observation requires either phase or phase_encoding"
            )
        phase_encoding = np.zeros(N_PHASES, dtype=float)
        phase_encoding[int(phase)] = 1.0
    else:
        phase_encoding = np.asarray(phase_encoding, dtype=float).reshape(-1)
    observation = np.concatenate(
        [velocity, endpoint_errors, ee_velocities, forces, prediction, phase_encoding]
    )
    expected_dim = observation_dim(phase_encoding.shape[0])
    if observation.shape != (expected_dim,):
        raise RuntimeError(f"internal observation shape {observation.shape} != {(expected_dim,)}")
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
    # None selects the unclipped policy-gradient objective for the final
    # PPO-clipping ablation; 0.15 preserves the validated default.
    clip_ratio: Optional[float] = 0.15
    value_loss_coefficient: float = 0.5
    normalize_returns: bool = False
    entropy_coef: float = 1e-3
    max_grad_norm: float = 1.0
    weight_delta_fraction: float = 0.65
    training_epochs: int = 4
    online_epochs: int = 1
    minibatch_size: int = 32
    target_kl: Optional[float] = 0.02
    minimum_online_rollout: int = 8
    maximum_online_actor_delta: Optional[float] = 0.02
    # Bounds cumulative L2 distance from the reference actor (the cold-init,
    # zero-residual state, persisted across checkpoint save/load -- see
    # OnlineActorCriticACMPC._reference_actor_state) across *all* updates,
    # not just this one. None (default) preserves existing behavior (no
    # cumulative cap) for scenarios that have not needed one. Set this when
    # many episodes' worth of online updates are expected (a curriculum
    # training loop), where maximum_online_actor_delta alone cannot prevent
    # slow drift away from a validated engineered prior.
    maximum_cumulative_actor_delta: Optional[float] = None
    # Cost Predictor output ablations. ``exp_residual`` implements
    # prior*exp(weight_delta_fraction*z), retaining positivity and the exact
    # engineered prior at z=0 without a final tanh.
    weight_parameterization: str = "bounded_residual"
    weight_clip_min: Optional[float] = 1e-3
    weight_clip_max: Optional[float] = 500.0
    initial_log_std: float = -3.2
    # Hard clamp bounds applied to log_std after every actor update (see
    # OnlineActorCriticACMPC._LOG_STD_MIN/_MAX, now sourced from here).
    # Defaults match this scenario's long-validated condition-2 behavior --
    # an exact no-op for every existing caller. A run whose exploration
    # collapses to log_std_min early (see acmpc_3way_ablation_condition3's
    # prior-free actor getting stuck near-identical to its cold-start init
    # across every phase) can raise log_std_min to keep more exploration
    # noise alive for longer, at the cost of noisier actions throughout.
    log_std_min: float = -5.0
    log_std_max: float = -1.8
    # Overrides AdaptiveCostActor._PHASE_PRIORS when set. Leave None for the
    # default handle-grasp priors; scenarios with different geometry/timing
    # (e.g. a fast ballistic catch needing much stronger grasp-separation
    # tracking) can supply their own without touching the shared default.
    phase_priors: Optional[tuple[tuple[float, ...], ...]] = None
    # Ablation switch: when True, the actor is a PriorFreeCostActor (cost
    # weights learned from scratch via softplus, no engineered phase-prior
    # anchor) instead of AdaptiveCostActor (bounded residual around
    # phase_priors). phase_priors/weight_delta_fraction are unused when this
    # is set. Default False is an exact no-op for every existing caller.
    use_prior_free_actor: bool = False
    # Scalar broadcasts to all 5 cost dims; a 5-tuple (object, grasp, force,
    # velocity, smoothness order -- see COST_NAMES) sets each independently,
    # the same across every phase/horizon step. See PriorFreeCostActor's
    # docstring for why a uniform scalar alone was not enough.
    prior_free_initial_weights: float | tuple[float, float, float, float, float] = 5.0
    # Number of phases the actor/critic input layer and observation's
    # phase-encoding slice are sized for. Defaults to the shared
    # BimanualPhase count (5, an exact no-op for existing callers). A
    # scenario with its own, smaller phase set (no per-phase table entry
    # needed for phases it never uses) can override this -- phase_priors
    # (when given) must then have exactly this many rows.
    n_phases: int = N_PHASES
    device: str = "auto"
    seed: int = 7
    # Ablation switch: force the actor's cost-weight residual to exactly zero
    # so the MPC runs on the engineered phase prior alone, while every other
    # part of the pipeline (observation, phase logic, MPC horizon, reference,
    # impedance, safety, termination) is untouched. The actor network and
    # checkpoint are left intact -- only its output is bypassed at the single
    # point where weights are handed to the MPC (see ``act``).
    residual_zero: bool = False
    # Recompute each minibatch's approximate KL *after* its optimizer step and
    # record it alongside the pre-step value. Off by default: it costs one
    # extra MPC-backed forward pass per minibatch. The KL that reaches
    # PPOUpdateSummary.approximate_kl is, and always was, a post-step
    # full-batch value (see _update's final_kl) -- this flag only adds the
    # per-minibatch pre/post pair.
    log_post_step_kl: bool = False
    # When True, ``act`` additionally solves the MPC with the zero-residual
    # (phase-prior) weights at the same state and reports the resulting
    # command gap in ACMPCAction.zero_residual_command_delta. Costs one extra
    # MPC solve per control step, so it is off by default.
    log_command_delta: bool = False


@dataclass
class ACMPCAction:
    velocity: np.ndarray
    mean_velocity: np.ndarray
    normalized_action: np.ndarray
    weights: dict[str, np.ndarray]
    # Values immediately before the optional hard weight clip. Keeping both
    # makes clip activation measurable instead of inferring it from values
    # that merely happen to be close to a bound.
    preclip_weights: dict[str, np.ndarray]
    value: float
    log_prob: float
    entropy: float
    # The (N_COSTS,) phase-prior row actually used by the actor for this
    # action -- either a hard per-phase lookup (existing behavior) or a
    # caller-supplied blend (e.g. a smoothstep interpolation across a phase
    # transition). Stored so a PPO replay of this transition (see
    # ACMPCRolloutBuffer/._distribution) reuses the exact same prior instead
    # of re-deriving it from a bare phase index, which would not reproduce a
    # mid-blend value.
    phase_prior: np.ndarray
    hessian_condition_number: float
    hessian_min_eigenvalue: float
    linear_solve_residual: float
    # ||u_mean - u_zero||_2: how far this action's MPC velocity command sits
    # from the command the same state would have produced with a zero actor
    # residual (phase prior only). None unless config.log_command_delta is
    # set; exactly 0.0 when config.residual_zero is set.
    zero_residual_command_delta: Optional[float] = None


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
        obs_dim: int = OBS_DIM,
        weight_parameterization: str = "bounded_residual",
        weight_clip_min: Optional[float] = 1e-3,
        weight_clip_max: Optional[float] = 500.0,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.delta_fraction = float(delta_fraction)
        self.weight_parameterization = str(weight_parameterization)
        self.weight_clip_min = weight_clip_min
        self.weight_clip_max = weight_clip_max
        if self.weight_parameterization not in {"bounded_residual", "exp_residual"}:
            raise ValueError(
                "weight_parameterization must be 'bounded_residual' or 'exp_residual'"
            )
        if self.weight_parameterization == "bounded_residual" and not 0.0 <= self.delta_fraction < 1.0:
            raise ValueError("bounded_residual requires 0 <= weight_delta_fraction < 1")
        if self.weight_parameterization == "exp_residual" and self.delta_fraction < 0.0:
            raise ValueError("exp_residual requires non-negative weight_delta_fraction")
        if weight_clip_min is not None and weight_clip_min <= 0.0:
            raise ValueError("weight_clip_min must be positive or None")
        if weight_clip_max is not None and weight_clip_max <= 0.0:
            raise ValueError("weight_clip_max must be positive or None")
        if (
            weight_clip_min is not None
            and weight_clip_max is not None
            and weight_clip_min >= weight_clip_max
        ):
            raise ValueError("weight_clip_min must be smaller than weight_clip_max")
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
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
        if not torch.isfinite(self.phase_priors).all() or torch.any(self.phase_priors <= 0.0):
            raise ValueError("phase priors must be finite and positive")
        # Learnable exploration noise on the resulting Cartesian velocity
        # action, in the same spirit as ppo_cost_adapter's _CostActor.log_std:
        # PPO's entropy bonus and clipped ratio need a real distribution
        # parameter to act on, not a fixed schedule.
        self.log_std = nn.Parameter(torch.full((6,), float(initial_log_std)))

    def forward_with_preclip(
        self, observation: torch.Tensor, phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict horizon-wise cost weights for a batch of observations.

        ``phase`` accepts two shapes:
        - ``(batch,)`` long tensor of ``BimanualPhase``-like indices (the
          original behavior): each row's prior is a hard lookup into
          ``self.phase_priors``.
        - ``(batch, N_COSTS)`` float tensor: used directly as each row's
          prior. This lets a caller (``OnlineActorCriticACMPC``) supply a
          value that is not a single phase's table row -- e.g. a smoothstep
          blend between two phases' rows across a transition -- without this
          class needing to know anything about blending itself.
        """

        raw = self.net(observation).reshape(-1, self.horizon, N_COSTS)
        if phase.dim() == 1:
            phase_index = phase.to(device=observation.device, dtype=torch.long).reshape(-1)
            base = self.phase_priors.to(device=observation.device, dtype=observation.dtype)[
                phase_index
            ]
        else:
            base = phase.to(device=observation.device, dtype=observation.dtype).reshape(-1, N_COSTS)
        if self.weight_parameterization == "bounded_residual":
            preclip = base.unsqueeze(1) * (
                1.0 + self.delta_fraction * torch.tanh(raw)
            )
        else:
            preclip = base.unsqueeze(1) * torch.exp(self.delta_fraction * raw)
        if not torch.isfinite(preclip).all() or torch.any(preclip <= 0.0):
            raise FloatingPointError("Cost Predictor produced non-finite or non-positive weights")
        weights = preclip
        if self.weight_clip_min is not None or self.weight_clip_max is not None:
            weights = torch.clamp(
                weights, min=self.weight_clip_min, max=self.weight_clip_max
            )
        return weights, preclip

    def forward(self, observation: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.forward_with_preclip(observation, phase)[0]


class PriorFreeCostActor(nn.Module):
    """Predict horizon-wise cost weights directly from PPO, no engineered prior.

    Ablation counterpart to ``AdaptiveCostActor``: that class computes
    ``weights = phase_prior * (1 + delta_fraction * tanh(net(obs)))``, i.e. a
    *bounded residual* around a hand-designed, per-phase engineered cost
    schedule -- the network can only ever nudge a strong human-designed
    strategy, not author its own. This class instead outputs
    ``softplus(net(obs))`` directly (guaranteeing positivity the way the QP
    requires, with no multiplicative anchor to any prior at all), so the
    entire cost-weight surface -- across cost dimension, horizon step, and
    (implicitly, since the phase one-hot is part of the observation) phase --
    is whatever PPO's reward signal shapes it into from scratch. This is a
    substantially harder exploration problem (see
    acmpc_box_catch_integration_status.md's notes on the planned 3-way
    ablation): no engineered starting point to fall back on if a training
    run/reward shaping choice doesn't pan out.

    ``initial_weights`` sets the output layer's bias so cold-start weights
    are ``softplus(bias) ~= initial_weights[c]`` for cost dimension ``c``,
    the *same* across every horizon step and phase (no per-phase table --
    that would just be AdaptiveCostActor's prior in disguise). A single
    scalar broadcasts to all five dimensions; passing a uniform value
    (e.g. 5.0 for all of object/grasp/force/velocity/smoothness) turns out
    to matter: scaling every cost term by the same constant is an exact
    no-op for the QP's solved velocity (the linear system's solution is
    scale-invariant), but *equal* weights also means the smoothness term
    (which resists any change from the previous velocity) competes
    one-for-one with the object/grasp tracking terms, and empirically this
    was never aggressive enough to close the gap to a fast-approaching box
    in the available time even when the box was slowed and moved closer
    (measured: endpoint error barely improved, 0.10 -> 0.083 m, over a full
    0.15 s -> 0.56 s window sweep). A single, phase-agnostic ratio (not
    zero, not a per-phase table) breaks that symmetry.
    """

    def __init__(
        self,
        horizon: int,
        hidden_dim: int,
        initial_log_std: float = -3.2,
        initial_weights: float | tuple[float, float, float, float, float] = 5.0,
        obs_dim: int = OBS_DIM,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.horizon * N_COSTS),
        )
        nn.init.zeros_(self.net[-1].weight)
        if isinstance(initial_weights, (int, float)):
            per_dim = [float(initial_weights)] * N_COSTS
        else:
            per_dim = [float(value) for value in initial_weights]
            if len(per_dim) != N_COSTS:
                raise ValueError(f"initial_weights must have {N_COSTS} entries")
        bias_per_dim = [float(np.log(np.expm1(max(value, 1e-6)))) for value in per_dim]
        bias_init = torch.tensor(bias_per_dim * self.horizon, dtype=torch.float32)
        with torch.no_grad():
            self.net[-1].bias.copy_(bias_init)
        self.log_std = nn.Parameter(torch.full((6,), float(initial_log_std)))

    def forward(self, observation: torch.Tensor, phase_prior: torch.Tensor) -> torch.Tensor:
        """Predict horizon-wise cost weights for a batch of observations.

        ``phase_prior`` is accepted for interface parity with
        ``AdaptiveCostActor`` (both are called the same way by
        ``OnlineActorCriticACMPC``) but is unused here -- the phase
        one-hot/soft-encoding already reaches the network as part of
        ``observation``, so any phase-dependent behavior is learned, not
        looked up from or blended against a table.
        """

        del phase_prior
        raw = self.net(observation).reshape(-1, self.horizon, N_COSTS)
        weights = nn.functional.softplus(raw)
        return torch.clamp(weights, min=1e-3, max=500.0)

    def forward_with_preclip(
        self, observation: torch.Tensor, phase_prior: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.forward(observation, phase_prior)
        return weights, weights


class ValueCritic(nn.Module):
    def __init__(self, hidden_dim: int, obs_dim: int = OBS_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
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
        self.collect_numerics = False
        self.last_condition_number = float("nan")
        self.last_min_eigenvalue = float("nan")
        self.last_solve_residual = float("nan")

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
        # COST_NAMES[2] ("compression") does not track any measured or
        # predicted contact force -- it pulls the relative left/right
        # separation toward a *compressed* position reference
        # (relative_reference shrunk by a fixed grasp_compression distance),
        # i.e. it is a compression-distance proxy for grip, not a
        # force-tracking term. Measured contact force (from the physics
        # engine) is only ever used by callers for their own observation/
        # success/safety logic, never fed back into this QP as a target.
        compressed_relative_ref = relative_reference.clone()
        rel_norm = torch.linalg.vector_norm(compressed_relative_ref, dim=1, keepdim=True).clamp_min(
            1e-6
        )
        compressed_relative_ref = compressed_relative_ref * (
            1.0 - self.config.grasp_compression / rel_norm
        )
        compressed_relative_ref = (
            compressed_relative_ref[:, None, :].expand(-1, n, -1).reshape(batch, 3 * n)
        )
        velocity_ref = object_velocities[:, None, :].expand(-1, n, -1)
        velocity_ref = torch.cat([velocity_ref, velocity_ref], dim=2).reshape(batch, 6 * n)

        base_state = torch.matmul(phi, ee_positions.unsqueeze(-1)).squeeze(-1)
        center_base = torch.matmul(center_stack, base_state.unsqueeze(-1)).squeeze(-1)
        relative_base = torch.matmul(relative_stack, base_state.unsqueeze(-1)).squeeze(-1)
        a_center = center_stack @ gamma
        a_relative = relative_stack @ gamma

        solutions: list[torch.Tensor] = []
        condition_numbers: list[float] = []
        minimum_eigenvalues: list[float] = []
        solve_residuals: list[float] = []
        eye = torch.eye(6 * n, device=device, dtype=dtype)
        for b in range(batch):
            w_object = torch.repeat_interleave(weights[b, :, 0], 3)
            w_grasp = torch.repeat_interleave(weights[b, :, 1], 3)
            w_compression = torch.repeat_interleave(weights[b, :, 2], 3)
            w_velocity = torch.repeat_interleave(weights[b, :, 3], 6)
            w_smooth = torch.repeat_interleave(weights[b, :, 4], 6)

            h = a_center.T @ (w_object[:, None] * a_center)
            rhs = a_center.T @ (w_object * (center_ref[b] - center_base[b]))
            h = h + a_relative.T @ ((w_grasp + w_compression)[:, None] * a_relative)
            rhs = rhs + a_relative.T @ (
                w_grasp * (relative_ref[b] - relative_base[b])
                + w_compression * (compressed_relative_ref[b] - relative_base[b])
            )
            h = h + torch.diag(w_velocity)
            rhs = rhs + w_velocity * velocity_ref[b]

            smooth_target = torch.zeros(6 * n, device=device, dtype=dtype)
            smooth_target[:6] = previous_velocity[b]
            h = h + difference.T @ (w_smooth[:, None] * difference)
            rhs = rhs + difference.T @ (w_smooth * smooth_target)
            h = 0.5 * (h + h.T) + self.config.regularization * eye
            if not torch.isfinite(h).all() or not torch.isfinite(rhs).all():
                raise FloatingPointError("MPC Hessian or right-hand side contains NaN or Inf")
            raw = torch.linalg.solve(h, rhs)
            if not torch.isfinite(raw).all():
                raise FloatingPointError("MPC linear solve produced NaN or Inf")
            if self.collect_numerics:
                with torch.no_grad():
                    eigenvalues = torch.linalg.eigvalsh(h.detach())
                    minimum_eigenvalue = float(eigenvalues[0].cpu())
                    condition_numbers.append(
                        float((eigenvalues[-1] / eigenvalues[0].clamp_min(1e-30)).cpu())
                    )
                    minimum_eigenvalues.append(minimum_eigenvalue)
                    solve_residuals.append(
                        float(
                            (
                                torch.linalg.vector_norm(h.detach() @ raw.detach() - rhs.detach())
                                / torch.linalg.vector_norm(rhs.detach()).clamp_min(1e-12)
                            ).cpu()
                        )
                    )
            bounded = self.config.velocity_limit * torch.tanh(
                raw / self.config.velocity_limit
            )
            solutions.append(bounded)

        sequence = torch.stack(solutions, dim=0).reshape(batch, n, 6)
        if self.collect_numerics:
            self.last_condition_number = max(condition_numbers)
            self.last_min_eigenvalue = min(minimum_eigenvalues)
            self.last_solve_residual = max(solve_residuals)
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
        # The (N_COSTS,) prior row the actor actually used for each
        # transition's action (see ACMPCAction.phase_prior) -- replayed
        # as-is during PPO update() instead of being re-derived from
        # ``phases``, since a mid-blend prior cannot be reconstructed from a
        # bare phase index alone.
        self.phase_priors: list[np.ndarray] = []
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
        self.phase_priors.append(np.asarray(action.phase_prior, dtype=np.float32).copy())
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
    phase_priors: torch.Tensor
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
            phase_priors=self.phase_priors[indices],
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

    def __init__(
        self,
        mpc_config: Optional[DifferentiableMPCConfig] = None,
        learning_config: Optional[OnlineActorCriticConfig] = None,
    ) -> None:
        self.mpc_config = mpc_config or DifferentiableMPCConfig()
        self.config = learning_config or OnlineActorCriticConfig()
        for name in (
            "clip_ratio",
            "target_kl",
            "maximum_online_actor_delta",
            "maximum_cumulative_actor_delta",
        ):
            value = getattr(self.config, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive or None")
        self.device = resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.obs_dim = observation_dim(self.config.n_phases)
        if (
            not self.config.use_prior_free_actor
            and self.config.phase_priors is not None
            and len(self.config.phase_priors) != self.config.n_phases
        ):
            raise ValueError(
                f"phase_priors has {len(self.config.phase_priors)} rows but "
                f"n_phases={self.config.n_phases}"
            )
        if self.config.use_prior_free_actor:
            self.actor: nn.Module = PriorFreeCostActor(
                self.mpc_config.horizon,
                self.config.hidden_dim,
                self.config.initial_log_std,
                self.config.prior_free_initial_weights,
                obs_dim=self.obs_dim,
            ).to(self.device)
        else:
            self.actor = AdaptiveCostActor(
                self.mpc_config.horizon,
                self.config.hidden_dim,
                self.config.weight_delta_fraction,
                self.config.initial_log_std,
                self.config.phase_priors,
                obs_dim=self.obs_dim,
                weight_parameterization=self.config.weight_parameterization,
                weight_clip_min=self.config.weight_clip_min,
                weight_clip_max=self.config.weight_clip_max,
            ).to(self.device)
        self.critic = ValueCritic(self.config.hidden_dim, obs_dim=self.obs_dim).to(self.device)
        self.mpc = DifferentiableBimanualMPC(self.mpc_config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.update_count = 0
        self.return_mean = 0.0
        self.return_variance = 1.0
        self.return_count = 0
        # maximum_online_actor_delta only bounds each *individual* update's
        # movement -- it does nothing to stop thousands of small, individually
        # legal updates from cumulatively drifting the actor far from the
        # known-safe engineered prior (observed: a 100-episode online run on
        # the hardest box-catch curriculum stage went from 81%/93% success in
        # the first ~40 episodes to ~38% in the back half, after ~1600+
        # cumulative updates, driven by INTERCEPT's velocity weight and
        # GRASPED's force weight drifting substantially off their tuned
        # values). This reference snapshot anchors an optional cumulative
        # cap in update() below. It reflects the cold-init (zero-residual)
        # actor unless overwritten by load() from a checkpoint that saved
        # its own reference.
        self._reference_actor_state = [p.detach().clone() for p in self.actor.parameters()]
        self.actor_parameter_path_length = 0.0

    def _resolve_phase_prior(
        self, phase: int, phase_prior: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Return the (1, N_COSTS) prior row to feed the actor for this action.

        ``phase_prior``, when given, is used directly -- this is how a caller
        injects a blended (e.g. smoothstep-interpolated across a phase
        transition) prior instead of a single phase's hard table row.
        Otherwise falls back to the original hard lookup by ``phase`` (an
        exact no-op for existing callers that never pass an override).
        ``PriorFreeCostActor`` has no ``phase_priors`` table at all -- its
        forward ignores this value entirely, so a zero placeholder is fine.
        """

        if phase_prior is not None:
            return self._tensor(np.asarray(phase_prior, dtype=np.float32)).reshape(1, N_COSTS)
        if hasattr(self.actor, "phase_priors"):
            return (
                self.actor.phase_priors[int(phase)]
                .to(device=self.device, dtype=torch.float32)
                .reshape(1, N_COSTS)
            )
        return torch.zeros(1, N_COSTS, device=self.device)

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
        phase_prior: Optional[np.ndarray] = None,
    ) -> ACMPCAction:
        obs = self._tensor(observation).reshape(1, self.obs_dim)
        prior_row = self._resolve_phase_prior(phase, phase_prior)
        state = self._tensor(ee_positions).reshape(1, 6)
        obj = self._tensor(object_position).reshape(1, 3)
        obj_vel = self._tensor(object_velocity).reshape(1, 3)
        rel = self._tensor(relative_reference).reshape(1, 3)
        previous = self._tensor(previous_velocity).reshape(1, 6)

        with torch.no_grad():
            weights, preclip_weights = self.actor.forward_with_preclip(obs, prior_row)
            zero_weights = prior_row.unsqueeze(1).expand_as(weights).contiguous()
            if self.config.residual_zero:
                # Bypass the residual at the one point where weights reach the
                # MPC. No clamp: the engineered prior must survive bit-exact
                # (see the residual_zero unit test's 1e-8 tolerance).
                weights = zero_weights
                preclip_weights = zero_weights
            self.mpc.collect_numerics = True
            try:
                mean_velocity, _ = self.mpc(
                    ee_positions=state,
                    object_positions=obj,
                    object_velocities=obj_vel,
                    relative_reference=rel,
                    weights=weights,
                    previous_velocity=previous,
                )
            finally:
                self.mpc.collect_numerics = False
            command_delta: Optional[float] = None
            if self.config.residual_zero:
                command_delta = 0.0
            elif self.config.log_command_delta:
                zero_velocity, _ = self.mpc(
                    ee_positions=state,
                    object_positions=obj,
                    object_velocities=obj_vel,
                    relative_reference=rel,
                    weights=zero_weights,
                    previous_velocity=previous,
                )
                command_delta = float(
                    torch.linalg.norm(mean_velocity - zero_velocity).cpu()
                )
            value = self._denormalize_value(self.critic(obs))
            normalized_mean = mean_velocity / self.mpc_config.velocity_limit
            log_std = torch.clamp(self.actor.log_std, self.config.log_std_min, self.config.log_std_max)
            std = log_std.exp().expand_as(normalized_mean)
            distribution = torch.distributions.Normal(normalized_mean, std)
            normalized_action = distribution.sample() if training else normalized_mean
            normalized_action = torch.clamp(normalized_action, -1.0, 1.0)
            velocity = normalized_action * self.mpc_config.velocity_limit
            log_prob = distribution.log_prob(normalized_action).sum(dim=1)
            entropy = distribution.entropy().sum(dim=1)

        weights_np = weights.detach().cpu().numpy()[0]
        preclip_weights_np = preclip_weights.detach().cpu().numpy()[0]
        return ACMPCAction(
            velocity=velocity.detach().cpu().numpy()[0],
            mean_velocity=mean_velocity.detach().cpu().numpy()[0],
            normalized_action=normalized_action.detach().cpu().numpy()[0],
            weights={name: weights_np[:, i].copy() for i, name in enumerate(COST_NAMES)},
            preclip_weights={
                name: preclip_weights_np[:, i].copy()
                for i, name in enumerate(COST_NAMES)
            },
            value=float(value.detach().cpu().item()),
            log_prob=float(log_prob.detach().cpu().item()),
            entropy=float(entropy.detach().cpu().item()),
            phase_prior=prior_row.detach().cpu().numpy()[0].copy(),
            hessian_condition_number=self.mpc.last_condition_number,
            hessian_min_eigenvalue=self.mpc.last_min_eigenvalue,
            linear_solve_residual=self.mpc.last_solve_residual,
            zero_residual_command_delta=command_delta,
        )

    def predict_value(self, observation: np.ndarray) -> float:
        """Critic value of a single observation, for a rollout's GAE bootstrap."""

        with torch.no_grad():
            obs = self._tensor(observation).reshape(1, self.obs_dim)
            value = self._denormalize_value(self.critic(obs))
            return float(value.detach().cpu().item())

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
        # Old (pre-update) critic predictions vs. GAE returns -- the
        # standard PPO diagnostic; no extra critic forward pass.
        update_explained_variance = explained_variance(returns, np.asarray(rollout.values))
        advantage_mean = float(advantages.mean()) if transitions else 0.0
        advantage_std = float(advantages.std()) if transitions > 1 else 0.0
        if transitions > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch = _MPCBatch(
            observations=self._tensor(np.stack(rollout.observations)),
            phases=torch.as_tensor(rollout.phases, dtype=torch.long, device=self.device),
            phase_priors=self._tensor(np.stack(rollout.phase_priors)),
            ee_positions=self._tensor(np.stack(rollout.ee_positions)),
            object_positions=self._tensor(np.stack(rollout.object_positions)),
            object_velocities=self._tensor(np.stack(rollout.object_velocities)),
            relative_references=self._tensor(np.stack(rollout.relative_references)),
            previous_velocities=self._tensor(np.stack(rollout.previous_velocities)),
            actions=self._tensor(np.stack(rollout.normalized_actions)),
            old_log_probabilities=self._tensor(np.asarray(rollout.log_probabilities)),
        )
        advantages_tensor = self._tensor(advantages)
        if self.config.normalize_returns:
            self._update_return_statistics(returns)
            returns_for_critic = (
                returns - self.return_mean
            ) / np.sqrt(self.return_variance + 1e-8)
        else:
            returns_for_critic = returns
        returns_tensor = self._tensor(returns_for_critic)

        actor_before = [parameter.detach().clone() for parameter in self.actor.parameters()]
        epochs = self.config.online_epochs if online else self.config.training_epochs
        epochs_completed = 0
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        actor_grad_norms: list[float] = []
        critic_grad_norms: list[float] = []
        clip_fractions: list[float] = []
        minibatch_records: list[dict[str, object]] = []
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
                if (
                    online
                    and self.config.target_kl is not None
                    and approximate_kl.item() > self.config.target_kl
                ):
                    stop_for_kl = True
                    break

                unclipped = ratio * advantages_tensor[indices]
                if self.config.clip_ratio is None:
                    policy_objective = unclipped
                    clip_fractions.append(0.0)
                else:
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    ) * advantages_tensor[indices]
                    policy_objective = torch.minimum(unclipped, clipped)
                    clip_fractions.append(
                        float(
                            ((ratio - 1.0).abs() > self.config.clip_ratio)
                            .float()
                            .mean()
                            .detach()
                            .cpu()
                        )
                    )
                entropy = distribution.entropy().sum(dim=-1).mean()
                # Kept as three named terms: the value historically logged as
                # "actor_loss"/"policy_loss" is the *total*, and with
                # entropy_coef=1e-3 against an entropy near -12.4 the entropy
                # bonus alone accounts for ~0.0124 of it -- i.e. the whole
                # logged magnitude. See the loss-decomposition unit test.
                policy_surrogate_loss = -policy_objective.mean()
                entropy_bonus = -self.config.entropy_coef * entropy
                actor_loss = policy_surrogate_loss + entropy_bonus

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                actor_grad_norms.append(float(grad_norm))
                self.actor_optimizer.step()
                with torch.no_grad():
                    self.actor.log_std.clamp_(self.config.log_std_min, self.config.log_std_max)

                predicted_value = self.critic(mini.observations)
                critic_loss = self.config.value_loss_coefficient * torch.mean(
                    (predicted_value - returns_tensor[indices]) ** 2
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                critic_grad_norms.append(float(critic_grad_norm))
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                minibatch_records.append(
                    {
                        "executed_ppo_epoch": epochs_completed,
                        "minibatch_index": len(minibatch_records),
                        "policy_surrogate_loss": float(
                            policy_surrogate_loss.detach().cpu()
                        ),
                        "entropy_bonus": float(entropy_bonus.detach().cpu()),
                        "total_actor_loss": float(actor_loss.detach().cpu()),
                        "entropy": float(entropy.detach().cpu()),
                        "entropy_coef": float(self.config.entropy_coef),
                        "pre_step_approximate_kl": float(approximate_kl.detach().cpu()),
                        "clip_fraction": clip_fractions[-1],
                        "actor_gradient_norm": float(grad_norm),
                        "post_step_approximate_kl": (
                            self._minibatch_kl(mini)
                            if self.config.log_post_step_kl
                            else None
                        ),
                    }
                )
            epochs_completed += 1
            if stop_for_kl:
                break

        raw_actor_delta = self._actor_delta(actor_before)
        actor_delta = raw_actor_delta
        online_delta_clipped = bool(
            online
            and self.config.maximum_online_actor_delta is not None
            and actor_delta > self.config.maximum_online_actor_delta
        )
        if online_delta_clipped:
            scale = self.config.maximum_online_actor_delta / max(actor_delta, 1e-12)
            with torch.no_grad():
                for parameter, previous in zip(self.actor.parameters(), actor_before):
                    parameter.copy_(previous + scale * (parameter - previous))
            actor_delta = self._actor_delta(actor_before)
        online_delta_removed = max(0.0, raw_actor_delta - actor_delta)

        pre_projection_kl = self._policy_kl(batch)
        final_kl = pre_projection_kl
        delta_before_projection = actor_delta
        projection_count = 0
        rollback_count = 0
        if (
            online
            and self.config.target_kl is not None
            and final_kl > self.config.target_kl
        ):
            # Parameter projection is the last line of defence if a single
            # small PPO step crosses the KL threshold before the next check.
            for _ in range(8):
                projection_count += 1
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
                rollback_count = 1
                with torch.no_grad():
                    for parameter, previous in zip(self.actor.parameters(), actor_before):
                        parameter.copy_(previous)
                final_kl = self._policy_kl(batch)
            actor_delta = self._actor_delta(actor_before)
        projection_removed_delta = max(0.0, delta_before_projection - actor_delta)

        cumulative_delta_clipped = False
        cumulative_projection_removed_delta = 0.0
        if online and self.config.maximum_cumulative_actor_delta is not None:
            cumulative_delta = self._actor_delta(self._reference_actor_state)
            if cumulative_delta > self.config.maximum_cumulative_actor_delta:
                cumulative_delta_clipped = True
                cumulative_projection_removed_delta = (
                    cumulative_delta - self.config.maximum_cumulative_actor_delta
                )
                scale = self.config.maximum_cumulative_actor_delta / max(cumulative_delta, 1e-12)
                with torch.no_grad():
                    for parameter, reference in zip(
                        self.actor.parameters(), self._reference_actor_state
                    ):
                        parameter.copy_(reference + scale * (parameter - reference))
                actor_delta = self._actor_delta(actor_before)

        self._assert_finite_actor()
        self.actor_parameter_path_length += actor_delta
        cumulative_delta = self._actor_delta(self._reference_actor_state)
        path_ratio = self.actor_parameter_path_length / max(cumulative_delta, 1e-12)
        self.update_count += 1
        return PPOUpdateSummary(
            applied=True,
            reason="target KL reached" if stop_for_kl else "updated",
            transitions=transitions,
            epochs=epochs_completed,
            actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.0,
            critic_loss=float(np.mean(critic_losses)) if critic_losses else 0.0,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
            minibatch_diagnostics=tuple(minibatch_records),
            policy_surrogate_loss=(
                float(np.mean([r["policy_surrogate_loss"] for r in minibatch_records]))
                if minibatch_records
                else 0.0
            ),
            entropy_bonus=(
                float(np.mean([r["entropy_bonus"] for r in minibatch_records]))
                if minibatch_records
                else 0.0
            ),
            approximate_kl=final_kl,
            actor_parameter_delta=actor_delta,
            explained_variance=update_explained_variance,
            advantage_std=advantage_std,
            actor_grad_norm=float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0,
            epochs_requested=epochs,
            target_kl_stopped=stop_for_kl,
            skipped_epochs=max(0, epochs - epochs_completed),
            raw_actor_parameter_delta=raw_actor_delta,
            online_delta_clipped=online_delta_clipped,
            actor_update_applied_fraction=actor_delta / max(raw_actor_delta, 1e-12),
            pre_projection_kl=pre_projection_kl,
            kl_projection_count=projection_count,
            kl_rollback_count=rollback_count,
            projection_removed_delta=projection_removed_delta,
            cumulative_actor_parameter_delta=cumulative_delta,
            cumulative_delta_clipped=cumulative_delta_clipped,
            actor_parameter_path_length=self.actor_parameter_path_length,
            actor_path_displacement_ratio=path_ratio,
            ppo_clip_fraction=float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            advantage_mean=advantage_mean,
            critic_grad_norm=(
                float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0
            ),
            online_delta_removed=online_delta_removed,
            cumulative_projection_removed_delta=cumulative_projection_removed_delta,
        )

    def _distribution(
        self, mini: _MPCBatch
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        # Replay uses the exact prior stored at collection time (mini.phase_priors),
        # not a fresh lookup by mini.phases -- a mid-blend prior cannot be
        # reconstructed from a bare phase index alone (see ACMPCAction.phase_prior).
        weights = self.actor(mini.observations, mini.phase_priors)
        mean_velocity, _ = self.mpc(
            ee_positions=mini.ee_positions,
            object_positions=mini.object_positions,
            object_velocities=mini.object_velocities,
            relative_reference=mini.relative_references,
            weights=weights,
            previous_velocity=mini.previous_velocities,
        )
        normalized_mean = mean_velocity / self.mpc_config.velocity_limit
        log_std = torch.clamp(self.actor.log_std, self.config.log_std_min, self.config.log_std_max)
        std = log_std.exp().expand_as(normalized_mean)
        distribution = torch.distributions.Normal(normalized_mean, std)
        new_log_probability = distribution.log_prob(mini.actions).sum(dim=-1)
        return distribution, new_log_probability

    def _minibatch_kl(self, mini: _MPCBatch) -> float:
        """Post-optimizer-step KL on one minibatch's own observations/actions."""
        return self._policy_kl(mini)

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

    def _assert_finite_actor(self) -> None:
        if not all(torch.isfinite(parameter).all() for parameter in self.actor.parameters()):
            raise FloatingPointError("Cost Predictor parameters became NaN or Inf")
        if not all(torch.isfinite(parameter).all() for parameter in self.critic.parameters()):
            raise FloatingPointError("Critic parameters became NaN or Inf")

    def _denormalize_value(self, value: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_returns:
            return value
        return value * np.sqrt(self.return_variance + 1e-8) + self.return_mean

    def _update_return_statistics(self, returns: np.ndarray) -> None:
        batch_count = int(len(returns))
        if batch_count == 0:
            return
        batch_mean = float(np.mean(returns))
        batch_variance = float(np.var(returns))
        if self.return_count == 0:
            self.return_mean = batch_mean
            self.return_variance = max(batch_variance, 1e-8)
            self.return_count = batch_count
            return
        total = self.return_count + batch_count
        delta = batch_mean - self.return_mean
        combined_m2 = (
            self.return_variance * self.return_count
            + batch_variance * batch_count
            + delta * delta * self.return_count * batch_count / total
        )
        self.return_mean += delta * batch_count / total
        self.return_variance = max(combined_m2 / total, 1e-8)
        self.return_count = total

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
                # The cumulative-drift anchor (see maximum_cumulative_actor_delta)
                # must survive across episodes/checkpoint reloads -- otherwise
                # each fresh run_box_catch() call would reset "cumulative
                # distance" back to zero and the cap would never actually
                # bind.
                "reference_actor": [p.detach().cpu() for p in self._reference_actor_state],
                "actor_parameter_path_length": self.actor_parameter_path_length,
                "return_normalizer": {
                    "mean": self.return_mean,
                    "variance": self.return_variance,
                    "count": self.return_count,
                },
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
        # Older checkpoints (saved before maximum_cumulative_actor_delta
        # existed) have no "reference_actor" -- keep this instance's cold-init
        # reference rather than erroring, since that is the best available
        # anchor in that case.
        if "reference_actor" in checkpoint:
            self._reference_actor_state = [
                tensor.to(self.device) for tensor in checkpoint["reference_actor"]
            ]
        self.actor_parameter_path_length = float(
            checkpoint.get("actor_parameter_path_length", 0.0)
        )
        normalizer = checkpoint.get("return_normalizer", {})
        self.return_mean = float(normalizer.get("mean", 0.0))
        self.return_variance = float(normalizer.get("variance", 1.0))
        self.return_count = int(normalizer.get("count", 0))

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)
