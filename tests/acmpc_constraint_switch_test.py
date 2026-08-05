"""Fast self-check for AC-MPC constraint-ablation switches.

Usage: python tests/acmpc_constraint_switch_test.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.mpc.online_actor_critic import (
    ACMPCRolloutBuffer,
    OBS_DIM,
    AdaptiveCostActor,
    BimanualPhase,
    DifferentiableMPCConfig,
    OnlineActorCriticACMPC,
    OnlineActorCriticConfig,
)


def _transition(rng: np.random.Generator) -> dict:
    return {
        "observation": rng.normal(size=OBS_DIM).astype(np.float32),
        "phase": BimanualPhase.PRE_CONTACT,
        "ee_positions": (rng.normal(size=6) * 0.1).astype(np.float32),
        "object_position": (rng.normal(size=3) * 0.1).astype(np.float32),
        "object_velocity": (rng.normal(size=3) * 0.05).astype(np.float32),
        "relative_reference": np.array([0.0, -0.5, 0.0], dtype=np.float32),
        "previous_velocity": (rng.normal(size=6) * 0.05).astype(np.float32),
    }


def _rollout(learner: OnlineActorCriticACMPC, seed: int = 4) -> ACMPCRolloutBuffer:
    rng = np.random.default_rng(seed)
    rollout = ACMPCRolloutBuffer()
    for index in range(8):
        transition = _transition(rng)
        action = learner.act(**transition, training=True)
        rollout.add(
            **transition,
            action=action,
            reward=float(rng.normal()),
            done=index == 7,
        )
    return rollout


def check_weight_parameterizations() -> None:
    observation = torch.zeros(1, OBS_DIM)
    phase = torch.tensor([int(BimanualPhase.PRE_CONTACT)])
    bounded = AdaptiveCostActor(3, 16, 0.65)
    exponential = AdaptiveCostActor(
        3,
        16,
        0.65,
        weight_parameterization="exp_residual",
        weight_clip_min=None,
        weight_clip_max=None,
    )
    expected = bounded.phase_priors[phase].unsqueeze(1).expand(-1, 3, -1)
    assert torch.allclose(bounded(observation, phase), expected)
    assert torch.allclose(exponential(observation, phase), expected)

    with torch.no_grad():
        bounded.net[-1].bias.fill_(2.0)
        exponential.net[-1].bias.fill_(2.0)
    bounded_weights = bounded(observation, phase)
    exponential_weights = exponential(observation, phase)
    assert torch.all(exponential_weights > 0.0)
    assert torch.all(exponential_weights > bounded_weights)
    assert torch.allclose(
        exponential_weights,
        expected * torch.exp(torch.tensor(0.65 * 2.0)),
    )

    clipped = AdaptiveCostActor(
        2, 16, 0.65, weight_clip_min=None, weight_clip_max=10.0
    )
    final, preclip = clipped.forward_with_preclip(observation, phase)
    assert torch.any(preclip > final)
    assert float(final.detach().max()) == 10.0

    with torch.no_grad():
        exponential.net[-1].bias.fill_(1e4)
    try:
        exponential(observation, phase)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("non-finite exponential weights were not rejected")


def check_optional_update_guards() -> None:
    config = OnlineActorCriticConfig(
        hidden_dim=24,
        device="cpu",
        seed=4,
        minibatch_size=4,
        minimum_online_rollout=4,
        target_kl=None,
        clip_ratio=None,
        maximum_online_actor_delta=None,
        maximum_cumulative_actor_delta=None,
        weight_parameterization="exp_residual",
        weight_clip_min=None,
        weight_clip_max=None,
        normalize_returns=True,
    )
    learner = OnlineActorCriticACMPC(
        DifferentiableMPCConfig(horizon=3, velocity_limit=0.2), config
    )
    summary = learner.update(_rollout(learner), online=True)
    assert summary.applied and np.isfinite(summary.approximate_kl)
    assert not summary.target_kl_stopped
    assert not summary.online_delta_clipped
    assert not summary.cumulative_delta_clipped
    assert summary.ppo_clip_fraction == 0.0
    assert np.isclose(
        summary.raw_actor_parameter_delta,
        summary.actor_parameter_delta,
        rtol=1e-4,
        atol=1e-7,
    )
    assert summary.actor_parameter_path_length > 0.0
    assert learner.return_count == 8
    assert np.isfinite(learner.return_mean) and learner.return_variance > 0.0


def _nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def check_frozen_policy_is_read_only() -> None:
    config = OnlineActorCriticConfig(
        hidden_dim=24,
        device="cpu",
        seed=9,
        minibatch_size=4,
        minimum_online_rollout=4,
        normalize_returns=True,
    )
    learner = OnlineActorCriticACMPC(
        DifferentiableMPCConfig(horizon=3, velocity_limit=0.2), config
    )
    rollout = _rollout(learner, seed=9)
    learner.update(rollout, online=True)
    actor_before = copy.deepcopy(learner.actor.state_dict())
    critic_before = copy.deepcopy(learner.critic.state_dict())
    actor_optimizer_before = copy.deepcopy(learner.actor_optimizer.state_dict())
    critic_optimizer_before = copy.deepcopy(learner.critic_optimizer.state_dict())
    counters_before = (
        learner.update_count,
        learner.actor_parameter_path_length,
        learner.return_mean,
        learner.return_variance,
        learner.return_count,
        len(rollout),
    )
    transition = _transition(np.random.default_rng(99))
    first = learner.act(**transition, training=False)
    second = learner.act(**transition, training=False)
    learner.predict_value(transition["observation"])
    counters_after = (
        learner.update_count,
        learner.actor_parameter_path_length,
        learner.return_mean,
        learner.return_variance,
        learner.return_count,
        len(rollout),
    )
    assert np.array_equal(first.normalized_action, second.normalized_action)
    assert _nested_equal(actor_before, learner.actor.state_dict())
    assert _nested_equal(critic_before, learner.critic.state_dict())
    assert _nested_equal(actor_optimizer_before, learner.actor_optimizer.state_dict())
    assert _nested_equal(critic_optimizer_before, learner.critic_optimizer.state_dict())
    assert counters_before == counters_after


if __name__ == "__main__":
    check_weight_parameterizations()
    check_optional_update_guards()
    check_frozen_policy_is_read_only()
    print("AC-MPC constraint switches: PASS")
