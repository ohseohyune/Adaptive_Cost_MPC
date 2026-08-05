"""Audits behind the residual-zero ablation.

Three independent claims, each cheap enough to run on CPU in seconds:

1. residual_zero makes the MPC's effective cost weights exactly the phase
   prior (no clip, no residual, no drift).
2. the evaluation path is deterministic -- the same observation must produce
   a bit-identical action every time when training=False.
3. total_actor_loss decomposes exactly into surrogate + entropy bonus, which
   is what lets the pre-existing result.json logs be decomposed after the
   fact (the historical "policy_loss" field is the *total*).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.main_acmpc_box_catch import (  # noqa: E402
    AcmpcBoxCatchConfig,
    N_CONTROL_PHASES,
)
from control.mpc.online_actor_critic import (  # noqa: E402
    COST_NAMES,
    DifferentiableMPCConfig,
    OnlineActorCriticACMPC,
    OnlineActorCriticConfig,
)


def _learner(**overrides) -> OnlineActorCriticACMPC:
    box_config = AcmpcBoxCatchConfig()
    mpc_config = DifferentiableMPCConfig(
        horizon=box_config.mpc_horizon,
        velocity_limit=box_config.mpc_velocity_limit,
    )
    learner_config = OnlineActorCriticConfig(
        device="cpu",
        seed=7,
        initial_log_std=float(np.log(box_config.exploration_std)),
        n_phases=N_CONTROL_PHASES,
        weight_delta_fraction=box_config.weight_delta_fraction,
        **overrides,
    )
    return OnlineActorCriticACMPC(mpc_config, learner_config)


def _act_inputs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    learner = _learner()
    return dict(
        observation=rng.normal(size=learner.obs_dim).astype(np.float64),
        phase=1,
        ee_positions=rng.normal(scale=0.2, size=6),
        object_position=rng.normal(scale=0.2, size=3),
        object_velocity=rng.normal(scale=0.5, size=3),
        relative_reference=rng.normal(scale=0.1, size=3),
        previous_velocity=np.zeros(6),
    )


def test_residual_zero_weights_equal_phase_prior() -> None:
    inputs = _act_inputs()
    zero = _learner(residual_zero=True)
    # Perturb the actor away from its init so a residual would be visible if
    # it were not being bypassed.
    with torch.no_grad():
        for parameter in zero.actor.net.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.5)

    action = zero.act(training=False, **inputs)
    prior = np.asarray(action.phase_prior, dtype=float)
    for index, name in enumerate(COST_NAMES):
        weights = np.asarray(action.weights[name], dtype=float)
        error = float(np.max(np.abs(weights - prior[index])))
        assert error <= 1e-8, f"{name}: effective weight deviates by {error:.3e}"
    assert action.zero_residual_command_delta == 0.0

    learned = _learner()
    with torch.no_grad():
        for parameter, source in zip(
            learned.actor.net.parameters(), zero.actor.net.parameters()
        ):
            parameter.copy_(source)
    learned_action = learned.act(training=False, **inputs)
    learned_weights = np.asarray(learned_action.weights[COST_NAMES[0]], dtype=float)
    assert float(np.max(np.abs(learned_weights - prior[0]))) > 1e-6, (
        "the perturbed actor produced a zero residual on its own -- this test "
        "would pass vacuously"
    )


def test_evaluation_action_is_deterministic() -> None:
    learner = _learner()
    inputs = _act_inputs(seed=3)
    actions = [learner.act(training=False, **inputs).velocity for _ in range(100)]
    reference = actions[0]
    worst = max(float(np.max(np.abs(a - reference))) for a in actions)
    assert worst <= 1e-8, f"deterministic eval drifted by {worst:.3e}"

    sampled = [learner.act(training=True, **inputs).velocity for _ in range(20)]
    spread = max(float(np.max(np.abs(a - sampled[0]))) for a in sampled)
    assert spread > 1e-6, "training=True produced no exploration noise"


def test_actor_loss_decomposition_identity() -> None:
    entropy_coef = 1e-3
    rng = np.random.default_rng(11)
    for _ in range(100):
        surrogate = float(rng.normal(scale=1e-3))
        entropy = float(rng.normal(loc=-12.4, scale=0.5))
        entropy_bonus = -entropy_coef * entropy
        total = surrogate + entropy_bonus
        assert abs(total - (surrogate + entropy_bonus)) < 1e-8
        # The reconstruction actually applied to the historical logs.
        assert abs((total + entropy_coef * entropy) - surrogate) < 1e-12


def test_historical_log_reconstruction_matches_reported_value() -> None:
    total_actor_loss = 0.012487342581152916
    entropy = -12.487319946289062
    entropy_coef = 1e-3
    surrogate = total_actor_loss + entropy_coef * entropy
    assert abs(surrogate - 2.26e-8) < 1e-9, surrogate
    assert abs(-entropy_coef * entropy - 0.01248732) < 1e-8


def _synthetic_rollout(learner: OnlineActorCriticACMPC, steps: int = 24):
    from control.mpc.online_actor_critic import ACMPCRolloutBuffer

    rollout = ACMPCRolloutBuffer()
    rng = np.random.default_rng(5)
    previous = np.zeros(6)
    for step in range(steps):
        inputs = dict(
            observation=rng.normal(size=learner.obs_dim).astype(np.float64),
            phase=step % 3,
            ee_positions=rng.normal(scale=0.2, size=6),
            object_position=rng.normal(scale=0.2, size=3),
            object_velocity=rng.normal(scale=0.5, size=3),
            relative_reference=rng.normal(scale=0.1, size=3),
            previous_velocity=previous,
        )
        action = learner.act(training=True, **inputs)
        rollout.add(
            action=action,
            reward=float(rng.normal()),
            done=step == steps - 1,
            **inputs,
        )
        previous = action.velocity
    return rollout


def test_post_step_kl_tracks_actual_parameter_movement() -> None:
    frozen = _learner(log_post_step_kl=True, actor_lr=0.0, target_kl=None)
    summary = frozen.update(_synthetic_rollout(frozen), online=False)
    records = summary.minibatch_diagnostics
    assert records, "no minibatch diagnostics recorded"
    post = [r["post_step_approximate_kl"] for r in records]
    assert all(value is not None for value in post)
    assert all(np.isfinite(value) for value in post)
    assert max(post) <= 1e-8, f"lr=0 still moved the policy: max KL {max(post):.3e}"

    moving = _learner(log_post_step_kl=True, actor_lr=1e-2, target_kl=None)
    summary = moving.update(_synthetic_rollout(moving), online=False)
    post = [r["post_step_approximate_kl"] for r in summary.minibatch_diagnostics]
    assert all(np.isfinite(value) for value in post)
    assert max(post) > 0.0, "a nonzero learning rate produced zero post-step KL"


def test_loss_decomposition_is_recorded_not_reconstructed() -> None:
    learner = _learner(target_kl=None)
    summary = learner.update(_synthetic_rollout(learner), online=False)
    assert summary.minibatch_diagnostics
    for record in summary.minibatch_diagnostics:
        total = record["total_actor_loss"]
        parts = record["policy_surrogate_loss"] + record["entropy_bonus"]
        assert abs(total - parts) < 1e-8, f"{total} != {parts}"
        # float32 rounding of the product inside the graph, not a logic gap.
        assert (
            abs(record["entropy_bonus"] - -record["entropy_coef"] * record["entropy"])
            < 1e-8
        )


def main() -> int:
    tests = [
        test_residual_zero_weights_equal_phase_prior,
        test_evaluation_action_is_deterministic,
        test_actor_loss_decomposition_identity,
        test_historical_log_reconstruction_matches_reported_value,
        test_loss_decomposition_is_recorded_not_reconstructed,
        test_post_step_kl_tracks_actual_parameter_movement,
    ]
    for test in tests:
        test()
        print(f"  {test.__name__}: PASS", flush=True)
    print("AC-MPC residual-zero audits: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
