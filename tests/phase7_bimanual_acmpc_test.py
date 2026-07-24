"""Phase 7 end-to-end online bimanual AC-MPC validation.

Usage:
    python tests/phase7_bimanual_acmpc_test.py

The final scenario is deliberately headless and deterministic.  It verifies
that both grippers physically contact the same moving free body, the grasp
survives into manipulation, and real MuJoCo rewards change the neural actor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.mpc.online_actor_critic import (
    OBS_DIM,
    AdaptiveCostActor,
    BimanualPhase,
    DifferentiableBimanualMPC,
    DifferentiableMPCConfig,
    OnlineActorCriticACMPC,
    build_bimanual_observation,
    resolve_device,
)
from acmpc.main_bimanual_acmpc import DemoConfig, run_demo


def sc1_observation() -> tuple[bool, str]:
    obs = build_bimanual_observation(
        object_velocity=np.array([0.1, 0.0, -0.1]),
        left_endpoint_error=np.array([0.02, 0.0, -0.03]),
        right_endpoint_error=np.array([0.02, 0.0, -0.03]),
        left_ee_velocity=np.zeros(3),
        right_ee_velocity=np.zeros(3),
        left_force=2.0,
        right_force=2.2,
        time_to_contact=0.3,
        phase=BimanualPhase.GRASPING,
    )
    ok = obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
    return ok, f"shape={obs.shape}, finite={np.isfinite(obs).all()}"


def sc2_differentiable_mpc() -> tuple[bool, str]:
    torch.manual_seed(2)
    cfg = DifferentiableMPCConfig(horizon=5, dt=0.02, velocity_limit=0.2)
    actor = AdaptiveCostActor(cfg.horizon, hidden_dim=32, delta_fraction=0.5)
    mpc = DifferentiableBimanualMPC(cfg)
    obs = torch.randn(1, OBS_DIM)
    weights = actor(obs, torch.tensor([int(BimanualPhase.PRE_CONTACT)]))
    mean, sequence = mpc(
        ee_positions=torch.tensor([[0.28, 0.26, 0.81, 0.28, -0.26, 0.81]]),
        object_positions=torch.tensor([[0.25, 0.0, 0.74]]),
        object_velocities=torch.tensor([[0.02, 0.0, 0.0]]),
        relative_reference=torch.tensor([[0.0, -0.52, 0.0]]),
        weights=weights,
        previous_velocity=torch.zeros(1, 6),
    )
    loss = mean.square().sum() + 0.01 * sequence.square().sum()
    loss.backward()
    grad_norm = sum(
        float(parameter.grad.norm())
        for parameter in actor.parameters()
        if parameter.grad is not None
    )
    bounded = float(mean.detach().abs().max()) <= cfg.velocity_limit + 1e-6
    ok = mean.shape == (1, 6) and sequence.shape == (1, cfg.horizon, 6)
    ok = ok and bounded and np.isfinite(grad_norm) and grad_norm > 0.0
    return ok, f"mean_shape={tuple(mean.shape)}, bounded={bounded}, actor_grad={grad_norm:.3e}"


def sc3_device_fallback() -> tuple[bool, str]:
    cpu = resolve_device("cpu")
    auto = resolve_device("auto")
    ok = cpu.type == "cpu" and auto.type in {"cpu", "cuda"}
    return ok, f"cpu={cpu}, auto={auto}, cuda_available={torch.cuda.is_available()}"


def sc4_online_physical_grasp() -> tuple[bool, str]:
    summary = run_demo(
        DemoConfig(
            duration_s=6.0,
            device="cpu",
            online_learning=True,
            exploration_std=0.015,
            seed=7,
            viewer=False,
        )
    )
    ok = (
        summary.success
        and summary.final_phase == "manipulation"
        and summary.grasp_time_s is not None
        and summary.left_contact_force_n > 0.005
        and summary.right_contact_force_n > 0.005
        # PPO updates in batches (rollout_size steps per update), not every
        # control step, so the meaningful "enough real experience" check is
        # on total buffered transitions; online_updates just needs to be
        # nonzero (at least one PPO update actually applied).
        and summary.total_transitions >= 250
        and summary.online_updates >= 1
        and summary.actor_weight_change_l2 > 1e-4
        and summary.object_displacement_m > 0.02
    )
    detail = (
        f"success={summary.success}, phase={summary.final_phase}, "
        f"grasp_t={summary.grasp_time_s}, force=({summary.left_contact_force_n:.2f},"
        f"{summary.right_contact_force_n:.2f})N, transitions={summary.total_transitions}, "
        f"updates={summary.online_updates}, "
        f"actor_delta={summary.actor_weight_change_l2:.3e}, "
        f"object_motion={summary.object_displacement_m:.3f}m"
    )
    return ok, detail


def sc5_ppo_update_mechanics() -> tuple[bool, str]:
    from control.mpc.online_actor_critic import ACMPCRolloutBuffer, OnlineActorCriticConfig

    mpc_cfg = DifferentiableMPCConfig(horizon=4, dt=0.02, velocity_limit=0.2)
    learning_cfg = OnlineActorCriticConfig(
        hidden_dim=32,
        device="cpu",
        seed=11,
        minibatch_size=4,
        minimum_online_rollout=4,
        maximum_online_actor_delta=0.01,
        training_epochs=3,
    )
    learner = OnlineActorCriticACMPC(mpc_cfg, learning_cfg)

    rng = np.random.default_rng(11)
    rollout = ACMPCRolloutBuffer()
    phases = [
        BimanualPhase.INTERCEPT,
        BimanualPhase.PRE_CONTACT,
        BimanualPhase.GRASPING,
        BimanualPhase.GRASPED,
    ]
    for index in range(8):
        phase = phases[index % len(phases)]
        observation = rng.normal(size=OBS_DIM).astype(np.float32)
        ee_positions = (rng.normal(size=6) * 0.1).astype(np.float32)
        object_position = (rng.normal(size=3) * 0.1).astype(np.float32)
        object_velocity = (rng.normal(size=3) * 0.05).astype(np.float32)
        relative_reference = np.array([0.0, -0.5, 0.0], dtype=np.float32)
        previous_velocity = (rng.normal(size=6) * 0.05).astype(np.float32)
        action = learner.act(
            observation=observation,
            phase=phase,
            ee_positions=ee_positions,
            object_position=object_position,
            object_velocity=object_velocity,
            relative_reference=relative_reference,
            previous_velocity=previous_velocity,
            training=True,
        )
        rollout.add(
            observation=observation,
            phase=phase,
            ee_positions=ee_positions,
            object_position=object_position,
            object_velocity=object_velocity,
            relative_reference=relative_reference,
            previous_velocity=previous_velocity,
            action=action,
            reward=float(rng.normal()),
            done=(index == 7),
        )

    online_summary = learner.update(rollout, online=True, next_value=0.0)
    online_ok = (
        online_summary.applied
        and online_summary.transitions == 8
        and online_summary.actor_parameter_delta <= learning_cfg.maximum_online_actor_delta + 1e-6
        and np.isfinite(online_summary.approximate_kl)
    )

    offline_summary = learner.update(rollout, online=False, next_value=0.0)
    offline_ok = offline_summary.applied and offline_summary.epochs == learning_cfg.training_epochs

    ok = online_ok and offline_ok
    detail = (
        f"online: applied={online_summary.applied}, "
        f"delta={online_summary.actor_parameter_delta:.4f}, "
        f"kl={online_summary.approximate_kl:.4f} | "
        f"offline: applied={offline_summary.applied}, epochs={offline_summary.epochs}"
    )
    return ok, detail


def main() -> None:
    scenarios = [
        ("SC1 observation", sc1_observation),
        ("SC2 differentiable MPC", sc2_differentiable_mpc),
        ("SC3 CPU/CUDA device selection", sc3_device_fallback),
        ("SC4 online physical bimanual grasp", sc4_online_physical_grasp),
        ("SC5 PPO update mechanics", sc5_ppo_update_mechanics),
    ]
    passed = 0
    print("\n=== Phase 7: Online Bimanual AC-MPC ===\n")
    for name, scenario in scenarios:
        ok, detail = scenario()
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}\n")
        passed += int(ok)
    print(f"Result: {passed}/{len(scenarios)} SCs passed")
    if passed != len(scenarios):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
