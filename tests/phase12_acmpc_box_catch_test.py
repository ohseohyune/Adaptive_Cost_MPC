"""Phase 12: bringing the ballistic box-catch scenario into the AC-MPC pipeline.

required_hold_s is 5.00 (matches the box-squeeze track's
required_dynamic_hold_s), not a quick 0.30 s touch-and-go.

With engineered phase priors only (online_learning=False), the pipeline
reliably catches and holds the fixed nominal box for the full 5 s (20/20
seeds tested manually, final box speed <0.01 m/s) -- SC3 asserts this.
online_learning=True needed two separate fixes to reach the same bar:

1. A 0.3 s-hold-era fix: per-step Cartesian velocity exploration noise
   (std ~0.14 m/s at the handle-grasp demo's inherited exploration_std=0.08)
   was disturbing the delicate compliant-contact hold on its own (a control
   sweep with rollout_size larger than any possible episode -- so ~zero
   mid-episode weight updates fire -- still failed, isolating the cause to
   noise, not actor drift). Lowering exploration_std to 0.03 fixed a 0.3 s
   hold: 30/30.
2. Extending the hold to 5 s reintroduced failures even at 0.03 (0/10), and
   even at exploration_std as low as 0.002 only recovered to ~5/10 --
   diminishing returns, since _LOG_STD_MIN in online_actor_critic.py floors
   the actor's std at ~0.0067 regardless of how low exploration_std is set.
   The real fix: once GRASPED, the ideal action is "hold still" -- there is
   nothing left to explore that outweighs the risk of a bad noise sample
   destabilizing an already-successful hold over ~500 steps, so
   exploration is now switched off (deterministic mean action) from
   GRASPED onward, while still exploring (and learning) through
   INTERCEPT/PRE_CONTACT/GRASPING. 20/20 with the full 5 s hold. SC4
   asserts this.

Domain randomization (BoxDomainParameters/apply_box_domain_randomization,
the same mass/friction/size/launch curriculum main_dynamic_box_squeeze.py
uses) is wired in via AcmpcBoxCatchConfig.domain_parameters. Manually swept
at the 5 s hold bar: warmup stage 20/20. (The earlier intermediate/full-stage
percentages recorded against the old 0.3 s bar are stale and have not yet
been re-measured against 5 s.) SC5 checks one fixed-seed warmup-stage sample
as a regression gate.

Across many episodes sharing a checkpoint (a curriculum training loop),
maximum_online_actor_delta alone was not enough: a 100-episode online run
on the "full" curriculum stage (paired against the same domain sequence
with online_learning=False) went 81% -> 61% overall and specifically 84%
(episodes 0-49) -> 38% (50-99), driven by cumulative drift away from the
tuned phase priors (INTERCEPT's first-horizon velocity weight fell ~62%
below its tuned value; GRASPED's force weight rose ~55% above it) even
though each individual update stayed within its bounded per-update delta.
maximum_cumulative_actor_delta (a cap on distance from the reference/
cold-init actor, persisted across checkpoint save/load) fixed it: the same
100-episode comparison went to 84% overall with the front/back halves
essentially flat (86% vs 82%). SC6 checks the capping mechanism itself
with fast synthetic updates (no physics), and that save/load round-trips
the reference anchor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.mpc.online_actor_critic import (
    ACMPCRolloutBuffer,
    BimanualPhase,
    DifferentiableBimanualMPC,
    DifferentiableMPCConfig,
    OBS_DIM,
    OnlineActorCriticACMPC,
    OnlineActorCriticConfig,
)
from control.squeeze import DynamicSideSqueezeConfig, default_curriculum
from acmpc.main_acmpc_box_catch import AcmpcBoxCatchConfig, run_box_catch


def _solve(gravity: tuple[float, float, float]) -> np.ndarray:
    mpc = DifferentiableBimanualMPC(
        DifferentiableMPCConfig(horizon=4, dt=0.05, velocity_limit=5.0, gravity=gravity)
    )
    weights = torch.ones(1, 4, 5) * torch.tensor([50.0, 1.0, 1.0, 1.0, 1.0]).view(1, 1, 5)
    with torch.no_grad():
        mean_velocity, _ = mpc(
            ee_positions=torch.zeros(1, 6),
            object_positions=torch.tensor([[1.0, 0.0, 0.5]]),
            object_velocities=torch.tensor([[-1.4, 0.0, 3.0]]),
            relative_reference=torch.zeros(1, 3),
            weights=weights,
            previous_velocity=torch.zeros(1, 6),
        )
    return mean_velocity[0, :3].numpy()


def sc1_gravity_changes_the_solved_reference() -> tuple[bool, str]:
    # A black-box (not closed-form-exact, since the QP jointly balances
    # several cost terms) sanity check: adding downward gravity to the
    # center_ref term should measurably pull the solved z-velocity down
    # relative to the same inputs with no gravity, since the actor is
    # heavily object-cost-weighted here and should track center_ref
    # (now curving downward) rather than a straight line.
    velocity_flat = _solve((0.0, 0.0, 0.0))
    velocity_ballistic = _solve((0.0, 0.0, -9.81))
    z_dropped = velocity_ballistic[2] < velocity_flat[2] - 0.01
    xy_similar = float(np.max(np.abs(velocity_ballistic[:2] - velocity_flat[:2]))) < 0.05
    ok = z_dropped and xy_similar
    return ok, (
        f"v_z flat={velocity_flat[2]:.4f} vs ballistic={velocity_ballistic[2]:.4f} "
        f"(expect ballistic lower), xy_diff={np.max(np.abs(velocity_ballistic[:2] - velocity_flat[:2])):.4f}"
    )


def sc2_gravity_default_is_noop() -> tuple[bool, str]:
    mpc = DifferentiableBimanualMPC(DifferentiableMPCConfig(horizon=6, dt=0.02, velocity_limit=0.25))
    ok = tuple(mpc.gravity.tolist()) == (0.0, 0.0, 0.0)
    return ok, f"gravity_buffer={mpc.gravity.tolist()}"


def sc3_box_catch_succeeds_with_engineered_priors() -> tuple[bool, str]:
    summary = run_box_catch(AcmpcBoxCatchConfig(seed=7, device="cpu", online_learning=False))
    ok = (
        summary.success
        and summary.hold_time_s >= 5.00
        and summary.final_box_speed_mps < 0.5
    )
    detail = (
        f"success={summary.success}, phase={summary.final_phase}, "
        f"hold={summary.hold_time_s:.3f}s, final_speed={summary.final_box_speed_mps:.3f} m/s, "
        f"min_endpoint_error={summary.minimum_endpoint_error_m:.4f} m"
    )
    return ok, detail


def sc4_online_learning_succeeds_and_still_learns() -> tuple[bool, str]:
    summary = run_box_catch(AcmpcBoxCatchConfig(seed=7, device="cpu", online_learning=True))
    ok = (
        summary.success
        and summary.hold_time_s >= 5.00
        and summary.total_transitions > 0
        and summary.online_updates >= 1
        and summary.actor_weight_change_l2 > 0.0
    )
    detail = (
        f"success={summary.success}, phase={summary.final_phase}, "
        f"hold={summary.hold_time_s:.3f}s, transitions={summary.total_transitions}, "
        f"updates={summary.online_updates}, actor_delta_l2={summary.actor_weight_change_l2:.4f}"
    )
    return ok, detail


def sc5_domain_randomization_warmup_stage_succeeds() -> tuple[bool, str]:
    seed = 3
    stage = default_curriculum()[0]  # warmup
    domain = stage.sample(np.random.default_rng(seed), 0)
    squeeze = DynamicSideSqueezeConfig(
        random_seed=seed,
        box_half_y=domain.half_size[1],
        launch_velocity_low=domain.launch_velocity,
        launch_velocity_high=domain.launch_velocity,
    )
    summary = run_box_catch(
        AcmpcBoxCatchConfig(seed=seed, device="cpu", squeeze=squeeze, domain_parameters=domain)
    )
    ok = summary.success and summary.hold_time_s >= 5.00
    detail = (
        f"mass={domain.mass:.3f} friction={domain.friction:.3f} "
        f"half_size={tuple(round(v, 4) for v in domain.half_size)}, "
        f"success={summary.success}, hold={summary.hold_time_s:.3f}s"
    )
    return ok, detail


def _synthetic_transition(rng: np.random.Generator, phase: BimanualPhase) -> dict:
    return dict(
        observation=rng.normal(size=OBS_DIM).astype(np.float32),
        phase=phase,
        ee_positions=(rng.normal(size=6) * 0.1).astype(np.float32),
        object_position=(rng.normal(size=3) * 0.1).astype(np.float32),
        object_velocity=(rng.normal(size=3) * 0.05).astype(np.float32),
        relative_reference=np.array([0.0, -0.5, 0.0], dtype=np.float32),
        previous_velocity=(rng.normal(size=6) * 0.05).astype(np.float32),
    )


def sc6_cumulative_actor_delta_cap_holds_and_round_trips() -> tuple[bool, str]:
    cap = 0.05
    mpc_cfg = DifferentiableMPCConfig(horizon=4, dt=0.02, velocity_limit=0.2)
    learning_cfg = OnlineActorCriticConfig(
        hidden_dim=32,
        device="cpu",
        seed=13,
        minibatch_size=4,
        minimum_online_rollout=4,
        maximum_online_actor_delta=0.02,  # looser than the cumulative cap
        maximum_cumulative_actor_delta=cap,
        training_epochs=3,
        target_kl=10.0,  # isolate the cumulative cap; don't let KL stop updates first
    )
    learner = OnlineActorCriticACMPC(mpc_cfg, learning_cfg)
    rng = np.random.default_rng(13)
    phases = [
        BimanualPhase.INTERCEPT,
        BimanualPhase.PRE_CONTACT,
        BimanualPhase.GRASPING,
        BimanualPhase.GRASPED,
    ]

    max_seen_delta = 0.0
    for update_index in range(20):
        rollout = ACMPCRolloutBuffer()
        for step in range(8):
            transition = _synthetic_transition(rng, phases[step % len(phases)])
            action = learner.act(**transition, training=True)
            rollout.add(
                **transition,
                action=action,
                reward=float(rng.normal()),
                done=(step == 7),
            )
        learner.update(rollout, online=True, next_value=0.0)
        max_seen_delta = max(
            max_seen_delta, learner._actor_delta(learner._reference_actor_state)
        )

    cap_held = max_seen_delta <= cap + 1e-6

    # save/load must round-trip the reference anchor, not reset it -- that
    # is the entire point (a fresh run_box_catch() call constructs a new
    # learner and loads a shared checkpoint every episode).
    ckpt_path = Path(__file__).resolve().parent / "_sc6_scratch_checkpoint.pt"
    try:
        learner.save(ckpt_path)
        reloaded = OnlineActorCriticACMPC(mpc_cfg, learning_cfg)
        reloaded.load(ckpt_path)
        reference_matches = all(
            torch.allclose(a, b)
            for a, b in zip(learner._reference_actor_state, reloaded._reference_actor_state)
        )
    finally:
        if ckpt_path.exists():
            ckpt_path.unlink()

    ok = cap_held and reference_matches
    detail = (
        f"cap={cap}, max cumulative actor_delta over 20 updates={max_seen_delta:.4f} "
        f"(held={cap_held}), reference round-trips through save/load={reference_matches}"
    )
    return ok, detail


def main() -> None:
    scenarios = [
        ("SC1 gravity changes the solved reference direction", sc1_gravity_changes_the_solved_reference),
        ("SC2 gravity=(0,0,0) default is a no-op", sc2_gravity_default_is_noop),
        ("SC3 box-catch succeeds with engineered priors (online_learning=False)", sc3_box_catch_succeeds_with_engineered_priors),
        ("SC4 box-catch succeeds with online_learning=True and still learns", sc4_online_learning_succeeds_and_still_learns),
        ("SC5 domain randomization (warmup stage) succeeds", sc5_domain_randomization_warmup_stage_succeeds),
        ("SC6 cumulative actor-delta cap holds and round-trips", sc6_cumulative_actor_delta_cap_holds_and_round_trips),
    ]
    passed = 0
    print("\n=== Phase 12: AC-MPC ballistic box catch (partial) ===\n")
    for name, scenario in scenarios:
        ok, detail = scenario()
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}\n")
        passed += int(ok)
    print(f"Result: {passed}/{len(scenarios)} SCs passed")
    if passed != len(scenarios):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
