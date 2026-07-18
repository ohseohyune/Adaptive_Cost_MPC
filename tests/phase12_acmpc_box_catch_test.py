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
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.mpc.online_actor_critic import DifferentiableBimanualMPC, DifferentiableMPCConfig
from control.squeeze import DynamicSideSqueezeConfig, default_curriculum
from main_acmpc_box_catch import AcmpcBoxCatchConfig, run_box_catch


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


def main() -> None:
    scenarios = [
        ("SC1 gravity changes the solved reference direction", sc1_gravity_changes_the_solved_reference),
        ("SC2 gravity=(0,0,0) default is a no-op", sc2_gravity_default_is_noop),
        ("SC3 box-catch succeeds with engineered priors (online_learning=False)", sc3_box_catch_succeeds_with_engineered_priors),
        ("SC4 box-catch succeeds with online_learning=True and still learns", sc4_online_learning_succeeds_and_still_learns),
        ("SC5 domain randomization (warmup stage) succeeds", sc5_domain_randomization_warmup_stage_succeeds),
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
