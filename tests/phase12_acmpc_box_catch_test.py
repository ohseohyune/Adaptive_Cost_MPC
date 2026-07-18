"""Phase 12: bringing the ballistic box-catch scenario into the AC-MPC pipeline.

This is a first, partial integration (see main_acmpc_box_catch.py's module
docstring for the known remaining limitation: world-Y hand-separation
convergence is too slow because CartesianImpedanceController applies K/D in
the *world* frame, and the world-Y axis is TTC-softened by the same impact
compliance schedule that also has to stay soft on the approach/normal axis).
Physical bilateral contact is not yet reliably achieved, so this suite
checks what is actually validated so far -- the gravity-aware MPC reference
term, and that the end-to-end pipeline runs and tracks the box down to a
known-good closest approach -- rather than asserting a catch that does not
happen yet. Tighten SC2's threshold (or add a bilateral-contact assertion)
once the world-frame stiffness-axis fix lands.
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
from main_acmpc_box_catch import AcmpcBoxCatchConfig, CatchPhase, run_box_catch


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


def sc3_box_catch_pipeline_runs_and_tracks() -> tuple[bool, str]:
    summary = run_box_catch(AcmpcBoxCatchConfig(seed=7, device="cpu", online_learning=True))
    reached_pre_contact = summary.final_phase in {
        CatchPhase.PRE_CONTACT.value,
        CatchPhase.GRASPING.value,
        CatchPhase.GRASPED.value,
        CatchPhase.SUCCESS.value,
    } or summary.minimum_endpoint_error_m < 0.15
    ok = (
        np.isfinite(summary.minimum_endpoint_error_m)
        and summary.minimum_endpoint_error_m < 0.15
        and summary.total_transitions > 0
        and summary.online_updates >= 1
        and reached_pre_contact
    )
    detail = (
        f"phase={summary.final_phase}, reason={summary.failure_reason}, "
        f"min_endpoint_error={summary.minimum_endpoint_error_m:.4f} m, "
        f"transitions={summary.total_transitions}, updates={summary.online_updates} "
        f"(physical bilateral contact not yet asserted -- known limitation)"
    )
    return ok, detail


def main() -> None:
    scenarios = [
        ("SC1 gravity changes the solved reference direction", sc1_gravity_changes_the_solved_reference),
        ("SC2 gravity=(0,0,0) default is a no-op", sc2_gravity_default_is_noop),
        ("SC3 box-catch pipeline runs and tracks the box", sc3_box_catch_pipeline_runs_and_tracks),
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
