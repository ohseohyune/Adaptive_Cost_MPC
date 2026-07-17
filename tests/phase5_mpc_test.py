"""Phase 5: Adaptive Cost MPC — success-criteria test suite.

SCs
---
SC1  Weight table correctness  — verify COST_WEIGHTS has the exact values
     specified for every GraspState.
SC2  QP feasibility            — AdaptiveCostMPC.solve() returns 'solved'
     (or 'solved inaccurate') for every GraspState with a non-trivial setup.
SC3  Weight switching effect   — changing GraspState changes cost structure:
     (a) APPROACH (w_pos=10) produces larger position-tracking signal than
         GRASPING (w_pos=1) when given the same position error.
     (b) GRASPING (w_force=5) produces larger force-tracking signal than
         APPROACH (w_force=0) when given the same f_desired.
SC4  Position tracking         — in APPROACH state, tau_cmd accelerates the
     EE toward x_target (J·τ has positive dot product with x_target − x_EE).
SC5  Force regulation          — in GRASPING state with non-zero f_desired,
     tau_cmd has larger projection onto J^T·f_desired than APPROACH state
     (where w_force=0).
"""

from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, ".")

from control.clik.grasp_state_machine import GraspState
from control.mpc.adaptive_cost_mpc import (
    COST_WEIGHTS,
    AdaptiveCostMPC,
    MPCConfig,
)

# ── expected weight schedule (matches spec) ────────────────────────────────────
_EXPECTED: dict[GraspState, dict[str, float]] = {
    GraspState.APPROACH:     {"pos": 10.0, "force": 0.0, "grasp": 0.0},
    GraspState.PRE_CONTACT:  {"pos":  5.0, "force": 1.0, "grasp": 2.0},
    GraspState.GRASPING:     {"pos":  1.0, "force": 5.0, "grasp": 5.0},
    GraspState.GRASPED:      {"pos":  8.0, "force": 3.0, "grasp": 3.0},
    GraspState.MANIPULATION: {"pos": 10.0, "force": 2.0, "grasp": 1.0},
}

# ── shared test fixtures ───────────────────────────────────────────────────────
_RNG   = np.random.default_rng(42)
_N_DOF = 7

# Fixed Jacobian that is full-rank (translational, scaled to ~mm/rad)
_J: np.ndarray = _RNG.standard_normal((3, _N_DOF)) * 0.05

_X_EE     = np.array([0.30, -0.30, 0.70])
_X_TARGET = np.array([0.35, -0.30, 0.70])    # 5 cm ahead in x
_F_DESIRED = np.array([10.0,  0.0,  0.0])    # 10 N in world-x
_QDOT = np.zeros(_N_DOF)

_MPC_CONFIG = MPCConfig(horizon=10, dt=0.01, w_reg=1e-3, verbose=False)


def _make_mpc() -> AdaptiveCostMPC:
    """Fresh MPC (no warm-start state) for each SC."""
    return AdaptiveCostMPC(n_dof=_N_DOF, config=_MPC_CONFIG)


# ── SC1: weight table correctness ─────────────────────────────────────────────

def sc1_weight_table() -> tuple[bool, str]:
    errors: list[str] = []
    for state, expected in _EXPECTED.items():
        actual = COST_WEIGHTS.get(state)
        if actual is None:
            errors.append(f"  {state.value}: missing from COST_WEIGHTS")
            continue
        for key in ("pos", "force", "grasp"):
            if abs(actual.get(key, float("nan")) - expected[key]) > 1e-9:
                errors.append(
                    f"  {state.value}[{key}]: got {actual.get(key)}, expected {expected[key]}"
                )
    if errors:
        return False, "Weight table mismatches:\n" + "\n".join(errors)
    # Verify all 5 states present
    if len(COST_WEIGHTS) != 5:
        return False, f"Expected 5 states in COST_WEIGHTS, found {len(COST_WEIGHTS)}"
    return True, "All 5 states × 3 weights match specification"


# ── SC2: QP feasibility for every GraspState ──────────────────────────────────

def sc2_qp_feasibility() -> tuple[bool, str]:
    failures: list[str] = []
    for state in GraspState:
        mpc = _make_mpc()
        try:
            res = mpc.solve(
                grasp_state=state,
                x_ee=_X_EE,
                qdot=_QDOT,
                jacobian=_J,
                x_target=_X_TARGET,
                f_desired=_F_DESIRED,
            )
        except Exception as exc:
            failures.append(f"  {state.value}: raised {type(exc).__name__}: {exc}")
            continue
        ok_statuses = {"solved", "solved inaccurate"}
        if not any(s in res.status.lower() for s in ok_statuses):
            failures.append(f"  {state.value}: unexpected status '{res.status}'")
        # Basic shape checks
        if res.tau_cmd.shape != (_N_DOF,):
            failures.append(f"  {state.value}: tau_cmd shape {res.tau_cmd.shape}")
        if res.state_trajectory.shape != (11, 3 + _N_DOF):
            failures.append(f"  {state.value}: traj shape {res.state_trajectory.shape}")
        if not np.all(np.isfinite(res.tau_cmd)):
            failures.append(f"  {state.value}: tau_cmd contains non-finite values")
    if failures:
        return False, "QP failures:\n" + "\n".join(failures)
    return True, "All 5 GraspStates return 'solved' with finite tau_cmd"


# ── SC3: weight switching changes cost structure ───────────────────────────────

def sc3_weight_switching() -> tuple[bool, str]:
    msgs: list[str] = []
    errors: list[str] = []

    # (a) APPROACH (w_pos=10) vs GRASPING (w_pos=1) — position signal
    # With same position error and no force desired, the position cost component
    # of the objective is proportional to w_pos.
    # We compare ||J·tau_cmd|| as a proxy for position-tracking effort.
    mpc_a = _make_mpc()
    mpc_g = _make_mpc()
    res_approach = mpc_a.solve(
        grasp_state=GraspState.APPROACH,
        x_ee=_X_EE, qdot=_QDOT, jacobian=_J, x_target=_X_TARGET,
        f_desired=None,   # disable force term for clean comparison
    )
    res_grasping = mpc_g.solve(
        grasp_state=GraspState.GRASPING,
        x_ee=_X_EE, qdot=_QDOT, jacobian=_J, x_target=_X_TARGET,
        f_desired=None,
    )
    ee_signal_approach = float(np.linalg.norm(_J @ res_approach.tau_cmd))
    ee_signal_grasping = float(np.linalg.norm(_J @ res_grasping.tau_cmd))
    msgs.append(f"  J·tau EE signal: APPROACH={ee_signal_approach:.5f}  GRASPING={ee_signal_grasping:.5f}")
    if ee_signal_approach <= ee_signal_grasping:
        errors.append(
            f"APPROACH position signal ({ee_signal_approach:.5f}) should exceed "
            f"GRASPING ({ee_signal_grasping:.5f}) because w_pos(APPROACH)=10 > w_pos(GRASPING)=1"
        )

    # (b) GRASPING (w_force=5) vs APPROACH (w_force=0) — force signal
    # Force reference is tau_fref = J.T @ f_desired.
    # In GRASPING, tau_cmd should have larger projection onto tau_fref.
    tau_fref = _J.T @ _F_DESIRED
    tau_fref_dir = tau_fref / (np.linalg.norm(tau_fref) + 1e-12)
    mpc_a2 = _make_mpc()
    mpc_g2 = _make_mpc()
    res_app2 = mpc_a2.solve(
        grasp_state=GraspState.APPROACH,
        x_ee=_X_TARGET,   # no position error so force is the distinguishing term
        qdot=_QDOT, jacobian=_J, x_target=_X_TARGET,
        f_desired=_F_DESIRED,
    )
    res_grasp2 = mpc_g2.solve(
        grasp_state=GraspState.GRASPING,
        x_ee=_X_TARGET,
        qdot=_QDOT, jacobian=_J, x_target=_X_TARGET,
        f_desired=_F_DESIRED,
    )
    proj_approach = float(tau_fref_dir @ res_app2.tau_cmd)
    proj_grasping = float(tau_fref_dir @ res_grasp2.tau_cmd)
    msgs.append(
        f"  force projection: APPROACH={proj_approach:.5f}  GRASPING={proj_grasping:.5f}"
    )
    if proj_grasping <= proj_approach:
        errors.append(
            f"GRASPING force projection ({proj_grasping:.5f}) should exceed "
            f"APPROACH ({proj_approach:.5f}) because w_force(GRASPING)=5 > w_force(APPROACH)=0"
        )

    detail = "\n".join(msgs)
    if errors:
        return False, detail + "\nFailed:\n  " + "\n  ".join(errors)
    return True, detail


# ── SC4: position tracking direction ──────────────────────────────────────────

def sc4_position_tracking() -> tuple[bool, str]:
    """In APPROACH, J·tau_cmd should point toward x_target − x_ee."""
    errors: list[str] = []
    mpc = _make_mpc()
    res = mpc.solve(
        grasp_state=GraspState.APPROACH,
        x_ee=_X_EE, qdot=_QDOT, jacobian=_J, x_target=_X_TARGET,
        f_desired=None,
    )
    pos_error_dir = _X_TARGET - _X_EE  # points toward target
    ee_accel      = _J @ res.tau_cmd   # acceleration direction from torque
    dot_val = float(np.dot(ee_accel, pos_error_dir))
    msgs = [
        f"  pos_error  = {pos_error_dir}",
        f"  J·tau_cmd  = {ee_accel.round(5)}",
        f"  dot(J·τ, Δx) = {dot_val:.6f}  (must be > 0)",
    ]
    if dot_val <= 0.0:
        errors.append(
            "J·tau_cmd does not point toward target "
            f"(dot={dot_val:.6f} ≤ 0)"
        )
    detail = "\n".join(msgs)
    if errors:
        return False, detail + "\nFailed: " + errors[0]
    return True, detail


# ── SC5: force regulation in GRASPING vs APPROACH ─────────────────────────────

def sc5_force_regulation() -> tuple[bool, str]:
    """GRASPING tau_cmd should project more strongly onto J^T·f_desired than APPROACH."""
    mpc_app  = _make_mpc()
    mpc_gras = _make_mpc()

    # Use x_ee == x_target to eliminate position error; only force term matters
    x_at_target = _X_TARGET.copy()

    res_app = mpc_app.solve(
        grasp_state=GraspState.APPROACH,
        x_ee=x_at_target, qdot=_QDOT, jacobian=_J,
        x_target=_X_TARGET, f_desired=_F_DESIRED,
    )
    res_gras = mpc_gras.solve(
        grasp_state=GraspState.GRASPING,
        x_ee=x_at_target, qdot=_QDOT, jacobian=_J,
        x_target=_X_TARGET, f_desired=_F_DESIRED,
    )

    tau_fref     = _J.T @ _F_DESIRED
    tau_fref_dir = tau_fref / (np.linalg.norm(tau_fref) + 1e-12)

    proj_app  = float(tau_fref_dir @ res_app.tau_cmd)
    proj_gras = float(tau_fref_dir @ res_gras.tau_cmd)

    msgs = [
        f"  tau_force_ref direction = {tau_fref_dir.round(4)}",
        f"  APPROACH  projection = {proj_app:.5f}  (w_force=0, expect ~0)",
        f"  GRASPING  projection = {proj_gras:.5f}  (w_force=5, expect > 0)",
    ]

    errors: list[str] = []
    # GRASPING must have strictly larger force projection
    if proj_gras <= proj_app + 1e-6:
        errors.append(
            f"GRASPING projection ({proj_gras:.5f}) not significantly larger than "
            f"APPROACH ({proj_app:.5f})"
        )
    # GRASPING projection should be positive
    if proj_gras <= 0.0:
        errors.append(f"GRASPING projection non-positive: {proj_gras:.5f}")
    # APPROACH projection should be near zero (w_force=0)
    if abs(proj_app) > 0.1 * abs(proj_gras) + 0.01:
        errors.append(
            f"APPROACH projection ({proj_app:.5f}) too large relative to GRASPING — "
            "w_force=0 should not produce force-tracking signal"
        )

    detail = "\n".join(msgs)
    if errors:
        return False, detail + "\nFailed:\n  " + "\n  ".join(errors)
    return True, detail


# ── runner ─────────────────────────────────────────────────────────────────────

_SCS = [
    ("SC1", "Weight table correctness",            sc1_weight_table),
    ("SC2", "QP feasibility (all GraspStates)",    sc2_qp_feasibility),
    ("SC3", "Weight switching changes cost",        sc3_weight_switching),
    ("SC4", "Position tracking direction",          sc4_position_tracking),
    ("SC5", "Force regulation in GRASPING",         sc5_force_regulation),
]

MAX_ATTEMPTS = 1   # pure algorithmic test — no stochastic elements


def run_all_scs() -> bool:
    print("\n=== Phase 5: Adaptive Cost MPC — SC Evaluation ===\n")
    results: dict[str, bool] = {}

    for tag, name, fn in _SCS:
        try:
            passed, detail = fn()
        except Exception as exc:
            passed = False
            detail = f"Exception: {type(exc).__name__}: {exc}"
        status = "PASS" if passed else "FAIL"
        results[tag] = passed
        print(f"[{status}] {tag}: {name}")
        for line in detail.splitlines():
            print(f"       {line}")
        print()

    total   = len(results)
    n_pass  = sum(results.values())
    print(f"Result: {n_pass}/{total} SCs passed")
    return n_pass == total


if __name__ == "__main__":
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n{'='*60}")
        print(f"Attempt {attempt}/{MAX_ATTEMPTS}")
        all_passed = run_all_scs()
        if all_passed:
            print("\nAll SCs passed.")
            sys.exit(0)
        if attempt < MAX_ATTEMPTS:
            print("\nSome SCs failed — retrying...")
    print("\nFailed to pass all SCs within attempt limit.")
    sys.exit(1)
