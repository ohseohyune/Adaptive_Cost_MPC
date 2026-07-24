"""Phase 6: NeuralCostMap — success-criteria test suite.

SCs
---
SC1  Instantiation & forward pass
     NeuralCostMap can be created; predict() returns (3,) in [0, W_MAX].

SC2  Data generation shape & range
     generate_supervised_data() returns X ∈ R^{N×13}, Y ∈ R^{N×3};
     Y values match COST_WEIGHTS targets exactly.

SC3  Training convergence
     After 500 epochs, final MSE < 0.5  (over [0, 10] weight scale).

SC4  Weight accuracy per GraspState
     Trained network's predictions are within 1.5 of COST_WEIGHTS targets
     for all 5 states × 3 weights, evaluated at neutral context.

SC5  CartesianMPC integration
     CartesianMPC constructed with a trained NeuralCostMap; solve() returns
     a valid CartesianMPCResult (finite x_target_next, status contains 'solved').
"""

from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, ".")

from control.clik.grasp_state_machine import GraspState
from control.mpc.adaptive_cost_mpc import COST_WEIGHTS
from control.mpc.cartesian_mpc import CartesianMPC, CartesianMPCConfig
from control.mpc.neural_cost_map import (
    INPUT_DIM,
    W_MAX,
    NeuralCostMap,
    generate_supervised_data,
)

# ── shared training fixture (trained once, reused in SC3–SC5) ─────────────────

_NCM: NeuralCostMap | None  = None
_LOSSES: list[float]        = []

N_PER_STATE  = 300     # small for fast tests; enough to converge
N_EPOCHS     = 500
LR           = 2e-3


def _get_trained_ncm() -> tuple[NeuralCostMap, list[float]]:
    global _NCM, _LOSSES
    if _NCM is None:
        ncm  = NeuralCostMap()
        X, Y = generate_supervised_data(n_per_state=N_PER_STATE, seed=0)
        losses = ncm.train(X, Y, n_epochs=N_EPOCHS, lr=LR, print_every=0)
        _NCM    = ncm
        _LOSSES = losses
    return _NCM, _LOSSES


# ── SC1: instantiation & forward pass ─────────────────────────────────────────

def sc1_instantiation() -> tuple[bool, str]:
    ncm = NeuralCostMap()
    # single input
    ctx  = np.zeros(INPUT_DIM)
    out  = ncm.predict(ctx)
    msgs = []
    errs = []

    if out.shape != (3,):
        errs.append(f"predict(single) shape {out.shape} != (3,)")
    if not np.all(out >= 0.0):
        errs.append(f"predict output < 0: {out}")
    if not np.all(out <= W_MAX):
        errs.append(f"predict output > W_MAX={W_MAX}: {out}")
    msgs.append(f"  predict(zeros) → {out.round(4)}")

    # batch input
    X_batch = np.random.default_rng(1).standard_normal((8, INPUT_DIM))
    out_b   = ncm.predict(X_batch)
    if out_b.shape != (8, 3):
        errs.append(f"predict(batch) shape {out_b.shape} != (8, 3)")
    if not np.all(out_b >= 0.0) or not np.all(out_b <= W_MAX):
        errs.append("batch predict output out of [0, W_MAX]")
    msgs.append(f"  predict(batch 8) → shape {out_b.shape}, range [{out_b.min():.3f}, {out_b.max():.3f}]")

    # weights_for_state
    w = ncm.weights_for_state(GraspState.APPROACH)
    if set(w.keys()) != {"pos", "force", "grasp"}:
        errs.append(f"weights_for_state keys wrong: {w.keys()}")
    msgs.append(f"  weights_for_state(APPROACH) = {w}")

    detail = "\n".join(msgs)
    if errs:
        return False, detail + "\nFailed:\n  " + "\n  ".join(errs)
    return True, detail


# ── SC2: data generation shape & range ────────────────────────────────────────

def sc2_data_generation() -> tuple[bool, str]:
    X, Y = generate_supervised_data(n_per_state=N_PER_STATE, seed=0)
    msgs = []
    errs = []

    n_total = 5 * N_PER_STATE   # 5 GraspStates
    if X.shape != (n_total, INPUT_DIM):
        errs.append(f"X shape {X.shape} != ({n_total}, {INPUT_DIM})")
    if Y.shape != (n_total, 3):
        errs.append(f"Y shape {Y.shape} != ({n_total}, 3)")
    msgs.append(f"  X shape: {X.shape},  Y shape: {Y.shape}")

    # Verify Y targets match COST_WEIGHTS exactly
    state_list = list(GraspState)
    for i, state in enumerate(state_list):
        cw      = COST_WEIGHTS[state]
        y_chunk = Y[i * N_PER_STATE: (i + 1) * N_PER_STATE]
        expected = np.array([cw["pos"], cw["force"], cw["grasp"]])
        if not np.allclose(y_chunk, expected[np.newaxis, :]):
            errs.append(f"Y targets for {state.value} don't match COST_WEIGHTS")
    msgs.append("  Y targets verified against COST_WEIGHTS for all 5 states")

    # Context values in expected normalised range
    if not np.all(X[:, 6] >= 0.0) or not np.all(X[:, 6] <= 1.0):
        errs.append("f_contact normalised value out of [0, 1]")
    if not np.all(X[:, 7] >= 0.0) or not np.all(X[:, 7] <= 1.0):
        errs.append("ttc normalised value out of [0, 1]")
    if not np.all(X[:, 8:13] >= 0.0) or not np.all(X[:, 8:13] <= 1.0):
        errs.append("one-hot entries out of [0, 1]")
    msgs.append("  context feature ranges verified")

    detail = "\n".join(msgs)
    if errs:
        return False, detail + "\nFailed:\n  " + "\n  ".join(errs)
    return True, detail


# ── SC3: training convergence ──────────────────────────────────────────────────

def sc3_training_convergence() -> tuple[bool, str]:
    ncm, losses = _get_trained_ncm()
    first = losses[0]
    final = losses[-1]
    msgs  = [
        f"  initial loss: {first:.4f}",
        f"  final   loss: {final:.4f}  (threshold < 0.5)",
    ]
    errs: list[str] = []
    if final >= 0.5:
        errs.append(f"final MSE {final:.4f} >= 0.5 — did not converge")
    if final >= first:
        errs.append("loss did not decrease during training")
    detail = "\n".join(msgs)
    if errs:
        return False, detail + "\nFailed:\n  " + "\n  ".join(errs)
    return True, detail


# ── SC4: per-state weight accuracy ────────────────────────────────────────────

def sc4_weight_accuracy() -> tuple[bool, str]:
    ncm, _ = _get_trained_ncm()
    errs: list[str] = []
    msgs: list[str] = []
    tol = 1.5   # max allowed error per weight after supervised training

    for state in GraspState:
        target = COST_WEIGHTS[state]
        pred   = ncm.weights_for_state(state)
        line   = (
            f"  {state.value:12s}  "
            f"pos: tgt={target['pos']:.1f} pred={pred['pos']:.2f} | "
            f"force: tgt={target['force']:.1f} pred={pred['force']:.2f} | "
            f"grasp: tgt={target['grasp']:.1f} pred={pred['grasp']:.2f}"
        )
        msgs.append(line)
        for key in ("pos", "force", "grasp"):
            err = abs(pred[key] - target[key])
            if err > tol:
                errs.append(
                    f"{state.value}.{key}: |{pred[key]:.2f} - {target[key]:.1f}| = {err:.2f} > tol={tol}"
                )

    detail = "\n".join(msgs)
    if errs:
        return False, detail + "\nFailed (> tol):\n  " + "\n  ".join(errs)
    return True, detail + f"\n  All errors < {tol}"


# ── SC5: CartesianMPC integration ─────────────────────────────────────────────

def sc5_cartesian_mpc_integration() -> tuple[bool, str]:
    ncm, _ = _get_trained_ncm()
    mpc = CartesianMPC(
        config=CartesianMPCConfig(horizon=10, dt=0.002, w_reg=0.005, v_max=0.15),
        cost_map=ncm,
    )

    x_ee    = np.array([0.3, -0.3, 0.7])
    x_obj   = np.array([0.3, -0.3, 0.62])
    x_target = x_obj + np.array([0, 0, 0.08])

    msgs: list[str] = []
    errs: list[str] = []

    for state in GraspState:
        res = mpc.solve(
            grasp_state=state,
            x_ee=x_ee,
            x_target=x_target,
            f_desired=np.array([0.0, 0.0, -3.0]),
            x_object=x_obj,
            K_pos=300.0,
            f_contact_total=2.0,
            ttc=float("inf"),
        )
        ok_status = "solved" in res.status.lower()
        ok_finite = np.all(np.isfinite(res.x_target_next))
        ok_shape  = res.x_target_next.shape == (3,)
        ok_wkeys  = set(res.weights.keys()) == {"pos", "force", "grasp"}
        msgs.append(
            f"  {state.value:12s}  status={res.status}  "
            f"x_next={res.x_target_next.round(5)}  "
            f"weights=(pos={res.weights['pos']:.2f}, "
            f"force={res.weights['force']:.2f}, grasp={res.weights['grasp']:.2f})"
        )
        if not ok_status:
            errs.append(f"{state.value}: status '{res.status}' not solved")
        if not ok_finite:
            errs.append(f"{state.value}: x_target_next not finite: {res.x_target_next}")
        if not ok_shape:
            errs.append(f"{state.value}: x_target_next shape {res.x_target_next.shape}")
        if not ok_wkeys:
            errs.append(f"{state.value}: weights keys wrong: {res.weights.keys()}")

    detail = "\n".join(msgs)
    if errs:
        return False, detail + "\nFailed:\n  " + "\n  ".join(errs)
    return True, detail


# ── runner ─────────────────────────────────────────────────────────────────────

_SCS = [
    ("SC1", "Instantiation & forward pass",          sc1_instantiation),
    ("SC2", "Data generation shape & range",         sc2_data_generation),
    ("SC3", "Training convergence (MSE < 0.5)",      sc3_training_convergence),
    ("SC4", "Weight accuracy per GraspState",        sc4_weight_accuracy),
    ("SC5", "CartesianMPC integration",              sc5_cartesian_mpc_integration),
]


def run_all_scs() -> bool:
    print("\n=== Phase 6: NeuralCostMap — SC Evaluation ===\n")
    results: dict[str, bool] = {}

    for tag, name, fn in _SCS:
        try:
            passed, detail = fn()
        except Exception as exc:
            import traceback
            passed = False
            detail = f"Exception: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        status = "PASS" if passed else "FAIL"
        results[tag] = passed
        print(f"[{status}] {tag}: {name}")
        for line in detail.splitlines():
            print(f"       {line}")
        print()

    n_pass = sum(results.values())
    total  = len(results)
    print(f"Result: {n_pass}/{total} SCs passed")
    return n_pass == total


if __name__ == "__main__":
    all_passed = run_all_scs()
    sys.exit(0 if all_passed else 1)
