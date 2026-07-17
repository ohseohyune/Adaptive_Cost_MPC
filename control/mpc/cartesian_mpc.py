"""Cartesian velocity MPC for EE position planning.

State:    x_EE ∈ R³             end-effector position [m]
Control:  v ∈ R³  per step      Cartesian velocity command [m/s]
Dynamics: x_{k+1} = x_k + dt·v_k   (single-integrator)

Decision variable: V = [v_0, ..., v_{N-1}] ∈ R^{3N}

All prediction matrices (Γ, Γ^T Γ, Γ^T Φ) are precomputed once in __init__
because they depend only on N and dt — not on the current state or Jacobian.

Cost per step k (weights adapted from GraspState via COST_WEIGHTS):
    w_total · ‖x_EE_k − x_ref‖²  +  w_reg · ‖v_k‖²

x_ref blends position target, grasp target, and force virtual target:
    x_ref = (w_pos·x_target + w_force·x_force_ref + w_grasp·x_grasp) / w_total
    x_force_ref = x_object + f_desired / K_pos   (position that generates f_desired
                                                   through the impedance controller)

Output: v_cmd = V*[0:3]  →  x_target_next = x_EE + dt·v_cmd
        (passed as position reference to the Cartesian impedance controller)

This is the "Method C" architecture:
    [High level]  CartesianMPC   → x_target_next  (where to move)
    [Low level]   ImpedanceCtrl  → joint torques   (how to get there)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import sparse

from control.clik.grasp_state_machine import GraspState
from control.mpc.adaptive_cost_mpc import COST_WEIGHTS

if TYPE_CHECKING:
    from control.mpc.neural_cost_map import NeuralCostMap


@dataclass
class CartesianMPCConfig:
    """Configuration for CartesianMPC."""
    horizon:  int   = 10     # prediction horizon N
    dt:       float = 0.002  # timestep [s] — should match simulation dt
    w_reg:    float = 0.005  # velocity regularisation weight
    v_max:    float = 0.15   # per-axis Cartesian velocity limit [m/s]
    eps_abs:  float = 1e-4
    eps_rel:  float = 1e-4
    max_iter: int   = 200
    verbose:  bool  = False


@dataclass
class CartesianMPCResult:
    """Output of one CartesianMPC solve."""
    v_cmd:          np.ndarray    # (3,)      first velocity command [m/s]
    x_target_next:  np.ndarray    # (3,)      = x_EE + dt·v_cmd  [m]
    trajectory:     np.ndarray    # (N+1, 3)  predicted EE positions
    cost:           float
    status:         str
    weights:        dict[str, float]


class CartesianMPC:
    """Single-integrator Cartesian velocity MPC for EE trajectory planning.

    All prediction matrices are precomputed in __init__ so each call only
    requires a matrix-vector multiply plus one OSQP warm-start update.

    Integration role:
        solve() returns x_target_next, which is the EE position reference
        fed to CartesianImpedanceController.  The impedance layer converts
        this Cartesian target to joint torques, handling contact compliance.

    Args:
        config: CartesianMPCConfig; defaults used when None.
    """

    def __init__(
        self,
        config:   Optional[CartesianMPCConfig] = None,
        cost_map: Optional["NeuralCostMap"]    = None,
    ) -> None:
        self.config    = config or CartesianMPCConfig()
        self._cost_map = cost_map   # None → use COST_WEIGHTS; else NeuralCostMap
        N  = self.config.horizon
        dt = self.config.dt

        # ── precompute prediction matrices ────────────────────────────────────
        # Γ = dt · (L ⊗ I₃)  where L is the N×N lower-triangular ones matrix
        # Φ = (1_N ⊗ I₃)      stacks N copies of I₃
        L     = np.tril(np.ones((N, N)))
        Gamma = dt * np.kron(L, np.eye(3))            # (3N, 3N)
        Phi   = np.kron(np.ones((N, 1)), np.eye(3))   # (3N, 3)

        self._GtG   = Gamma.T @ Gamma                  # (3N, 3N) — precomputed
        self._GtPhi = Gamma.T @ Phi                    # (3N, 3)  — precomputed
        self._n_var = 3 * N

        # ── OSQP warm-start state ─────────────────────────────────────────────
        self._solver     = None
        self._p_rows     = None
        self._p_cols     = None
        self._p_template = None
        self._last_V     = np.zeros(self._n_var)

    # ── public API ─────────────────────────────────────────────────────────────

    def solve(
        self,
        *,
        grasp_state:       GraspState,
        x_ee:              np.ndarray,                    # (3,) current EE position [m]
        x_target:          np.ndarray,                    # (3,) kinematic target [m]
        f_desired:         Optional[np.ndarray] = None,   # (3,) desired contact force [N]
        x_object:          Optional[np.ndarray] = None,   # (3,) object position [m]
        xdot_object:       Optional[np.ndarray] = None,   # (3,) object velocity [m/s]
        x_grasp:           Optional[np.ndarray] = None,   # (3,) ideal grasp EE pos [m]
        K_pos:             float = 500.0,                 # impedance stiffness [N/m]
        f_contact_total:   float = 0.0,                   # total contact force [N] (NeuralCostMap)
        ttc:               float = float("inf"),           # time-to-contact [s]  (NeuralCostMap)
    ) -> CartesianMPCResult:
        """Solve one MPC step and return the optimal Cartesian velocity command.

        The returned x_target_next should be passed as the position component
        of T_des to CartesianImpedanceController.apply().
        """
        x_ee     = np.asarray(x_ee,     dtype=float).reshape(3)
        x_target = np.asarray(x_target, dtype=float).reshape(3)
        x_grasp  = x_target if x_grasp is None else np.asarray(x_grasp, float).reshape(3)

        if self._cost_map is not None:
            from control.mpc.neural_cost_map import build_context as _build_ctx
            ctx    = _build_ctx(
                x_ee=x_ee,
                x_object=x_object if x_object is not None else np.zeros(3),
                xdot_object=xdot_object if xdot_object is not None else np.zeros(3),
                f_contact=f_contact_total,
                ttc=ttc,
                grasp_state=grasp_state,
            )
            w_pred  = self._cost_map.predict(ctx)
            w_pos   = float(w_pred[0])
            w_force = float(w_pred[1])
            w_grasp = float(w_pred[2])
        else:
            _cw     = COST_WEIGHTS[grasp_state]
            w_pos   = float(_cw["pos"])
            w_force = float(_cw["force"])
            w_grasp = float(_cw["grasp"])
        weights_dict = {"pos": w_pos, "force": w_force, "grasp": w_grasp}
        w_reg   = float(self.config.w_reg)
        w_total = w_pos + w_force + w_grasp

        # ── blended reference position ────────────────────────────────────────
        if w_total > 1e-12:
            num = w_pos * x_target + w_grasp * x_grasp
            if f_desired is not None and x_object is not None and w_force > 1e-12:
                # Impedance: f = K_pos * (x_target - x_obj)
                # Virtual target to produce f_desired: x_obj + f_desired / K_pos
                x_force_ref = (np.asarray(x_object, float).reshape(3)
                               + np.asarray(f_desired, float).reshape(3)
                               / max(float(K_pos), 1.0))
                num += w_force * x_force_ref
            else:
                num += w_force * x_target
            x_ref = num / w_total
        else:
            x_ref = x_target.copy()

        # ── QP matrices ───────────────────────────────────────────────────────
        # Since Q_bar = w_total · I, H simplifies to:
        #   H = w_total · Γ^T Γ + w_reg · I   (precomputed Γ^T Γ)
        # and g = w_total · Γ^T Φ · (x_ee - x_ref)  (single error vector)
        e0 = x_ee - x_ref                                        # (3,)
        H  = w_total * self._GtG + w_reg * np.eye(self._n_var)
        H  = 0.5 * (H + H.T)                                     # symmetrise
        g  = w_total * (self._GtPhi @ e0)                        # (3N,)

        # ── velocity bounds ───────────────────────────────────────────────────
        v = float(self.config.v_max)
        l = -v * np.ones(self._n_var)
        u =  v * np.ones(self._n_var)

        # ── solve ─────────────────────────────────────────────────────────────
        V_opt, status = self._solve_qp(H, g, l, u)

        v_cmd         = V_opt[:3].copy()
        x_target_next = x_ee + self.config.dt * v_cmd
        traj          = self._rollout(x_ee, V_opt.reshape(-1, 3))
        cost          = float(0.5 * V_opt @ H @ V_opt + g @ V_opt)

        return CartesianMPCResult(
            v_cmd=v_cmd,
            x_target_next=x_target_next,
            trajectory=traj,
            cost=cost,
            status=status,
            weights=weights_dict,
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    def _rollout(self, x0: np.ndarray, V_seq: np.ndarray) -> np.ndarray:
        N, dt = V_seq.shape[0], self.config.dt
        traj  = np.zeros((N + 1, 3))
        traj[0] = x0
        for k in range(N):
            traj[k + 1] = traj[k] + dt * V_seq[k]
        return traj

    # ── OSQP interface (same warm-start pattern as NullSpaceQPController) ──────

    def _solve_qp(
        self,
        H: np.ndarray,
        g: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        import osqp
        self._ensure_solver(H, g, l, u, osqp)
        self._update_solver(H, g, l, u)
        res    = self._solver.solve()
        status = str(res.info.status)

        if res.x is not None and np.all(np.isfinite(res.x)):
            V_opt = np.asarray(res.x, dtype=float)
            self._last_V = V_opt.copy()
        else:
            V_opt  = np.zeros(self._n_var)
            status = f"{status} (fallback V=0)"

        return V_opt, status

    def _ensure_solver(self, H, g, l, u, osqp_module) -> None:
        if self._solver is not None:
            return
        n = self._n_var
        # Dense upper-triangular sparsity template (fixed for all future calls)
        p_rows, p_cols = np.triu_indices(n)
        self._p_template = sparse.csc_matrix(
            (np.ones(len(p_rows), dtype=float), (p_rows, p_cols)),
            shape=(n, n),
        )
        self._p_rows = self._p_template.indices.copy()
        self._p_cols = np.repeat(np.arange(n), np.diff(self._p_template.indptr))

        P_init = self._make_P(H)
        A_con  = sparse.eye(n, format="csc")   # box constraint A = I

        self._solver = osqp_module.OSQP()
        self._solver.setup(
            P=P_init, q=g, A=A_con, l=l, u=u,
            warm_starting=True, verbose=self.config.verbose,
            eps_abs=self.config.eps_abs, eps_rel=self.config.eps_rel,
            max_iter=self.config.max_iter, polish=True, adaptive_rho=True,
        )
        self._solver.warm_start(x=self._last_V)

    def _update_solver(self, H, g, l, u) -> None:
        self._solver.update(
            Px=H[self._p_rows, self._p_cols],
            q=g, l=l, u=u,
        )
        self._solver.warm_start(x=self._last_V)

    def _make_P(self, H: np.ndarray) -> sparse.csc_matrix:
        m = self._p_template.copy()
        m.data = H[self._p_rows, self._p_cols].copy()
        return m
