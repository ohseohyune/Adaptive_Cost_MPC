"""Adaptive Cost MPC for robot grasping.

Formulation
-----------
Tracking state:  s = [x_EE (3),  q̇ (n_dof)]   dim n_s = 3 + n_dof
Control:         u = τ (joint torques)           dim n_u = n_dof
Horizon:         N steps

The full system state passed to the controller is
    [q (n_dof), q̇ (n_dof), x_object (3), ẋ_object (3)]
The Jacobian J (3 × n_dof) is computed externally each control step.

Linearised dynamics (at current J):
    x_EE_{k+1} = x_EE_k + dt · J · q̇_k
    q̇_{k+1}   = q̇_k   + dt · τ_k

Cost (per step k, weights from active GraspState):
    w_pos   · ‖x_EE_k − x_target‖²
  + w_grasp · ‖x_EE_k − x_grasp‖²
  + w_force · ‖τ_k    − J^T f_desired‖²
  + w_reg   · ‖τ_k‖²

The condensed QP is solved with OSQP (warm-started between calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse

from control.clik.grasp_state_machine import GraspState

# ── adaptive cost weight schedule ─────────────────────────────────────────────

COST_WEIGHTS: dict[GraspState, dict[str, float]] = {
    GraspState.APPROACH:     {"pos": 10.0, "force": 0.0, "grasp": 0.0},
    GraspState.PRE_CONTACT:  {"pos":  5.0, "force": 1.0, "grasp": 2.0},
    GraspState.GRASPING:     {"pos":  1.0, "force": 5.0, "grasp": 5.0},
    GraspState.GRASPED:      {"pos":  8.0, "force": 3.0, "grasp": 3.0},
    GraspState.MANIPULATION: {"pos": 10.0, "force": 2.0, "grasp": 1.0},
}


@dataclass
class MPCConfig:
    """Configuration for AdaptiveCostMPC."""
    horizon:  int   = 10      # prediction horizon N
    dt:       float = 0.01    # timestep [s]
    w_reg:    float = 1e-3    # torque regularisation weight
    tau_max:  float = 50.0    # torque upper bound [N·m]
    tau_min:  float = -50.0   # torque lower bound [N·m]
    eps_abs:  float = 1e-4
    eps_rel:  float = 1e-4
    max_iter: int   = 500
    verbose:  bool  = False


@dataclass
class MPCResult:
    """Output of one MPC solve."""
    tau_sequence:      np.ndarray   # (N, n_dof) full optimal torque sequence
    tau_cmd:           np.ndarray   # (n_dof,)   first torque command to apply
    state_trajectory:  np.ndarray   # (N+1, n_s) predicted tracking-state trajectory
    cost:              float
    status:            str
    weights:           dict[str, float]


class AdaptiveCostMPC:
    """N-step Adaptive Cost MPC with GraspState-dependent cost weights.

    Usage::

        mpc = AdaptiveCostMPC(n_dof=7)
        result = mpc.solve(
            grasp_state=GraspState.APPROACH,
            x_ee=x_ee, qdot=qdot, jacobian=J,
            x_target=x_target,
        )
        torque = result.tau_cmd   # apply to robot

    Args:
        n_dof:  Number of robot joints (7 for FFW right arm).
        config: MPCConfig; defaults used when None.
    """

    def __init__(self, n_dof: int = 7, config: Optional[MPCConfig] = None) -> None:
        self.n_dof  = int(n_dof)
        self.config = config or MPCConfig()
        self._n_s   = 3 + self.n_dof          # tracking state dim
        self._n_u   = self.n_dof               # control dim
        self._n_var = self.config.horizon * self._n_u  # total QP variable dim
        # OSQP solver state
        self._solver    = None
        self._p_rows    = None
        self._p_cols    = None
        self._p_template = None
        self._last_U    = np.zeros(self._n_var)

    # ── public API ─────────────────────────────────────────────────────────────

    def solve(
        self,
        *,
        grasp_state:  GraspState,
        x_ee:         np.ndarray,           # (3,)     current EE position [m]
        qdot:         np.ndarray,           # (n_dof,) current joint velocity [rad/s]
        jacobian:     np.ndarray,           # (3, n_dof) translational Jacobian (world)
        x_target:     np.ndarray,           # (3,)     desired EE position [m]
        f_desired:    Optional[np.ndarray] = None,  # (3,) desired contact force [N]
        x_grasp:      Optional[np.ndarray] = None,  # (3,) ideal grasp EE position [m]
        # following are stored for context only, not used in QP
        q:            Optional[np.ndarray] = None,
        x_object:     Optional[np.ndarray] = None,
        xdot_object:  Optional[np.ndarray] = None,
    ) -> MPCResult:
        n_dof = self.n_dof
        n_s   = self._n_s
        N     = self.config.horizon
        dt    = self.config.dt

        x_ee     = np.asarray(x_ee,     dtype=float).reshape(3)
        qdot     = np.asarray(qdot,     dtype=float).reshape(n_dof)
        jacobian = np.asarray(jacobian, dtype=float).reshape(3, n_dof)
        x_target = np.asarray(x_target, dtype=float).reshape(3)
        x_grasp  = x_target if x_grasp is None else np.asarray(x_grasp, float).reshape(3)

        weights = COST_WEIGHTS[grasp_state]
        w_pos   = float(weights["pos"])
        w_force = float(weights["force"])
        w_grasp = float(weights["grasp"])
        w_reg   = float(self.config.w_reg)

        # ── linearised dynamics ─────────────────────────────────────────────
        A, B = self._build_dynamics(jacobian, dt)

        # ── condensed prediction matrices ───────────────────────────────────
        Phi, Gamma = self._build_condensed(A, B, N)

        # ── state cost Q (n_s × n_s) ────────────────────────────────────────
        # Both position and grasp terms penalise EE position error.
        # They share the same selection matrix C = [I_3, 0_{3×n_dof}].
        C   = np.hstack([np.eye(3), np.zeros((3, n_dof))])   # (3, n_s)
        CtC = C.T @ C                                          # (n_s, n_s)
        Q   = (w_pos + w_grasp) * CtC                        # per-step state weight

        Q_bar = np.kron(np.eye(N), Q)                         # (N·n_s, N·n_s)

        # Weighted reference position
        total_pos_w = w_pos + w_grasp
        if total_pos_w > 1e-12:
            x_ref = (w_pos * x_target + w_grasp * x_grasp) / total_pos_w
        else:
            x_ref = x_target.copy()
        s_ref = np.concatenate([x_ref, np.zeros(n_dof)])
        X_ref = np.tile(s_ref, N)                             # (N·n_s,)

        # ── initial error propagated through horizon ─────────────────────────
        s0 = np.concatenate([x_ee, qdot])
        E0 = Phi @ s0 - X_ref                                 # (N·n_s,)

        # ── force reference torques ──────────────────────────────────────────
        if f_desired is not None and w_force > 1e-12:
            tau_fref = jacobian.T @ np.asarray(f_desired, float).reshape(3)
        else:
            tau_fref = np.zeros(n_dof)
        U_fref = np.tile(tau_fref, N)                         # (N·n_u,)

        # ── QP matrices H (n_var × n_var), g (n_var,) ───────────────────────
        H  = Gamma.T @ Q_bar @ Gamma                           # position+grasp
        H += w_force * np.eye(self._n_var)                     # force tracking
        H += w_reg   * np.eye(self._n_var)                     # regularisation
        H  = 0.5 * (H + H.T)                                  # symmetrise

        g  = Gamma.T @ Q_bar @ E0                              # position+grasp
        g += -w_force * U_fref                                  # force tracking

        # ── bounds ───────────────────────────────────────────────────────────
        l = self.config.tau_min * np.ones(self._n_var)
        u = self.config.tau_max * np.ones(self._n_var)

        # ── solve ─────────────────────────────────────────────────────────────
        U_opt, status = self._solve_qp(H, g, l, u)

        # ── unpack ────────────────────────────────────────────────────────────
        tau_seq  = U_opt.reshape(N, self._n_u)
        tau_cmd  = tau_seq[0].copy()
        traj     = self._rollout(s0, A, B, tau_seq)
        cost     = float(0.5 * U_opt @ H @ U_opt + g @ U_opt)

        return MPCResult(
            tau_sequence=tau_seq,
            tau_cmd=tau_cmd,
            state_trajectory=traj,
            cost=cost,
            status=status,
            weights=dict(weights),
        )

    # ── dynamics helpers ───────────────────────────────────────────────────────

    def _build_dynamics(
        self, J: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        n_s = self._n_s
        n_u = self._n_u
        A = np.eye(n_s)
        A[:3, 3:] = dt * J             # x_EE_{k+1} += dt·J·qdot_k
        B = np.zeros((n_s, n_u))
        B[3:, :] = dt * np.eye(n_u)   # qdot_{k+1} += dt·τ_k
        return A, B

    def _build_condensed(
        self, A: np.ndarray, B: np.ndarray, N: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Condensed prediction: X = Phi·s0 + Gamma·U."""
        n_s = A.shape[0]
        n_u = B.shape[1]

        # Precompute A^0, A^1, ..., A^N
        A_pows = [np.eye(n_s)]
        for _ in range(N):
            A_pows.append(A_pows[-1] @ A)

        Phi   = np.zeros((N * n_s, n_s))
        Gamma = np.zeros((N * n_s, N * n_u))

        for i in range(N):
            Phi[i*n_s:(i+1)*n_s, :] = A_pows[i + 1]         # A^{i+1}
            for j in range(i + 1):
                Gamma[i*n_s:(i+1)*n_s, j*n_u:(j+1)*n_u] = A_pows[i - j] @ B

        return Phi, Gamma

    def _rollout(
        self,
        s0: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        tau_seq: np.ndarray,           # (N, n_u)
    ) -> np.ndarray:
        N   = tau_seq.shape[0]
        n_s = s0.shape[0]
        traj = np.zeros((N + 1, n_s))
        traj[0] = s0
        for k in range(N):
            traj[k + 1] = A @ traj[k] + B @ tau_seq[k]
        return traj

    # ── OSQP interface ─────────────────────────────────────────────────────────

    def _solve_qp(
        self,
        H: np.ndarray,   # (n_var, n_var) symmetric PD
        g: np.ndarray,   # (n_var,)
        l: np.ndarray,   # (n_var,) lower bounds on U
        u: np.ndarray,   # (n_var,) upper bounds on U
    ) -> tuple[np.ndarray, str]:
        import osqp  # lazy import so module can be imported without osqp installed

        n = self._n_var
        self._ensure_solver(H, g, l, u, osqp)
        self._update_solver(H, g, l, u)
        res    = self._solver.solve()
        status = str(res.info.status)

        if res.x is not None and np.all(np.isfinite(res.x)):
            U_opt = np.asarray(res.x, dtype=float)
            self._last_U = U_opt.copy()
        else:
            U_opt  = np.zeros(n)
            status = f"{status} (fallback U=0)"

        return U_opt, status

    def _ensure_solver(
        self,
        H: np.ndarray,
        g: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        osqp_module,
    ) -> None:
        if self._solver is not None:
            return
        n = self._n_var

        # Build dense upper-triangular sparsity template (fixed for all calls)
        p_rows, p_cols = np.triu_indices(n)
        self._p_template = sparse.csc_matrix(
            (np.ones(len(p_rows), dtype=float), (p_rows, p_cols)),
            shape=(n, n),
        )
        # Store CSC-ordered coordinates for value extraction
        self._p_rows = self._p_template.indices.copy()
        self._p_cols = np.repeat(
            np.arange(n), np.diff(self._p_template.indptr)
        )

        A_con = sparse.eye(n, format="csc")
        P_init = self._make_P(H)

        self._solver = osqp_module.OSQP()
        self._solver.setup(
            P=P_init,
            q=g,
            A=A_con,
            l=l,
            u=u,
            warm_starting=True,
            verbose=self.config.verbose,
            eps_abs=self.config.eps_abs,
            eps_rel=self.config.eps_rel,
            max_iter=self.config.max_iter,
            polish=True,
            adaptive_rho=True,
        )
        self._solver.warm_start(x=self._last_U)

    def _update_solver(
        self,
        H: np.ndarray,
        g: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
    ) -> None:
        self._solver.update(
            Px=H[self._p_rows, self._p_cols],
            q=g,
            l=l,
            u=u,
        )
        self._solver.warm_start(x=self._last_U)

    def _make_P(self, H: np.ndarray) -> sparse.csc_matrix:
        """Fill the stored sparsity template with new values from H."""
        m = self._p_template.copy()
        m.data = H[self._p_rows, self._p_cols].copy()
        return m

    # ── utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def get_weights(grasp_state: GraspState) -> dict[str, float]:
        """Return the cost weight dict for a given GraspState."""
        return dict(COST_WEIGHTS[grasp_state])
