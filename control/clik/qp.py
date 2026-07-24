"""Quadratic-programming joint-velocity controllers."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class QPJointVelocityConfig:
    """
    We solve for joint velocity x = qdot with the standard QP form:

        minimize    0.5 * x.T P x + q.T x
        subject to  G x <= h

    The task term is soft:

        0.5 * w_task * ||dt * J x - e||^2

    Position and velocity limits are hard:

        q_min <= q + dt * x <= q_max
        qdot_min <= x <= qdot_max
    """

    task_weight: float = 1.0
    manipulability_weight: float = 0.02
    regularization: float = 1e-6
    solver: str = "osqp"


@dataclass(frozen=True)
class QPJointVelocityResult:
    """QP solution and matrices used to compute it."""

    qdot: np.ndarray
    joint_delta: np.ndarray
    P: np.ndarray
    q: np.ndarray
    G: np.ndarray
    h: np.ndarray
    task_residual: np.ndarray
    objective_value: float
    solver: str


class QPJointVelocityController:
    """Build and solve a soft-task, hard-limit joint-velocity QP."""

    def __init__(self, config: QPJointVelocityConfig | None = None) -> None:
        self.config = config or QPJointVelocityConfig()

    def solve(
        self,
        *,
        jacobian: np.ndarray,
        task_error: np.ndarray,
        q_current: np.ndarray,
        dt: float,
        q_min: np.ndarray,
        q_max: np.ndarray,
        qdot_min: np.ndarray,
        qdot_max: np.ndarray,
        manipulability_gradient: np.ndarray | None = None,
        task_weight: float | None = None,
        manipulability_weight: float | None = None,
    ) -> QPJointVelocityResult:
        """Return qdot and dq=qdot*dt for the current robot state."""

        jacobian = np.asarray(jacobian, dtype=float)
        task_error = np.asarray(task_error, dtype=float).reshape(jacobian.shape[0])
        q_current = np.asarray(q_current, dtype=float).reshape(jacobian.shape[1])
        q_min = np.asarray(q_min, dtype=float).reshape(q_current.shape)
        q_max = np.asarray(q_max, dtype=float).reshape(q_current.shape)
        qdot_min = np.asarray(qdot_min, dtype=float).reshape(q_current.shape)
        qdot_max = np.asarray(qdot_max, dtype=float).reshape(q_current.shape)
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive for QP joint-velocity control.")

        n_dof = q_current.size
        task_weight = self.config.task_weight if task_weight is None else task_weight
        manipulability_weight = (
            self.config.manipulability_weight
            if manipulability_weight is None
            else manipulability_weight
        )

        P = float(self.config.regularization) * np.eye(n_dof)
        q = np.zeros(n_dof)

        # Soft task tracking: 0.5*w*||dt*J*qdot - e||^2.
        weighted_task = float(task_weight)
        P += weighted_task * (dt**2) * (jacobian.T @ jacobian)
        q += -weighted_task * dt * (jacobian.T @ task_error)

        # Manipulability maximization by first-order ascent:
        # maximize grad(w).T * dt*qdot  <=>  minimize -grad(w).T * dt*qdot.
        if manipulability_gradient is not None and manipulability_weight > 0.0:
            manipulability_gradient = np.asarray(
                manipulability_gradient, dtype=float
            ).reshape(q_current.shape)
            q += -float(manipulability_weight) * dt * manipulability_gradient

        G, h = self._build_inequality_constraints(
            q_current=q_current,
            dt=dt,
            q_min=q_min,
            q_max=q_max,
            qdot_min=qdot_min,
            qdot_max=qdot_max,
        )

        qdot = self._solve_qp(P, q, G, h)
        joint_delta = qdot * dt
        task_residual = dt * jacobian @ qdot - task_error
        objective_value = float(0.5 * qdot.T @ P @ qdot + q.T @ qdot)

        return QPJointVelocityResult(
            qdot=qdot,
            joint_delta=joint_delta,
            P=P,
            q=q,
            G=G,
            h=h,
            task_residual=task_residual,
            objective_value=objective_value,
            solver=self.config.solver,
        )

    def _solve_qp(
        self,
        P: np.ndarray,
        q: np.ndarray,
        G: np.ndarray,
        h: np.ndarray,
    ) -> np.ndarray:
        """Call qpsolvers lazily so importing this module does not require OSQP."""

        try:
            from qpsolvers import solve_qp
        except ImportError as exc:
            raise ImportError(
                "QP mode requires qpsolvers and OSQP. Install them with "
                "`pip install qpsolvers osqp`."
            ) from exc

        P = 0.5 * (P + P.T)
        solution = solve_qp(P, q, G, h, solver=self.config.solver)
        if solution is None:
            raise RuntimeError("QP solver returned no solution; constraints are infeasible.")
        return np.asarray(solution, dtype=float).reshape(q.shape)

    @staticmethod
    def _build_inequality_constraints(
        *,
        q_current: np.ndarray,
        dt: float,
        q_min: np.ndarray,
        q_max: np.ndarray,
        qdot_min: np.ndarray,
        qdot_max: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build Gx <= h for hard position and velocity bounds."""

        n_dof = q_current.size
        eye = np.eye(n_dof)

        position_upper = (q_max - q_current) / dt
        position_lower = (q_min - q_current) / dt

        G = np.vstack(
            [
                eye,
                -eye,
                eye,
                -eye,
            ]
        )
        h = np.concatenate(
            [
                position_upper,
                -position_lower,
                qdot_max,
                -qdot_min,
            ]
        )
        return G, h


@dataclass(frozen=True)
class NullSpaceQPConfig:
    """
    OSQP controller for qdot = qdot_task + N z.

    The OSQP workspace is built with fixed dense sparsity for P and G, then each
    control step updates only matrix/vector values.
    """

    posture_weight: float = 6.0
    manipulability_weight: float = 0.02
    damping_weight: float = 1e-4
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    max_iter: int = 200
    verbose: bool = False


@dataclass(frozen=True)
class NullSpaceQPResult:
    """Null-space QP solution and diagnostics."""

    z: np.ndarray
    qdot_task: np.ndarray
    qdot_task_unscaled: np.ndarray
    qdot: np.ndarray
    joint_delta: np.ndarray
    nullspace_projector: np.ndarray
    task_scaling: float
    P: np.ndarray
    q: np.ndarray
    G: np.ndarray
    l: np.ndarray
    u: np.ndarray
    objective_value: float
    status: str


class NullSpaceQPController:
    """
    Realtime null-space QP controller.

    The optimization variable is z, and the final command is:

        qdot = qdot_task + N z

    Position and velocity limits are enforced as hard OSQP bounds:

        l <= G z <= u
    """

    def __init__(self, n_dof: int, config: NullSpaceQPConfig | None = None) -> None:
        self.n_dof = int(n_dof)
        if self.n_dof <= 0:
            raise ValueError("n_dof must be positive.")
        self.config = config or NullSpaceQPConfig()
        self._solver = None
        self._p_template = None
        self._g_template = None
        self._p_rows = None
        self._p_cols = None
        self._g_rows = None
        self._g_cols = None
        self._last_z = np.zeros(self.n_dof)

    def solve(
        self,
        *,
        q_current: np.ndarray,
        qdot_task: np.ndarray,
        nullspace_projector: np.ndarray,
        dt: float,
        q_min: np.ndarray,
        q_max: np.ndarray,
        qdot_min: np.ndarray,
        qdot_max: np.ndarray,
        q_nominal: np.ndarray | None = None,
        posture_weights: np.ndarray | None = None,
        joint_limit_velocity: np.ndarray | None = None,
        manipulability_gradient: np.ndarray | None = None,
        posture_weight: float | None = None,
        manipulability_weight: float | None = None,
        damping_weight: float | None = None,
    ) -> NullSpaceQPResult:
        """Solve one null-space QP update using the existing OSQP workspace."""

        q_current = np.asarray(q_current, dtype=float).reshape(self.n_dof)
        qdot_task_unscaled = np.asarray(qdot_task, dtype=float).reshape(self.n_dof)
        nullspace_projector = np.asarray(
            nullspace_projector, dtype=float
        ).reshape(self.n_dof, self.n_dof)
        q_min = np.asarray(q_min, dtype=float).reshape(self.n_dof)
        q_max = np.asarray(q_max, dtype=float).reshape(self.n_dof)
        qdot_min = np.asarray(qdot_min, dtype=float).reshape(self.n_dof)
        qdot_max = np.asarray(qdot_max, dtype=float).reshape(self.n_dof)
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive for null-space QP control.")

        posture_weight = (
            self.config.posture_weight
            if posture_weight is None
            else float(posture_weight)
        )
        manipulability_weight = (
            self.config.manipulability_weight
            if manipulability_weight is None
            else float(manipulability_weight)
        )
        damping_weight = (
            self.config.damping_weight
            if damping_weight is None
            else float(damping_weight)
        )

        task_scaling = self._task_scaling_factor(
            q_current=q_current,
            qdot_task=qdot_task_unscaled,
            dt=dt,
            q_min=q_min,
            q_max=q_max,
            qdot_min=qdot_min,
            qdot_max=qdot_max,
        )
        qdot_task = task_scaling * qdot_task_unscaled

        P, q_vector = self._build_objective(
            q_current=q_current,
            qdot_task=qdot_task,
            nullspace_projector=nullspace_projector,
            dt=dt,
            q_nominal=q_nominal,
            posture_weights=posture_weights,
            joint_limit_velocity=joint_limit_velocity,
            manipulability_gradient=manipulability_gradient,
            posture_weight=posture_weight,
            manipulability_weight=manipulability_weight,
            damping_weight=damping_weight,
        )
        G, lower, upper = self._build_constraints(
            q_current=q_current,
            qdot_task=qdot_task,
            nullspace_projector=nullspace_projector,
            dt=dt,
            q_min=q_min,
            q_max=q_max,
            qdot_min=qdot_min,
            qdot_max=qdot_max,
        )

        self._ensure_solver(P, q_vector, G, lower, upper)
        self._update_solver(P, q_vector, G, lower, upper)
        result = self._solver.solve()
        # Accept an inaccurate-but-feasible iterate rather than silently
        # falling back to z=0, which would otherwise mask marginal OSQP runs.
        status = str(result.info.status)
        if status.lower() in {"solved", "solved inaccurate"}:
            z = np.asarray(result.x, dtype=float).reshape(self.n_dof)
        elif result.x is not None and self._is_feasible_z(result.x, G, lower, upper):
            z = np.asarray(result.x, dtype=float).reshape(self.n_dof)
            status = f"{status} (accepted feasible iterate)"
        else:
            z = np.zeros(self.n_dof)
            status = f"{status} (fallback z=0)"
        self._last_z = z
        qdot = qdot_task + nullspace_projector @ z
        joint_delta = qdot * dt
        objective_value = float(0.5 * z.T @ P @ z + q_vector.T @ z)

        return NullSpaceQPResult(
            z=z,
            qdot_task=qdot_task,
            qdot_task_unscaled=qdot_task_unscaled,
            qdot=qdot,
            joint_delta=joint_delta,
            nullspace_projector=nullspace_projector,
            task_scaling=float(task_scaling),
            P=P,
            q=q_vector,
            G=G,
            l=lower,
            u=upper,
            objective_value=objective_value,
            status=status,
        )

    def _ensure_solver(
        self,
        P: np.ndarray,
        q_vector: np.ndarray,
        G: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        if self._solver is not None:
            return
        try:
            import osqp
        except ImportError as exc:
            raise ImportError(
                "Null-space QP mode requires OSQP. Install it with `pip install osqp`."
            ) from exc

        p_rows, p_cols = np.triu_indices(self.n_dof)
        self._p_template = sparse.csc_matrix(
            (np.ones_like(p_rows, dtype=float), (p_rows, p_cols)),
            shape=(self.n_dof, self.n_dof),
        )
        self._p_rows, self._p_cols = self._data_coordinates(self._p_template)

        self._g_template = sparse.csc_matrix(np.ones((2 * self.n_dof, self.n_dof)))
        self._g_rows, self._g_cols = self._data_coordinates(self._g_template)

        self._solver = osqp.OSQP()
        self._solver.setup(
            P=self._matrix_with_values(
                self._p_template,
                P[self._p_rows, self._p_cols],
            ),
            q=q_vector,
            A=self._matrix_with_values(
                self._g_template,
                G[self._g_rows, self._g_cols],
            ),
            l=lower,
            u=upper,
            warm_starting=True,
            verbose=self.config.verbose,
            eps_abs=self.config.eps_abs,
            eps_rel=self.config.eps_rel,
            max_iter=self.config.max_iter,
            polish=False,
            adaptive_rho=True,
        )
        self._solver.warm_start(x=self._last_z)

    def _update_solver(
        self,
        P: np.ndarray,
        q_vector: np.ndarray,
        G: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        self._solver.update(
            Px=P[self._p_rows, self._p_cols],
            q=q_vector,
            Ax=G[self._g_rows, self._g_cols],
            l=lower,
            u=upper,
        )
        self._solver.warm_start(x=self._last_z)

    def _build_objective(
        self,
        *,
        q_current: np.ndarray,
        qdot_task: np.ndarray,
        nullspace_projector: np.ndarray,
        dt: float,
        q_nominal: np.ndarray | None,
        posture_weights: np.ndarray | None,
        joint_limit_velocity: np.ndarray | None,
        manipulability_gradient: np.ndarray | None,
        posture_weight: float,
        manipulability_weight: float,
        damping_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_dof = self.n_dof
        P = max(float(damping_weight), 1e-12) * np.eye(n_dof)
        q_vector = np.zeros(n_dof)

        if q_nominal is not None and posture_weight > 0.0:
            q_nominal = np.asarray(q_nominal, dtype=float).reshape(n_dof)
            if posture_weights is None:
                posture_diag = np.ones(n_dof)
            else:
                posture_diag = np.asarray(posture_weights, dtype=float).reshape(n_dof)
            weighted_nullspace = posture_diag[:, None] * nullspace_projector
            next_posture_error = q_current + dt * qdot_task - q_nominal
            P += float(posture_weight) * (dt**2) * (
                nullspace_projector.T @ weighted_nullspace
            )
            q_vector += float(posture_weight) * dt * (
                nullspace_projector.T @ (posture_diag * next_posture_error)
            )

        if joint_limit_velocity is not None:
            joint_limit_velocity = np.asarray(
                joint_limit_velocity, dtype=float
            ).reshape(n_dof)
            if np.linalg.norm(joint_limit_velocity) > 0.0:
                P += nullspace_projector.T @ nullspace_projector
                q_vector += -(nullspace_projector.T @ joint_limit_velocity)

        if manipulability_gradient is not None and manipulability_weight > 0.0:
            manipulability_gradient = np.asarray(
                manipulability_gradient, dtype=float
            ).reshape(n_dof)
            q_vector += -float(manipulability_weight) * (
                nullspace_projector.T @ manipulability_gradient
            )

        return 0.5 * (P + P.T), q_vector

    @staticmethod
    def _build_constraints(
        *,
        q_current: np.ndarray,
        qdot_task: np.ndarray,
        nullspace_projector: np.ndarray,
        dt: float,
        q_min: np.ndarray,
        q_max: np.ndarray,
        qdot_min: np.ndarray,
        qdot_max: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        G = np.vstack([nullspace_projector, nullspace_projector])
        position_lower = (q_min - q_current) / dt - qdot_task
        position_upper = (q_max - q_current) / dt - qdot_task
        velocity_lower = qdot_min - qdot_task
        velocity_upper = qdot_max - qdot_task
        lower = np.concatenate([position_lower, velocity_lower])
        upper = np.concatenate([position_upper, velocity_upper])
        return G, lower, upper

    @staticmethod
    def _task_scaling_factor(
        *,
        q_current: np.ndarray,
        qdot_task: np.ndarray,
        dt: float,
        q_min: np.ndarray,
        q_max: np.ndarray,
        qdot_min: np.ndarray,
        qdot_max: np.ndarray,
    ) -> float:
        alpha = 1.0
        for index, velocity in enumerate(qdot_task):
            if velocity > 0.0:
                alpha = min(
                    alpha,
                    (q_max[index] - q_current[index]) / (dt * velocity),
                )
                alpha = min(alpha, qdot_max[index] / velocity)
            elif velocity < 0.0:
                alpha = min(
                    alpha,
                    (q_min[index] - q_current[index]) / (dt * velocity),
                )
                alpha = min(alpha, qdot_min[index] / velocity)
        return float(np.clip(alpha, 0.0, 1.0))

    @staticmethod
    def _is_feasible_z(
        z: np.ndarray,
        G: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        tolerance: float = 5e-3,
    ) -> bool:
        z = np.asarray(z, dtype=float).reshape(G.shape[1])
        if not np.all(np.isfinite(z)):
            return False
        value = G @ z
        return bool(
            np.all(value >= lower - tolerance)
            and np.all(value <= upper + tolerance)
        )

    @staticmethod
    def _data_coordinates(matrix: sparse.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
        rows = matrix.indices.copy()
        cols = np.repeat(np.arange(matrix.shape[1]), np.diff(matrix.indptr))
        return rows, cols

    @staticmethod
    def _matrix_with_values(
        template: sparse.csc_matrix,
        values: np.ndarray,
    ) -> sparse.csc_matrix:
        matrix = template.copy()
        matrix.data = np.asarray(values, dtype=float).copy()
        return matrix
