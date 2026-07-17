"""OSQP-based null-space QP controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

try:
    from .null_config import NullSpaceQPConfig
except ImportError:
    from null_config import NullSpaceQPConfig


@dataclass(frozen=True)
class NullSpaceQPResult:
    z: np.ndarray
    qdot_task: np.ndarray
    qdot: np.ndarray
    dq: np.ndarray
    task_scaling: float
    status: str


class NullSpaceQPController:
    """Realtime OSQP controller for qdot = qdot_task + N z."""

    def __init__(self, n_dof: int, config: NullSpaceQPConfig) -> None:
        try:
            import osqp
        except ImportError as exc:
            raise ImportError("Install OSQP first: pip install osqp") from exc

        self.n_dof = int(n_dof)
        self.config = config
        self._last_z = np.zeros(self.n_dof)

        p_rows, p_cols = np.triu_indices(self.n_dof)
        self._p_template = sparse.csc_matrix(
            (np.ones_like(p_rows, dtype=float), (p_rows, p_cols)),
            shape=(self.n_dof, self.n_dof),
        )
        self._p_rows, self._p_cols = self._data_coordinates(self._p_template)

        self._g_template = sparse.csc_matrix(np.ones((2 * self.n_dof, self.n_dof)))
        self._g_rows, self._g_cols = self._data_coordinates(self._g_template)

        initial_p = np.eye(self.n_dof)
        initial_q = np.zeros(self.n_dof)
        initial_g = np.vstack([np.eye(self.n_dof), np.eye(self.n_dof)])
        initial_l = -np.ones(2 * self.n_dof)
        initial_u = np.ones(2 * self.n_dof)

        self._solver = osqp.OSQP()
        self._solver.setup(
            P=self._matrix_with_values(
                self._p_template,
                initial_p[self._p_rows, self._p_cols],
            ),
            q=initial_q,
            A=self._matrix_with_values(
                self._g_template,
                initial_g[self._g_rows, self._g_cols],
            ),
            l=initial_l,
            u=initial_u,
            warm_starting=True,
            adaptive_rho=True,
            polish=False,
            eps_abs=config.eps_abs,
            eps_rel=config.eps_rel,
            max_iter=config.max_iter,
            verbose=config.verbose,
        )

    def solve(
        self,
        *,
        q_current: np.ndarray,
        qdot_task: np.ndarray,
        nullspace_projector: np.ndarray,
        q_nominal: np.ndarray,
        dt: float,
        q_min: np.ndarray,
        q_max: np.ndarray,
        qdot_min: np.ndarray,
        qdot_max: np.ndarray,
    ) -> NullSpaceQPResult:
        q_current = np.asarray(q_current, dtype=float).reshape(self.n_dof)
        qdot_task_raw = np.asarray(qdot_task, dtype=float).reshape(self.n_dof)
        nullspace_projector = np.asarray(nullspace_projector, dtype=float).reshape(
            self.n_dof, self.n_dof
        )
        q_nominal = np.asarray(q_nominal, dtype=float).reshape(self.n_dof)

        task_scaling = self._task_scaling_factor(
            q_current=q_current,
            qdot_task=qdot_task_raw,
            dt=dt,
            q_min=q_min,
            q_max=q_max,
            qdot_min=qdot_min,
            qdot_max=qdot_max,
        )
        qdot_task = task_scaling * qdot_task_raw

        p_matrix, q_vector = self._objective(
            q_current=q_current,
            qdot_task=qdot_task,
            nullspace_projector=nullspace_projector,
            q_nominal=q_nominal,
            dt=dt,
        )
        g_matrix, lower, upper = self._constraints(
            q_current=q_current,
            qdot_task=qdot_task,
            nullspace_projector=nullspace_projector,
            dt=dt,
            q_min=q_min,
            q_max=q_max,
            qdot_min=qdot_min,
            qdot_max=qdot_max,
        )

        self._solver.update(
            Px=p_matrix[self._p_rows, self._p_cols],
            q=q_vector,
            Ax=g_matrix[self._g_rows, self._g_cols],
            l=lower,
            u=upper,
        )
        self._solver.warm_start(x=self._last_z)
        result = self._solver.solve()
        status = str(result.info.status)

        if status.lower() in {"solved", "solved inaccurate"}:
            z = np.asarray(result.x, dtype=float).reshape(self.n_dof)
        elif result.x is not None and self._is_feasible_z(result.x, g_matrix, lower, upper):
            z = np.asarray(result.x, dtype=float).reshape(self.n_dof)
            status = f"{status} (accepted feasible iterate)"
        else:
            z = np.zeros(self.n_dof)
            status = f"{status} (fallback z=0)"

        self._last_z = z
        qdot = qdot_task + nullspace_projector @ z
        return NullSpaceQPResult(
            z=z,
            qdot_task=qdot_task,
            qdot=qdot,
            dq=dt * qdot,
            task_scaling=float(task_scaling),
            status=status,
        )

    def _objective(
        self,
        *,
        q_current: np.ndarray,
        qdot_task: np.ndarray,
        nullspace_projector: np.ndarray,
        q_nominal: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Minimize posture error after one step:
        # 0.5*w_p*||q + dt*(qdot_task + N*z) - q_nom||^2 + 0.5*w_d*||z||^2
        next_error_without_null = q_current + dt * qdot_task - q_nominal
        p_matrix = (
            self.config.damping_weight * np.eye(self.n_dof)
            + self.config.posture_weight
            * (dt**2)
            * (nullspace_projector.T @ nullspace_projector)
        )
        q_vector = (
            self.config.posture_weight
            * dt
            * (nullspace_projector.T @ next_error_without_null)
        )
        return 0.5 * (p_matrix + p_matrix.T), q_vector

    @staticmethod
    def _constraints(
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
        # l <= G*z <= u, where qdot = qdot_task + N*z.
        g_matrix = np.vstack([nullspace_projector, nullspace_projector])
        position_lower = (q_min - q_current) / dt - qdot_task
        position_upper = (q_max - q_current) / dt - qdot_task
        velocity_lower = qdot_min - qdot_task
        velocity_upper = qdot_max - qdot_task
        lower = np.concatenate([position_lower, velocity_lower])
        upper = np.concatenate([position_upper, velocity_upper])
        return g_matrix, lower, upper

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
        for i, velocity in enumerate(qdot_task):
            if velocity > 0.0:
                alpha = min(alpha, (q_max[i] - q_current[i]) / (dt * velocity))
                alpha = min(alpha, qdot_max[i] / velocity)
            elif velocity < 0.0:
                alpha = min(alpha, (q_min[i] - q_current[i]) / (dt * velocity))
                alpha = min(alpha, qdot_min[i] / velocity)
        return float(np.clip(alpha, 0.0, 1.0))

    @staticmethod
    def _is_feasible_z(
        z: np.ndarray,
        g_matrix: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        tolerance: float = 5e-3,
    ) -> bool:
        z = np.asarray(z, dtype=float).reshape(g_matrix.shape[1])
        if not np.all(np.isfinite(z)):
            return False
        value = g_matrix @ z
        return bool(np.all(value >= lower - tolerance) and np.all(value <= upper + tolerance))

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
