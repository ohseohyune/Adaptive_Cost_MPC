"""Constrained two-contact wrench allocation with slip and spin costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from control.squeeze.config import RotatingSideSqueezeConfig
from control.squeeze.rotation import skew


@dataclass(frozen=True)
class WrenchAllocation:
    left_wrench: np.ndarray
    right_wrench: np.ndarray
    achieved_wrench: np.ndarray
    desired_wrench: np.ndarray
    status: str
    tracking_error: float
    slip_cost: float
    angular_velocity_cost: float


class BimanualWrenchAllocator:
    """Allocate desired object wrench to two bounded soft-finger contacts."""

    def __init__(self, config: RotatingSideSqueezeConfig) -> None:
        self.config = config
        self._previous_solution: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_solution = None

    def solve(
        self,
        *,
        box_rotation: np.ndarray,
        left_contact_position: np.ndarray,
        right_contact_position: np.ndarray,
        box_center: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        object_mass: float | None = None,
        box_half_size: np.ndarray | tuple[float, float, float] | None = None,
        friction_coefficient: float | None = None,
        dt: float | None = None,
        capture_phase: bool = False,
    ) -> WrenchAllocation:
        rotation = np.asarray(box_rotation, dtype=float).reshape(3, 3)
        x_axis, y_axis, z_axis = rotation[:, 0], rotation[:, 1], rotation[:, 2]
        inward = (-y_axis, y_axis)
        positions = (
            np.asarray(left_contact_position) - np.asarray(box_center),
            np.asarray(right_contact_position) - np.asarray(box_center),
        )

        grasp = np.zeros((6, 12))
        for index, arm in enumerate(positions):
            column = 6 * index
            grasp[:3, column : column + 3] = np.eye(3)
            grasp[3:, column : column + 3] = skew(arm)
            grasp[3:, column + 3 : column + 6] = np.eye(3)

        mass = max(
            1e-6,
            float(self.config.box_mass if object_mass is None else object_mass),
        )
        half_size = np.asarray(
            (
                (0.055, self.config.box_half_y, 0.055)
                if box_half_size is None
                else box_half_size
            ),
            dtype=float,
        )
        mu = max(
            1e-4,
            float(
                self.config.wrench_friction_coefficient
                if friction_coefficient is None
                else min(
                    self.config.wrench_friction_coefficient,
                    0.90 * friction_coefficient,
                )
            ),
        )
        prediction_dt = max(
            1e-5, float(self.config.qp_prediction_dt if dt is None else dt)
        )
        desired_force = -self.config.linear_wrench_damping * np.asarray(linear_velocity)
        desired_force = desired_force - mass * np.asarray(self.config.gravity)
        desired_moment = -self.config.angular_wrench_damping * np.asarray(angular_velocity)
        desired = np.concatenate([desired_force, desired_moment])
        weights = np.diag(
            [self.config.wrench_tracking_weight] * 3
            + [self.config.angular_wrench_weight] * 3
        )

        nominal = np.zeros(12)
        for index, normal in enumerate(inward):
            nominal[6 * index : 6 * index + 3] = (
                self.config.nominal_qp_normal_force * normal
            )
        slip_matrix = np.zeros((12, 12))
        tangent_projector = np.outer(x_axis, x_axis) + np.outer(z_axis, z_axis)
        for index in range(2):
            slip_matrix[6 * index : 6 * index + 3, 6 * index : 6 * index + 3] = tangent_projector
        hessian = (
            grasp.T @ weights @ grasp
            + self.config.wrench_regularization * np.eye(12)
            + self.config.slip_cost_weight * slip_matrix
        )
        gradient = (
            -grasp.T @ weights @ desired
            - self.config.wrench_regularization * nominal
        )

        # Penalize predicted object twist, so the learned angular/slip weights
        # alter the QP solution rather than only the reported diagnostics.
        force_map = grasp[:3]
        moment_map = grasp[3:]
        gravity = np.asarray(self.config.gravity, dtype=float)
        linear_map = prediction_dt / mass * force_map
        linear_bias = np.asarray(linear_velocity, dtype=float) + prediction_dt * gravity
        hx, hy, hz = half_size
        body_inertia = mass / 3.0 * np.diag(
            [hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy]
        )
        world_inertia = rotation @ body_inertia @ rotation.T
        inverse_inertia = np.linalg.inv(world_inertia + 1e-9 * np.eye(3))
        angular_map = prediction_dt * inverse_inertia @ moment_map
        angular_bias = np.asarray(angular_velocity, dtype=float)
        angular_weight = self.config.angular_velocity_cost_weight * (
            self.config.capture_angular_weight_scale if capture_phase else 1.0
        )
        hessian += (
            self.config.linear_velocity_cost_weight
            * (linear_map.T @ linear_map)
            + angular_weight * (angular_map.T @ angular_map)
        )
        gradient += (
            self.config.linear_velocity_cost_weight
            * (linear_map.T @ linear_bias)
            + angular_weight * (angular_map.T @ angular_bias)
        )
        if self._previous_solution is not None:
            hessian += self.config.wrench_rate_weight * np.eye(12)
            gradient -= self.config.wrench_rate_weight * self._previous_solution

        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        minimum_normal_force = min(
            self.config.maximum_qp_normal_force,
            1.15 * mass * float(np.linalg.norm(gravity)) / (2.0 * mu),
        )
        for index, normal in enumerate(inward):
            base = 6 * index
            normal_row = np.zeros(12)
            normal_row[base : base + 3] = normal
            rows.append(normal_row)
            lower.append(minimum_normal_force)
            upper.append(self.config.maximum_qp_normal_force)
            for tangent in (x_axis, z_axis):
                for sign in (-1.0, 1.0):
                    row = np.zeros(12)
                    row[base : base + 3] = sign * tangent - mu * normal
                    rows.append(row)
                    lower.append(-np.inf)
                    upper.append(0.0)
            for axis_index in range(3):
                row = np.zeros(12)
                row[base + 3 + axis_index] = 1.0
                rows.append(row)
                lower.append(-self.config.maximum_contact_moment)
                upper.append(self.config.maximum_contact_moment)
            # Soft-finger torsional moment capacity grows with normal force.
            for sign in (-1.0, 1.0):
                row = np.zeros(12)
                row[base : base + 3] = (
                    -self.config.torsional_friction_coefficient * normal
                )
                row[base + 3 : base + 6] = sign * normal
                rows.append(row)
                lower.append(-np.inf)
                upper.append(0.0)

        try:
            import osqp

            solver = osqp.OSQP()
            solver.setup(
                P=sparse.csc_matrix(0.5 * (hessian + hessian.T) * 2.0),
                q=gradient * 2.0,
                A=sparse.csc_matrix(np.vstack(rows)),
                l=np.asarray(lower),
                u=np.asarray(upper),
                verbose=False,
                polishing=False,
                eps_abs=1e-5,
                eps_rel=1e-5,
                max_iter=200,
            )
            result = solver.solve()
            solution = np.asarray(result.x) if result.x is not None else nominal
            status = str(result.info.status).lower()
        except Exception:  # pragma: no cover - nominal fallback for optional solver failure
            solution = nominal
            status = "fallback"

        if "solved" not in status or not np.all(np.isfinite(solution)):
            solution = nominal
        self._previous_solution = solution.copy()

        achieved = grasp @ solution
        slip_cost = 0.0
        for index, normal in enumerate(inward):
            force = solution[6 * index : 6 * index + 3]
            normal_force = max(1e-6, float(np.dot(normal, force)))
            tangential = force - normal_force * normal
            slip_cost += float(
                (np.linalg.norm(tangential) / (mu * normal_force)) ** 2
            )
        return WrenchAllocation(
            left_wrench=solution[:6].copy(),
            right_wrench=solution[6:].copy(),
            achieved_wrench=achieved,
            desired_wrench=desired,
            status=status,
            tracking_error=float(np.linalg.norm(achieved - desired)),
            slip_cost=slip_cost,
            angular_velocity_cost=(
                self.config.angular_velocity_cost_weight
                * float(np.dot(angular_velocity, angular_velocity))
            ),
        )
