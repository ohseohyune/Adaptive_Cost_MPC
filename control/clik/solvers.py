"""Numerical IK solver utilities."""

import numpy as np


def damped_pseudoinverse(jacobian: np.ndarray, damping: float) -> np.ndarray:
    """
    Return the damped least-squares pseudo-inverse of a task Jacobian.

    J_dls^+ = J.T @ inv(J @ J.T + lambda^2 I)
    """

    task_dim = jacobian.shape[0]
    lhs = jacobian @ jacobian.T + (damping**2) * np.eye(task_dim)
    return np.linalg.solve(lhs, jacobian).T


def damped_least_squares(
    jacobian: np.ndarray,
    error: np.ndarray,
    damping: float,
) -> np.ndarray:
    """
    Convert task-space error into a joint increment.

    Damped least squares (DLS):
        dq = J.T @ inv(J @ J.T + lambda^2 I) @ e

    This is a damped pseudo-inverse, not the plain Jacobian-transpose method.
    The damping term lambda^2 I prevents very large joint updates when the
    Jacobian is near-singular or poorly conditioned.
    """

    return damped_pseudoinverse(jacobian, damping) @ error


def damped_least_squares_with_nullspace(
    jacobian: np.ndarray,
    error: np.ndarray,
    damping: float,
    q_current: np.ndarray,
    q_reference: np.ndarray,
    posture_gain: float,
    posture_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve a DLS IK step with a null-space posture objective.

    dq = J^+ e + (I - J^+ J) dq_posture
    dq_posture = k * W * (q_ref - q)

    The null-space term nudges redundant joints toward q_reference while mostly
    preserving the end-effector task handled by the first term.
    """

    q_current = np.asarray(q_current, dtype=float).reshape(-1)
    q_reference = np.asarray(q_reference, dtype=float).reshape(q_current.shape)

    if posture_weights is None:
        weights = np.ones_like(q_current)
    else:
        weights = np.asarray(posture_weights, dtype=float).reshape(q_current.shape)

    jacobian_pinv = damped_pseudoinverse(jacobian, damping)
    dq_task = jacobian_pinv @ error
    dq_posture = float(posture_gain) * weights * (q_reference - q_current)
    null_projector = np.eye(q_current.size) - jacobian_pinv @ jacobian
    dq_null = null_projector @ dq_posture
    return dq_task + dq_null, dq_task, dq_null


def pseudoinverse(jacobian: np.ndarray, error: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Solve dq = pinv(J) @ error with NumPy's SVD-based pseudo-inverse."""
    return np.linalg.pinv(jacobian, rcond=rcond) @ error
