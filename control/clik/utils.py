"""Pure NumPy SE(3)/SO(3) math utilities."""

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def transform_inverse(transform: np.ndarray) -> np.ndarray:
    """Invert a homogeneous transform T in SE(3)."""
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    rotation = transform[:3, :3]
    position = transform[:3, 3]

    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ position
    return inverse


def matrix_exp_se3(screw_axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Return exp([S] theta) in SE(3) for a screw axis S = [omega; v].

    Revolute joints use the Modern Robotics closed form. Prismatic joints are
    handled by omega ~= 0.
    """

    screw_axis = np.asarray(screw_axis, dtype=float).reshape(6)
    omega = screw_axis[:3]
    linear = screw_axis[3:]
    omega_norm = np.linalg.norm(omega)

    transform = np.eye(4)
    if omega_norm < 1e-12:
        transform[:3, 3] = linear * theta
        return transform

    omega_hat = skew(omega)
    omega_hat_squared = omega_hat @ omega_hat
    omega_theta = omega_norm * theta

    transform[:3, :3] = (
        np.eye(3)
        + (np.sin(omega_theta) / omega_norm) * omega_hat
        + ((1.0 - np.cos(omega_theta)) / omega_norm**2) * omega_hat_squared
    )

    g_theta = (
        theta * np.eye(3)
        + ((1.0 - np.cos(omega_theta)) / omega_norm**2) * omega_hat
        + ((omega_theta - np.sin(omega_theta)) / omega_norm**3)
        * omega_hat_squared
    )
    transform[:3, 3] = g_theta @ linear
    return transform


def adjoint(transform: np.ndarray) -> np.ndarray:
    """Return the 6x6 adjoint representation Ad_T for T in SE(3)."""

    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    rotation = transform[:3, :3]
    position = transform[:3, 3]

    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, :3] = skew(position) @ rotation
    adjoint_matrix[3:, 3:] = rotation
    return adjoint_matrix


def matrix_log_se3(transform: np.ndarray) -> np.ndarray:
    """
    Matrix logarithm on SE(3), returned as [omega * theta, v * theta].

    Orientation errors live on SO(3), not in a vector space, so subtracting two
    rotation matrices or Euler angles is not a reliable pose error. The SE(3)
    logarithm maps the relative transform T_cur^{-1} T_des into a 6D twist
    coordinate that can be used by a Jacobian-based CLIK update.
    """

    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    rotation = transform[:3, :3]
    position = transform[:3, 3]

    cos_theta = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        return np.concatenate([np.zeros(3), position.copy()])

    if abs(theta - np.pi) < 1e-6:
        omega_outer = (rotation + np.eye(3)) / 2.0
        axis_index = int(np.argmax(np.diag(omega_outer)))
        omega = np.zeros(3)
        omega[axis_index] = np.sqrt(max(omega_outer[axis_index, axis_index], 0.0))
        if omega[axis_index] < 1e-10:
            omega[axis_index] = 1.0
        for i in range(3):
            if i != axis_index:
                omega[i] = omega_outer[axis_index, i] / omega[axis_index]
    else:
        omega_hat = (rotation - rotation.T) / (2.0 * np.sin(theta))
        omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])

    omega_theta = omega * theta

    omega_skew = skew(omega)
    tan_half_theta = np.tan(theta / 2.0)
    if abs(tan_half_theta) < 1e-10:
        coefficient = 1.0
    else:
        coefficient = 1.0 - (theta / 2.0) / tan_half_theta

    g_inv_theta = (
        np.eye(3)
        - (theta / 2.0) * omega_skew
        + coefficient * (omega_skew @ omega_skew)
    )
    v_theta = g_inv_theta @ position

    return np.concatenate([omega_theta, v_theta])
