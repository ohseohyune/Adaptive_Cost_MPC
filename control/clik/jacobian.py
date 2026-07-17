"""Modern Robotics PoE body-Jacobian helpers."""

import mujoco
import numpy as np

from control.clik.mujoco_utils import body_screw_axes_from_model
from control.clik.types import SerialArm
from control.clik.utils import adjoint, matrix_exp_se3, transform_inverse


def body_jacobian(theta: np.ndarray, body_screw_axes: list[np.ndarray]) -> np.ndarray:
    """
    Compute the Modern Robotics body Jacobian J_b(theta).

    Parameters
    ----------
    theta
        Joint coordinates ordered from base to end effector.
    body_screw_axes
        Body-frame screw axes B_i = [omega_i; v_i] at the home pose, ordered to
        match theta.
    """

    theta = np.asarray(theta, dtype=float).reshape(-1)
    if len(body_screw_axes) != theta.size:
        raise ValueError(
            f"Expected {theta.size} screw axes, got {len(body_screw_axes)}."
        )

    space_jacobian = np.zeros((6, theta.size))
    transform = np.eye(4)

    for index, (screw_axis, joint_angle) in enumerate(zip(body_screw_axes, theta)):
        body_axis = np.asarray(screw_axis, dtype=float).reshape(6)
        space_jacobian[:, index] = adjoint(transform) @ body_axis
        transform = transform @ matrix_exp_se3(body_axis, joint_angle)

    return adjoint(transform_inverse(transform)) @ space_jacobian


def compute_full_body_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
) -> np.ndarray:
    """
    Compute a 6xn Modern Robotics body Jacobian for full-pose CLIK.


    Implementation:
    1. Build a temporary home state by setting this arm's seven joint angles to
       zero while preserving non-arm coordinates such as the floating base.
    2. Extract each hinge/slide screw axis S_i from MuJoCo's home joint axis and
       anchor.
    3. Convert S_i to body screw axes B_i through B_i = Ad_{M^-1} S_i, where M
       is the selected body/site end-effector home transform.
    4. Evaluate J_b(theta) with the Modern Robotics PoE body-Jacobian formula.
    """

    theta = data.qpos[arm.qpos_indices].copy()
    body_axes = body_screw_axes_from_model(model, data, arm)
    return body_jacobian(theta, body_axes)
