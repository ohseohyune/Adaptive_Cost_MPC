"""Robot state to geometric representation helpers."""

import mujoco
import numpy as np

from control.clik.types import SerialArm
from control.clik.utils import matrix_log_se3, transform_inverse


def get_ee_position(data: mujoco.MjData, arm: SerialArm) -> np.ndarray:
    """
    Return the current end-effector position in world coordinates.

    If a site is configured, the site position is used. This is useful when the
    desired control point is a grasp center offset from the final link frame.
    Otherwise the body origin is used.
    """

    if arm.ee_site_id is not None:
        return data.site_xpos[arm.ee_site_id].copy()
    return data.xpos[arm.ee_body_id].copy()


def get_ee_transform(data: mujoco.MjData, arm: SerialArm) -> np.ndarray:
    """
    Return the current end-effector pose as a 4x4 world-frame transform.

    If an end-effector site is configured, both its position and orientation are
    used. Otherwise the body frame is used. For FFW, the red EE site marks the
    gripper grasp center, so pose control is evaluated at that site.
    """

    transform = np.eye(4)
    if arm.ee_site_id is not None:
        transform[:3, :3] = data.site_xmat[arm.ee_site_id].reshape(3, 3).copy()
        transform[:3, 3] = data.site_xpos[arm.ee_site_id].copy()
    else:
        transform[:3, :3] = data.xmat[arm.ee_body_id].reshape(3, 3).copy()
        transform[:3, 3] = data.xpos[arm.ee_body_id].copy()
    return transform


def make_transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from position and rotation matrix."""
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return transform


def compute_pose_error(
    current_transform: np.ndarray,
    desired_transform: np.ndarray,
) -> np.ndarray:
    """
    Compute body-frame SE(3) pose error e_b in R^6.

    The old OMY implementation used the same core idea:
        T_err = T_cur^{-1} @ T_des
        e_b   = log(T_err)^vee

    This error is body-frame because the relative transform is left-multiplied
    by T_cur^{-1}; it describes the motion from the current EE frame to the
    desired EE frame in current EE coordinates.

    A world-frame error like x_des - x_cur is fine for pure position control,
    but full orientation control needs SO(3)/SE(3) geometry. Rotation matrices
    and quaternions cannot be subtracted like ordinary vectors without creating
    representation-dependent errors.
    """

    transform_error = transform_inverse(current_transform) @ np.asarray(
        desired_transform, dtype=float
    ).reshape(4, 4)
    return matrix_log_se3(transform_error)
