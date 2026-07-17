"""Utilities for null-space QP self-motion tests."""

from __future__ import annotations

import mujoco
import numpy as np


def limit_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)


def null_motion_error_scale(
    position_error_norm: float,
    start: float,
    stop: float,
) -> float:
    """Fade null-space motion out when the EE drifts away from the target."""

    if position_error_norm <= start:
        return 1.0
    if position_error_norm >= stop:
        return 0.0
    return float((stop - position_error_norm) / (stop - start))


def _cylinder_from_to(
    geom,
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    rgba: list[float],
) -> None:
    """Draw a cylinder from origin along direction."""
    z = np.asarray(direction, dtype=float)
    z = z / np.linalg.norm(z)
    mid = np.asarray(origin, dtype=float) + z * (length / 2.0)

    # Build rotation matrix with local z → direction
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
    x = ref - np.dot(ref, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.column_stack([x, y, z])

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([radius, radius, length / 2.0]),
        mid,
        rot.flatten(),
        np.array(rgba, dtype=float),
    )


_AXIS_COLORS = [
    [1.0, 0.1, 0.1, 1.0],  # x → red
    [0.1, 1.0, 0.1, 1.0],  # y → green
    [0.1, 0.1, 1.0, 1.0],  # z → blue
]
_WORLD_AXES = np.eye(3)  # world frame rotation is identity


def update_scene_visuals(
    viewer,
    ee_transform: np.ndarray,
    fixed_position: np.ndarray,
    world_origin: np.ndarray | None = None,
    axis_length: float = 0.12,
    axis_radius: float = 0.006,
    sphere_radius: float = 0.025,
) -> None:
    """Draw fixed-position marker, EE local frame axes, and world frame axes.

    Axes: x=red, y=green, z=blue (RGB = XYZ convention).
    world_origin: where to draw the world frame (default: [0, 0, 0]).
    """
    viewer.user_scn.ngeom = 0
    geoms = viewer.user_scn.geoms
    n = 0

    # Red sphere at fixed target position
    mujoco.mjv_initGeom(
        geoms[n],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([sphere_radius, 0.0, 0.0]),
        np.asarray(fixed_position, dtype=float),
        np.eye(3).flatten(),
        np.array([1.0, 0.05, 0.05, 0.85]),
    )
    n += 1

    # EE local frame axes (columns of rotation matrix)
    ee_origin = ee_transform[:3, 3]
    ee_rot = ee_transform[:3, :3]
    for i in range(3):
        _cylinder_from_to(geoms[n], ee_origin, ee_rot[:, i], axis_length, axis_radius, _AXIS_COLORS[i])
        n += 1

    # World frame axes (identity rotation, fixed at world_origin)
    w_origin = np.zeros(3) if world_origin is None else np.asarray(world_origin, dtype=float)
    for i in range(3):
        _cylinder_from_to(geoms[n], w_origin, _WORLD_AXES[:, i], axis_length, axis_radius, _AXIS_COLORS[i])
        n += 1

    viewer.user_scn.ngeom = n


def log_step(
    sim_time: float,
    reconfigure_started: bool,
    ori_err: float,
    pos_err: float,
    task_vel_err: np.ndarray,
    null_vel: np.ndarray,
    null_scale: float,
    min_margin: float,
    task_scaling: float,
    status: str,
) -> None:
    phase = "reconfig" if reconfigure_started else "hold"
    print(
        f"t={sim_time:6.3f} | {phase:8s} | "
        f"ori_err={ori_err:.3e} | "
        f"pos_err={pos_err:.3e} | "
        f"vel_task_err={np.linalg.norm(task_vel_err):.3e} | "
        f"null_vel={np.linalg.norm(null_vel):.3e} | "
        f"null_scale={null_scale:.2f} | "
        f"min_margin={min_margin:.3f} | "
        f"alpha={task_scaling:.3f} | "
        f"{status}",
        flush=True,
    )


# kept for backward compatibility
def add_red_sphere(viewer, position: np.ndarray, radius: float = 0.025) -> None:
    viewer.user_scn.ngeom = 0
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.array([1.0, 0.05, 0.05, 0.85]),
    )
    viewer.user_scn.ngeom = 1
