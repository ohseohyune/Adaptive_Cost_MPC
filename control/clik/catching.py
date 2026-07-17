"""Dynamic catching utilities.

This module bridges the prediction / interception pipeline with the
Cartesian impedance controller for impact-absorption catching.

API
---
compute_catching_orientation
    Aligns the gripper's local forward axis with the incoming object's
    approach direction so the jaw plane is perpendicular to the flight path.

adaptive_stiffness
    Linearly interpolates from K_nominal down to K_soft over the
    t_soften window before predicted impact.
"""

from __future__ import annotations

import numpy as np

from control.clik.orientation import normalize, rotation_from_axis_alignment


# ── gripper convention ─────────────────────────────────────────────────────────
# FFW / RH-P12-RN: local x-axis is gripper forward, local y-axis is up.
_DEFAULT_GRIPPER_FORWARD = np.array([1.0, 0.0, 0.0])
_DEFAULT_GRIPPER_UP      = np.array([0.0, 1.0, 0.0])


def compute_catching_orientation(
    object_velocity: np.ndarray,
    interception_position: np.ndarray,
    *,
    gripper_forward_local: np.ndarray = _DEFAULT_GRIPPER_FORWARD,
    gripper_up_local: np.ndarray = _DEFAULT_GRIPPER_UP,
) -> np.ndarray:
    """Compute gripper rotation so the jaw plane faces the incoming object.

    The gripper's local primary axis (``gripper_forward_local``) is rotated
    to align with ``-object_velocity`` (the direction the object is coming
    *from*).  A secondary axis is projected perpendicular to the primary so
    the complete SO(3) frame is valid.

    Args:
        object_velocity:       Estimated object velocity in world frame [m/s].
                               If near-zero, the identity rotation is returned.
        interception_position: Interception point in world frame [m].
                               Currently unused but kept for future use
                               (e.g. look-at orientation from EE to object).
        gripper_forward_local: Gripper-frame axis that should face the object.
        gripper_up_local:      Gripper-frame secondary axis (orthogonalised).

    Returns:
        R (3×3) rotation matrix (world_R_gripper) satisfying det(R)=+1.
    """
    vel = np.asarray(object_velocity, dtype=float).reshape(3)
    speed = float(np.linalg.norm(vel))
    if speed < 1e-6:
        return np.eye(3)

    # Gripper should face *into* the incoming object, so point along -velocity.
    approach_world = -vel / speed

    # Choose a world secondary axis perpendicular to the approach direction.
    # Prefer world +z (up); fall back to world +x if approach is near-vertical.
    world_secondary = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(approach_world, world_secondary))) > 0.9:
        world_secondary = np.array([1.0, 0.0, 0.0])

    return rotation_from_axis_alignment(
        local_primary_axis=np.asarray(gripper_forward_local, dtype=float),
        world_primary_axis=approach_world,
        local_secondary_axis=np.asarray(gripper_up_local, dtype=float),
        world_secondary_axis=world_secondary,
    )


def adaptive_stiffness(
    time_to_contact: float,
    K_nominal: float | np.ndarray,
    K_soft: float | np.ndarray,
    t_soften: float = 0.1,
) -> float | np.ndarray:
    """Linearly blend from K_nominal to K_soft as impact approaches.

    When ``time_to_contact > t_soften`` the full stiffness is used.
    Within the last ``t_soften`` seconds the stiffness ramps linearly down
    to ``K_soft``, so the gripper is compliant right at contact.

    Args:
        time_to_contact: Seconds until predicted object contact [s].
        K_nominal:       Full stiffness (scalar or array) [N/m].
        K_soft:          Minimum stiffness at contact (scalar or array) [N/m].
        t_soften:        Softening window duration [s].  Default 0.1 s.

    Returns:
        Effective stiffness (same type and shape as K_nominal).
    """
    ttc = float(time_to_contact)
    if t_soften <= 0.0 or ttc >= t_soften:
        return K_nominal
    alpha = float(np.clip(ttc / t_soften, 0.0, 1.0))
    return alpha * np.asarray(K_nominal) + (1.0 - alpha) * np.asarray(K_soft)
