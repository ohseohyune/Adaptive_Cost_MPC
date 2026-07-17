"""Contact detection and friction cone utilities.

contact frame convention (MuJoCo mj_contactForce output)
---------------------------------------------------------
force[0]   = normal force [N]   (positive = compression)
force[1:3] = tangential force components [N]
force[3:6] = torque components [N·m]

Friction cone (Coulomb):
    ||f_tangent|| ≤ μ * f_normal   (and f_normal ≥ 0)
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class ContactInfo:
    """Single contact between gripper and object."""

    geom1: int
    geom2: int
    pos: np.ndarray      # contact point [m], world frame
    normal: np.ndarray   # contact normal (geom1 → geom2 surface), world frame
    force: np.ndarray    # 6D wrench in contact frame
    f_normal: float      # normal force [N] (positive = compression)
    f_tangent: float     # tangential force magnitude [N]


def get_contact_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_geom_ids: set[int],
) -> list[ContactInfo]:
    """Extract all contacts that involve at least one gripper geom.

    Returns an empty list if no gripper contacts exist.
    """
    contacts: list[ContactInfo] = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in gripper_geom_ids or c.geom2 in gripper_geom_ids:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            contacts.append(
                ContactInfo(
                    geom1=int(c.geom1),
                    geom2=int(c.geom2),
                    pos=c.pos.copy(),
                    normal=c.frame[:3].copy(),
                    force=force.copy(),
                    f_normal=float(force[0]),
                    f_tangent=float(np.linalg.norm(force[1:3])),
                )
            )
    return contacts


def get_contacts_between(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first_geom_ids: set[int],
    second_geom_ids: set[int],
) -> list[ContactInfo]:
    """Extract contacts whose two geoms belong to the requested two sets.

    Unlike :func:`get_contact_forces`, this excludes finger/finger, robot/floor,
    and unrelated self contacts.  That distinction is required when the two
    grippers independently validate a physical bimanual grasp.
    """

    contacts: list[ContactInfo] = []
    for i in range(data.ncon):
        c = data.contact[i]
        forward = c.geom1 in first_geom_ids and c.geom2 in second_geom_ids
        reverse = c.geom2 in first_geom_ids and c.geom1 in second_geom_ids
        if not (forward or reverse):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        normal = c.frame[:3].copy()
        if reverse:
            normal = -normal
        contacts.append(
            ContactInfo(
                geom1=int(c.geom1),
                geom2=int(c.geom2),
                pos=c.pos.copy(),
                normal=normal,
                force=force.copy(),
                f_normal=float(force[0]),
                f_tangent=float(np.linalg.norm(force[1:3])),
            )
        )
    return contacts


def body_geom_ids(model: mujoco.MjModel, body_name: str) -> set[int]:
    """Return collision geom IDs directly attached to a named body."""

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {body_name}")
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id and model.geom_contype[geom_id] != 0
    }


def check_friction_cone(f_normal: float, f_tangent: float, mu: float = 0.5) -> bool:
    """Return True if the contact lies inside the Coulomb friction cone.

    Args:
        f_normal:  normal force [N], positive = compression
        f_tangent: tangential force magnitude [N]
        mu:        friction coefficient
    """
    if f_normal <= 0.0:
        return False
    return f_tangent <= mu * f_normal


def total_normal_force(contacts: list[ContactInfo]) -> float:
    """Sum of positive (compressive) normal forces [N]."""
    return sum(max(0.0, c.f_normal) for c in contacts)


def all_in_friction_cone(contacts: list[ContactInfo], mu: float = 0.5) -> bool:
    """Return True if every contact satisfies the friction cone constraint.

    Returns False for an empty contact list (no grasp to validate).
    """
    if not contacts:
        return False
    return all(check_friction_cone(c.f_normal, c.f_tangent, mu) for c in contacts)


def finger_geom_ids(model: mujoco.MjModel, finger_body_names: tuple[str, ...]) -> set[int]:
    """Resolve gripper body names to the set of collision geom IDs."""
    ids: set[int] = set()
    for bname in finger_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            continue
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == bid and model.geom_contype[g] != 0:
                ids.add(g)
    return ids
