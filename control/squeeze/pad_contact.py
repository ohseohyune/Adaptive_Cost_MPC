"""Object-only contact measurements for the two side-squeeze pads."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from control.clik.contact import ContactInfo, get_contacts_between, total_normal_force


@dataclass(frozen=True)
class PadContactMeasurement:
    active: bool
    count: int
    normal_force: float
    tangential_force: float
    peak_normal_force: float
    mean_position: np.ndarray
    contacts: tuple[ContactInfo, ...]


@dataclass(frozen=True)
class BilateralPadContact:
    left: PadContactMeasurement
    right: PadContactMeasurement

    @property
    def bilateral(self) -> bool:
        return self.left.active and self.right.active

    @property
    def force_imbalance(self) -> float:
        return abs(self.left.normal_force - self.right.normal_force)


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"MuJoCo geom not found: {name}")
    return int(geom_id)


def _measurement(contacts: list[ContactInfo]) -> PadContactMeasurement:
    if contacts:
        position = np.mean([contact.pos for contact in contacts], axis=0)
        peak = max(max(0.0, contact.f_normal) for contact in contacts)
    else:
        position = np.full(3, np.nan)
        peak = 0.0
    return PadContactMeasurement(
        active=bool(contacts),
        count=len(contacts),
        normal_force=float(total_normal_force(contacts)),
        tangential_force=float(sum(contact.f_tangent for contact in contacts)),
        peak_normal_force=float(peak),
        mean_position=position,
        contacts=tuple(contacts),
    )


def read_bilateral_pad_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_geom_name: str = "flying_box_geom",
) -> BilateralPadContact:
    """Measure only left-pad/box and right-pad/box contacts."""

    box = {_geom_id(model, box_geom_name)}
    left = get_contacts_between(model, data, {_geom_id(model, "left_squeeze_pad")}, box)
    right = get_contacts_between(model, data, {_geom_id(model, "right_squeeze_pad")}, box)
    return BilateralPadContact(left=_measurement(left), right=_measurement(right))

