"""Minimum normal-force calculation for symmetric side squeezing."""

from __future__ import annotations

import numpy as np


def minimum_symmetric_squeeze_force(
    *,
    mass: float,
    friction: float,
    gravity: np.ndarray | tuple[float, float, float],
    calibration_factor: float,
) -> float:
    """Return the minimum normal force per pad for gravity support.

    The ideal two-contact bound is ``m |g| / (2 mu)``.  The calibration
    factor accounts for finite pad area, rotating contact, solver compliance,
    and the margin needed to avoid visible creep rather than merely delay it.
    """

    if mass <= 0.0:
        raise ValueError("mass must be positive")
    if friction <= 0.0:
        raise ValueError("friction must be positive")
    if calibration_factor < 1.0:
        raise ValueError("calibration_factor must be at least one")
    gravity_norm = float(np.linalg.norm(np.asarray(gravity, dtype=float)))
    return float(calibration_factor * mass * gravity_norm / (2.0 * friction))
