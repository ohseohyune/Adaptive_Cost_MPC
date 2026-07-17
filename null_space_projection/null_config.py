"""Configuration for the null-space QP self-motion test."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
XML_PATH = PROJECT_ROOT / "model" / "robotis_ffw" / "scene_ffw_sg2.xml"
MOTION_SPEED_SCALE = 4.0


@dataclass(frozen=True)
class NullSpaceQPConfig:
    posture_weight: float = 8.0
    damping_weight: float = 1e-4
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    max_iter: int = 200
    verbose: bool = False


@dataclass(frozen=True)
class SimParams:
    physics_dt: float
    gain_vector: np.ndarray
    qdot_limit: np.ndarray
    switch_threshold: float
    null_scale_start: float
    null_scale_stop: float
    max_null_velocity: float
