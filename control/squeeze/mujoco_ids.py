"""Small MuJoCo model-lookup helpers shared by the box-squeeze entry scripts."""

from __future__ import annotations

import mujoco
import numpy as np


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def require_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(value)


def actuator_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names],
        dtype=int,
    )
