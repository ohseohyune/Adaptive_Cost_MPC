"""Runnable self-check for make_transform input validation.

    python tests/kinematics_transform_test.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.clik.kinematics import make_transform


def expect_value_error(position, rotation):
    try:
        make_transform(position, rotation)
    except ValueError:
        return
    raise AssertionError("make_transform accepted an invalid input")


transform = make_transform([1.0, 2.0, 3.0], np.eye(3))
assert np.allclose(transform[:3, 3], [1.0, 2.0, 3.0])
assert np.allclose(transform[:3, :3], np.eye(3))

expect_value_error([1.0, 2.0], np.eye(3))
expect_value_error([1.0, 2.0, 3.0], np.ones((3, 3)))
expect_value_error([1.0, 2.0, 3.0], np.diag([1.0, 1.0, -1.0]))

print("OK -- make_transform validates position and rotation")
