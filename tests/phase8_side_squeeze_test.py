"""Phase 8 acceptance tests for the stationary box side-squeeze milestone."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.squeeze.config import SideSqueezeConfig
from control.squeeze.hybrid_controller import HybridSqueezeController
from control.squeeze.pad_contact import BilateralPadContact, PadContactMeasurement
from main_box_squeeze import RunConfig, run_side_squeeze


SCENE = ROOT / "model/robotis_ffw/scene_ffw_sg2_fixed_base_box_squeeze.xml"


def _empty_contact() -> BilateralPadContact:
    empty = PadContactMeasurement(
        active=False,
        count=0,
        normal_force=0.0,
        tangential_force=0.0,
        peak_normal_force=0.0,
        mean_position=np.full(3, np.nan),
        contacts=(),
    )
    return BilateralPadContact(
        left=empty,
        right=empty,
    )


def test_scene_geometry_and_collision_masks() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))

    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_squeeze_pad")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_squeeze_pad")
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "flying_box_geom")
    fixture_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "box_fixture")

    assert min(left_id, right_id, box_id, fixture_id) >= 0
    for pad_id in (left_id, right_id):
        assert model.geom_contype[pad_id] == 8
        assert model.geom_conaffinity[pad_id] == 16
    assert model.geom_contype[box_id] == 16
    assert model.geom_conaffinity[box_id] == 8


def test_force_error_moves_both_pads_inward() -> None:
    config = SideSqueezeConfig()
    controller = HybridSqueezeController(config)
    initial_left = controller.left_compression
    initial_right = controller.right_compression

    for _ in range(50):
        command = controller.update(
            box_center=np.zeros(3),
            contact=_empty_contact(),
            dt=0.002,
        )

    assert command.left_compression > initial_left
    assert command.right_compression > initial_right
    assert abs(command.left_compression - command.right_compression) < 1e-12


def test_stationary_box_is_held_after_fixture_release() -> None:
    config = SideSqueezeConfig()
    result = run_side_squeeze(RunConfig(viewer=False, squeeze=config))

    assert result.success, result
    assert result.fixture_release_time_s is not None
    assert result.free_hold_time_s >= config.required_free_hold_s
    assert result.final_left_force_n >= config.minimum_contact_force
    assert result.final_right_force_n >= config.minimum_contact_force
    assert result.final_box_drop_m <= config.maximum_box_drop
    assert result.final_box_lateral_drift_m <= config.maximum_box_lateral_drift
    assert result.grippers_remained_open


def main() -> int:
    tests = [
        test_scene_geometry_and_collision_masks,
        test_force_error_moves_both_pads_inward,
        test_stationary_box_is_held_after_fixture_release,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - standalone acceptance runner
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
