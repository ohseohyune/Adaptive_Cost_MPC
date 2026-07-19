"""Phase 9 acceptance tests for ballistic broad-pad box interception."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.clik.catching import adaptive_stiffness
from control.squeeze import (
    BallisticBoxPredictor,
    BallisticPrediction,
    BoxFaceInterceptionPlanner,
    DynamicSideSqueezeConfig,
    minimum_symmetric_squeeze_force,
    adaptive_impact_command,
)
from box_squeeze.main_dynamic_box_squeeze import DynamicRunConfig, run_dynamic_side_squeeze


def test_ballistic_predictor_matches_known_gravity_trajectory() -> None:
    gravity = np.array([0.0, 0.0, -9.81])
    velocity_0 = np.array([-0.94, 0.007, 0.65])
    position_0 = np.array([0.5, 0.0, 1.0])
    predictor = BallisticBoxPredictor(
        gravity=gravity,
        velocity_alpha=1.0,
        max_speed=3.0,
    )
    prediction = None
    for time_s in np.linspace(0.0, 0.10, 11):
        position = position_0 + velocity_0 * time_s + 0.5 * gravity * time_s**2
        prediction = predictor.update(float(time_s), position)
    assert prediction is not None
    true_velocity = velocity_0 + gravity * 0.10
    assert np.linalg.norm(prediction.velocity - true_velocity) < 1e-9
    assert prediction.confidence == 1.0


def test_face_interception_and_ttc_stiffness() -> None:
    config = DynamicSideSqueezeConfig()
    planner = BoxFaceInterceptionPlanner(config)
    prediction = BallisticPrediction(
        time_s=0.0,
        position=np.array([0.5, 0.0, 1.0]),
        velocity=np.array([-1.0, 0.0, 0.65]),
        gravity=np.array(config.gravity),
        confidence=1.0,
    )
    expected_center = prediction.position_after(0.2)
    offset = config.box_half_y + config.pad_half_thickness + config.precontact_gap
    plan = planner.plan(
        prediction,
        left_pad_position=expected_center + np.array([0.0, offset, 0.0]),
        right_pad_position=expected_center - np.array([0.0, offset, 0.0]),
    )
    assert abs(plan.time_to_contact_s - 0.2) < 1e-12
    assert np.allclose(plan.box_center, expected_center)
    assert np.allclose(plan.left_face_center - plan.right_face_center, [0.0, 0.3, 0.0])
    assert plan.reachable

    softened = float(
        adaptive_stiffness(
            0.0,
            config.tangential_stiffness,
            config.impact_tangential_stiffness,
            config.ttc_soften_window_s,
        )
    )
    assert softened == config.impact_tangential_stiffness
    assert softened < config.tangential_stiffness


def test_minimum_no_slip_force_uses_mass_and_friction() -> None:
    ideal = minimum_symmetric_squeeze_force(
        mass=0.50,
        friction=1.20,
        gravity=(0.0, 0.0, -9.81),
        calibration_factor=1.0,
    )
    calibrated = minimum_symmetric_squeeze_force(
        mass=0.50,
        friction=1.20,
        gravity=(0.0, 0.0, -9.81),
        calibration_factor=4.9,
    )
    assert np.isclose(ideal, 2.04375)
    assert np.isclose(calibrated, 10.014375)


def test_impact_schedule_softens_high_energy_contact() -> None:
    config = DynamicSideSqueezeConfig()
    slow = adaptive_impact_command(
        config,
        object_mass=0.35,
        contact_face_area=0.0121,
        relative_normal_speed=0.05,
    )
    fast = adaptive_impact_command(
        config,
        object_mass=0.70,
        contact_face_area=0.0080,
        relative_normal_speed=0.60,
    )
    assert fast.normal_stiffness < slow.normal_stiffness
    assert fast.tangential_stiffness < slow.tangential_stiffness
    assert fast.desired_force <= slow.desired_force


def test_random_velocity_dynamic_catches() -> None:
    assert DynamicSideSqueezeConfig().required_dynamic_hold_s == 5.0
    assert DynamicSideSqueezeConfig().timeout_s > 5.0
    velocities: list[tuple[float, float, float]] = []
    launch_positions: list[tuple[float, float, float]] = []
    for seed in (0, 3, 5):
        config = DynamicSideSqueezeConfig(random_seed=seed)
        result = run_dynamic_side_squeeze(
            DynamicRunConfig(viewer=False, squeeze=config)
        )
        velocities.append(result.launch_velocity_mps)
        launch_positions.append(result.launch_position_m)
        assert result.success, result
        assert result.simultaneous_start
        assert abs(result.initial_ttc_s - 0.56) < 1e-9
        assert 1.35 <= abs(result.launch_velocity_mps[0]) <= 1.47
        assert 1.05 <= result.launch_position_m[0] <= 1.13
        assert result.reachable_intercept_seen
        assert result.first_contact_time_s is not None
        assert result.bilateral_contact_time_s is not None
        assert result.first_contact_peak_force_n <= config.first_contact_force_limit
        assert np.isclose(result.minimum_no_slip_force_per_pad_n, 10.014375)
        assert result.final_hold_force_per_pad_n >= result.minimum_no_slip_force_per_pad_n
        assert result.final_hold_force_per_pad_n <= config.maximum_hold_normal_force
        assert result.minimum_ttc_stiffness_npm < config.tangential_stiffness
        assert result.prediction_velocity_error_mps < 0.10
        assert result.interception_center_error_m < 0.060
        assert result.final_box_speed_mps <= config.maximum_capture_speed
        assert result.dynamic_hold_time_s >= config.required_dynamic_hold_s
        assert result.grippers_remained_open
    assert len(set(velocities)) == len(velocities)
    assert len(set(launch_positions)) == len(launch_positions)


def main() -> int:
    tests = [
        test_ballistic_predictor_matches_known_gravity_trajectory,
        test_face_interception_and_ttc_stiffness,
        test_minimum_no_slip_force_uses_mass_and_friction,
        test_impact_schedule_softens_high_energy_contact,
        test_random_velocity_dynamic_catches,
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
