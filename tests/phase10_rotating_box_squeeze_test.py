"""Phase 10 acceptance tests for rotating-box SE(3) stabilization."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.squeeze import (
    BimanualWrenchAllocator,
    QuaternionAngularVelocityPredictor,
    RotatingBoxPrediction,
    RotatingSideSqueezeConfig,
    SE3BoxFaceTargetPlanner,
    rotation_exp,
    rotation_to_quaternion,
)
from control.squeeze.ballistic import BallisticPrediction
from box_squeeze.main_dynamic_box_squeeze import DynamicRunConfig, run_dynamic_side_squeeze


def test_quaternion_angular_velocity_prediction() -> None:
    config = RotatingSideSqueezeConfig(
        predictor_angular_velocity_alpha=1.0,
        maximum_predicted_angular_speed=3.0,
    )
    predictor = QuaternionAngularVelocityPredictor(config)
    omega = np.array([0.07, -0.04, 0.12])
    position = np.array([0.5, 0.0, 1.0])
    prediction = None
    for time_s in np.linspace(0.0, 0.10, 11):
        rotation = rotation_exp(omega * time_s)
        prediction = predictor.update(
            float(time_s), position, rotation_to_quaternion(rotation)
        )
    assert prediction is not None
    assert np.linalg.norm(prediction.angular_velocity - omega) < 1e-8
    assert prediction.angular_confidence == 1.0


def test_se3_face_targets_follow_box_rotation() -> None:
    config = RotatingSideSqueezeConfig()
    rotation = rotation_exp(np.array([0.08, -0.03, 0.12]))
    ballistic = BallisticPrediction(
        time_s=0.0,
        position=np.array([0.5, 0.0, 1.0]),
        velocity=np.array([-1.0, 0.0, 0.65]),
        gravity=np.array(config.gravity),
        confidence=1.0,
    )
    prediction = RotatingBoxPrediction(
        ballistic=ballistic,
        quaternion=rotation_to_quaternion(rotation),
        rotation=rotation,
        angular_velocity=np.zeros(3),
        angular_confidence=1.0,
    )
    planner = SE3BoxFaceTargetPlanner(
        config,
        left_ee_to_pad_rotation=np.eye(3),
        right_ee_to_pad_rotation=np.eye(3),
    )
    target = planner.plan(prediction)
    expected_face_delta = 2.0 * config.box_half_y * rotation[:, 1]
    assert np.allclose(
        target.left_face_center - target.right_face_center,
        expected_face_delta,
    )
    for transform in (target.left_pad_transform, target.right_pad_transform):
        assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
        assert abs(np.linalg.det(transform[:3, :3]) - 1.0) < 1e-10

    unreachable = planner.plan(
        prediction,
        left_pad_position=np.array([3.0, 3.0, 3.0]),
        right_pad_position=np.array([-3.0, -3.0, -3.0]),
    )
    assert not unreachable.reachable
    assert unreachable.reachability_margin < 0.0


def test_wrench_qp_respects_friction_and_costs() -> None:
    config = RotatingSideSqueezeConfig()
    allocator = BimanualWrenchAllocator(config)
    omega = np.array([0.08, -0.03, 0.12])
    result = allocator.solve(
        box_rotation=np.eye(3),
        left_contact_position=np.array([0.0, config.box_half_y, 0.0]),
        right_contact_position=np.array([0.0, -config.box_half_y, 0.0]),
        box_center=np.zeros(3),
        linear_velocity=np.array([-0.2, 0.01, -0.3]),
        angular_velocity=omega,
    )
    assert "solved" in result.status
    for wrench, inward in (
        (result.left_wrench, np.array([0.0, -1.0, 0.0])),
        (result.right_wrench, np.array([0.0, 1.0, 0.0])),
    ):
        normal = float(np.dot(inward, wrench[:3]))
        tangential = wrench[:3] - normal * inward
        assert -1e-5 <= normal <= config.maximum_qp_normal_force + 1e-5
        assert np.linalg.norm(tangential, ord=np.inf) <= (
            config.wrench_friction_coefficient * normal + 1e-4
        )
    assert result.slip_cost >= 0.0
    assert np.isclose(
        result.angular_velocity_cost,
        config.angular_velocity_cost_weight * np.dot(omega, omega),
    )
    low = BimanualWrenchAllocator(
        replace(config, angular_velocity_cost_weight=0.5)
    ).solve(
        box_rotation=np.eye(3),
        left_contact_position=np.array([0.0, config.box_half_y, 0.0]),
        right_contact_position=np.array([0.0, -config.box_half_y, 0.0]),
        box_center=np.zeros(3),
        linear_velocity=np.array([-0.2, 0.01, -0.3]),
        angular_velocity=omega,
        capture_phase=True,
    )
    high = BimanualWrenchAllocator(
        replace(config, angular_velocity_cost_weight=8.0)
    ).solve(
        box_rotation=np.eye(3),
        left_contact_position=np.array([0.0, config.box_half_y, 0.0]),
        right_contact_position=np.array([0.0, -config.box_half_y, 0.0]),
        box_center=np.zeros(3),
        linear_velocity=np.array([-0.2, 0.01, -0.3]),
        angular_velocity=omega,
        capture_phase=True,
    )
    assert np.linalg.norm(high.achieved_wrench[3:]) > np.linalg.norm(
        low.achieved_wrench[3:]
    )


def test_random_rotating_box_stabilization() -> None:
    for seed in (0, 3, 7):
        config = RotatingSideSqueezeConfig(random_seed=seed)
        result = run_dynamic_side_squeeze(
            DynamicRunConfig(viewer=False, squeeze=config)
        )
        assert result.success, result
        assert result.simultaneous_start
        assert abs(result.initial_ttc_s - 0.56) < 1e-9
        assert 1.35 <= abs(result.launch_velocity_mps[0]) <= 1.47
        assert 1.05 <= result.launch_position_m[0] <= 1.13
        assert np.linalg.norm(result.launch_angular_velocity_radps) > 0.0
        assert result.angular_prediction_error_radps < 0.01
        assert "solved" in result.wrench_qp_status
        assert result.slip_cost <= 2.0 * config.maximum_hold_slip_ratio**2
        assert result.angular_velocity_cost >= 0.0
        assert result.final_angular_speed_radps <= config.maximum_capture_angular_speed
        assert result.final_angular_speed_radps < result.peak_angular_speed_radps
        assert result.first_contact_peak_force_n <= config.first_contact_force_limit
        assert result.minimum_no_slip_force_per_pad_n > 11.0
        assert result.final_hold_force_per_pad_n >= result.minimum_no_slip_force_per_pad_n
        assert result.dynamic_hold_time_s >= config.required_dynamic_hold_s


def main() -> int:
    tests = [
        test_quaternion_angular_velocity_prediction,
        test_se3_face_targets_follow_box_rotation,
        test_wrench_qp_respects_friction_and_costs,
        test_random_rotating_box_stabilization,
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
