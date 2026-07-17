"""Controller-level configuration dataclasses."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoseCLIKConfig:
    """
    Parameter bundle for full SE(3) CLIK.

    The pose error is ordered as [rot_x, rot_y, rot_z, pos_x, pos_y, pos_z].
    """

    gain: np.ndarray = field(
        default_factory=lambda: np.array([3, 3, 3, 3, 3, 3]) * 6.0
    )
    damping: float = 0.03
    max_joint_step: float = 0.05


@dataclass
class Level3AcceptanceConfig:
    """Pass/fail thresholds for kinematic grasp-consistency diagnostics."""

    enabled: bool = True
    object_error_max: float = 0.03
    relative_position_error_max: float = 0.02
    relative_orientation_error_max: float = 0.08
    min_manipulability_min: float = 1e-4
    qp_task_residual_max: float = 0.05
    command_clip_delta_max: float = 1e-6
    allow_dq_clipping: bool = False
    require_qp: bool = True


@dataclass
class FFWDemoConfig:
    """Runtime parameters for the FFW target-tracking demo."""

    arm_name: str = "right"

    target_centers: dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "right": np.array([0.2855, -0.0775, 0.8076]),
            "left": np.array([0.0855, 0.2775, 0.7076]),
        }
    )
    target_trajectory: str = "circle"
    target_frequency_hz: float = 0.35
    target_circle_radius: float = 0.10
    target_sine_x_amplitude: float = 0.08
    target_sine_z_amplitude: float = 0.04
    target_path_samples: int = 80
    prediction_enabled: bool = True
    prediction_horizon_s: float = 0.25
    prediction_smoothing_alpha: float = 0.2
    prediction_max_speed: float | None = 1.5
    prediction_log_enabled: bool = True
    prediction_log_stride: int = 1

    interception_enabled: bool = False
    interception_min_lead_s: float = 0.05
    interception_max_lead_s: float = 1.0
    interception_lead_samples: int = 20
    interception_reach_speed: float = 0.3
    interception_blend_alpha: float = 0.5
    interception_log_enabled: bool = True
    interception_log_stride: int = 1

    clik: PoseCLIKConfig = field(default_factory=PoseCLIKConfig)
    bimanual_damping: float = 0.04
    bimanual_solver_mode: str = "nullspace_qp"
    bimanual_qp_posture_weight: float = 1.0
    bimanual_qp_manipulability_weight: float | None = None
    bimanual_qp_regularization: float = 1e-4
    bimanual_qp_eps_abs: float = 1e-4
    bimanual_qp_eps_rel: float = 1e-4
    bimanual_qp_max_iter: int = 200
    bimanual_joint_velocity_limit: float | None = 12.0
    bimanual_qp_max_null_velocity: float | None = 6.0
    bimanual_qp_null_error_scale_start: float = 0.008
    bimanual_qp_null_error_scale_stop: float = 0.030
    bimanual_orientation_task_weight: float = 0.0
    bimanual_relative_orientation_weight: float = 0.8
    bimanual_relative_axis_local: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, 0.0, 0.0])
    )
    bimanual_relative_axis_face_each_other: bool = True
    bimanual_start_from_zero_home: bool = True
    bimanual_initial_hold_s: float = 0.0
    bimanual_prealign_s: float = 3.0
    bimanual_prealign_relative_orientation_weight: float = 0.0
    bimanual_prealign_orientation_task_weight: float = 0.0
    bimanual_relative_orientation_ramp_s: float = 2.0
    bimanual_reset_nominal_after_prealign: bool = True
    bimanual_use_prealign_nominal: bool = True
    bimanual_joint_space_prealign: bool = True
    bimanual_joint_limit_gain: float = 0.2
    bimanual_joint_limit_activation_margin: float = 0.30
    bimanual_manipulability_gain: float = 0.01
    bimanual_manipulability_gradient_step: float = 1e-4
    bimanual_max_manipulability_command: float = 0.005
    bimanual_adaptive_damping_alpha: float = 0.0005
    bimanual_adaptive_damping_epsilon: float = 1e-4
    bimanual_adaptive_damping_max: float = 0.25
    bimanual_singularity_warning_threshold: float = 1e-4
    bimanual_visualize_nullspace_comparison: bool = True
    bimanual_max_object_command: float = 0.04
    bimanual_max_relative_position_command: float = 0.005
    bimanual_max_relative_rotation_command: float = 0.12
    bimanual_target_ramp_s: float = 5.0
    bimanual_kinematic_control: bool = True
    bimanual_kinematic_alpha: float = 0.15
    bimanual_control_log_enabled: bool = True
    bimanual_control_log_stride: int = 1
    level3_acceptance: Level3AcceptanceConfig = field(
        default_factory=Level3AcceptanceConfig
    )
    zero_arm_home: np.ndarray = field(default_factory=lambda: np.zeros(7))

    right_arm_home: np.ndarray = field(
        default_factory=lambda: np.array([
            -0.534804,
            -0.460945,
            -1.278942,
            0.871459,
            -0.281033,
            -0.379103,
            0.050507,
        ])
    )

    left_arm_home: np.ndarray = field(
        default_factory=lambda: np.array([
            -0.534802,
            0.460945,
            1.278943,
            0.871459,
            0.281032,
            -0.379103,
            -0.050513,
        ])
    )

    left_arm_prealign: np.ndarray = field(
        default_factory=lambda: np.array([
            -0.534802,
            0.460945,
            1.278943,
            0.871459,
            0.281032,
            -0.379103,
            -0.050513,
        ])
    )

    right_arm_prealign: np.ndarray = field(
        default_factory=lambda: np.array([
            -0.534804,
            -0.460945,
            -1.278942,
            0.871459,
            -0.281033,
            -0.379103,
            0.050507,
        ])
    )

    orientation_target_mode: str = "axis_alignment"

    # Single-arm fallback axes. The bimanual demo uses the explicit left/right
    # local axes below.

    # gripper 자기 좌표계에서 “앞 방향”이 어느 축인지 / gripper/EE frame 안에서의 축
    gripper_forward_axis_local: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    gripper_up_axis_local: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0])
    )

    # 월드 좌표계에서 gripper forward가 향해야 하는 목표 방향
    gripper_forward_axis_world: np.ndarray = field( 
        default_factory=lambda: np.array([0.0, -1.0, 0.0])
    )
    gripper_up_axis_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )


    # Edit these four fields to tune each hand's local EE frame convention.
    left_gripper_forward_axis_local: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    left_gripper_up_axis_local: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0])
    )
    right_gripper_forward_axis_local: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    right_gripper_up_axis_local: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0])
    )


    gripper_forward_axis_world_tilt: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    gripper_up_axis_world_tilt: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )

    world_axis_length: float = 0.35
    world_axis_width: float = 0.012
    gripper_axis_length: float = 0.12
    gripper_axis_width: float = 0.006
    visualize_gripper_axes: bool = True
    visualize_target_gripper_axes: bool = True

    object_center: np.ndarray = field(
        default_factory=lambda: np.array([0.1855, 0.0, 0.8076])
    )
    object_rotation_world: np.ndarray = field(default_factory=lambda: np.eye(3))
    hand_offsets: dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "right": np.array([0.0, -0.26586, 0.0]),
            "left": np.array([0.0, 0.26586, 0.0]),
        }
    )
    bimanual_face_object_center: bool = False
