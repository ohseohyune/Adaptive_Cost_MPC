"""Configuration for the first stationary-box side-squeeze milestone."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SideSqueezeConfig:
    # Geometry [m]. Values must match the MuJoCo scene and pad geoms.
    box_half_y: float = 0.150
    pad_half_thickness: float = 0.006
    precontact_gap: float = 0.010
    max_compression: float = 0.015

    # Timing [s].
    initial_settle_s: float = 0.30
    approach_s: float = 2.50
    bilateral_stable_s: float = 0.50
    required_free_hold_s: float = 1.00
    timeout_s: float = 7.00

    # Force/admittance control.
    desired_normal_force: float = 10.0
    minimum_contact_force: float = 2.5
    maximum_safe_force: float = 35.0
    maximum_force_imbalance: float = 8.0
    admittance_gain: float = 0.0015       # [m/(N*s)]
    maximum_compression_rate: float = 0.015  # [m/s]
    force_filter_alpha: float = 0.15

    # Hold success/failure thresholds.
    maximum_box_drop: float = 0.035
    maximum_box_lateral_drift: float = 0.040

    # Cartesian impedance gains.
    tangential_stiffness: float = 480.0
    normal_stiffness: float = 180.0
    rotational_stiffness: float = 55.0
    tangential_damping: float = 85.0
    normal_damping: float = 45.0
    rotational_damping: float = 14.0


@dataclass
class DynamicSideSqueezeConfig(SideSqueezeConfig):
    """Configuration for intercepting a linearly launched ballistic box."""

    # Launch distribution. The box travels toward decreasing world-x. In the
    # default long-flight mode vz is derived so the ballistic arc crosses the
    # catch plane at ``target_catch_height``; the z entries below are ignored.
    random_seed: int = 7
    launch_position: tuple[float, float, float] = (0.820, 0.0, 0.150)
    launch_velocity_low: tuple[float, float, float] = (-1.47, -0.010, 0.0)
    launch_velocity_high: tuple[float, float, float] = (-1.35, 0.010, 0.0)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    simultaneous_start: bool = True
    # Tuned for the nominal box_half_y=0.150 (see SideSqueezeConfig.box_half_y
    # / generalization.py's NOMINAL_BOX_HALF_SIZE[1]). __post_init__ below
    # scales this by box_half_y/0.150 so a larger randomized box -- which
    # needs the arms to spread to a proportionally wider target separation
    # (pad_offset_scalar in main_acmpc_box_catch.py) -- gets proportionally
    # more flight-time margin to reach it before the fixed
    # interception-workspace deadline, instead of a bigger box always
    # getting the same tight window a small box was tuned for.
    target_flight_time_s: float | None = 0.56
    derive_vertical_launch_velocity: bool = True
    target_catch_height: float = 0.800

    # Ready pose and face interception.
    prepare_s: float = 1.60
    catch_plane_x: float = 0.300
    minimum_intercept_ttc: float = 0.015
    maximum_intercept_ttc: float = 0.85
    maximum_pad_reach_speed: float = 0.80
    maximum_pad_reach_acceleration: float = 4.0
    interception_candidate_count: int = 24
    interception_lock_delay_s: float = 0.05
    minimum_reachability_margin: float = -0.015
    # Reachability is checked against the impedance-controller's effective
    # response lag, not an instant-start kinematic bound. Without this, the
    # planner can call a target "reachable" that the compliant pad tracking
    # cannot actually catch up to before the box arrives.
    pad_tracking_delay_s: float = 0.05
    # Bounded per-episode bias applied to the locked interception time
    # (seconds). Left at 0.0 by engineered defaults; PPO nudges it via
    # ``apply_cost_action``.
    catch_ttc_offset_s: float = 0.0
    preferred_catch_height: float = 0.80
    minimum_catch_z: float = 0.68
    maximum_catch_z: float = 0.98
    predictor_velocity_alpha: float = 0.35
    predictor_max_speed: float = 4.5
    squeeze_lead_s: float = 0.080

    # Dynamic force control overrides the slower stationary defaults.
    precontact_gap: float = 0.006
    desired_normal_force: float = 12.0
    minimum_contact_force: float = 0.75
    maximum_safe_force: float = 32.0
    maximum_force_imbalance: float = 10.0
    admittance_gain: float = 0.008
    maximum_compression_rate: float = 0.080
    tangential_stiffness: float = 700.0
    tangential_damping: float = 140.0
    normal_stiffness: float = 800.0
    normal_damping: float = 100.0

    # TTC-based compliance and impact limiting.
    ttc_soften_window_s: float = 0.18
    # A ballistic catch's vertical fall rate keeps increasing right up to
    # contact (~1.5+ m/s by the time the box arrives). Softening tangential
    # stiffness this much (was 85, ~8x below nominal) starting a full
    # ttc_soften_window_s before predicted contact left the pad unable to
    # track that closing rate: it fell ~1.5 m/s behind the box by the time of
    # touch, producing a glancing, asymmetric first contact that spun the box
    # up violently. 300 keeps meaningfully more tracking authority through
    # the approach while still well below the nominal 700 N/m.
    impact_tangential_stiffness: float = 300.0
    impact_normal_stiffness: float = 45.0
    # A perfectly simultaneous bilateral touch is rare: one pad typically
    # registers contact a step or two before the other. If EE rotational
    # impedance stays at its rigid post-capture value through that window,
    # the stiff orientation tracking fights the box's residual spin from the
    # one-sided touch and can pump angular velocity up sharply before contact
    # is lost entirely. Soften it during first contact like the linear axes.
    impact_rotational_stiffness: float = 6.0
    minimum_adaptive_impact_rotational_stiffness: float = 3.0
    maximum_adaptive_impact_rotational_stiffness: float = 30.0
    impact_desired_force: float = 4.0
    first_contact_window_s: float = 0.12
    first_contact_force_limit: float = 18.0
    emergency_contact_force: float = 36.0
    impact_relief_gain: float = 0.0012  # [m/N]
    maximum_impact_relief: float = 0.012
    predictive_force_guard_ratio: float = 0.78
    force_prediction_horizon_s: float = 0.025
    relative_normal_speed_limit: float = 0.30
    expected_contact_deflection: float = 0.010
    minimum_adaptive_impact_stiffness: float = 30.0
    maximum_adaptive_impact_stiffness: float = 180.0
    maximum_adaptive_impact_force: float = 8.0
    hold_contact_time_constant_s: float = 0.012
    hold_contact_damping_ratio: float = 1.0
    hold_contact_transition_s: float = 0.50

    # Capture acceptance.
    capture_stable_s: float = 0.04
    required_dynamic_hold_s: float = 5.00
    maximum_capture_speed: float = 0.18
    maximum_capture_drop: float = 0.060
    maximum_capture_drift: float = 0.150
    # Minimum-force hold control. 4.9 is calibrated from the 5 s no-creep
    # threshold in the compliant MuJoCo broad-pad contact (10 N/pad for the
    # nominal 0.5 kg, mu=1.2 box; the ideal rigid-contact bound is 2.04 N).
    minimum_hold_force_calibration: float = 4.9
    maximum_hold_normal_force: float = 24.0
    # Absent a genuine detected slip, the adaptive hold force is capped near
    # the computed minimum no-slip force rather than being allowed to creep
    # all the way to maximum_hold_normal_force. Real slip (measured
    # tangential force) can still push the request above this ceiling, up to
    # the hard maximum_hold_normal_force safety limit.
    hold_force_margin_ratio: float = 1.6
    hold_slip_speed_threshold: float = 0.005
    hold_force_increase_rate: float = 10.0
    hold_force_release_rate: float = 2.0
    hold_slip_recovery_gain: float = 6.0
    maximum_hold_recovery_offset: float = 0.080
    # Capture stabilization.  A moving reference absorbs the remaining object
    # twist instead of snapping to the first-touch pose.
    capture_linear_time_constant_s: float = 0.18
    capture_angular_time_constant_s: float = 0.15
    single_contact_timeout_s: float = 0.20
    maximum_contact_slip_speed: float = 0.08
    hold_instability_grace_s: float = 0.025
    # Includes the ballistic approach, capture transient, and five-second hold.
    timeout_s: float = 7.50

    def __post_init__(self) -> None:
        # No-op at the nominal box_half_y (ratio == 1.0) -- every existing
        # caller that never randomizes box_half_y keeps exactly 0.56s.
        if self.target_flight_time_s is not None:
            self.target_flight_time_s = self.target_flight_time_s * (self.box_half_y / 0.150)


@dataclass
class RotatingSideSqueezeConfig(DynamicSideSqueezeConfig):
    """Configuration for phase-3 rotating-box SE(3) stabilization."""

    minimum_hold_force_calibration: float = 5.5
    random_angular_velocity_low: tuple[float, float, float] = (-0.005, -0.005, -0.025)
    random_angular_velocity_high: tuple[float, float, float] = (0.005, 0.005, 0.025)
    predictor_angular_velocity_alpha: float = 0.35
    maximum_predicted_angular_speed: float = 2.0
    rotational_stiffness: float = 40.0
    rotational_damping: float = 20.0

    box_mass: float = 0.50
    linear_wrench_damping: float = 5.0
    angular_wrench_damping: float = 0.22
    nominal_qp_normal_force: float = 5.0
    maximum_qp_normal_force: float = 16.0
    wrench_friction_coefficient: float = 0.90
    maximum_contact_moment: float = 0.18
    wrench_feedforward_scale: float = 0.002
    wrench_normal_force_blend: float = 0.05
    wrench_tracking_weight: float = 8.0
    angular_wrench_weight: float = 16.0
    wrench_regularization: float = 0.04
    slip_cost_weight: float = 0.30
    angular_velocity_cost_weight: float = 2.0
    linear_velocity_cost_weight: float = 1.0
    wrench_rate_weight: float = 0.12
    capture_angular_weight_scale: float = 3.0
    qp_prediction_dt: float = 0.010
    torsional_friction_coefficient: float = 0.012

    wrench_position_compliance: float = 0.0
    wrench_rotation_compliance: float = 0.0
    maximum_wrench_position_offset: float = 0.018
    maximum_wrench_rotation_offset: float = 0.10
    maximum_capture_angular_speed: float = 0.25
    maximum_hold_slip_ratio: float = 0.95
