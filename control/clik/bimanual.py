"""Object-centric bimanual target and relative-pose helpers."""

from dataclasses import dataclass

import mujoco
import numpy as np

from control.clik.config import FFWDemoConfig, Level3AcceptanceConfig
from control.clik.jacobian import compute_full_body_jacobian
from control.clik.kinematics import compute_pose_error, get_ee_transform, make_transform
from control.clik.orientation import rotation_from_axis_alignment
from control.clik.qp import (
    NullSpaceQPConfig,
    NullSpaceQPController,
    NullSpaceQPResult,
)
from control.clik.solvers import (
    damped_least_squares,
    damped_pseudoinverse,
)
from control.clik.types import SerialArm
from control.clik.utils import matrix_log_se3, skew, transform_inverse


_NULLSPACE_QP_CONTROLLERS: dict[
    tuple[float, float, float, float, float, int, int],
    NullSpaceQPController,
] = {}


def _get_nullspace_qp_controller(
    n_dof: int,
    config: NullSpaceQPConfig,
) -> NullSpaceQPController:
    """Reuse one OSQP workspace for each fixed controller configuration."""

    key = (
        float(config.posture_weight),
        float(config.manipulability_weight),
        float(config.damping_weight),
        float(config.eps_abs),
        float(config.eps_rel),
        int(config.max_iter),
        int(n_dof),
    )
    controller = _NULLSPACE_QP_CONTROLLERS.get(key)
    if controller is None:
        controller = NullSpaceQPController(n_dof, config)
        _NULLSPACE_QP_CONTROLLERS[key] = controller
    return controller


@dataclass(frozen=True)
class BimanualTargets:
    """Desired virtual object frame and fixed left/right EE target poses."""

    object_center: np.ndarray
    object_transform: np.ndarray
    left_transform: np.ndarray
    right_transform: np.ndarray


@dataclass(frozen=True)
class RelativePoseState:
    """Current bimanual relative transform and SE(3) constraint error."""

    left_transform: np.ndarray
    right_transform: np.ndarray
    relative_transform: np.ndarray
    relative_pose_error: np.ndarray
    translational_error_norm: float
    rotational_error_norm: float


@dataclass(frozen=True)
class RelativeOrientationState:
    """Relative hand orientation state and SO(3) constraint error."""

    relative_rotation: np.ndarray
    desired_relative_rotation: np.ndarray
    rotation_error_matrix: np.ndarray
    rotation_error: np.ndarray
    rotation_error_norm: float


@dataclass(frozen=True)
class RelativeAxisState:
    """Axis-only hand relation for selected local EE axes."""

    local_axis: np.ndarray
    left_axis_world: np.ndarray
    right_axis_world: np.ndarray
    left_desired_axis_world: np.ndarray
    right_desired_axis_world: np.ndarray
    axis_error: np.ndarray
    axis_error_norm: float
    axis_dot: float
    face_each_other: bool


@dataclass(frozen=True)
class BimanualTaskStack:
    """Stacked bimanual IK task ready for a DLS solve."""

    jacobian: np.ndarray
    error: np.ndarray
    object_error: np.ndarray
    relative_position_error: np.ndarray
    left_rotation_error: np.ndarray
    right_rotation_error: np.ndarray
    relative_orientation_state: RelativeOrientationState
    relative_axis_state: RelativeAxisState
    commanded_object_error: np.ndarray
    commanded_relative_position_error: np.ndarray
    commanded_relative_orientation_error: np.ndarray


@dataclass(frozen=True)
class NullspaceSecondaryState:
    """Secondary objectives projected into the primary task nullspace."""

    q_current: np.ndarray
    joint_limit_velocity: np.ndarray
    manipulability_velocity: np.ndarray
    secondary_velocity: np.ndarray
    projected_velocity: np.ndarray
    nullspace_projector: np.ndarray
    joint_limit_margin: np.ndarray
    secondary_velocity_norm: float
    projected_velocity_norm: float
    min_joint_limit_margin: float
    null_motion_scale: float = 1.0
    projected_velocity_before_scale: np.ndarray | None = None
    max_projected_velocity_norm: float | None = None


@dataclass(frozen=True)
class BimanualSolveResult:
    """Detailed stacked IK solve split into primary and nullspace motion."""

    joint_delta: np.ndarray
    task_delta: np.ndarray
    nullspace_delta: np.ndarray
    pseudoinverse: np.ndarray
    nullspace: NullspaceSecondaryState
    solver_mode: str = "dls_nullspace"
    qp_result: NullSpaceQPResult | None = None


@dataclass(frozen=True)
class ManipulabilityState:
    """Per-arm Yoshikawa manipulability and gradient information."""

    left: float
    right: float
    minimum: float
    left_gradient: np.ndarray
    right_gradient: np.ndarray
    stacked_gradient: np.ndarray
    velocity: np.ndarray
    velocity_norm: float
    left_condition_number: float
    right_condition_number: float
    adaptive_damping: float
    is_near_singularity: bool


@dataclass(frozen=True)
class Level3AcceptanceMetrics:
    """Level-3 grasp-consistency acceptance result for one control step."""

    passed: bool
    object_error_norm: float
    relative_position_error_norm: float
    relative_orientation_error_norm: float
    min_manipulability: float
    qp_task_residual_norm: float
    dq_was_clipped: bool
    command_clip_delta_norm: float
    failed_terms: tuple[str, ...]


def _is_qp_status_acceptable(status: str | None, require_qp: bool) -> bool:
    if status is None:
        return not require_qp
    status_lower = status.lower()
    return (
        status_lower.startswith("solved")
        or "accepted feasible iterate" in status_lower
    )


def evaluate_level3_acceptance(
    *,
    object_error_norm: float,
    relative_position_error_norm: float,
    relative_orientation_error_norm: float,
    min_manipulability: float,
    qp_task_residual_norm: float | None,
    qp_status: str | None,
    dq_was_clipped: bool,
    command_clip_delta_norm: float,
    config: Level3AcceptanceConfig,
) -> Level3AcceptanceMetrics:
    """Evaluate whether the current step satisfies Level-3 criteria."""

    qp_residual = (
        float("nan")
        if qp_task_residual_norm is None
        else float(qp_task_residual_norm)
    )
    if not config.enabled:
        return Level3AcceptanceMetrics(
            passed=True,
            object_error_norm=float(object_error_norm),
            relative_position_error_norm=float(relative_position_error_norm),
            relative_orientation_error_norm=float(relative_orientation_error_norm),
            min_manipulability=float(min_manipulability),
            qp_task_residual_norm=qp_residual,
            dq_was_clipped=bool(dq_was_clipped),
            command_clip_delta_norm=float(command_clip_delta_norm),
            failed_terms=(),
        )

    failed_terms: list[str] = []

    if float(object_error_norm) > config.object_error_max:
        failed_terms.append("object_error")
    if float(relative_position_error_norm) > config.relative_position_error_max:
        failed_terms.append("relative_position_error")
    if float(relative_orientation_error_norm) > config.relative_orientation_error_max:
        failed_terms.append("relative_orientation_error")
    if float(min_manipulability) < config.min_manipulability_min:
        failed_terms.append("min_manipulability")
    if not _is_qp_status_acceptable(qp_status, config.require_qp):
        failed_terms.append("qp_status")
    if config.require_qp and (
        not np.isfinite(qp_residual)
        or qp_residual > config.qp_task_residual_max
    ):
        failed_terms.append("qp_task_residual")
    if bool(dq_was_clipped) and not config.allow_dq_clipping:
        failed_terms.append("dq_clipping")
    if float(command_clip_delta_norm) > config.command_clip_delta_max:
        failed_terms.append("command_clipping")

    return Level3AcceptanceMetrics(
        passed=not failed_terms,
        object_error_norm=float(object_error_norm),
        relative_position_error_norm=float(relative_position_error_norm),
        relative_orientation_error_norm=float(relative_orientation_error_norm),
        min_manipulability=float(min_manipulability),
        qp_task_residual_norm=qp_residual,
        dq_was_clipped=bool(dq_was_clipped),
        command_clip_delta_norm=float(command_clip_delta_norm),
        failed_terms=tuple(failed_terms),
    )


def _limit_vector_norm(vector: np.ndarray, max_norm: float | None) -> np.ndarray:
    """Return vector scaled to max_norm without changing its direction."""

    vector = np.asarray(vector, dtype=float).reshape(-1)
    if max_norm is None or max_norm <= 0.0:
        return vector

    norm = np.linalg.norm(vector)
    if norm <= max_norm or norm < 1e-12:
        return vector
    return vector * (float(max_norm) / norm)


def _null_motion_error_scale(
    error_norm: float,
    start: float | None,
    stop: float | None,
) -> float:
    """Fade out null-space motion when primary task error grows."""

    if start is None or stop is None:
        return 1.0
    start = float(start)
    stop = float(stop)
    if stop <= start:
        return 1.0 if float(error_norm) <= start else 0.0
    error_norm = float(error_norm)
    if error_norm <= start:
        return 1.0
    if error_norm >= stop:
        return 0.0
    return float((stop - error_norm) / (stop - start))


def geometric_jacobian_world(
    position_jacobian: np.ndarray,
    angular_jacobian: np.ndarray,
) -> np.ndarray:
    """Return a 6xn world-frame EE Jacobian ordered as [angular; linear]."""

    return np.vstack([angular_jacobian, position_jacobian])


def yoshikawa_manipulability(jacobian: np.ndarray) -> float:
    """
    Return Yoshikawa manipulability w(q) = sqrt(det(J J.T)).

    Numerically this is computed as the product of singular values, which is
    equivalent for a full-row-rank task Jacobian and more stable near singular
    configurations where det(JJ.T) becomes tiny.
    """

    singular_values = np.linalg.svd(np.asarray(jacobian, dtype=float), compute_uv=False)
    return float(np.prod(np.maximum(singular_values, 0.0)))


def jacobian_condition_number(jacobian: np.ndarray, eps: float = 1e-12) -> float:
    """Return a finite condition number for a task Jacobian."""

    singular_values = np.linalg.svd(np.asarray(jacobian, dtype=float), compute_uv=False)
    if singular_values.size == 0:
        return float("inf")
    smallest = singular_values[-1]
    if smallest < eps:
        return float("inf")
    return float(singular_values[0] / smallest)


def arm_manipulability(model: mujoco.MjModel, data: mujoco.MjData, arm: SerialArm) -> float:
    """Compute Yoshikawa manipulability for one arm's EE Jacobian."""

    position_jacobian, angular_jacobian = _ee_jacobians_world(model, data, arm)
    return yoshikawa_manipulability(
        geometric_jacobian_world(position_jacobian, angular_jacobian)
    )


def manipulability_gradient(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
    step: float,
) -> np.ndarray:
    """Finite-difference gradient of one arm's manipulability."""

    step = float(step)
    if step <= 0.0:
        raise ValueError("manipulability gradient step must be positive.")

    qpos_saved = data.qpos.copy()
    qvel_saved = data.qvel.copy()
    gradient = np.zeros(arm.dof)

    for local_index, qpos_index in enumerate(arm.qpos_indices):
        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        data.qpos[qpos_index] += step
        mujoco.mj_forward(model, data)
        plus = arm_manipulability(model, data, arm)

        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        data.qpos[qpos_index] -= step
        mujoco.mj_forward(model, data)
        minus = arm_manipulability(model, data, arm)

        gradient[local_index] = (plus - minus) / (2.0 * step)

    data.qpos[:] = qpos_saved
    data.qvel[:] = qvel_saved
    mujoco.mj_forward(model, data)
    return gradient


def adaptive_manipulability_damping(
    base_damping: float,
    manipulability_min: float,
    alpha: float,
    epsilon: float,
    max_damping: float | None = None,
) -> float:
    """Return lambda = lambda0 + alpha / (w + epsilon), optionally capped."""

    damping = float(base_damping) + float(alpha) / (
        float(manipulability_min) + float(epsilon)
    )
    if max_damping is not None and max_damping > 0.0:
        damping = min(damping, float(max_damping))
    return damping


def _compute_manipulability_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_arm: SerialArm,
    right_arm: SerialArm,
    left_geometric_jacobian: np.ndarray,
    right_geometric_jacobian: np.ndarray,
    *,
    damping: float,
    manipulability_gain: float,
    manipulability_gradient_step: float,
    max_manipulability_command: float | None,
    singularity_warning_threshold: float,
    adaptive_damping_alpha: float,
    adaptive_damping_epsilon: float,
    adaptive_damping_max: float | None,
) -> "ManipulabilityState":
    """Manipulability, its gradient, and the resulting adaptive damping/secondary velocity."""

    left_manipulability = yoshikawa_manipulability(left_geometric_jacobian)
    right_manipulability = yoshikawa_manipulability(right_geometric_jacobian)
    min_manipulability = min(left_manipulability, right_manipulability)
    adaptive_damping = adaptive_manipulability_damping(
        base_damping=damping,
        manipulability_min=min_manipulability,
        alpha=adaptive_damping_alpha,
        epsilon=adaptive_damping_epsilon,
        max_damping=adaptive_damping_max,
    )
    is_near_singularity = min_manipulability < singularity_warning_threshold

    if manipulability_gain > 0.0:
        left_manip_gradient = manipulability_gradient(
            model,
            data,
            left_arm,
            manipulability_gradient_step,
        )
        right_manip_gradient = manipulability_gradient(
            model,
            data,
            right_arm,
            manipulability_gradient_step,
        )
    else:
        left_manip_gradient = np.zeros(left_arm.dof)
        right_manip_gradient = np.zeros(right_arm.dof)
    stacked_manip_gradient = np.concatenate(
        [left_manip_gradient, right_manip_gradient]
    )
    manipulability_velocity = _limit_vector_norm(
        float(manipulability_gain) * stacked_manip_gradient,
        max_manipulability_command,
    )
    return ManipulabilityState(
        left=left_manipulability,
        right=right_manipulability,
        minimum=min_manipulability,
        left_gradient=left_manip_gradient,
        right_gradient=right_manip_gradient,
        stacked_gradient=stacked_manip_gradient,
        velocity=manipulability_velocity,
        velocity_norm=float(np.linalg.norm(manipulability_velocity)),
        left_condition_number=jacobian_condition_number(left_geometric_jacobian),
        right_condition_number=jacobian_condition_number(right_geometric_jacobian),
        adaptive_damping=adaptive_damping,
        is_near_singularity=is_near_singularity,
    )


def joint_limit_margin(
    q_current: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> np.ndarray:
    """Return each joint's signed distance to its nearest position limit."""

    q_current = np.asarray(q_current, dtype=float).reshape(-1)
    q_low = np.asarray(q_low, dtype=float).reshape(q_current.shape)
    q_high = np.asarray(q_high, dtype=float).reshape(q_current.shape)
    return np.minimum(q_current - q_low, q_high - q_current)


def joint_limit_barrier_velocity(
    q_current: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    gain: float,
    activation_margin: float,
) -> np.ndarray:
    """
    Return a secondary velocity that pushes joints away from nearby limits.

    This is intentionally simple and optional.  With gain=0 it contributes
    nothing, but the function makes the later barrier-term extension explicit.
    """

    q_current = np.asarray(q_current, dtype=float).reshape(-1)
    q_low = np.asarray(q_low, dtype=float).reshape(q_current.shape)
    q_high = np.asarray(q_high, dtype=float).reshape(q_current.shape)
    gain = float(gain)
    activation_margin = float(activation_margin)
    if gain <= 0.0 or activation_margin <= 0.0:
        return np.zeros_like(q_current)

    lower_margin = q_current - q_low
    upper_margin = q_high - q_current
    velocity = np.zeros_like(q_current)

    lower_active = lower_margin < activation_margin
    upper_active = upper_margin < activation_margin
    velocity[lower_active] += (
        gain * (activation_margin - lower_margin[lower_active]) / activation_margin
    )
    velocity[upper_active] -= (
        gain * (activation_margin - upper_margin[upper_active]) / activation_margin
    )
    return velocity


def build_nullspace_secondary_state(
    jacobian_pinv: np.ndarray,
    jacobian: np.ndarray,
    q_current: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    joint_limit_gain: float = 0.0,
    joint_limit_activation_margin: float = 0.20,
    manipulability_velocity: np.ndarray | None = None,
) -> NullspaceSecondaryState:
    """Build qdot_null = N(qdot_joint_limit + qdot_manip)."""

    q_current = np.asarray(q_current, dtype=float).reshape(-1)
    q_low = np.asarray(q_low, dtype=float).reshape(q_current.shape)
    q_high = np.asarray(q_high, dtype=float).reshape(q_current.shape)

    nullspace_projector = np.eye(q_current.size) - jacobian_pinv @ jacobian
    limit_velocity = joint_limit_barrier_velocity(
        q_current,
        q_low,
        q_high,
        joint_limit_gain,
        joint_limit_activation_margin,
    )
    if manipulability_velocity is None:
        manipulability_velocity = np.zeros_like(q_current)
    else:
        manipulability_velocity = np.asarray(
            manipulability_velocity, dtype=float
        ).reshape(q_current.shape)
    secondary_velocity = limit_velocity + manipulability_velocity
    projected_velocity = nullspace_projector @ secondary_velocity
    margin = joint_limit_margin(q_current, q_low, q_high)

    return NullspaceSecondaryState(
        q_current=q_current,
        joint_limit_velocity=limit_velocity,
        manipulability_velocity=manipulability_velocity,
        secondary_velocity=secondary_velocity,
        projected_velocity=projected_velocity,
        nullspace_projector=nullspace_projector,
        joint_limit_margin=margin,
        secondary_velocity_norm=float(np.linalg.norm(secondary_velocity)),
        projected_velocity_norm=float(np.linalg.norm(projected_velocity)),
        min_joint_limit_margin=float(np.min(margin)),
    )


def gripper_local_axes_for_arm(
    config: FFWDemoConfig,
    arm_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the configured local forward/up axes for one gripper."""

    if arm_name == "left":
        return config.left_gripper_forward_axis_local, config.left_gripper_up_axis_local
    if arm_name == "right":
        return config.right_gripper_forward_axis_local, config.right_gripper_up_axis_local
    return config.gripper_forward_axis_local, config.gripper_up_axis_local


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Return the SO(3) logarithm as an axis-angle rotation vector."""

    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    return matrix_log_se3(transform)[:3]


def object_center_at_time(config: FFWDemoConfig, t: float) -> np.ndarray:
    """Return the shared moving object-center target in world coordinates."""

    phase = 2.0 * np.pi * config.target_frequency_hz * t

    if config.target_trajectory == "circle":
        offset = config.target_circle_radius * np.array(
            [np.cos(phase), 0.0, np.sin(phase)]
        )
    elif config.target_trajectory == "sine":
        offset = np.array([
            config.target_sine_x_amplitude * np.sin(phase),
            0.0,
            config.target_sine_z_amplitude * np.sin(2.0 * phase),
        ])
    else:
        raise ValueError(f"Unknown target trajectory: {config.target_trajectory}")

    return config.object_center + offset


def object_rotation_world(config: FFWDemoConfig) -> np.ndarray:
    """Return the fixed world orientation of the virtual object frame."""

    return np.asarray(config.object_rotation_world, dtype=float).reshape(3, 3)


def object_transform_at_time(config: FFWDemoConfig, t: float) -> np.ndarray:
    """Return the desired virtual object transform in the world frame."""

    return make_transform(object_center_at_time(config, t), object_rotation_world(config))


def hand_target_from_object(
    object_transform: np.ndarray,
    hand_offset_object: np.ndarray,
    hand_rotation_world: np.ndarray,
) -> np.ndarray:
    """Build one EE target from a virtual object frame and grasp offset."""

    object_transform = np.asarray(object_transform, dtype=float).reshape(4, 4)
    hand_offset_object = np.asarray(hand_offset_object, dtype=float).reshape(3)
    hand_position = (
        object_transform[:3, 3] + object_transform[:3, :3] @ hand_offset_object
    )
    return make_transform(hand_position, hand_rotation_world)


def fixed_grasp_rotations(config: FFWDemoConfig) -> dict[str, np.ndarray]:
    """
    Build fixed hand orientations for the bimanual grasp.

    The local gripper axes and world-up constraint reuse the existing
    axis-alignment orientation definition. For the bimanual grasp, each hand's
    forward axis points toward the object center, so the two end effectors face
    each other while the desired orientations remain fixed during motion.
    """

    rotations = {}
    for arm_name, hand_offset in config.hand_offsets.items():
        local_forward, local_up = gripper_local_axes_for_arm(config, arm_name)

        if config.bimanual_face_object_center:
            world_forward = -np.asarray(hand_offset, dtype=float).reshape(3)
        else:
            world_forward = config.gripper_forward_axis_world

        world_forward = world_forward + config.gripper_forward_axis_world_tilt
        world_forward_norm = np.linalg.norm(world_forward)
        if world_forward_norm < 1e-12:
            raise ValueError(f"{arm_name} world forward axis must be nonzero.")
        world_forward = world_forward / world_forward_norm
        world_up = config.gripper_up_axis_world + config.gripper_up_axis_world_tilt

        rotations[arm_name] = rotation_from_axis_alignment(
            local_primary_axis=local_forward,
            world_primary_axis=world_forward,
            local_secondary_axis=local_up,
            world_secondary_axis=world_up,
        )

    return rotations


def bimanual_targets_at_time(
    config: FFWDemoConfig,
    t: float,
    fixed_rotations: dict[str, np.ndarray],
) -> BimanualTargets:
    """Generate left/right EE targets from one shared virtual object frame."""

    object_transform = object_transform_at_time(config, t)
    object_center = object_transform[:3, 3]

    return BimanualTargets(
        object_center=object_center,
        object_transform=object_transform,
        left_transform=hand_target_from_object(
            object_transform,
            config.hand_offsets["left"],
            fixed_rotations["left"],
        ),
        right_transform=hand_target_from_object(
            object_transform,
            config.hand_offsets["right"],
            fixed_rotations["right"],
        ),
    )


def desired_relative_transform(targets: BimanualTargets) -> np.ndarray:
    """Return T_rel_desired = inv(T_left_desired) @ T_right_desired."""

    return transform_inverse(targets.left_transform) @ targets.right_transform


def desired_relative_rotation(
    left_transform: np.ndarray,
    right_transform: np.ndarray,
) -> np.ndarray:
    """Return R_rel_desired = R_left.T @ R_right for the grasp relation."""

    left_rotation = np.asarray(left_transform, dtype=float).reshape(4, 4)[:3, :3]
    right_rotation = np.asarray(right_transform, dtype=float).reshape(4, 4)[:3, :3]
    return left_rotation.T @ right_rotation


def relative_orientation_state(
    left_transform: np.ndarray,
    right_transform: np.ndarray,
    relative_rotation_desired: np.ndarray,
) -> RelativeOrientationState:
    """
    Compute relative orientation error between both hands.

    R_rel_current = R_left.T @ R_right
    R_err = R_rel_current.T @ R_rel_desired
    e_rel_rot = log(R_err)
    """

    left_rotation = np.asarray(left_transform, dtype=float).reshape(4, 4)[:3, :3]
    right_rotation = np.asarray(right_transform, dtype=float).reshape(4, 4)[:3, :3]
    relative_rotation_desired = np.asarray(
        relative_rotation_desired, dtype=float
    ).reshape(3, 3)

    relative_rotation = left_rotation.T @ right_rotation
    rotation_error_matrix = relative_rotation.T @ relative_rotation_desired
    rotation_error = so3_log(rotation_error_matrix)

    return RelativeOrientationState(
        relative_rotation=relative_rotation,
        desired_relative_rotation=relative_rotation_desired.copy(),
        rotation_error_matrix=rotation_error_matrix,
        rotation_error=rotation_error,
        rotation_error_norm=float(np.linalg.norm(rotation_error)),
    )


def relative_local_axis_state(
    left_transform: np.ndarray,
    right_transform: np.ndarray,
    local_axis: np.ndarray,
    face_each_other: bool = False,
) -> RelativeAxisState:
    """
    Compute an axis-only grasp relation for a selected local EE axis.

    With face_each_other=False, the desired relation is anti-parallel only:
        axis_left_world + axis_right_world = 0

    With face_each_other=True, the desired relation also includes direction:
        axis_left_world  points from the left hand to the right hand
        axis_right_world points from the right hand to the left hand

    That second mode distinguishes "-X axes face each other" from "+X axes
    face each other"; both look identical to a pure anti-parallel constraint.
    """

    left_transform = np.asarray(left_transform, dtype=float).reshape(4, 4)
    right_transform = np.asarray(right_transform, dtype=float).reshape(4, 4)
    left_rotation = left_transform[:3, :3]
    right_rotation = right_transform[:3, :3]
    local_axis = np.asarray(local_axis, dtype=float).reshape(3)
    local_axis_norm = np.linalg.norm(local_axis)
    if local_axis_norm < 1e-12:
        raise ValueError("relative local axis must be nonzero.")
    local_axis = local_axis / local_axis_norm
    left_axis = left_rotation @ local_axis
    right_axis = right_rotation @ local_axis
    if face_each_other:
        hand_direction = right_transform[:3, 3] - left_transform[:3, 3]
        hand_direction_norm = np.linalg.norm(hand_direction)
        if hand_direction_norm < 1e-12:
            raise ValueError("Cannot build facing axis constraint for coincident hands.")
        left_desired_axis = hand_direction / hand_direction_norm
        right_desired_axis = -left_desired_axis
        axis_error = np.concatenate(
            [
                left_axis - left_desired_axis,
                right_axis - right_desired_axis,
            ]
        )
    else:
        left_desired_axis = -right_axis
        right_desired_axis = -left_axis
        axis_error = left_axis + right_axis

    return RelativeAxisState(
        local_axis=local_axis.copy(),
        left_axis_world=left_axis,
        right_axis_world=right_axis,
        left_desired_axis_world=left_desired_axis.copy(),
        right_desired_axis_world=right_desired_axis.copy(),
        axis_error=axis_error,
        axis_error_norm=float(np.linalg.norm(axis_error)),
        axis_dot=float(np.dot(left_axis, right_axis)),
        face_each_other=bool(face_each_other),
    )


def relative_z_axis_state(
    left_transform: np.ndarray,
    right_transform: np.ndarray,
) -> RelativeAxisState:
    """Backward-compatible helper for the old EE +Z anti-parallel relation."""

    return relative_local_axis_state(
        left_transform,
        right_transform,
        np.array([0.0, 0.0, 1.0]),
    )


def relative_pose_state(
    left_transform: np.ndarray,
    right_transform: np.ndarray,
    relative_transform_desired: np.ndarray,
) -> RelativePoseState:
    """
    Compute the current bimanual relative transform and SE(3) error.

    T_rel_current = inv(T_left) @ T_right
    T_err = inv(T_rel_current) @ T_rel_desired
    e_rel = log(T_err)
    """

    left_transform = np.asarray(left_transform, dtype=float).reshape(4, 4)
    right_transform = np.asarray(right_transform, dtype=float).reshape(4, 4)
    relative_transform_desired = np.asarray(
        relative_transform_desired, dtype=float
    ).reshape(4, 4)

    relative_transform = transform_inverse(left_transform) @ right_transform
    transform_error = transform_inverse(relative_transform) @ relative_transform_desired
    relative_pose_error = matrix_log_se3(transform_error)

    return RelativePoseState(
        left_transform=left_transform.copy(),
        right_transform=right_transform.copy(),
        relative_transform=relative_transform,
        relative_pose_error=relative_pose_error,
        translational_error_norm=float(np.linalg.norm(relative_pose_error[3:])),
        rotational_error_norm=float(np.linalg.norm(relative_pose_error[:3])),
    )


def _ee_jacobians_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world-frame EE position and angular Jacobians for one arm."""

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    if arm.ee_site_id is not None:
        mujoco.mj_jacSite(model, data, jacp, jacr, arm.ee_site_id)
    else:
        mujoco.mj_jacBody(model, data, jacp, jacr, arm.ee_body_id)
    return jacp[:, arm.qvel_indices].copy(), jacr[:, arm.qvel_indices].copy()


def _apply_arm_joint_delta(
    data: mujoco.MjData,
    arm: SerialArm,
    joint_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one arm's joint increment through position actuators."""

    q_current = data.qpos[arm.qpos_indices].copy()
    q_command_unclipped = q_current + joint_delta
    q_command = np.clip(q_command_unclipped, arm.ctrl_low, arm.ctrl_high)
    data.ctrl[arm.actuator_ids] = q_command
    return q_command, q_command_unclipped


def build_bimanual_task_stack(
    object_jacobian: np.ndarray,
    relative_position_jacobian: np.ndarray,
    relative_orientation_jacobian: np.ndarray,
    object_error: np.ndarray,
    relative_position_error: np.ndarray,
    relative_orientation: RelativeOrientationState,
    relative_axis: RelativeAxisState,
    left_rotation_error: np.ndarray,
    right_rotation_error: np.ndarray,
    position_gain: np.ndarray,
    rotation_gain: np.ndarray,
    relative_orientation_weight: float,
    orientation_task_weight: float,
    max_object_command: float | None = None,
    max_relative_position_command: float | None = None,
    max_relative_rotation_command: float | None = None,
    left_rotation_jacobian: np.ndarray | None = None,
    right_rotation_jacobian: np.ndarray | None = None,
) -> BimanualTaskStack:
    """Build the object/relative-orientation stacked IK task."""

    commanded_object_error = _limit_vector_norm(
        position_gain * object_error,
        max_object_command,
    )
    commanded_relative_position_error = _limit_vector_norm(
        position_gain * relative_position_error,
        max_relative_position_command,
    )
    if relative_axis.axis_error.shape[0] == rotation_gain.shape[0]:
        relative_axis_gain = rotation_gain
    else:
        repeats = int(np.ceil(relative_axis.axis_error.shape[0] / rotation_gain.shape[0]))
        relative_axis_gain = np.tile(rotation_gain, repeats)[
            : relative_axis.axis_error.shape[0]
        ]
    commanded_relative_orientation_error = _limit_vector_norm(
        -relative_axis_gain * relative_axis.axis_error,
        max_relative_rotation_command,
    )

    task_jacobians = [object_jacobian, relative_position_jacobian]
    task_errors = [
        commanded_object_error,
        commanded_relative_position_error,
    ]

    if relative_orientation_weight > 0.0:
        task_jacobians.append(
            relative_orientation_weight * relative_orientation_jacobian
        )
        task_errors.append(
            relative_orientation_weight * commanded_relative_orientation_error
        )

    if orientation_task_weight > 0.0:
        if left_rotation_jacobian is None or right_rotation_jacobian is None:
            raise ValueError("Absolute orientation rows require both rotation Jacobians.")
        commanded_left_rotation_error = _limit_vector_norm(
            rotation_gain * left_rotation_error,
            max_relative_rotation_command,
        )
        commanded_right_rotation_error = _limit_vector_norm(
            rotation_gain * right_rotation_error,
            max_relative_rotation_command,
        )
        task_jacobians.extend(
            [
                orientation_task_weight * left_rotation_jacobian,
                orientation_task_weight * right_rotation_jacobian,
            ]
        )
        task_errors.extend(
            [
                orientation_task_weight * commanded_left_rotation_error,
                orientation_task_weight * commanded_right_rotation_error,
            ]
        )

    return BimanualTaskStack(
        jacobian=np.vstack(task_jacobians),
        error=np.concatenate(task_errors),
        object_error=object_error,
        relative_position_error=relative_position_error,
        left_rotation_error=left_rotation_error,
        right_rotation_error=right_rotation_error,
        relative_orientation_state=relative_orientation,
        relative_axis_state=relative_axis,
        commanded_object_error=commanded_object_error,
        commanded_relative_position_error=commanded_relative_position_error,
        commanded_relative_orientation_error=commanded_relative_orientation_error,
    )


def solve_bimanual_task_stack(
    task_stack: BimanualTaskStack,
    damping: float,
) -> np.ndarray:
    """Solve one stacked bimanual IK system with damped least squares."""

    return damped_least_squares(task_stack.jacobian, task_stack.error, damping)


def solve_bimanual_task_stack_with_nullspace(
    task_stack: BimanualTaskStack,
    damping: float,
    q_current: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    joint_limit_gain: float = 0.0,
    joint_limit_activation_margin: float = 0.20,
    manipulability_velocity: np.ndarray | None = None,
) -> BimanualSolveResult:
    """
    Solve dq = J^+ e + (I - J^+ J) qdot_secondary.

    The primary task handles object/relative constraints. The secondary task
    uses redundant 14-DOF bimanual motion for joint-limit avoidance and
    manipulability improvement.
    """

    jacobian_pinv = damped_pseudoinverse(task_stack.jacobian, damping)
    task_delta = jacobian_pinv @ task_stack.error
    nullspace = build_nullspace_secondary_state(
        jacobian_pinv=jacobian_pinv,
        jacobian=task_stack.jacobian,
        q_current=q_current,
        q_low=q_low,
        q_high=q_high,
        joint_limit_gain=joint_limit_gain,
        joint_limit_activation_margin=joint_limit_activation_margin,
        manipulability_velocity=manipulability_velocity,
    )
    joint_delta = task_delta + nullspace.projected_velocity

    return BimanualSolveResult(
        joint_delta=joint_delta,
        task_delta=task_delta,
        nullspace_delta=nullspace.projected_velocity,
        pseudoinverse=jacobian_pinv,
        nullspace=nullspace,
        solver_mode="dls_nullspace",
    )


def solve_bimanual_task_stack_with_qp(
    task_stack: BimanualTaskStack,
    q_current: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    qdot_low: np.ndarray,
    qdot_high: np.ndarray,
    dt: float,
    q_nominal: np.ndarray | None = None,
    posture_weight: float = 6.0,
    posture_weights: np.ndarray | None = None,
    joint_limit_velocity: np.ndarray | None = None,
    manipulability_gradient: np.ndarray | None = None,
    manipulability_weight: float = 0.0,
    regularization: float = 1e-4,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
    max_iter: int = 200,
    max_null_velocity: float | None = None,
    null_error_norm: float | None = None,
    null_error_scale_start: float | None = None,
    null_error_scale_stop: float | None = None,
    damping_for_diagnostics: float = 1e-3,
) -> BimanualSolveResult:
    """
    Solve a null-space QP over z and return dq=(qdot_task + N z)*dt.

    The primary task is computed by DLS pseudo-inverse first. The QP only
    optimizes the redundant null-space component.
    """

    jacobian_pinv = damped_pseudoinverse(task_stack.jacobian, damping_for_diagnostics)
    task_delta_unscaled = jacobian_pinv @ task_stack.error
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive for null-space QP.")
    qdot_task_unscaled = task_delta_unscaled / dt
    nullspace_projector = np.eye(q_current.size) - jacobian_pinv @ task_stack.jacobian

    config = NullSpaceQPConfig(
        posture_weight=posture_weight,
        manipulability_weight=manipulability_weight,
        damping_weight=regularization,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
    )
    qp_controller = _get_nullspace_qp_controller(q_current.size, config)
    qp_result = qp_controller.solve(
        q_current=q_current,
        qdot_task=qdot_task_unscaled,
        nullspace_projector=nullspace_projector,
        dt=dt,
        q_min=q_low,
        q_max=q_high,
        qdot_min=qdot_low,
        qdot_max=qdot_high,
        q_nominal=q_nominal,
        posture_weights=posture_weights,
        joint_limit_velocity=joint_limit_velocity,
        manipulability_gradient=manipulability_gradient,
        posture_weight=posture_weight,
        manipulability_weight=manipulability_weight,
        damping_weight=regularization,
    )

    task_qdot = qp_result.qdot_task
    null_qdot_unscaled = nullspace_projector @ qp_result.z
    null_qdot_limited = _limit_vector_norm(null_qdot_unscaled, max_null_velocity)
    if null_error_norm is None:
        null_motion_scale = 1.0
    else:
        null_motion_scale = _null_motion_error_scale(
            null_error_norm,
            null_error_scale_start,
            null_error_scale_stop,
        )
    null_qdot = null_motion_scale * null_qdot_limited
    qdot = task_qdot + null_qdot

    task_delta = task_qdot * dt
    nullspace_delta = null_qdot * dt
    joint_delta = qdot * dt
    margin = joint_limit_margin(q_current, q_low, q_high)
    if joint_limit_velocity is None:
        joint_limit_velocity = np.zeros_like(q_current)
    else:
        joint_limit_velocity = np.asarray(
            joint_limit_velocity, dtype=float
        ).reshape(q_current.shape)
    nullspace = NullspaceSecondaryState(
        q_current=np.asarray(q_current, dtype=float).reshape(-1),
        joint_limit_velocity=joint_limit_velocity,
        manipulability_velocity=(
            np.zeros_like(q_current)
            if manipulability_gradient is None
            else float(manipulability_weight)
            * np.asarray(manipulability_gradient, dtype=float).reshape(q_current.shape)
        ),
        secondary_velocity=joint_limit_velocity + qp_result.z,
        projected_velocity=nullspace_delta,
        nullspace_projector=nullspace_projector,
        joint_limit_margin=margin,
        secondary_velocity_norm=float(
            np.linalg.norm(joint_limit_velocity + qp_result.z)
        ),
        projected_velocity_norm=float(np.linalg.norm(nullspace_delta)),
        min_joint_limit_margin=float(np.min(margin)),
        null_motion_scale=float(null_motion_scale),
        projected_velocity_before_scale=dt * null_qdot_unscaled,
        max_projected_velocity_norm=max_null_velocity,
    )

    return BimanualSolveResult(
        joint_delta=joint_delta,
        task_delta=task_delta,
        nullspace_delta=nullspace_delta,
        pseudoinverse=jacobian_pinv,
        nullspace=nullspace,
        solver_mode="nullspace_qp",
        qp_result=qp_result,
    )


def bimanual_object_clik_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_arm: SerialArm,
    right_arm: SerialArm,
    targets: BimanualTargets,
    desired_relative_translation: np.ndarray,
    desired_relative_rotation: np.ndarray,
    gain: float | np.ndarray,
    damping: float,
    max_joint_step: float,
    solver_mode: str = "dls_nullspace",
    control_dt: float | None = None,
    joint_velocity_limit: float | np.ndarray | None = None,
    q_nominal: np.ndarray | None = None,
    qp_posture_weight: float = 6.0,
    posture_weights: np.ndarray | None = None,
    qp_manipulability_weight: float | None = None,
    qp_regularization: float = 1e-4,
    qp_eps_abs: float = 1e-4,
    qp_eps_rel: float = 1e-4,
    qp_max_iter: int = 200,
    qp_max_null_velocity: float | None = None,
    qp_null_error_scale_start: float | None = None,
    qp_null_error_scale_stop: float | None = None,
    orientation_task_weight: float = 0.0,
    relative_orientation_weight: float = 2.0,
    joint_limit_gain: float = 0.0,
    joint_limit_activation_margin: float = 0.20,
    manipulability_gain: float = 0.0,
    manipulability_gradient_step: float = 1e-4,
    max_manipulability_command: float | None = None,
    adaptive_damping_alpha: float = 0.0,
    adaptive_damping_epsilon: float = 1e-4,
    adaptive_damping_max: float | None = None,
    singularity_warning_threshold: float = 1e-4,
    max_object_command: float | None = None,
    max_relative_position_command: float | None = None,
    max_relative_rotation_command: float | None = None,
    relative_axis_local: np.ndarray | None = None,
    relative_axis_face_each_other: bool = False,
) -> dict:
    """
    Run one coordinated bimanual DLS step.

    The required STEP-1 task is object-centric translation:
        p_obj = 0.5 * (p_left + p_right)
        p_rel = p_right - p_left

    Instead of solving each arm separately, this builds one stacked system over
    q_stack = [q_left, q_right]. The first three rows move the virtual object
    center; the next three rows preserve the hand-to-hand translation.

    The relative orientation row preserves the initial rigid grasp relation:
        R_rel = R_left.T @ R_right
        R_err = R_rel_current.T @ R_rel_desired
        J_rel_rot = R_right.T @ [-Jw_left, Jw_right]

    Existing fixed EE orientation targets are still computed and reported. They
    can be softly included by setting orientation_task_weight > 0, but the
    default first-step controller leaves them out of the solve to avoid a large
    initial 180-degree orientation correction dominating the translational
    coordination task.
    """

    gain_array = np.asarray(gain, dtype=float)
    if gain_array.ndim == 0:
        rotation_gain = np.full(3, float(gain_array))
        position_gain = np.full(3, float(gain_array))
    else:
        gain_array = gain_array.reshape(6)
        rotation_gain = gain_array[:3]
        position_gain = gain_array[3:]

    left_transform = get_ee_transform(data, left_arm)
    right_transform = get_ee_transform(data, right_arm)
    left_position = left_transform[:3, 3]
    right_position = right_transform[:3, 3]
    object_position = 0.5 * (left_position + right_position)
    relative_translation = right_position - left_position

    object_error = targets.object_transform[:3, 3] - object_position
    relative_error = (
        np.asarray(desired_relative_translation, dtype=float).reshape(3)
        - relative_translation
    )

    left_pose_error = compute_pose_error(left_transform, targets.left_transform)
    right_pose_error = compute_pose_error(right_transform, targets.right_transform)
    left_rotation_error = left_pose_error[:3]
    right_rotation_error = right_pose_error[:3]
    rel_orientation = relative_orientation_state(
        left_transform,
        right_transform,
        desired_relative_rotation,
    )
    if relative_axis_local is None:
        relative_axis_local = np.array([0.0, 0.0, 1.0])
    rel_axis = relative_local_axis_state(
        left_transform,
        right_transform,
        relative_axis_local,
        face_each_other=relative_axis_face_each_other,
    )

    left_position_jacobian, left_angular_jacobian = _ee_jacobians_world(
        model, data, left_arm
    )
    right_position_jacobian, right_angular_jacobian = _ee_jacobians_world(
        model, data, right_arm
    )
    left_geometric_jacobian = geometric_jacobian_world(
        left_position_jacobian,
        left_angular_jacobian,
    )
    right_geometric_jacobian = geometric_jacobian_world(
        right_position_jacobian,
        right_angular_jacobian,
    )
    manipulability_state = _compute_manipulability_state(
        model,
        data,
        left_arm,
        right_arm,
        left_geometric_jacobian,
        right_geometric_jacobian,
        damping=damping,
        manipulability_gain=manipulability_gain,
        manipulability_gradient_step=manipulability_gradient_step,
        max_manipulability_command=max_manipulability_command,
        singularity_warning_threshold=singularity_warning_threshold,
        adaptive_damping_alpha=adaptive_damping_alpha,
        adaptive_damping_epsilon=adaptive_damping_epsilon,
        adaptive_damping_max=adaptive_damping_max,
    )
    left_manipulability = manipulability_state.left
    right_manipulability = manipulability_state.right
    min_manipulability = manipulability_state.minimum
    left_manip_gradient = manipulability_state.left_gradient
    right_manip_gradient = manipulability_state.right_gradient
    stacked_manip_gradient = manipulability_state.stacked_gradient
    manipulability_velocity = manipulability_state.velocity
    adaptive_damping = manipulability_state.adaptive_damping
    is_near_singularity = manipulability_state.is_near_singularity

    left_dof = left_arm.dof
    right_dof = right_arm.dof

    # Geometric intuition:
    # moving both hands the same way moves the virtual object center, while
    # moving them oppositely changes the grasp width.
    object_jacobian = np.hstack(
        [0.5 * left_position_jacobian, 0.5 * right_position_jacobian]
    )
    relative_position_jacobian = np.hstack(
        [-left_position_jacobian, right_position_jacobian]
    )
    # Axis-only grasp consistency. In pure anti-parallel mode the task has
    # three rows: axis_left + axis_right -> 0. In facing mode it has six rows:
    # selected left axis points to the right hand, and selected right axis
    # points to the left hand. The baseline direction derivative is ignored
    # here; hand distance/center tasks already regulate the baseline.
    if rel_axis.face_each_other:
        zeros_left_axis = np.zeros_like(left_angular_jacobian)
        zeros_right_axis = np.zeros_like(right_angular_jacobian)
        relative_orientation_jacobian_world = np.vstack(
            [
                np.hstack(
                    [
                        -skew(rel_axis.left_axis_world) @ left_angular_jacobian,
                        zeros_right_axis,
                    ]
                ),
                np.hstack(
                    [
                        zeros_left_axis,
                        -skew(rel_axis.right_axis_world) @ right_angular_jacobian,
                    ]
                ),
            ]
        )
    else:
        relative_orientation_jacobian_world = np.hstack(
            [
                -skew(rel_axis.left_axis_world) @ left_angular_jacobian,
                -skew(rel_axis.right_axis_world) @ right_angular_jacobian,
            ]
        )
    relative_orientation_jacobian = relative_orientation_jacobian_world

    left_rotation_jacobian = None
    right_rotation_jacobian = None
    if orientation_task_weight > 0.0:
        left_body_jacobian = compute_full_body_jacobian(model, data, left_arm)
        right_body_jacobian = compute_full_body_jacobian(model, data, right_arm)
        zeros_left = np.zeros((3, left_dof))
        zeros_right = np.zeros((3, right_dof))
        left_rotation_jacobian = np.hstack([left_body_jacobian[:3], zeros_right])
        right_rotation_jacobian = np.hstack([zeros_left, right_body_jacobian[:3]])
    task_stack = build_bimanual_task_stack(
        object_jacobian=object_jacobian,
        relative_position_jacobian=relative_position_jacobian,
        relative_orientation_jacobian=relative_orientation_jacobian,
        object_error=object_error,
        relative_position_error=relative_error,
        relative_orientation=rel_orientation,
        relative_axis=rel_axis,
        left_rotation_error=left_rotation_error,
        right_rotation_error=right_rotation_error,
        position_gain=position_gain,
        rotation_gain=rotation_gain,
        relative_orientation_weight=relative_orientation_weight,
        orientation_task_weight=orientation_task_weight,
        max_object_command=max_object_command,
        max_relative_position_command=max_relative_position_command,
        max_relative_rotation_command=max_relative_rotation_command,
        left_rotation_jacobian=left_rotation_jacobian,
        right_rotation_jacobian=right_rotation_jacobian,
    )
    stacked_jacobian = task_stack.jacobian
    stacked_error = task_stack.error
    stacked_condition_number = float(np.linalg.cond(stacked_jacobian))
    nullspace_error_norm = max(
        float(np.linalg.norm(task_stack.object_error)),
        float(np.linalg.norm(task_stack.relative_position_error)),
    )

    q_current = np.concatenate(
        [
            data.qpos[left_arm.qpos_indices].copy(),
            data.qpos[right_arm.qpos_indices].copy(),
        ]
    )
    q_low = np.concatenate([left_arm.ctrl_low, right_arm.ctrl_low])
    q_high = np.concatenate([left_arm.ctrl_high, right_arm.ctrl_high])

    if solver_mode.lower() in {"qp", "osqp", "nullspace_qp"}:
        dt = float(model.opt.timestep if control_dt is None else control_dt)
        if joint_velocity_limit is None:
            qdot_limit = np.full_like(q_current, float(max_joint_step) / dt)
        else:
            qdot_limit = np.asarray(joint_velocity_limit, dtype=float)
            if qdot_limit.ndim == 0:
                qdot_limit = np.full_like(q_current, float(qdot_limit))
            else:
                qdot_limit = qdot_limit.reshape(q_current.shape)
        joint_limit_velocity = joint_limit_barrier_velocity(
            q_current,
            q_low,
            q_high,
            joint_limit_gain,
            joint_limit_activation_margin,
        )
        solve_result = solve_bimanual_task_stack_with_qp(
            task_stack,
            q_current=q_current,
            q_low=q_low,
            q_high=q_high,
            qdot_low=-qdot_limit,
            qdot_high=qdot_limit,
            dt=dt,
            q_nominal=q_nominal,
            posture_weight=qp_posture_weight,
            posture_weights=posture_weights,
            joint_limit_velocity=joint_limit_velocity,
            manipulability_gradient=stacked_manip_gradient,
            manipulability_weight=(
                manipulability_gain
                if qp_manipulability_weight is None
                else float(qp_manipulability_weight)
            ),
            regularization=qp_regularization,
            eps_abs=qp_eps_abs,
            eps_rel=qp_eps_rel,
            max_iter=qp_max_iter,
            max_null_velocity=qp_max_null_velocity,
            null_error_norm=nullspace_error_norm,
            null_error_scale_start=qp_null_error_scale_start,
            null_error_scale_stop=qp_null_error_scale_stop,
            damping_for_diagnostics=adaptive_damping,
        )
    else:
        solve_result = solve_bimanual_task_stack_with_nullspace(
            task_stack,
            adaptive_damping,
            q_current=q_current,
            q_low=q_low,
            q_high=q_high,
            joint_limit_gain=joint_limit_gain,
            joint_limit_activation_margin=joint_limit_activation_margin,
            manipulability_velocity=manipulability_velocity,
        )
    joint_delta_unclipped = solve_result.joint_delta
    joint_delta = np.clip(joint_delta_unclipped, -max_joint_step, max_joint_step)
    left_delta = joint_delta[:left_dof]
    right_delta = joint_delta[left_dof:]
    left_command, left_command_unclipped = _apply_arm_joint_delta(
        data, left_arm, left_delta
    )
    right_command, right_command_unclipped = _apply_arm_joint_delta(
        data, right_arm, right_delta
    )
    object_position_task_only_preview = object_position + (
        object_jacobian @ solve_result.task_delta
    )
    object_position_with_nullspace_preview = object_position + (
        object_jacobian @ solve_result.joint_delta
    )

    return {
        "left_transform": left_transform,
        "right_transform": right_transform,
        "target_left_transform": targets.left_transform.copy(),
        "target_right_transform": targets.right_transform.copy(),
        "object_position": object_position,
        "target_object_position": targets.object_center.copy(),
        "target_object_transform": targets.object_transform.copy(),
        "relative_translation": relative_translation,
        "desired_relative_translation": np.asarray(
            desired_relative_translation, dtype=float
        ).reshape(3).copy(),
        "object_error": object_error,
        "relative_position_error": relative_error,
        "left_rotation_error": left_rotation_error,
        "right_rotation_error": right_rotation_error,
        "relative_orientation_error": rel_axis.axis_error,
        "full_relative_orientation_error": rel_orientation.rotation_error,
        "commanded_object_error": task_stack.commanded_object_error,
        "commanded_relative_position_error": (
            task_stack.commanded_relative_position_error
        ),
        "commanded_relative_orientation_error": (
            task_stack.commanded_relative_orientation_error
        ),
        "object_error_norm": float(np.linalg.norm(object_error)),
        "relative_position_error_norm": float(np.linalg.norm(relative_error)),
        "left_rotation_error_norm": float(np.linalg.norm(left_rotation_error)),
        "right_rotation_error_norm": float(np.linalg.norm(right_rotation_error)),
        "relative_orientation_error_norm": rel_axis.axis_error_norm,
        "full_relative_orientation_error_norm": rel_orientation.rotation_error_norm,
        "commanded_object_error_norm": float(
            np.linalg.norm(task_stack.commanded_object_error)
        ),
        "commanded_relative_position_error_norm": float(
            np.linalg.norm(task_stack.commanded_relative_position_error)
        ),
        "commanded_relative_orientation_error_norm": float(
            np.linalg.norm(task_stack.commanded_relative_orientation_error)
        ),
        "orientation_task_weight": float(orientation_task_weight),
        "relative_orientation_weight": float(relative_orientation_weight),
        "relative_orientation_state": rel_orientation,
        "relative_axis_state": rel_axis,
        "relative_axis_local": rel_axis.local_axis,
        "relative_axis_dot": rel_axis.axis_dot,
        "relative_orientation_jacobian": relative_orientation_jacobian,
        "relative_orientation_jacobian_world": relative_orientation_jacobian_world,
        "stacked_jacobian": stacked_jacobian,
        "stacked_error": stacked_error,
        "stacked_condition_number": stacked_condition_number,
        "nullspace_error_norm": nullspace_error_norm,
        "base_damping": float(damping),
        "adaptive_damping": float(adaptive_damping),
        "manipulability_state": manipulability_state,
        "left_manipulability": left_manipulability,
        "right_manipulability": right_manipulability,
        "min_manipulability": min_manipulability,
        "left_manipulability_gradient": left_manip_gradient,
        "right_manipulability_gradient": right_manip_gradient,
        "manipulability_velocity": manipulability_velocity,
        "manipulability_velocity_norm": manipulability_state.velocity_norm,
        "left_jacobian_condition_number": (
            manipulability_state.left_condition_number
        ),
        "right_jacobian_condition_number": (
            manipulability_state.right_condition_number
        ),
        "is_near_singularity": bool(is_near_singularity),
        "q_current": q_current,
        "joint_limit_margin": solve_result.nullspace.joint_limit_margin,
        "min_joint_limit_margin": solve_result.nullspace.min_joint_limit_margin,
        "joint_limit_velocity": solve_result.nullspace.joint_limit_velocity,
        "joint_limit_velocity_norm": float(
            np.linalg.norm(solve_result.nullspace.joint_limit_velocity)
        ),
        "nullspace_manipulability_velocity": (
            solve_result.nullspace.manipulability_velocity
        ),
        "nullspace_manipulability_velocity_norm": float(
            np.linalg.norm(solve_result.nullspace.manipulability_velocity)
        ),
        "nullspace_projector": solve_result.nullspace.nullspace_projector,
        "nullspace_motion_scale": solve_result.nullspace.null_motion_scale,
        "nullspace_projected_before_scale": (
            solve_result.nullspace.projected_velocity_before_scale
        ),
        "dq_task": solve_result.task_delta,
        "dq_null": solve_result.nullspace_delta,
        "dq_task_norm": float(np.linalg.norm(solve_result.task_delta)),
        "dq_null_norm": float(np.linalg.norm(solve_result.nullspace_delta)),
        "dq_null_before_scale_norm": (
            None
            if solve_result.nullspace.projected_velocity_before_scale is None
            else float(
                np.linalg.norm(
                    solve_result.nullspace.projected_velocity_before_scale
                )
            )
        ),
        "dq_secondary_norm": solve_result.nullspace.secondary_velocity_norm,
        "solver_mode": solve_result.solver_mode,
        "qp_result": solve_result.qp_result,
        "qp_status": (
            None
            if solve_result.qp_result is None
            else solve_result.qp_result.status
        ),
        "qp_objective": (
            None
            if solve_result.qp_result is None
            else solve_result.qp_result.objective_value
        ),
        "qp_task_scaling": (
            None
            if solve_result.qp_result is None
            else solve_result.qp_result.task_scaling
        ),
        "qp_task_residual": (
            None
            if solve_result.qp_result is None
            else task_stack.jacobian @ solve_result.joint_delta - task_stack.error
        ),
        "qp_task_residual_norm": (
            None
            if solve_result.qp_result is None
            else float(
                np.linalg.norm(
                    task_stack.jacobian @ solve_result.joint_delta - task_stack.error
                )
            )
        ),
        "object_position_task_only_preview": object_position_task_only_preview,
        "object_position_with_nullspace_preview": (
            object_position_with_nullspace_preview
        ),
        "dq": joint_delta,
        "dq_unclipped": joint_delta_unclipped,
        "dq_was_clipped": bool(not np.allclose(joint_delta, joint_delta_unclipped)),
        "left_dq": left_delta,
        "right_dq": right_delta,
        "left_q_command": left_command,
        "right_q_command": right_command,
        "left_q_command_unclipped": left_command_unclipped,
        "right_q_command_unclipped": right_command_unclipped,
    }
