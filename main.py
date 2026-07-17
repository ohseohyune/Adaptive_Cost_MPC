from pathlib import Path
import csv
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.clik import (
    BimanualTargets,
    FFWDemoConfig,
    bimanual_object_clik_step,
    bimanual_targets_at_time,
    build_serial_arm,
    desired_relative_rotation,
    desired_relative_transform,
    evaluate_level3_acceptance,
    fixed_grasp_rotations,
    get_ee_position,
    get_ee_transform,
    gripper_local_axes_for_arm,
    hand_target_from_object,
    InterceptionPlan,
    InterceptionPlanner,
    make_transform,
    MovingTargetPrediction,
    MovingTargetPredictor,
    relative_pose_state,
    rotation_from_axis_alignment,
)
from robot.ffw_config import FFW_ARMS


DEMO_CONFIG = FFWDemoConfig()

XYZ_LABELS = ("x", "y", "z")
LEFT_RIGHT_XYZ_LABELS = (
    "left_x",
    "left_y",
    "left_z",
    "right_x",
    "right_y",
    "right_z",
)
BIMANUAL_JOINT_LABELS = tuple(
    [f"left_j{i}" for i in range(1, 8)]
    + [f"right_j{i}" for i in range(1, 8)]
)


def set_arm_joint_positions(model, data, arm, q) -> None:
    """Set both qpos and position-actuator targets for one arm."""
    q = np.asarray(q, dtype=float)
    data.qpos[arm.qpos_indices] = q
    data.ctrl[arm.actuator_ids] = np.clip(q, arm.ctrl_low, arm.ctrl_high)


def target_position_at_time(arm_name: str, t: float) -> np.ndarray:
    """Return the moving end-effector target position at simulation time t."""
    phase = 2.0 * np.pi * DEMO_CONFIG.target_frequency_hz * t

    if DEMO_CONFIG.target_trajectory == "circle":
        # Vertical circle in the x-z plane.
        offset = DEMO_CONFIG.target_circle_radius * np.array(
            [np.cos(phase), 0.0, np.sin(phase)]
        )
    elif DEMO_CONFIG.target_trajectory == "sine":
        # Lying sine wave in the x-z plane.
        offset = np.array([
            DEMO_CONFIG.target_sine_x_amplitude * np.sin(phase),
            0.0,
            DEMO_CONFIG.target_sine_z_amplitude * np.sin(2.0 * phase),
        ])
    else:
        raise ValueError(f"Unknown target trajectory: {DEMO_CONFIG.target_trajectory}")

    return DEMO_CONFIG.target_centers[arm_name] + offset


def target_path_points(arm_name: str) -> np.ndarray:
    """Return sampled points along one period of the target path."""
    period = 1.0 / DEMO_CONFIG.target_frequency_hz
    times = np.linspace(0.0, period, DEMO_CONFIG.target_path_samples, endpoint=False)
    return np.array([target_position_at_time(arm_name, t) for t in times])


def target_rotation_for_grasp(initial_ee_rotation: np.ndarray) -> np.ndarray:
    """Return the desired fixed grasp orientation in world coordinates."""
    if DEMO_CONFIG.orientation_target_mode == "initial":
        return initial_ee_rotation.copy()

    if DEMO_CONFIG.orientation_target_mode == "axis_alignment":
        return rotation_from_axis_alignment(
            local_primary_axis=DEMO_CONFIG.gripper_forward_axis_local,
            world_primary_axis=DEMO_CONFIG.gripper_forward_axis_world,
            local_secondary_axis=DEMO_CONFIG.gripper_up_axis_local,
            world_secondary_axis=DEMO_CONFIG.gripper_up_axis_world,
        )

    raise ValueError(
        f"Unknown orientation target mode: {DEMO_CONFIG.orientation_target_mode}"
    )


def add_marker_sphere(
    viewer,
    position: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    """Add one sphere to the viewer's user scene."""
    geom_id = viewer.user_scn.ngeom
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_id],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=float),
    )
    viewer.user_scn.ngeom += 1


def add_marker_arrow(
    viewer,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    rgba: np.ndarray,
) -> None:
    """Add one arrow connector to the viewer's user scene."""
    geom_id = viewer.user_scn.ngeom
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_id],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=float),
    )
    mujoco.mjv_connector(
        viewer.user_scn.geoms[geom_id],
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    viewer.user_scn.ngeom += 1


def add_world_axis_markers(viewer) -> None:
    """Draw world-frame +x, +y, and +z axes at the origin."""
    origin = np.zeros(3)
    axes = (
        (
            np.array([DEMO_CONFIG.world_axis_length, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
        ),
        (
            np.array([0.0, DEMO_CONFIG.world_axis_length, 0.0]),
            np.array([0.0, 0.8, 0.0, 1.0]),
        ),
        (
            np.array([0.0, 0.0, DEMO_CONFIG.world_axis_length]),
            np.array([0.1, 0.3, 1.0, 1.0]),
        ),
    )

    add_marker_sphere(
        viewer,
        origin,
        DEMO_CONFIG.world_axis_width * 1.5,
        np.array([1.0, 1.0, 1.0, 1.0]),
    )
    for endpoint, color in axes:
        add_marker_arrow(viewer, origin, endpoint, DEMO_CONFIG.world_axis_width, color)


def add_frame_axis_markers(
    viewer,
    transform: np.ndarray,
    *,
    length: float,
    width: float,
    alpha: float = 1.0,
) -> None:
    """Draw local +x, +y, and +z axes of one transform in world coordinates."""

    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    origin = transform[:3, 3]
    rotation = transform[:3, :3]
    axes = (
        (rotation[:, 0], np.array([1.0, 0.0, 0.0, alpha])),
        (rotation[:, 1], np.array([0.0, 0.8, 0.0, alpha])),
        (rotation[:, 2], np.array([0.1, 0.3, 1.0, alpha])),
    )

    add_marker_sphere(
        viewer,
        origin,
        width * 1.6,
        np.array([1.0, 1.0, 1.0, alpha]),
    )
    for axis_world, color in axes:
        add_marker_arrow(viewer, origin, origin + length * axis_world, width, color)


def add_gripper_axis_markers(
    viewer,
    left_transform: np.ndarray,
    right_transform: np.ndarray,
) -> None:
    """Draw actual left/right EE local coordinate frames."""

    if not DEMO_CONFIG.visualize_gripper_axes:
        return
    for transform in (left_transform, right_transform):
        add_frame_axis_markers(
            viewer,
            transform,
            length=DEMO_CONFIG.gripper_axis_length,
            width=DEMO_CONFIG.gripper_axis_width,
        )


def add_target_gripper_axis_markers(viewer, targets: BimanualTargets) -> None:
    """Draw desired left/right EE coordinate frames from config target rotations."""

    if not DEMO_CONFIG.visualize_target_gripper_axes:
        return
    for transform in (targets.left_transform, targets.right_transform):
        add_frame_axis_markers(
            viewer,
            transform,
            length=DEMO_CONFIG.gripper_axis_length * 1.25,
            width=DEMO_CONFIG.gripper_axis_width * 0.55,
            alpha=0.45,
        )


def update_target_markers(viewer, arm_name: str, target_position: np.ndarray) -> None:
    """Draw world axes, the target path, and the current target."""
    viewer.user_scn.ngeom = 0
    add_world_axis_markers(viewer)

    for path_position in target_path_points(arm_name):
        add_marker_sphere(
            viewer,
            path_position,
            0.008,
            np.array([0.1, 0.8, 0.1, 0.35]),
        )

    add_marker_sphere(
        viewer,
        target_position,
        0.013,
        np.array([0.0, 1.0, 0.0, 1.0]),
    )


def update_bimanual_target_markers(viewer, targets, fixed_rotations) -> None:
    """Draw object-center and left/right hand target paths."""
    viewer.user_scn.ngeom = 0
    add_world_axis_markers(viewer)

    period = 1.0 / DEMO_CONFIG.target_frequency_hz
    times = np.linspace(0.0, period, DEMO_CONFIG.target_path_samples, endpoint=False)
    for t in times:
        path_targets = bimanual_targets_at_time(DEMO_CONFIG, t, fixed_rotations)
        add_marker_sphere(
            viewer,
            path_targets.object_center,
            0.006,
            np.array([1.0, 0.85, 0.05, 0.35]),
        )
        add_marker_sphere(
            viewer,
            path_targets.left_transform[:3, 3],
            0.005,
            np.array([0.2, 0.4, 1.0, 0.25]),
        )
        add_marker_sphere(
            viewer,
            path_targets.right_transform[:3, 3],
            0.005,
            np.array([0.0, 1.0, 0.2, 0.25]),
        )

    add_marker_sphere(
        viewer,
        targets.object_center,
        0.014,
        np.array([1.0, 0.85, 0.05, 1.0]),
    )
    add_marker_sphere(
        viewer,
        targets.left_transform[:3, 3],
        0.012,
        np.array([0.2, 0.4, 1.0, 1.0]),
    )
    add_marker_sphere(
        viewer,
        targets.right_transform[:3, 3],
        0.012,
        np.array([0.0, 1.0, 0.2, 1.0]),
    )


def add_nullspace_comparison_markers(viewer, clik_info: dict) -> None:
    """Draw one-step object-center previews with and without nullspace motion."""

    add_marker_sphere(
        viewer,
        clik_info["object_position_task_only_preview"],
        0.009,
        np.array([1.0, 0.2, 0.8, 0.9]),
    )
    add_marker_sphere(
        viewer,
        clik_info["object_position_with_nullspace_preview"],
        0.009,
        np.array([0.1, 0.9, 1.0, 0.9]),
    )


def _add_vector_columns(
    row: dict[str, float | int | str],
    prefix: str,
    vector: np.ndarray | None,
    labels: tuple[str, ...],
) -> None:
    """Add fixed vector columns to one CSV row."""

    if vector is None:
        values = np.full(len(labels), np.nan)
    else:
        values = np.asarray(vector, dtype=float).reshape(-1)
        if values.size != len(labels):
            raise ValueError(
                f"{prefix} has {values.size} values, expected {len(labels)}."
            )
    for label, value in zip(labels, values):
        row[f"{prefix}_{label}"] = float(value)


def _safe_norm(vector: np.ndarray | None) -> float:
    if vector is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(vector, dtype=float).reshape(-1)))


def level3_acceptance_from_clik_info(clik_info: dict):
    """Build the Level-3 acceptance result from one CLIK diagnostic snapshot."""

    q_current = np.asarray(clik_info["q_current"], dtype=float).reshape(-1)
    q_command = np.concatenate(
        [clik_info["left_q_command"], clik_info["right_q_command"]]
    )
    q_command_unclipped = np.concatenate(
        [
            clik_info["left_q_command_unclipped"],
            clik_info["right_q_command_unclipped"],
        ]
    )
    q_command_clip_delta = q_command - q_command_unclipped
    if q_command.shape != q_current.shape:
        raise ValueError("Bimanual command vector does not match q_current shape.")

    return evaluate_level3_acceptance(
        object_error_norm=clik_info["object_error_norm"],
        relative_position_error_norm=clik_info["relative_position_error_norm"],
        relative_orientation_error_norm=clik_info["relative_orientation_error_norm"],
        min_manipulability=clik_info["min_manipulability"],
        qp_task_residual_norm=clik_info["qp_task_residual_norm"],
        qp_status=clik_info["qp_status"],
        dq_was_clipped=clik_info["dq_was_clipped"],
        command_clip_delta_norm=_safe_norm(q_command_clip_delta),
        config=DEMO_CONFIG.level3_acceptance,
    )


def build_bimanual_control_log_row(
    *,
    step_count: int,
    time_s: float,
    control_phase: str,
    ramp: float,
    raw_targets: BimanualTargets,
    targets: BimanualTargets,
    clik_info: dict,
    applied_delta: np.ndarray,
    kinematic_alpha: float,
    control_dt: float,
) -> dict[str, float | int | str]:
    """Create one CSV row with task-space and joint-space control terms."""

    q_current = np.asarray(clik_info["q_current"], dtype=float).reshape(-1)
    dq_task = np.asarray(clik_info["dq_task"], dtype=float).reshape(-1)
    dq_null = np.asarray(clik_info["dq_null"], dtype=float).reshape(-1)
    dq = np.asarray(clik_info["dq"], dtype=float).reshape(-1)
    dq_unclipped = np.asarray(clik_info["dq_unclipped"], dtype=float).reshape(-1)
    applied_delta = np.asarray(applied_delta, dtype=float).reshape(-1)

    q_command = np.concatenate(
        [clik_info["left_q_command"], clik_info["right_q_command"]]
    )
    q_command_unclipped = np.concatenate(
        [
            clik_info["left_q_command_unclipped"],
            clik_info["right_q_command_unclipped"],
        ]
    )
    q_command_delta = q_command - q_current
    q_command_clip_delta = q_command - q_command_unclipped
    dq_clip_delta = dq - dq_unclipped
    level3_metrics = level3_acceptance_from_clik_info(clik_info)

    stacked_jacobian = np.asarray(clik_info["stacked_jacobian"], dtype=float)
    stacked_error = np.asarray(clik_info["stacked_error"], dtype=float).reshape(-1)
    task_effect = stacked_jacobian @ dq_task
    null_effect = stacked_jacobian @ dq_null
    total_effect = stacked_jacobian @ dq
    residual_after_clipped = total_effect - stacked_error

    qp_result = clik_info.get("qp_result")
    qdot_task_unscaled = None if qp_result is None else qp_result.qdot_task_unscaled
    qdot_task_scaled = None if qp_result is None else qp_result.qdot_task
    qdot_qp_raw = None if qp_result is None else qp_result.qdot
    qdot_null_qp_raw = (
        None if qp_result is None else qp_result.qdot - qp_result.qdot_task
    )

    row: dict[str, float | int | str] = {
        "step": int(step_count),
        "time_s": float(time_s),
        "control_phase": str(control_phase),
        "control_dt_s": float(control_dt),
        "target_ramp": float(np.clip(ramp, 0.0, 1.0)),
        "solver_mode": str(clik_info["solver_mode"]),
        "kinematic_control": int(DEMO_CONFIG.bimanual_kinematic_control),
        "kinematic_alpha": float(kinematic_alpha),
        "qp_status": "" if clik_info["qp_status"] is None else clik_info["qp_status"],
        "qp_task_scaling": (
            float("nan")
            if clik_info["qp_task_scaling"] is None
            else float(clik_info["qp_task_scaling"])
        ),
        "qp_objective": (
            float("nan")
            if clik_info["qp_objective"] is None
            else float(clik_info["qp_objective"])
        ),
        "qp_task_residual_norm": (
            float("nan")
            if clik_info["qp_task_residual_norm"] is None
            else float(clik_info["qp_task_residual_norm"])
        ),
        "stacked_error_norm": _safe_norm(stacked_error),
        "stacked_condition_number": float(clik_info["stacked_condition_number"]),
        "object_error_norm": float(clik_info["object_error_norm"]),
        "relative_position_error_norm": float(
            clik_info["relative_position_error_norm"]
        ),
        "relative_orientation_error_norm": float(
            clik_info["relative_orientation_error_norm"]
        ),
        "relative_axis_dot": float(clik_info["relative_axis_dot"]),
        "left_rotation_error_norm": float(clik_info["left_rotation_error_norm"]),
        "right_rotation_error_norm": float(clik_info["right_rotation_error_norm"]),
        "commanded_object_error_norm": float(
            clik_info["commanded_object_error_norm"]
        ),
        "commanded_relative_position_error_norm": float(
            clik_info["commanded_relative_position_error_norm"]
        ),
        "commanded_relative_orientation_error_norm": float(
            clik_info["commanded_relative_orientation_error_norm"]
        ),
        "dq_task_norm": float(clik_info["dq_task_norm"]),
        "dq_null_norm": float(clik_info["dq_null_norm"]),
        "dq_total_unclipped_norm": _safe_norm(dq_unclipped),
        "dq_total_clipped_norm": _safe_norm(dq),
        "dq_clip_delta_norm": _safe_norm(dq_clip_delta),
        "dq_was_clipped": int(clik_info["dq_was_clipped"]),
        "dq_null_before_scale_norm": (
            float("nan")
            if clik_info["dq_null_before_scale_norm"] is None
            else float(clik_info["dq_null_before_scale_norm"])
        ),
        "dq_secondary_norm": float(clik_info["dq_secondary_norm"]),
        "nullspace_motion_scale": float(clik_info["nullspace_motion_scale"]),
        "q_command_delta_norm": _safe_norm(q_command_delta),
        "q_command_clip_delta_norm": _safe_norm(q_command_clip_delta),
        "applied_delta_norm": _safe_norm(applied_delta),
        "max_abs_dq_task": float(np.max(np.abs(dq_task))),
        "max_abs_dq_null": float(np.max(np.abs(dq_null))),
        "max_abs_dq_unclipped": float(np.max(np.abs(dq_unclipped))),
        "max_abs_dq_clipped": float(np.max(np.abs(dq))),
        "max_abs_applied_delta": float(np.max(np.abs(applied_delta))),
        "min_joint_limit_margin": float(clik_info["min_joint_limit_margin"]),
        "joint_limit_velocity_norm": float(clik_info["joint_limit_velocity_norm"]),
        "nullspace_manipulability_velocity_norm": float(
            clik_info["nullspace_manipulability_velocity_norm"]
        ),
        "manipulability_velocity_norm": float(
            clik_info["manipulability_velocity_norm"]
        ),
        "left_manipulability": float(clik_info["left_manipulability"]),
        "right_manipulability": float(clik_info["right_manipulability"]),
        "min_manipulability": float(clik_info["min_manipulability"]),
        "base_damping": float(clik_info["base_damping"]),
        "adaptive_damping": float(clik_info["adaptive_damping"]),
        "object_task_effect_norm": _safe_norm(task_effect[:3]),
        "object_null_effect_norm": _safe_norm(null_effect[:3]),
        "object_total_effect_norm": _safe_norm(total_effect[:3]),
        "object_residual_after_clipped_norm": _safe_norm(
            residual_after_clipped[:3]
        ),
        "relative_position_task_effect_norm": _safe_norm(task_effect[3:6]),
        "relative_position_null_effect_norm": _safe_norm(null_effect[3:6]),
        "relative_position_total_effect_norm": _safe_norm(total_effect[3:6]),
        "relative_position_residual_after_clipped_norm": _safe_norm(
            residual_after_clipped[3:6]
        ),
        "qdot_task_unscaled_norm": _safe_norm(qdot_task_unscaled),
        "qdot_task_scaled_norm": _safe_norm(qdot_task_scaled),
        "qdot_null_qp_raw_norm": _safe_norm(qdot_null_qp_raw),
        "qdot_qp_raw_norm": _safe_norm(qdot_qp_raw),
        "level3_passed": int(level3_metrics.passed),
        "level3_failed_terms": ",".join(level3_metrics.failed_terms),
        "level3_object_error_norm": level3_metrics.object_error_norm,
        "level3_object_error_max": float(
            DEMO_CONFIG.level3_acceptance.object_error_max
        ),
        "level3_relative_position_error_norm": (
            level3_metrics.relative_position_error_norm
        ),
        "level3_relative_position_error_max": float(
            DEMO_CONFIG.level3_acceptance.relative_position_error_max
        ),
        "level3_relative_orientation_error_norm": (
            level3_metrics.relative_orientation_error_norm
        ),
        "level3_relative_orientation_error_max": float(
            DEMO_CONFIG.level3_acceptance.relative_orientation_error_max
        ),
        "level3_min_manipulability": level3_metrics.min_manipulability,
        "level3_min_manipulability_min": float(
            DEMO_CONFIG.level3_acceptance.min_manipulability_min
        ),
        "level3_qp_task_residual_norm": level3_metrics.qp_task_residual_norm,
        "level3_qp_task_residual_max": float(
            DEMO_CONFIG.level3_acceptance.qp_task_residual_max
        ),
        "level3_dq_was_clipped": int(level3_metrics.dq_was_clipped),
        "level3_command_clip_delta_norm": (
            level3_metrics.command_clip_delta_norm
        ),
        "level3_command_clip_delta_max": float(
            DEMO_CONFIG.level3_acceptance.command_clip_delta_max
        ),
    }

    if stacked_error.size >= 9:
        row.update(
            {
                "relative_orientation_weighted_task_effect_norm": _safe_norm(
                    task_effect[6:9]
                ),
                "relative_orientation_weighted_null_effect_norm": _safe_norm(
                    null_effect[6:9]
                ),
                "relative_orientation_weighted_total_effect_norm": _safe_norm(
                    total_effect[6:9]
                ),
                "relative_orientation_weighted_residual_after_clipped_norm": (
                    _safe_norm(residual_after_clipped[6:9])
                ),
            }
        )
    else:
        row.update(
            {
                "relative_orientation_weighted_task_effect_norm": float("nan"),
                "relative_orientation_weighted_null_effect_norm": float("nan"),
                "relative_orientation_weighted_total_effect_norm": float("nan"),
                "relative_orientation_weighted_residual_after_clipped_norm": (
                    float("nan")
                ),
            }
        )

    _add_vector_columns(row, "raw_target_object", raw_targets.object_center, XYZ_LABELS)
    _add_vector_columns(row, "target_object", targets.object_center, XYZ_LABELS)
    _add_vector_columns(row, "actual_object", clik_info["object_position"], XYZ_LABELS)
    _add_vector_columns(row, "object_error", clik_info["object_error"], XYZ_LABELS)
    _add_vector_columns(
        row, "commanded_object_error", clik_info["commanded_object_error"], XYZ_LABELS
    )
    _add_vector_columns(
        row,
        "actual_relative_translation",
        clik_info["relative_translation"],
        XYZ_LABELS,
    )
    _add_vector_columns(
        row,
        "desired_relative_translation",
        clik_info["desired_relative_translation"],
        XYZ_LABELS,
    )
    _add_vector_columns(
        row,
        "relative_position_error",
        clik_info["relative_position_error"],
        XYZ_LABELS,
    )
    _add_vector_columns(
        row,
        "commanded_relative_position_error",
        clik_info["commanded_relative_position_error"],
        XYZ_LABELS,
    )
    _add_vector_columns(
        row,
        "relative_orientation_error",
        clik_info["relative_orientation_error"],
        LEFT_RIGHT_XYZ_LABELS,
    )
    _add_vector_columns(
        row,
        "commanded_relative_orientation_error",
        clik_info["commanded_relative_orientation_error"],
        LEFT_RIGHT_XYZ_LABELS,
    )
    _add_vector_columns(
        row,
        "relative_axis_local",
        clik_info["relative_axis_local"],
        XYZ_LABELS,
    )
    relative_axis_state = clik_info["relative_axis_state"]
    _add_vector_columns(
        row,
        "left_relative_axis_world",
        relative_axis_state.left_axis_world,
        XYZ_LABELS,
    )
    _add_vector_columns(
        row,
        "right_relative_axis_world",
        relative_axis_state.right_axis_world,
        XYZ_LABELS,
    )
    _add_vector_columns(
        row,
        "left_desired_relative_axis_world",
        relative_axis_state.left_desired_axis_world,
        XYZ_LABELS,
    )
    _add_vector_columns(
        row,
        "right_desired_relative_axis_world",
        relative_axis_state.right_desired_axis_world,
        XYZ_LABELS,
    )
    _add_vector_columns(
        row, "left_rotation_error", clik_info["left_rotation_error"], XYZ_LABELS
    )
    _add_vector_columns(
        row, "right_rotation_error", clik_info["right_rotation_error"], XYZ_LABELS
    )
    _add_vector_columns(row, "object_task_effect", task_effect[:3], XYZ_LABELS)
    _add_vector_columns(row, "object_null_effect", null_effect[:3], XYZ_LABELS)
    _add_vector_columns(row, "object_total_effect", total_effect[:3], XYZ_LABELS)
    _add_vector_columns(
        row, "object_residual_after_clipped", residual_after_clipped[:3], XYZ_LABELS
    )
    _add_vector_columns(
        row, "relative_position_task_effect", task_effect[3:6], XYZ_LABELS
    )
    _add_vector_columns(
        row, "relative_position_null_effect", null_effect[3:6], XYZ_LABELS
    )
    _add_vector_columns(
        row, "relative_position_total_effect", total_effect[3:6], XYZ_LABELS
    )
    _add_vector_columns(
        row,
        "relative_position_residual_after_clipped",
        residual_after_clipped[3:6],
        XYZ_LABELS,
    )

    _add_vector_columns(row, "q_current", q_current, BIMANUAL_JOINT_LABELS)
    _add_vector_columns(row, "dq_task", dq_task, BIMANUAL_JOINT_LABELS)
    _add_vector_columns(row, "dq_null", dq_null, BIMANUAL_JOINT_LABELS)
    _add_vector_columns(row, "dq_unclipped", dq_unclipped, BIMANUAL_JOINT_LABELS)
    _add_vector_columns(row, "dq_clipped", dq, BIMANUAL_JOINT_LABELS)
    _add_vector_columns(row, "dq_clip_delta", dq_clip_delta, BIMANUAL_JOINT_LABELS)
    _add_vector_columns(row, "q_command_delta", q_command_delta, BIMANUAL_JOINT_LABELS)
    _add_vector_columns(
        row, "q_command_clip_delta", q_command_clip_delta, BIMANUAL_JOINT_LABELS
    )
    _add_vector_columns(row, "applied_delta", applied_delta, BIMANUAL_JOINT_LABELS)
    _add_vector_columns(
        row, "qdot_task_unscaled", qdot_task_unscaled, BIMANUAL_JOINT_LABELS
    )
    _add_vector_columns(
        row, "qdot_task_scaled", qdot_task_scaled, BIMANUAL_JOINT_LABELS
    )
    _add_vector_columns(
        row, "qdot_null_qp_raw", qdot_null_qp_raw, BIMANUAL_JOINT_LABELS
    )
    _add_vector_columns(row, "qdot_qp_raw", qdot_qp_raw, BIMANUAL_JOINT_LABELS)

    return row


def build_prediction_log_row(
    *,
    step_count: int,
    time_s: float,
    task_time_s: float,
    prediction: MovingTargetPrediction,
    future_actual_position: np.ndarray,
) -> dict[str, float | int | str]:
    """Create one CSV row for moving-target prediction diagnostics."""

    future_actual_position = np.asarray(future_actual_position, dtype=float).reshape(3)
    prediction_error = prediction.predicted_position - future_actual_position
    row: dict[str, float | int | str] = {
        "step": int(step_count),
        "time_s": float(time_s),
        "task_time_s": float(task_time_s),
        "prediction_dt_s": float(prediction.dt_s),
        "prediction_horizon_s": float(prediction.horizon_s),
        "estimated_speed": _safe_norm(prediction.velocity),
        "estimated_acceleration_norm": _safe_norm(prediction.acceleration),
        "prediction_error_norm": _safe_norm(prediction_error),
        "prediction_confidence": float(prediction.confidence),
        "prediction_speed_was_clipped": int(prediction.speed_was_clipped),
    }
    _add_vector_columns(row, "target", prediction.position, XYZ_LABELS)
    _add_vector_columns(row, "estimated_velocity", prediction.velocity, XYZ_LABELS)
    _add_vector_columns(
        row, "estimated_acceleration", prediction.acceleration, XYZ_LABELS
    )
    _add_vector_columns(
        row, "predicted_target", prediction.predicted_position, XYZ_LABELS
    )
    _add_vector_columns(row, "future_actual_target", future_actual_position, XYZ_LABELS)
    _add_vector_columns(row, "prediction_error", prediction_error, XYZ_LABELS)
    return row


def build_interception_log_row(
    *,
    step_count: int,
    time_s: float,
    task_time_s: float,
    plan: InterceptionPlan,
    blend_alpha: float,
    blended_target_position: np.ndarray,
    previous_interception_position: np.ndarray | None,
) -> dict[str, float | int | str]:
    """Create one CSV row for interception-planning diagnostics."""

    blended_target_position = np.asarray(
        blended_target_position, dtype=float
    ).reshape(3)
    target_shift = plan.interception_position - plan.current_target_position
    if previous_interception_position is None:
        step_delta = np.zeros(3)
    else:
        step_delta = plan.interception_position - np.asarray(
            previous_interception_position, dtype=float
        ).reshape(3)
    row: dict[str, float | int | str] = {
        "step": int(step_count),
        "time_s": float(time_s),
        "task_time_s": float(task_time_s),
        "lead_time_s": float(plan.lead_time_s),
        "time_to_contact_s": float(plan.time_to_contact_s),
        "distance_to_intercept": float(plan.distance_to_intercept),
        "reachable_distance": float(plan.reachable_distance),
        "reachability_margin": float(plan.reachability_margin),
        "reachable": int(plan.reachable),
        "confidence": float(plan.confidence),
        "blend_alpha": float(blend_alpha),
        "target_shift_norm": _safe_norm(target_shift),
        "interception_step_delta_norm": _safe_norm(step_delta),
    }
    _add_vector_columns(
        row, "current_object", plan.current_object_position, XYZ_LABELS
    )
    _add_vector_columns(
        row, "current_target", plan.current_target_position, XYZ_LABELS
    )
    _add_vector_columns(
        row, "interception_target", plan.interception_position, XYZ_LABELS
    )
    _add_vector_columns(row, "blended_target", blended_target_position, XYZ_LABELS)
    _add_vector_columns(row, "target_shift", target_shift, XYZ_LABELS)
    _add_vector_columns(row, "interception_step_delta", step_delta, XYZ_LABELS)
    return row


def write_bimanual_control_log(rows: list[dict[str, float | int | str]]) -> None:
    """Write the per-step bimanual control decomposition to CSV."""

    if not rows:
        return

    output_dir = PROJECT_ROOT / "sweep_results" / "bimanual_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bimanual_control_terms.csv"

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved bimanual control terms CSV: {output_path}")


def print_level3_acceptance_summary(history: list) -> None:
    """Print aggregate Level-3 acceptance statistics after the viewer closes."""

    if not history:
        return

    pass_count = sum(1 for sample in history if sample.passed)
    pass_rate = pass_count / len(history)
    object_peak = max(sample.object_error_norm for sample in history)
    relative_position_peak = max(
        sample.relative_position_error_norm for sample in history
    )
    relative_orientation_peak = max(
        sample.relative_orientation_error_norm for sample in history
    )
    min_manipulability = min(sample.min_manipulability for sample in history)
    qp_residual_values = [
        sample.qp_task_residual_norm
        for sample in history
        if np.isfinite(sample.qp_task_residual_norm)
    ]
    qp_residual_peak = (
        max(qp_residual_values) if qp_residual_values else float("nan")
    )
    clipped_steps = sum(1 for sample in history if sample.dq_was_clipped)
    command_clipped_steps = sum(
        1
        for sample in history
        if sample.command_clip_delta_norm
        > DEMO_CONFIG.level3_acceptance.command_clip_delta_max
    )

    failed_terms: dict[str, int] = {}
    for sample in history:
        for term in sample.failed_terms:
            failed_terms[term] = failed_terms.get(term, 0) + 1
    failed_text = (
        "-"
        if not failed_terms
        else ", ".join(
            f"{term}:{count}"
            for term, count in sorted(
                failed_terms.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
    )

    print(
        "Level-3 acceptance summary: "
        f"{pass_count}/{len(history)} passed ({100.0 * pass_rate:.1f}%) | "
        f"peak obj={object_peak:.4f} m | "
        f"peak rel_pos={relative_position_peak:.4f} m | "
        f"peak rel_rot={relative_orientation_peak:.4f} rad | "
        f"min manip={min_manipulability:.3e} | "
        f"peak qp_res={qp_residual_peak:.3e} | "
        f"dq_clipped_steps={clipped_steps} | "
        f"cmd_clipped_steps={command_clipped_steps} | "
        f"failures={failed_text}"
    )


def print_prediction_summary(history: list[dict[str, float | int | str]]) -> None:
    """Print aggregate moving-target prediction diagnostics."""

    if not history:
        return

    errors = np.array(
        [float(sample["prediction_error_norm"]) for sample in history], dtype=float
    )
    speeds = np.array([float(sample["estimated_speed"]) for sample in history])
    accelerations = np.array(
        [float(sample["estimated_acceleration_norm"]) for sample in history],
        dtype=float,
    )
    confidences = np.array(
        [float(sample["prediction_confidence"]) for sample in history], dtype=float
    )
    valid_mask = np.isfinite(errors) & (confidences > 0.0)
    valid_errors = errors[valid_mask]
    if valid_errors.size == 0:
        print("Prediction summary: no finite prediction errors.")
        return

    clipped_steps = sum(
        int(sample["prediction_speed_was_clipped"]) for sample in history
    )
    print(
        "Prediction summary: "
        f"samples={int(np.sum(valid_mask))}/{len(history)} valid | "
        f"err_p95={np.percentile(valid_errors, 95):.4f} m | "
        f"err_max={np.max(valid_errors):.4f} m | "
        f"speed_p95={np.percentile(speeds, 95):.4f} m/s | "
        f"acc_p95={np.percentile(accelerations, 95):.4f} m/s^2 | "
        f"confidence_min={np.min(confidences):.2f} | "
        f"speed_clipped_steps={clipped_steps} | "
        f"warmup_or_nan_samples={len(history) - int(np.sum(valid_mask))}"
    )


def print_interception_summary(history: list[dict[str, float | int | str]]) -> None:
    """Print aggregate interception-planning diagnostics."""

    if not history:
        return

    reachable = np.array([int(sample["reachable"]) for sample in history], dtype=float)
    margins = np.array(
        [float(sample["reachability_margin"]) for sample in history], dtype=float
    )
    lead_times = np.array([float(sample["lead_time_s"]) for sample in history])
    target_shifts = np.array(
        [float(sample["target_shift_norm"]) for sample in history], dtype=float
    )
    step_deltas = np.array(
        [float(sample["interception_step_delta_norm"]) for sample in history],
        dtype=float,
    )
    distances = np.array(
        [float(sample["distance_to_intercept"]) for sample in history],
        dtype=float,
    )

    print(
        "Interception summary: "
        f"samples={len(history)} | "
        f"reachable_rate={100.0 * float(np.mean(reachable)):.1f}% | "
        f"margin_p05={np.percentile(margins, 5):.4f} m | "
        f"margin_min={np.min(margins):.4f} m | "
        f"lead_p50={np.percentile(lead_times, 50):.3f} s | "
        f"lead_p95={np.percentile(lead_times, 95):.3f} s | "
        f"distance_p95={np.percentile(distances, 95):.4f} m | "
        f"target_shift_p95={np.percentile(target_shifts, 95):.4f} m | "
        f"step_delta_p95={np.percentile(step_deltas, 95):.4f} m"
    )


def plot_bimanual_diagnostics(history: list[dict]) -> None:
    """Save manipulability, damping, and condition-number diagnostics."""

    if not history:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping bimanual diagnostics plot.")
        return

    times = np.array([sample["time"] for sample in history])
    left_manip = np.array([sample["left_manipulability"] for sample in history])
    right_manip = np.array([sample["right_manipulability"] for sample in history])
    damping = np.array([sample["adaptive_damping"] for sample in history])
    stack_cond = np.array([sample["stacked_condition_number"] for sample in history])
    left_cond = np.array([sample["left_condition_number"] for sample in history])
    right_cond = np.array([sample["right_condition_number"] for sample in history])

    output_dir = PROJECT_ROOT / "sweep_results" / "bimanual_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manipulability_damping_condition.png"

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(times, left_manip, label="left")
    axes[0].plot(times, right_manip, label="right")
    axes[0].set_ylabel("manipulability")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, damping, color="tab:orange")
    axes[1].set_ylabel("damping")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(times, stack_cond, label="stack")
    axes[2].semilogy(times, left_cond, label="left")
    axes[2].semilogy(times, right_cond, label="right")
    axes[2].set_ylabel("condition")
    axes[2].set_xlabel("time [s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved bimanual diagnostics plot: {output_path}")


def ramp_bimanual_targets(
    raw_targets: BimanualTargets,
    initial_object_position: np.ndarray,
    ramp: float,
) -> BimanualTargets:
    """Blend the object target from the current midpoint onto the trajectory."""

    ramp = float(np.clip(ramp, 0.0, 1.0))
    object_center = (1.0 - ramp) * initial_object_position + ramp * raw_targets.object_center
    object_transform = make_transform(
        object_center,
        raw_targets.object_transform[:3, :3],
    )
    return BimanualTargets(
        object_center=object_center,
        object_transform=object_transform,
        left_transform=hand_target_from_object(
            object_transform,
            DEMO_CONFIG.hand_offsets["left"],
            raw_targets.left_transform[:3, :3],
        ),
        right_transform=hand_target_from_object(
            object_transform,
            DEMO_CONFIG.hand_offsets["right"],
            raw_targets.right_transform[:3, :3],
        ),
    )


def main() -> None:
    xml_path = PROJECT_ROOT / "model" / "robotis_ffw" / "scene_ffw_sg2.xml"

    # Load the MuJoCo model from the XML file.
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    right_arm = build_serial_arm(model, FFW_ARMS["right"])
    left_arm = build_serial_arm(model, FFW_ARMS["left"])

    if DEMO_CONFIG.bimanual_start_from_zero_home:
        start_posture_label = "zero_arm_home"
        right_start_q = np.asarray(DEMO_CONFIG.zero_arm_home, dtype=float)
        left_start_q = np.asarray(DEMO_CONFIG.zero_arm_home, dtype=float)
    else:
        start_posture_label = "configured arm_home"
        right_start_q = np.asarray(DEMO_CONFIG.right_arm_home, dtype=float)
        left_start_q = np.asarray(DEMO_CONFIG.left_arm_home, dtype=float)
    set_arm_joint_positions(model, data, right_arm, right_start_q)
    set_arm_joint_positions(model, data, left_arm, left_start_q)
    mujoco.mj_forward(model, data)

    fixed_rotations = fixed_grasp_rotations(DEMO_CONFIG)
    initial_targets = bimanual_targets_at_time(DEMO_CONFIG, 0.0, fixed_rotations)
    relative_transform_desired = desired_relative_transform(initial_targets)
    initial_left_transform = get_ee_transform(data, left_arm)
    initial_right_transform = get_ee_transform(data, right_arm)
    initial_object_position = 0.5 * (
        initial_left_transform[:3, 3] + initial_right_transform[:3, 3]
    )
    desired_relative_translation = (
        initial_targets.right_transform[:3, 3] - initial_targets.left_transform[:3, 3]
    )
    desired_relative_rotation_matrix = desired_relative_rotation(
        initial_targets.left_transform,
        initial_targets.right_transform,
    )
    initial_hand_distance = np.linalg.norm(
        desired_relative_translation
    )
    bimanual_q_nominal = np.concatenate(
        [
            data.qpos[left_arm.qpos_indices].copy(),
            data.qpos[right_arm.qpos_indices].copy(),
        ]
    )
    bimanual_initial_q = bimanual_q_nominal.copy()
    bimanual_prealign_q_nominal = np.concatenate(
        [
            np.asarray(DEMO_CONFIG.left_arm_prealign, dtype=float).reshape(left_arm.dof),
            np.asarray(DEMO_CONFIG.right_arm_prealign, dtype=float).reshape(right_arm.dof),
        ]
    )

    print("Controlling both arms")
    print(f"Initial left EE position: {get_ee_position(data, left_arm)}")
    print(f"Initial right EE position: {get_ee_position(data, right_arm)}")
    print(f"Moving object center: {DEMO_CONFIG.object_center}")
    print(f"Initial desired hand distance: {initial_hand_distance:.4f} m")
    print(f"Target trajectory: {DEMO_CONFIG.target_trajectory}")
    print(f"Bimanual solver mode: {DEMO_CONFIG.bimanual_solver_mode}")
    print(f"Start posture: {start_posture_label}")
    print(
        "Null-space QP: "
        f"posture_w={DEMO_CONFIG.bimanual_qp_posture_weight:.3g}, "
        f"reg={DEMO_CONFIG.bimanual_qp_regularization:.1e}, "
        f"qdot_limit={DEMO_CONFIG.bimanual_joint_velocity_limit}, "
        f"max_null_vel={DEMO_CONFIG.bimanual_qp_max_null_velocity}, "
        f"gate=({DEMO_CONFIG.bimanual_qp_null_error_scale_start:.3f}, "
        f"{DEMO_CONFIG.bimanual_qp_null_error_scale_stop:.3f})"
    )
    print(
        "Control phases: "
        f"initial_hold={DEMO_CONFIG.bimanual_initial_hold_s:.2f}s, "
        f"prealign={DEMO_CONFIG.bimanual_prealign_s:.2f}s "
        f"(rel_ori_w={DEMO_CONFIG.bimanual_prealign_relative_orientation_weight:.3g}), "
        f"prealign_nominal={DEMO_CONFIG.bimanual_use_prealign_nominal}, "
        f"joint_space_prealign={DEMO_CONFIG.bimanual_joint_space_prealign}, "
        f"active_rel_ori_w={DEMO_CONFIG.bimanual_relative_orientation_weight:.3g}, "
        f"rel_ori_ramp={DEMO_CONFIG.bimanual_relative_orientation_ramp_s:.2f}s"
    )
    print(
        "Relative axis-only constraint: "
        f"local_axis={DEMO_CONFIG.bimanual_relative_axis_local}, "
        f"face_each_other={DEMO_CONFIG.bimanual_relative_axis_face_each_other}"
    )
    print(
        "Level-3 acceptance: "
        f"obj<={DEMO_CONFIG.level3_acceptance.object_error_max:.3f} m, "
        f"rel_pos<={DEMO_CONFIG.level3_acceptance.relative_position_error_max:.3f} m, "
        f"rel_rot<={DEMO_CONFIG.level3_acceptance.relative_orientation_error_max:.3f} rad, "
        f"manip>={DEMO_CONFIG.level3_acceptance.min_manipulability_min:.1e}, "
        f"qp_res<={DEMO_CONFIG.level3_acceptance.qp_task_residual_max:.3f}, "
        f"clip_allowed={DEMO_CONFIG.level3_acceptance.allow_dq_clipping}"
    )
    print(f"Orientation target mode: {DEMO_CONFIG.orientation_target_mode}")
    for arm_name in ("left", "right"):
        local_forward, local_up = gripper_local_axes_for_arm(DEMO_CONFIG, arm_name)
        print(
            f"{arm_name} local axes: forward={local_forward}, up={local_up}"
        )
    print("Close the viewer window to stop.")

    diagnostics_history = []
    level3_acceptance_history = []
    prediction_history = []
    interception_history = []
    target_predictor = None
    if DEMO_CONFIG.prediction_enabled:
        target_predictor = MovingTargetPredictor(
            horizon_s=DEMO_CONFIG.prediction_horizon_s,
            smoothing_alpha=DEMO_CONFIG.prediction_smoothing_alpha,
            max_prediction_speed=DEMO_CONFIG.prediction_max_speed,
        )
    interception_planner = None
    previous_interception_position = None
    if DEMO_CONFIG.interception_enabled:
        interception_planner = InterceptionPlanner(
            min_lead_s=DEMO_CONFIG.interception_min_lead_s,
            max_lead_s=DEMO_CONFIG.interception_max_lead_s,
            lead_samples=DEMO_CONFIG.interception_lead_samples,
            reach_speed=DEMO_CONFIG.interception_reach_speed,
        )
    control_log_stride = max(1, int(DEMO_CONFIG.bimanual_control_log_stride))
    prediction_log_stride = max(1, int(DEMO_CONFIG.prediction_log_stride))
    interception_log_stride = max(1, int(DEMO_CONFIG.interception_log_stride))
    control_log_file = None
    control_log_writer = None
    prediction_log_file = None
    prediction_log_writer = None
    interception_log_file = None
    interception_log_writer = None
    control_log_path = (
        PROJECT_ROOT
        / "sweep_results"
        / "bimanual_diagnostics"
        / "bimanual_control_terms.csv"
    )
    prediction_log_path = (
        PROJECT_ROOT
        / "sweep_results"
        / "bimanual_diagnostics"
        / "target_prediction_terms.csv"
    )
    interception_log_path = (
        PROJECT_ROOT
        / "sweep_results"
        / "bimanual_diagnostics"
        / "interception_terms.csv"
    )
    if DEMO_CONFIG.bimanual_control_log_enabled:
        control_log_path.parent.mkdir(parents=True, exist_ok=True)
        control_log_file = control_log_path.open("w", newline="")
        print(f"Streaming bimanual control CSV: {control_log_path}")
    if DEMO_CONFIG.prediction_enabled and DEMO_CONFIG.prediction_log_enabled:
        prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_log_file = prediction_log_path.open("w", newline="")
        print(f"Streaming target prediction CSV: {prediction_log_path}")
    if DEMO_CONFIG.interception_enabled and DEMO_CONFIG.interception_log_enabled:
        interception_log_path.parent.mkdir(parents=True, exist_ok=True)
        interception_log_file = interception_log_path.open("w", newline="")
        print(f"Streaming interception CSV: {interception_log_path}")

    # Open a passive viewer window for visualization.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        prealign_completed = False
        while viewer.is_running():
            step_start = time.time()
            control_time = float(data.time)
            initial_hold_s = max(0.0, float(DEMO_CONFIG.bimanual_initial_hold_s))
            prealign_s = max(0.0, float(DEMO_CONFIG.bimanual_prealign_s))
            prealign_time = control_time - initial_hold_s
            in_initial_hold = control_time < initial_hold_s
            in_prealign = (not in_initial_hold) and prealign_time < prealign_s
            if in_initial_hold:
                control_phase = "initial_hold"
            elif in_prealign:
                control_phase = "prealign"
            else:
                control_phase = "active"
            task_time = (
                0.0
                if control_phase != "active"
                else control_time - initial_hold_s - prealign_s
            )

            if in_initial_hold:
                data.qpos[left_arm.qpos_indices] = bimanual_initial_q[:left_arm.dof]
                data.qpos[right_arm.qpos_indices] = bimanual_initial_q[left_arm.dof:]
                data.ctrl[left_arm.actuator_ids] = bimanual_initial_q[:left_arm.dof]
                data.ctrl[right_arm.actuator_ids] = bimanual_initial_q[left_arm.dof:]
                data.qvel[:] = 0.0
                data.time += model.opt.timestep
                mujoco.mj_forward(model, data)

                raw_targets = bimanual_targets_at_time(
                    DEMO_CONFIG,
                    0.0,
                    fixed_rotations,
                )
                targets = ramp_bimanual_targets(
                    raw_targets,
                    initial_object_position,
                    0.0,
                )
                with viewer.lock():
                    update_bimanual_target_markers(viewer, targets, fixed_rotations)
                    add_target_gripper_axis_markers(viewer, targets)
                    add_gripper_axis_markers(
                        viewer,
                        get_ee_transform(data, left_arm),
                        get_ee_transform(data, right_arm),
                    )
                viewer.sync()

                if step_count % 250 == 0:
                    print(
                        f"t={data.time:6.3f} s | phase=initial_hold | "
                        f"holding {start_posture_label} posture",
                        flush=True,
                    )
                step_count += 1
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                continue

            if in_prealign and DEMO_CONFIG.bimanual_joint_space_prealign:
                if prealign_s > 1e-9:
                    phase = np.clip(prealign_time / prealign_s, 0.0, 1.0)
                else:
                    phase = 1.0
                smooth_phase = phase * phase * (3.0 - 2.0 * phase)
                q_prealign = (
                    (1.0 - smooth_phase) * bimanual_initial_q
                    + smooth_phase * bimanual_prealign_q_nominal
                )
                data.qpos[left_arm.qpos_indices] = q_prealign[:left_arm.dof]
                data.qpos[right_arm.qpos_indices] = q_prealign[left_arm.dof:]
                data.ctrl[left_arm.actuator_ids] = q_prealign[:left_arm.dof]
                data.ctrl[right_arm.actuator_ids] = q_prealign[left_arm.dof:]
                data.qvel[:] = 0.0
                data.time += model.opt.timestep
                mujoco.mj_forward(model, data)

                raw_targets = bimanual_targets_at_time(
                    DEMO_CONFIG,
                    0.0,
                    fixed_rotations,
                )
                targets = ramp_bimanual_targets(
                    raw_targets,
                    initial_object_position,
                    0.0,
                )
                with viewer.lock():
                    update_bimanual_target_markers(viewer, targets, fixed_rotations)
                    add_target_gripper_axis_markers(viewer, targets)
                    add_gripper_axis_markers(
                        viewer,
                        get_ee_transform(data, left_arm),
                        get_ee_transform(data, right_arm),
                    )
                viewer.sync()

                if step_count % 250 == 0:
                    print(
                        f"t={data.time:6.3f} s | phase=prealign | "
                        f"joint_s={smooth_phase:.3f}",
                        flush=True,
                    )
                step_count += 1
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                continue

            if (
                control_phase == "active"
                and not prealign_completed
                and DEMO_CONFIG.bimanual_reset_nominal_after_prealign
            ):
                initial_left_transform = get_ee_transform(data, left_arm)
                initial_right_transform = get_ee_transform(data, right_arm)
                initial_object_position = 0.5 * (
                    initial_left_transform[:3, 3] + initial_right_transform[:3, 3]
                )
                bimanual_q_nominal = np.concatenate(
                    [
                        data.qpos[left_arm.qpos_indices].copy(),
                        data.qpos[right_arm.qpos_indices].copy(),
                    ]
                )
                prealign_completed = True
                print(
                    f"Prealign complete at t={control_time:.3f}s; "
                    "reset bimanual nominal posture.",
                    flush=True,
                )
            elif control_phase == "active":
                prealign_completed = True

            if in_prealign:
                phase_relative_orientation_weight = (
                    DEMO_CONFIG.bimanual_prealign_relative_orientation_weight
                )
                phase_orientation_task_weight = (
                    DEMO_CONFIG.bimanual_prealign_orientation_task_weight
                )
                phase_q_nominal = (
                    bimanual_prealign_q_nominal
                    if DEMO_CONFIG.bimanual_use_prealign_nominal
                    else bimanual_q_nominal
                )
            else:
                ramp_s = max(
                    1e-9,
                    float(DEMO_CONFIG.bimanual_relative_orientation_ramp_s),
                )
                orientation_ramp = np.clip(task_time / ramp_s, 0.0, 1.0)
                phase_relative_orientation_weight = (
                    orientation_ramp * DEMO_CONFIG.bimanual_relative_orientation_weight
                )
                phase_orientation_task_weight = (
                    DEMO_CONFIG.bimanual_orientation_task_weight
                )
                phase_q_nominal = bimanual_q_nominal
            raw_targets = bimanual_targets_at_time(
                DEMO_CONFIG,
                task_time,
                fixed_rotations,
            )
            if target_predictor is not None and control_phase == "active":
                prediction = target_predictor.update(
                    task_time,
                    raw_targets.object_center,
                )
                future_targets = bimanual_targets_at_time(
                    DEMO_CONFIG,
                    task_time + prediction.horizon_s,
                    fixed_rotations,
                )
                prediction_log_row = build_prediction_log_row(
                    step_count=step_count,
                    time_s=control_time,
                    task_time_s=task_time,
                    prediction=prediction,
                    future_actual_position=future_targets.object_center,
                )
                prediction_history.append(prediction_log_row)
                if (
                    DEMO_CONFIG.prediction_log_enabled
                    and step_count % prediction_log_stride == 0
                ):
                    if prediction_log_writer is None:
                        prediction_log_writer = csv.DictWriter(
                            prediction_log_file,
                            fieldnames=list(prediction_log_row.keys()),
                        )
                        prediction_log_writer.writeheader()
                    prediction_log_writer.writerow(prediction_log_row)
                    prediction_log_file.flush()
            if DEMO_CONFIG.bimanual_target_ramp_s > 0.0:
                ramp = task_time / DEMO_CONFIG.bimanual_target_ramp_s
            else:
                ramp = 1.0
            targets = ramp_bimanual_targets(
                raw_targets,
                initial_object_position,
                ramp,
            )
            if interception_planner is not None and control_phase == "active":
                left_transform_for_plan = get_ee_transform(data, left_arm)
                right_transform_for_plan = get_ee_transform(data, right_arm)
                current_object_position_for_plan = 0.5 * (
                    left_transform_for_plan[:3, 3]
                    + right_transform_for_plan[:3, 3]
                )

                def target_position_at_future_time(future_task_time: float) -> np.ndarray:
                    return bimanual_targets_at_time(
                        DEMO_CONFIG,
                        future_task_time,
                        fixed_rotations,
                    ).object_center

                interception_plan = interception_planner.plan(
                    time_s=task_time,
                    current_object_position=current_object_position_for_plan,
                    current_target_position=raw_targets.object_center,
                    target_position_at_time=target_position_at_future_time,
                )
                interception_blend_alpha = float(
                    np.clip(DEMO_CONFIG.interception_blend_alpha, 0.0, 1.0)
                )
                blended_interception_target = (
                    (1.0 - interception_blend_alpha) * raw_targets.object_center
                    + interception_blend_alpha
                    * interception_plan.interception_position
                )
                interception_log_row = build_interception_log_row(
                    step_count=step_count,
                    time_s=control_time,
                    task_time_s=task_time,
                    plan=interception_plan,
                    blend_alpha=interception_blend_alpha,
                    blended_target_position=blended_interception_target,
                    previous_interception_position=previous_interception_position,
                )
                interception_history.append(interception_log_row)
                previous_interception_position = (
                    interception_plan.interception_position.copy()
                )
                if (
                    DEMO_CONFIG.interception_log_enabled
                    and step_count % interception_log_stride == 0
                ):
                    if interception_log_writer is None:
                        interception_log_writer = csv.DictWriter(
                            interception_log_file,
                            fieldnames=list(interception_log_row.keys()),
                        )
                        interception_log_writer.writeheader()
                    interception_log_writer.writerow(interception_log_row)
                    interception_log_file.flush()

            with viewer.lock():
                update_bimanual_target_markers(viewer, targets, fixed_rotations)
                add_target_gripper_axis_markers(viewer, targets)
                add_gripper_axis_markers(
                    viewer,
                    get_ee_transform(data, left_arm),
                    get_ee_transform(data, right_arm),
                )

            clik_info = bimanual_object_clik_step(
                model=model,
                data=data,
                left_arm=left_arm,
                right_arm=right_arm,
                targets=targets,
                desired_relative_translation=desired_relative_translation,
                desired_relative_rotation=desired_relative_rotation_matrix,
                gain=DEMO_CONFIG.clik.gain,
                damping=DEMO_CONFIG.bimanual_damping,
                max_joint_step=DEMO_CONFIG.clik.max_joint_step,
                solver_mode=DEMO_CONFIG.bimanual_solver_mode,
                joint_velocity_limit=DEMO_CONFIG.bimanual_joint_velocity_limit,
                q_nominal=phase_q_nominal,
                qp_posture_weight=DEMO_CONFIG.bimanual_qp_posture_weight,
                qp_manipulability_weight=(
                    DEMO_CONFIG.bimanual_qp_manipulability_weight
                ),
                qp_regularization=DEMO_CONFIG.bimanual_qp_regularization,
                qp_eps_abs=DEMO_CONFIG.bimanual_qp_eps_abs,
                qp_eps_rel=DEMO_CONFIG.bimanual_qp_eps_rel,
                qp_max_iter=DEMO_CONFIG.bimanual_qp_max_iter,
                qp_max_null_velocity=DEMO_CONFIG.bimanual_qp_max_null_velocity,
                qp_null_error_scale_start=(
                    DEMO_CONFIG.bimanual_qp_null_error_scale_start
                ),
                qp_null_error_scale_stop=(
                    DEMO_CONFIG.bimanual_qp_null_error_scale_stop
                ),
                orientation_task_weight=phase_orientation_task_weight,
                relative_orientation_weight=phase_relative_orientation_weight,
                joint_limit_gain=DEMO_CONFIG.bimanual_joint_limit_gain,
                joint_limit_activation_margin=(
                    DEMO_CONFIG.bimanual_joint_limit_activation_margin
                ),
                manipulability_gain=DEMO_CONFIG.bimanual_manipulability_gain,
                manipulability_gradient_step=(
                    DEMO_CONFIG.bimanual_manipulability_gradient_step
                ),
                max_manipulability_command=(
                    DEMO_CONFIG.bimanual_max_manipulability_command
                ),
                adaptive_damping_alpha=DEMO_CONFIG.bimanual_adaptive_damping_alpha,
                adaptive_damping_epsilon=(
                    DEMO_CONFIG.bimanual_adaptive_damping_epsilon
                ),
                adaptive_damping_max=DEMO_CONFIG.bimanual_adaptive_damping_max,
                singularity_warning_threshold=(
                    DEMO_CONFIG.bimanual_singularity_warning_threshold
                ),
                max_object_command=DEMO_CONFIG.bimanual_max_object_command,
                max_relative_position_command=(
                    DEMO_CONFIG.bimanual_max_relative_position_command
                ),
                max_relative_rotation_command=(
                    DEMO_CONFIG.bimanual_max_relative_rotation_command
                ),
                relative_axis_local=DEMO_CONFIG.bimanual_relative_axis_local,
                relative_axis_face_each_other=(
                    DEMO_CONFIG.bimanual_relative_axis_face_each_other
                ),
            )

            if DEMO_CONFIG.bimanual_visualize_nullspace_comparison:
                with viewer.lock():
                    add_nullspace_comparison_markers(viewer, clik_info)

            diagnostics_history.append(
                {
                    "time": float(data.time),
                    "left_manipulability": clik_info["left_manipulability"],
                    "right_manipulability": clik_info["right_manipulability"],
                    "adaptive_damping": clik_info["adaptive_damping"],
                    "stacked_condition_number": clik_info[
                        "stacked_condition_number"
                    ],
                    "left_condition_number": clik_info[
                        "left_jacobian_condition_number"
                    ],
                    "right_condition_number": clik_info[
                        "right_jacobian_condition_number"
                    ],
                }
            )

            if clik_info["is_near_singularity"] and step_count % 100 == 0:
                print(
                    "WARNING: approaching singularity | "
                    f"w_left={clik_info['left_manipulability']:.3e} | "
                    f"w_right={clik_info['right_manipulability']:.3e} | "
                    f"damping={clik_info['adaptive_damping']:.3f}",
                    flush=True,
                )

            q_before_apply = np.concatenate(
                [
                    data.qpos[left_arm.qpos_indices].copy(),
                    data.qpos[right_arm.qpos_indices].copy(),
                ]
            )
            kinematic_alpha = 0.0
            if DEMO_CONFIG.bimanual_kinematic_control:
                kinematic_alpha = float(
                    np.clip(DEMO_CONFIG.bimanual_kinematic_alpha, 0.0, 1.0)
                )
                left_command = (
                    (1.0 - kinematic_alpha) * data.qpos[left_arm.qpos_indices]
                    + kinematic_alpha * clik_info["left_q_command"]
                )
                right_command = (
                    (1.0 - kinematic_alpha) * data.qpos[right_arm.qpos_indices]
                    + kinematic_alpha * clik_info["right_q_command"]
                )
                data.qpos[left_arm.qpos_indices] = left_command
                data.qpos[right_arm.qpos_indices] = right_command
                data.ctrl[left_arm.actuator_ids] = left_command
                data.ctrl[right_arm.actuator_ids] = right_command
                data.qvel[:] = 0.0
                data.time += model.opt.timestep
                mujoco.mj_forward(model, data)
            else:
                # Advance the physics simulation by one step.
                mujoco.mj_step(model, data)

            q_after_apply = np.concatenate(
                [
                    data.qpos[left_arm.qpos_indices].copy(),
                    data.qpos[right_arm.qpos_indices].copy(),
                ]
            )
            applied_delta = q_after_apply - q_before_apply
            level3_metrics = level3_acceptance_from_clik_info(clik_info)
            level3_acceptance_history.append(level3_metrics)
            if (
                DEMO_CONFIG.bimanual_control_log_enabled
                and step_count % control_log_stride == 0
            ):
                control_log_row = build_bimanual_control_log_row(
                    step_count=step_count,
                    time_s=control_time,
                    control_phase=control_phase,
                    ramp=ramp,
                    raw_targets=raw_targets,
                    targets=targets,
                    clik_info=clik_info,
                    applied_delta=applied_delta,
                    kinematic_alpha=kinematic_alpha,
                    control_dt=float(model.opt.timestep),
                )
                if control_log_writer is None:
                    control_log_writer = csv.DictWriter(
                        control_log_file,
                        fieldnames=list(control_log_row.keys()),
                    )
                    control_log_writer.writeheader()
                control_log_writer.writerow(control_log_row)
                control_log_file.flush()

            # Update the viewer with the latest simulation state.
            viewer.sync()

            if step_count % 250 == 0:
                left_transform = get_ee_transform(data, left_arm)
                right_transform = get_ee_transform(data, right_arm)
                relative_state = relative_pose_state(
                    left_transform,
                    right_transform,
                    relative_transform_desired,
                )
                qp_task_scaling = clik_info["qp_task_scaling"]
                qp_task_scaling_text = (
                    "n/a" if qp_task_scaling is None else f"{qp_task_scaling:.3f}"
                )
                qp_status = clik_info["qp_status"] or "n/a"
                level3_text = "PASS" if level3_metrics.passed else "FAIL"
                level3_failed_terms = (
                    "-"
                    if level3_metrics.passed
                    else ",".join(level3_metrics.failed_terms)
                )
                print(
                    f"t={data.time:6.3f} s | "
                    f"phase={control_phase} | "
                    f"L3={level3_text}({level3_failed_terms}) | "
                    f"obj={clik_info['object_error_norm']:.4f} m | "
                    f"rel_pos={clik_info['relative_position_error_norm']:.4f} m | "
                    f"rel_rot={clik_info['relative_orientation_error_norm']:.4f} rad | "
                    f"rel_w={phase_relative_orientation_weight:.3f} | "
                    f"L_rot={clik_info['left_rotation_error_norm']:.4f} rad | "
                    f"R_rot={clik_info['right_rotation_error_norm']:.4f} rad | "
                    f"null={clik_info['dq_null_norm']:.4f} | "
                    f"null_scale={clik_info['nullspace_motion_scale']:.2f} | "
                    f"limit={clik_info['min_joint_limit_margin']:.4f} | "
                    f"qp_alpha={qp_task_scaling_text} | "
                    f"qp={qp_status} | "
                    f"manip=({clik_info['left_manipulability']:.2e},"
                    f"{clik_info['right_manipulability']:.2e}) | "
                    f"lambda={clik_info['adaptive_damping']:.3f} | "
                    f"cond={clik_info['stacked_condition_number']:.2e} | "
                    f"rel_tf_pos={relative_state.translational_error_norm:.4f} m | "
                    f"rel_tf_rot={relative_state.rotational_error_norm:.4f} rad | "
                    f"object={targets.object_center}",
                    flush=True,
                )
            step_count += 1

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if control_log_file is not None:
        control_log_file.close()
        print(f"Saved bimanual control terms CSV: {control_log_path}")
    if prediction_log_file is not None:
        prediction_log_file.close()
        print(f"Saved target prediction CSV: {prediction_log_path}")
    if interception_log_file is not None:
        interception_log_file.close()
        print(f"Saved interception CSV: {interception_log_path}")
    print_level3_acceptance_summary(level3_acceptance_history)
    print_prediction_summary(prediction_history)
    print_interception_summary(interception_history)
    plot_bimanual_diagnostics(diagnostics_history)


if __name__ == "__main__":
    main()
