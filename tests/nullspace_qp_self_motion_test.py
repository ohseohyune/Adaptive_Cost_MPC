"""Null-space QP self-motion test in MuJoCo.

Scenario:
    0-2 s: keep the right end-effector fixed and hold the initial posture.
    2+ s: keep the same end-effector position fixed, but switch q_nom to a
          distant posture so only null-space motion can reconfigure the arm.

Run:
    python tests/nullspace_qp_self_motion_test.py
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from control.clik.mujoco_utils import build_serial_arm
from control.clik.nullspace_qp_controller import NullSpaceQPConfig, NullSpaceQPController
from control.clik.nullspace_qp_utils import (
    ee_position,
    ee_position_jacobian,
    joint_limit_margin,
    limit_vector_norm,
    null_motion_error_scale,
    set_arm_position,
    svd_pseudoinverse,
)
from control.clik.types import SerialArm
from robot.ffw_config import FFW_ARMS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
XML_PATH = PROJECT_ROOT / "model" / "robotis_ffw" / "scene_ffw_sg2.xml"
CONTROL_SUBSTEPS = 4
POSITION_COMMAND_LOOKAHEAD = 0.04
MOTION_SPEED_SCALE = 0.2

# ── Scenario timing ───────────────────────────────────────────────────────────
SWITCH_TIME: float = 2.0
# [s] time after which q_nom switch is allowed

SWITCH_POSITION_ERROR_THRESHOLD: float = 0.015
# [m] EE must be within this distance of target before q_nom switches

# ── Null-space motion gating ──────────────────────────────────────────────────
NULL_ERROR_SCALE_START: float = 0.012
# [m] null-space motion fully active when EE error is below this

NULL_ERROR_SCALE_STOP: float = 0.035
# [m] null-space motion fully suppressed when EE error exceeds this

MAX_NULL_VELOCITY: float = 1.5 * MOTION_SPEED_SCALE
# [rad/s] Euclidean norm cap on null-space velocity vector

# ── Task-space controller ─────────────────────────────────────────────────────
POSITION_GAIN: float = 64.0 * MOTION_SPEED_SCALE
# [1/s] proportional gain mapping EE position error [m] to xdot_desired [m/s]

QDOT_LIMIT: float = 3.0 * MOTION_SPEED_SCALE
# [rad/s] symmetric joint velocity limit applied to all joints

# ── Viewer ────────────────────────────────────────────────────────────────────
TARGET_SPHERE_RADIUS: float = 0.025
# [m] radius of the red target marker sphere in the MuJoCo viewer


class SimScenario:
    """Owns the two-phase test scenario: posture-hold → null-space reconfiguration.

    Phase transitions are irreversible: once reconfiguration starts, it does not
    revert even if the EE drifts back above the threshold.
    """

    def __init__(
        self,
        q_nom_hold: np.ndarray,
        q_nom_reconfigure: np.ndarray,
        switch_time: float,
        switch_error_threshold: float,
    ) -> None:
        self._q_nom_hold = q_nom_hold
        self._q_nom_reconfigure = q_nom_reconfigure
        self._switch_time = switch_time
        self._switch_error_threshold = switch_error_threshold
        self._reconfigure_started: bool = False

    @property
    def phase(self) -> str:
        """Return 'hold' or 'reconfig'."""
        return "reconfig" if self._reconfigure_started else "hold"

    @property
    def q_nominal(self) -> np.ndarray:
        """Return the active nominal posture for the QP."""
        return self._q_nom_reconfigure if self._reconfigure_started else self._q_nom_hold

    def try_switch(self, sim_time: float, position_error_norm: float) -> bool:
        """Trigger phase transition if time and error conditions are both met.
        Returns True on the step the transition occurs, False otherwise."""
        if (
            not self._reconfigure_started
            and sim_time >= self._switch_time
            and position_error_norm <= self._switch_error_threshold
        ):
            self._reconfigure_started = True
            return True
        return False


class SimLogger:
    """Decoupled stdout logger for the simulation loop.

    Separating logging from control logic means the control loop can run
    at full rate without conditional print logic cluttering the hot path.
    """

    def __init__(self, print_interval: float = 0.1) -> None:
        self._last_print_time: float = -1.0
        self._print_interval = print_interval

    def print_header(
        self,
        xml_path: Path,
        fixed_position: np.ndarray,
        q_nom_reconfigure: np.ndarray,
        switch_time: float,
        switch_threshold: float,
        control_dt: float,
    ) -> None:
        """Print the one-time startup summary."""
        print("Null-space QP self-motion test")
        print(f"XML: {xml_path}")
        print(f"Fixed right EE position: {fixed_position}")
        print(
            f"q_nom switches after {switch_time:.1f} s and after EE error is below "
            f"{switch_threshold:.3f} m."
        )
        print(f"q_nom reconfigure target: {q_nom_reconfigure}")
        print(
            f"control_dt={control_dt:.4f} s, "
            f"position_command_lookahead={POSITION_COMMAND_LOOKAHEAD:.3f} s, "
            f"motion_speed_scale={MOTION_SPEED_SCALE:.2f}"
        )
        print(
            "Columns: t | phase | pos_err | vel_task_err | null_vel | null_scale | "
            "min_margin | alpha | status"
        )

    def log_switch_event(self, sim_time: float, position_error_norm: float) -> None:
        """Print the q_nom switch event (called by SimScenario or the loop)."""
        print(
            f"--- q_nom switched at t={sim_time:.3f} s "
            f"(EE error={position_error_norm:.4f} m) ---",
            flush=True,
        )

    def log_step(
        self,
        sim_time: float,
        phase: str,
        position_error_norm: float,
        task_velocity_error_norm: float,
        null_velocity_norm: float,
        null_velocity_scale: float,
        min_margin: float,
        task_scaling: float,
        status: str,
    ) -> None:
        """Print one telemetry row if the print interval has elapsed."""
        if sim_time - self._last_print_time >= self._print_interval:
            print(
                f"t={sim_time:6.3f} | {phase:8s} | "
                f"pos_err={position_error_norm:.3e} | "
                f"vel_task_err={task_velocity_error_norm:.3e} | "
                f"null_vel={null_velocity_norm:.3e} | "
                f"null_scale={null_velocity_scale:.2f} | "
                f"min_margin={min_margin:.3f} | "
                f"alpha={task_scaling:.3f} | "
                f"{status}",
                flush=True,
            )
            self._last_print_time = sim_time


def run_simulation_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    right_arm: SerialArm,
    left_arm: SerialArm,
    q_left_initial: np.ndarray,
    fixed_position: np.ndarray,
    controller: NullSpaceQPController,
    scenario: SimScenario,
    logger: SimLogger,
    viewer: Any,
) -> None:
    """Run the MuJoCo simulation loop until the viewer is closed.

    Separated from main() so the loop can be called with a mock viewer
    in unit tests without spawning a real window.
    """
    qdot_limit = np.full(right_arm.dof, QDOT_LIMIT)
    control_dt = float(model.opt.timestep) * CONTROL_SUBSTEPS

    while viewer.is_running():
        step_start = time.time()
        sim_time = float(data.time)
        q_current = data.qpos[right_arm.qpos_indices].copy()
        current_position = ee_position(data, right_arm)
        position_error = fixed_position - current_position

        jacobian = ee_position_jacobian(model, data, right_arm)
        jacobian_pinv = svd_pseudoinverse(jacobian)
        xdot_desired = POSITION_GAIN * position_error
        qdot_task = jacobian_pinv @ xdot_desired
        nullspace_projector = np.eye(right_arm.dof) - jacobian_pinv @ jacobian

        position_error_norm = float(np.linalg.norm(position_error))
        if scenario.try_switch(sim_time, position_error_norm):
            logger.log_switch_event(sim_time, position_error_norm)

        result = controller.solve(
            q_current=q_current,
            qdot_task=qdot_task,
            nullspace_projector=nullspace_projector,
            q_nominal=scenario.q_nominal,
            dt=control_dt,
            q_min=right_arm.ctrl_low,
            q_max=right_arm.ctrl_high,
            qdot_min=-qdot_limit,
            qdot_max=qdot_limit,
        )

        task_qdot = result.qdot_task
        null_qdot = limit_vector_norm(
            nullspace_projector @ result.z,
            MAX_NULL_VELOCITY,
        )
        if scenario.phase == "reconfig":
            null_velocity_scale = null_motion_error_scale(
                position_error_norm,
                NULL_ERROR_SCALE_START,
                NULL_ERROR_SCALE_STOP,
            )
        else:
            null_velocity_scale = 0.0
        qdot_command = task_qdot + null_velocity_scale * null_qdot
        q_command = np.clip(
            q_current + POSITION_COMMAND_LOOKAHEAD * qdot_command,
            right_arm.ctrl_low,
            right_arm.ctrl_high,
        )
        data.ctrl[right_arm.actuator_ids] = q_command
        data.ctrl[left_arm.actuator_ids] = q_left_initial

        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)

        with viewer.lock():
            add_red_sphere(viewer, fixed_position)
        viewer.sync()

        task_velocity_error = jacobian @ qdot_command - xdot_desired
        null_velocity = null_velocity_scale * null_qdot
        min_margin = float(
            np.min(joint_limit_margin(q_current, right_arm.ctrl_low, right_arm.ctrl_high))
        )
        logger.log_step(
            sim_time=sim_time,
            phase=scenario.phase,
            position_error_norm=position_error_norm,
            task_velocity_error_norm=float(np.linalg.norm(task_velocity_error)),
            null_velocity_norm=float(np.linalg.norm(null_velocity)),
            null_velocity_scale=null_velocity_scale,
            min_margin=min_margin,
            task_scaling=result.task_scaling,
            status=result.status,
        )

        sleep_time = control_dt - (time.time() - step_start)
        if sleep_time > 0.0:
            time.sleep(sleep_time)


def add_red_sphere(viewer: Any, position: np.ndarray) -> None:
    viewer.user_scn.ngeom = 0
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([TARGET_SPHERE_RADIUS, 0.0, 0.0]),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.array([1.0, 0.05, 0.05, 0.85]),
    )
    viewer.user_scn.ngeom = 1


def make_distant_nominal(q_initial: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
    offset = np.array([0.9, -0.8, 1.1, -1.2, 0.9, -0.8, 0.7])
    return np.clip(q_initial + offset, q_min + 0.05, q_max - 0.05)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    right_arm = build_serial_arm(model, FFW_ARMS["right"])
    left_arm = build_serial_arm(model, FFW_ARMS["left"])

    # Avoid starting exactly on a joint limit. The right arm's second joint has
    # upper limit 0, so an all-zero pose makes the task-scaling guard overly
    # conservative and can freeze recovery motion.
    q_right_initial = np.array([0.0, -0.8, 0.4, -0.7, 0.3, 0.5, 0.0])
    q_left_initial = np.zeros(left_arm.dof)
    set_arm_position(data, right_arm, q_right_initial)
    set_arm_position(data, left_arm, q_left_initial)
    mujoco.mj_forward(model, data)

    fixed_position = ee_position(data, right_arm)
    q_initial = data.qpos[right_arm.qpos_indices].copy()
    q_nom_reconfigure = make_distant_nominal(q_initial, right_arm.ctrl_low, right_arm.ctrl_high)
    control_dt = float(model.opt.timestep) * CONTROL_SUBSTEPS

    controller = NullSpaceQPController(
        right_arm.dof,
        NullSpaceQPConfig(
            posture_weight=6.0,
            damping_weight=1e-4,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=200,
        ),
    )
    scenario = SimScenario(
        q_nom_hold=q_initial.copy(),
        q_nom_reconfigure=q_nom_reconfigure,
        switch_time=SWITCH_TIME,
        switch_error_threshold=SWITCH_POSITION_ERROR_THRESHOLD,
    )
    logger = SimLogger()
    logger.print_header(
        xml_path=XML_PATH,
        fixed_position=fixed_position,
        q_nom_reconfigure=q_nom_reconfigure,
        switch_time=SWITCH_TIME,
        switch_threshold=SWITCH_POSITION_ERROR_THRESHOLD,
        control_dt=control_dt,
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        run_simulation_loop(
            model=model,
            data=data,
            right_arm=right_arm,
            left_arm=left_arm,
            q_left_initial=q_left_initial,
            fixed_position=fixed_position,
            controller=controller,
            scenario=scenario,
            logger=logger,
            viewer=viewer,
        )


if __name__ == "__main__":
    main()
