"""Catch a ballistically-launched box with the differentiable AC-MPC pipeline.

This reuses the box scene, ballistic launch/prediction, and first-contact
impact-safety machinery already validated in ``main_dynamic_box_squeeze.py``,
but replaces its hand-engineered interception/squeeze state machine with the
same actor -> differentiable-MPC -> Cartesian-velocity pipeline used by
``main_bimanual_acmpc.py`` (now PPO+GAE trained). Box orientation is tracked
directly from the measured rotation rather than through the MPC's own state:
``DifferentiableBimanualMPC`` is a position-only model, and the curriculum's
angular velocities are gentle enough that a decoupled orientation overlay
(mirroring ``_ee_rotations_for_box`` in ``main_dynamic_box_squeeze.py``) is a
reasonable simplification for this first integration pass.

Phase A validated the ballistic-aware MPC reference and impact-safety
scheduling on a fixed nominal box (mass/friction/size held constant): 12/12
seeds succeed with engineered priors alone, and 30/30 with online_learning
enabled (see tests/phase12_acmpc_box_catch_test.py). Domain randomization
(``BoxDomainParameters``/``apply_box_domain_randomization``, the same
mass/friction/size/launch curriculum ``main_dynamic_box_squeeze.py`` uses)
is now wired in via ``AcmpcBoxCatchConfig.domain_parameters`` -- pass
``None`` (the default) for the original fixed-nominal-box behavior, or a
sampled ``BoxDomainParameters`` to catch a randomized box.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from control.clik import build_serial_arm, get_ee_transform
from control.clik.catching import adaptive_stiffness
from control.clik.impedance import CartesianImpedanceConfig, CartesianImpedanceController
from control.mpc import (
    ACMPCRolloutBuffer,
    BimanualPhase,
    DifferentiableMPCConfig,
    OnlineActorCriticACMPC,
    OnlineActorCriticConfig,
    PPOUpdateSummary,
    build_bimanual_observation,
)
from control.squeeze import (
    BallisticBoxPredictor,
    BoxDomainParameters,
    DynamicSideSqueezeConfig,
    FirstContactForceLimiter,
    adaptive_impact_command,
    apply_box_domain_randomization,
    read_bilateral_pad_contact,
    resolve_ballistic_launch_position,
    resolve_ballistic_launch_velocity,
)
from main_box_squeeze import LEFT_HOME_Q, RIGHT_HOME_Q
from main_dynamic_box_squeeze import (
    HAND_CAMERA_COLLISION_BIT,
    _disable_duplicate_end_effector_collisions,
)
from robot.ffw_config import FFW_ARMS, FFW_GRIPPERS


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "model/robotis_ffw/scene_ffw_sg2_fixed_base_box_dynamic_squeeze.xml"

_NOMINAL_BOX_MASS = 0.50
_NOMINAL_BOX_FRICTION = 1.20
_NOMINAL_BOX_HALF_SIZE = (0.055, 0.150, 0.055)


class CatchPhase(Enum):
    INTERCEPT = "intercept"
    PRE_CONTACT = "pre_contact"
    GRASPING = "grasping"
    GRASPED = "grasped"
    SUCCESS = "success"
    FAILED = "failed"


_PHASE_TO_BIMANUAL = {
    CatchPhase.INTERCEPT: BimanualPhase.INTERCEPT,
    CatchPhase.PRE_CONTACT: BimanualPhase.PRE_CONTACT,
    CatchPhase.GRASPING: BimanualPhase.GRASPING,
    CatchPhase.GRASPED: BimanualPhase.GRASPED,
    CatchPhase.SUCCESS: BimanualPhase.GRASPED,
    CatchPhase.FAILED: BimanualPhase.GRASPED,
}

# The handle-grasp demo's grasp weight (10) gives a hand-separation gain that
# is far too weak to close a large initial separation error within a ~0.5 s
# ballistic flight (measured: ~0.03 m/s of closing velocity per 0.1 m of
# error at weight 10, vs ~0.5 m/s at weight 250) -- the home pose's natural
# hand separation does not match the box's target pad separation, and this
# scenario has no prepare/ready-pose phase to close that gap in advance the
# way the box-squeeze track's does. Object/velocity/force/smoothness priors
# are left close to the original INTERCEPT/PRE_CONTACT values.
#
# The "force" column pulls the relative separation toward a *compressed*
# reference (relative_reference shrunk by grasp_compression), while "grasp"
# pulls it toward the raw, uncompressed contact separation -- the two
# compete, and the QP's effective target compression is the force-weighted
# share of grasp_compression (force / (grasp + force)) -- so, e.g., grasp=60,
# force=12 at grasp_compression=0.03 only ever pulls ~16% (~5 mm) of the way
# to full compression, well short of the several-mm-at-hundreds-of-N/m
# needed for a secure friction grip. Once actually touching (GRASPING
# onward), there is no reason to still weight exact-contact-surface
# tracking anywhere near force: raised force to dominate grasp instead.
_BOX_CATCH_PHASE_PRIORS = (
    (30.0, 250.0, 0.05, 4.0, 0.4),
    (30.0, 250.0, 1.5, 3.0, 0.5),
    (16.0, 40.0, 55.0, 1.5, 1.0),
    (18.0, 20.0, 65.0, 2.0, 1.5),
    (32.0, 18.0, 7.0, 3.0, 1.5),
)


@dataclass
class AcmpcBoxCatchConfig:
    seed: int = 7
    device: str = "auto"
    online_learning: bool = True
    # 0.08 (the handle-grasp demo's default) adds ~0.14 m/s of per-step
    # Cartesian velocity noise (std * mpc_velocity_limit) -- enough on its
    # own, with zero weight updates, to occasionally break the delicate
    # compliant-contact hold in this fast ballistic catch (a 30-seed sweep
    # at 0.08 only succeeded 7/10; the failures traced to exploration noise
    # during GRASPING/GRASPED, not to actor weight drift -- see
    # actor-weight-change tests in tests/phase12_acmpc_box_catch_test.py).
    # 0.03 was verified 30/30 across seeds while still producing nonzero
    # online_updates/actor_weight_change_l2 (genuine learning still happens,
    # just without destabilizing the hold).
    exploration_std: float = 0.03
    rollout_size: int = 16
    offline_training: bool = False
    maximum_online_actor_delta: float = 0.02
    mpc_horizon: int = 6
    mpc_velocity_limit: float = 1.8
    # A too-small lookahead starves the impedance controller's spring force
    # regardless of the nominal commanded velocity (CartesianImpedanceController
    # applies K*(desired-measured) - D*dx, and desired = measured + lookahead
    # *velocity, so the position error the spring actually sees scales
    # directly with this). 0.02 s (the original handle-grasp demo's slow,
    # near-static-approach default) was verified too small for this fast
    # ballistic catch to generate enough approach/grip force; 0.3 s is the
    # value the full seed sweep (10/10 with online_learning=False) was
    # validated against.
    command_lookahead_s: float = 0.3
    precontact_distance: float = 0.10
    # Separate from precontact_distance (which also gates the INTERCEPT ->
    # PRE_CONTACT phase transition -- widening that reintroduces the
    # too-early-switch bug fixed earlier this session). This one only feeds
    # the pre-contact stiffness-softening ratio below.
    stiffness_softening_distance: float = 0.10
    contact_force: float = 0.05
    stable_grasp_force: float = 1.0
    stable_grasp_time: float = 0.05
    # Matches the box-squeeze track's required_dynamic_hold_s=5.00 (see
    # control/squeeze/config.py). timeout_s must cover the ~0.45 s approach
    # + this hold, plus margin.
    required_hold_s: float = 5.00
    timeout_s: float = 7.0
    viewer: bool = False
    log_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    squeeze: DynamicSideSqueezeConfig = None  # type: ignore[assignment]
    # None (default) keeps the original fixed nominal box
    # (_NOMINAL_BOX_MASS/_NOMINAL_BOX_FRICTION/_NOMINAL_BOX_HALF_SIZE).
    # Passing a sampled BoxDomainParameters (e.g. via
    # control.squeeze.default_curriculum()[stage].sample(rng, stage)) mass/
    # friction/size/randomizes the box the same way
    # main_dynamic_box_squeeze.py does.
    domain_parameters: Optional[BoxDomainParameters] = None

    def __post_init__(self) -> None:
        if self.squeeze is None:
            self.squeeze = DynamicSideSqueezeConfig(random_seed=self.seed)
        if self.domain_parameters is not None and not np.isclose(
            self.squeeze.box_half_y, self.domain_parameters.half_size[1]
        ):
            raise ValueError(
                "squeeze.box_half_y must match domain_parameters.half_size[1] "
                "-- pad_offset_scalar (the pad-separation target) assumes a "
                "fixed grip-axis half-size, so randomization curricula must "
                "keep the Y half-size close to the configured nominal (this "
                "is why default_curriculum()'s stages only vary "
                "axis_scale[1] by a few percent)"
            )


@dataclass
class BoxCatchSummary:
    success: bool
    final_phase: str
    failure_reason: str
    simulated_time_s: float
    first_contact_time_s: Optional[float]
    first_contact_peak_force_n: float
    bilateral_contact_time_s: Optional[float]
    hold_time_s: float
    minimum_endpoint_error_m: float
    final_box_speed_mps: float
    online_updates: int
    total_transitions: int
    actor_weight_change_l2: float
    device: str


def _require_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(value)


def _limit_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm <= 1e-12:
        return vector
    return vector * (maximum / norm)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _reward(
    *,
    previous_endpoint_error: float,
    endpoint_error: float,
    left_force: float,
    right_force: float,
    velocity: np.ndarray,
    phase: CatchPhase,
    force_limit_exceeded: bool,
    emergency: bool,
) -> float:
    progress = 25.0 * (previous_endpoint_error - endpoint_error)
    endpoint_penalty = 2.5 * endpoint_error
    force_sum = min(left_force, 20.0) + min(right_force, 20.0)
    contact_reward = 0.025 * force_sum
    bilateral_bonus = 0.8 if left_force > 0.35 and right_force > 0.35 else 0.0
    force_balance_penalty = 0.006 * abs(left_force - right_force)
    effort_penalty = 0.01 * float(np.linalg.norm(velocity))
    phase_bonus = {
        CatchPhase.INTERCEPT: 0.0,
        CatchPhase.PRE_CONTACT: 0.10,
        CatchPhase.GRASPING: 0.30,
        CatchPhase.GRASPED: 0.60,
        CatchPhase.SUCCESS: 0.60,
        CatchPhase.FAILED: 0.0,
    }[phase]
    safety_penalty = 3.0 if force_limit_exceeded else 0.0
    safety_penalty += 5.0 if emergency else 0.0
    return float(
        progress
        - endpoint_penalty
        + contact_reward
        + bilateral_bonus
        + phase_bonus
        - force_balance_penalty
        - effort_penalty
        - safety_penalty
    )


def run_box_catch(config: Optional[AcmpcBoxCatchConfig] = None) -> BoxCatchSummary:
    config = config or AcmpcBoxCatchConfig()
    cfg = config.squeeze
    rng = np.random.default_rng(cfg.random_seed)
    sampled_launch_velocity = rng.uniform(cfg.launch_velocity_low, cfg.launch_velocity_high)
    launch_position = resolve_ballistic_launch_position(cfg, sampled_launch_velocity)
    launch_velocity = resolve_ballistic_launch_velocity(
        cfg, sampled_launch_velocity, launch_position=launch_position
    )
    gravity = np.asarray(cfg.gravity, dtype=float)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    if config.domain_parameters is not None:
        apply_box_domain_randomization(model, config.domain_parameters)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    control_stride = max(1, int(round(0.01 / dt)))
    control_dt = control_stride * dt

    arms = {name: build_serial_arm(model, FFW_ARMS[name]) for name in ("left", "right")}
    homes = {"left": LEFT_HOME_Q.copy(), "right": RIGHT_HOME_Q.copy()}
    gripper_ids = {
        name: np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                for actuator_name in FFW_GRIPPERS[name].actuator_names
            ],
            dtype=int,
        )
        for name in ("left", "right")
    }
    for name, arm in arms.items():
        data.qpos[arm.qpos_indices] = homes[name]
        data.ctrl[arm.actuator_ids] = homes[name]
        data.ctrl[gripper_ids[name]] = FFW_GRIPPERS[name].open_ctrl
    mujoco.mj_forward(model, data)

    box_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_box")
    box_geom_id = _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
    model.geom_conaffinity[box_geom_id] |= HAND_CAMERA_COLLISION_BIT | 1
    box_joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "dynamic_box_joint")
    box_qpos_address = int(model.jnt_qposadr[box_joint_id])
    box_dof_address = int(model.jnt_dofadr[box_joint_id])
    fixture_id = _require_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "box_launch_fixture")
    pad_ids = {
        name: _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_squeeze_pad")
        for name in ("left", "right")
    }
    _disable_duplicate_end_effector_collisions(model, set(pad_ids.values()))
    pad_box_pair_ids = tuple(
        _require_id(model, mujoco.mjtObj.mjOBJ_PAIR, name)
        for name in ("dynamic_left_pad_box", "dynamic_right_pad_box")
    )
    impact_pair_time_constants = {
        pair_id: float(model.pair_solref[pair_id, 0]) for pair_id in pad_box_pair_ids
    }

    data.qpos[box_qpos_address : box_qpos_address + 3] = launch_position
    mujoco.mj_forward(model, data)

    impedance = {
        name: CartesianImpedanceController(
            CartesianImpedanceConfig(
                K_pos=cfg.tangential_stiffness,
                K_rot=cfg.rotational_stiffness,
                D_pos=cfg.tangential_damping,
                D_rot=cfg.rotational_damping,
                tau_limit=280.0,
                Kp_ns=6.0,
                Kd_ns=2.5,
            ),
            arms[name],
            model,
            homes[name],
        )
        for name in ("left", "right")
    }
    desired_transforms = {
        name: get_ee_transform(data, arms[name]).copy() for name in ("left", "right")
    }
    # The MPC/actor only replan every control_stride physics substeps
    # (10 ms), but a fast-closing box can build a large contact force within
    # that same 10 ms window before the next replan has a chance to react --
    # base_commanded_position lets the first-contact force relief below be
    # applied every substep (reusing the impact/contact state already
    # computed at substep granularity) on top of the control-stride's
    # planned target, instead of only every 10 ms.
    base_commanded_position = {
        name: desired_transforms[name][:3, 3].copy() for name in ("left", "right")
    }

    predictor = BallisticBoxPredictor(
        gravity=gravity,
        velocity_alpha=cfg.predictor_velocity_alpha,
        max_speed=cfg.predictor_max_speed,
    )
    limiter = FirstContactForceLimiter(cfg)
    learner = OnlineActorCriticACMPC(
        DifferentiableMPCConfig(
            horizon=config.mpc_horizon,
            dt=control_dt,
            velocity_limit=config.mpc_velocity_limit,
            gravity=tuple(float(value) for value in gravity),
            # The MPC's "force" cost pulls the relative separation toward
            # relative_reference*(1 - grasp_compression/|relative_reference|)
            # -- a fixed compression *ceiling* no combination of weights can
            # exceed (raising w_force can only interpolate the solve's
            # effective target closer to this ceiling, never past it). The
            # handle-grasp demo's default (0.008 m) caps real achievable
            # normal force at a few N regardless of phase-prior weights,
            # nowhere near what a 0.5 kg box's own weight needs from
            # friction alone (~10 N/pad, see minimum_symmetric_squeeze_force
            # in control/squeeze/friction.py) to avoid slipping through the
            # pads under gravity during the hold.
            grasp_compression=0.030,
        ),
        OnlineActorCriticConfig(
            device=config.device,
            seed=config.seed,
            initial_log_std=float(np.log(max(config.exploration_std, 1e-6))),
            phase_priors=_BOX_CATCH_PHASE_PRIORS,
            maximum_online_actor_delta=config.maximum_online_actor_delta,
        ),
    )
    if config.checkpoint_path and Path(config.checkpoint_path).exists():
        learner.load(config.checkpoint_path)
    initial_actor = np.concatenate(
        [parameter.detach().cpu().numpy().ravel() for parameter in learner.actor.parameters()]
    )

    pad_offset_scalar = cfg.box_half_y + cfg.pad_half_thickness + cfg.precontact_gap
    if config.domain_parameters is not None:
        object_mass = config.domain_parameters.mass
        object_friction = config.domain_parameters.friction
        half_x, _, half_z = config.domain_parameters.half_size
    else:
        object_mass = _NOMINAL_BOX_MASS
        object_friction = _NOMINAL_BOX_FRICTION
        half_x, _, half_z = _NOMINAL_BOX_HALF_SIZE

    data.eq_active[fixture_id] = 0
    data.qvel[box_dof_address : box_dof_address + 3] = launch_velocity
    mujoco.mj_forward(model, data)
    predictor.reset(initial_velocity=launch_velocity)

    phase = CatchPhase.INTERCEPT
    phase_started = 0.0
    hold_timer = 0.0
    bilateral_contact_time: Optional[float] = None
    capture_center: Optional[np.ndarray] = None
    capture_reference_velocity: Optional[np.ndarray] = None
    capture_rotation: Optional[np.ndarray] = None
    failure_reason = ""
    previous_control_positions = np.concatenate(
        [desired_transforms["left"][:3, 3], desired_transforms["right"][:3, 3]]
    )
    previous_velocity = np.zeros(6)
    command_velocity = np.zeros(6)
    previous_endpoint_error = 0.0
    minimum_endpoint_error = float("inf")

    rollout = ACMPCRolloutBuffer()
    updates: list[PPOUpdateSummary] = []
    total_transitions = 0
    latest_update: Optional[PPOUpdateSummary] = None
    last_observation: Optional[np.ndarray] = None
    last_phase: Optional[BimanualPhase] = None
    last_ee_positions: Optional[np.ndarray] = None
    last_object_position: Optional[np.ndarray] = None
    last_object_velocity: Optional[np.ndarray] = None
    last_relative_reference: Optional[np.ndarray] = None
    last_previous_velocity: Optional[np.ndarray] = None
    last_action = None
    rows: list[dict[str, float | str]] = []

    viewer = None
    if config.viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)

    total_steps = int(np.ceil(config.timeout_s / dt))
    try:
        for step in range(total_steps):
            if viewer is not None and not viewer.is_running():
                failure_reason = "viewer closed"
                break
            time_s = float(data.time)
            box_position = data.xpos[box_body_id].copy()
            box_velocity = data.qvel[box_dof_address : box_dof_address + 3].copy()
            box_rotation = data.xmat[box_body_id].reshape(3, 3).copy()
            contact = read_bilateral_pad_contact(model, data, box_geom_name="dynamic_box_geom")
            impact = limiter.update(time_s, contact)
            if impact.first_contact_time_s is not None:
                # The XML pair is deliberately soft during impact (absorbs
                # the initial hit). Left this soft for the whole grip, the
                # box can bounce/oscillate against a springy contact instead
                # of being held -- this was missing here entirely (present
                # in main_dynamic_box_squeeze.py) and is a strong candidate
                # for the observed spin-and-escape right after bilateral
                # contact. Ramp to a firmer hold contact once the
                # force-limited window ends, smoothly rather than switching
                # discontinuously.
                transition_elapsed = max(
                    0.0,
                    time_s - impact.first_contact_time_s - cfg.first_contact_window_s,
                )
                contact_blend = _smoothstep(
                    transition_elapsed / max(cfg.hold_contact_transition_s, dt)
                )
                for pair_id in pad_box_pair_ids:
                    model.pair_solref[pair_id, 0] = (
                        (1.0 - contact_blend) * impact_pair_time_constants[pair_id]
                        + contact_blend * cfg.hold_contact_time_constant_s
                    )
                    model.pair_solref[pair_id, 1] = cfg.hold_contact_damping_ratio

            if step % control_stride == 0:
                prediction = predictor.update(time_s, box_position)
                # The MPC's own horizon lookahead is only
                # horizon*dt (~tens of ms) -- far shorter than the ~0.5 s
                # flight time -- so handing it the box's *current* position
                # as "object_position" leaves it always reacting to a target
                # that is still ~1 m away and closing fast, never given
                # enough real time to converge. Instead, aim at the
                # predicted *interception point* (mirroring
                # BoxFaceInterceptionPlanner in control/squeeze/ballistic.py)
                # and hand that to the MPC as a near-static reference
                # (zero object_velocity) that it tracks and settles onto
                # over the whole remaining flight, re-solved fresh each step
                # as the TTC estimate refines.
                # (catch_plane_x - position_x) and velocity_x are both
                # negative while approaching, so their ratio is the correct
                # positive time-to-plane -- do not negate velocity_x here
                # (that flips the sign and was clamping every estimate to
                # the minimum_intercept_ttc floor).
                remaining_ttc = float(
                    np.clip(
                        (cfg.catch_plane_x - prediction.position[0])
                        / min(prediction.velocity[0], -1e-3),
                        cfg.minimum_intercept_ttc,
                        cfg.maximum_intercept_ttc,
                    )
                )
                intercept_center = prediction.position_after(remaining_ttc)

                # Snapshot the captured reference (both position AND
                # rotation) once, exactly at the *instant* of first contact --
                # mirroring main_dynamic_box_squeeze.py, which captures the
                # moment impact.first_contact_time_s is first set (its
                # TRACK -> IMPACT/SINGLE_CONTACT transition), not at the end
                # of the compliant window. Capturing at window-end instead
                # (an earlier version of this file) left the box-catch
                # target chasing the box's live (still-settling) position for
                # the whole 0.12 s window, so a real position error was still
                # open exactly when tangential_k/normal_k step from their
                # soft impact-window values to full cfg stiffness below --
                # that step is a hard branch switch, not a ramp, so a
                # residual error at that instant produces a large
                # instantaneous force spike and rings/bounces the contact
                # (observed as force oscillating 0<->25N and 1-2 rad/s of
                # spin-up starting right at window-end). Capturing
                # immediately at contact gives the target the *entire*
                # window to converge under soft stiffness first, so little
                # to no error remains by the time full stiffness kicks in.
                #
                # Freezing rotation the same way matters independently: pad
                # orientation previously kept tracking the box's *live*
                # rotation unconditionally, so rotational impedance error was
                # ~always zero and provided no resistance to spin -- any
                # small left/right force asymmetry then had nothing opposing
                # the torque it induced, and the box's tiny rotational
                # inertia let that alone ramp angular speed up across the
                # grasp (same failure mode fixed for translation above, and
                # for the box-squeeze track earlier this session).
                if capture_center is None and impact.first_contact_time_s is not None:
                    capture_center = box_position.copy()
                    capture_reference_velocity = _limit_norm(
                        box_velocity, cfg.maximum_capture_speed
                    )
                    capture_rotation = box_rotation.copy()

                effective_rotation = (
                    capture_rotation if capture_rotation is not None else box_rotation
                )
                y_axis = effective_rotation[:, 1]
                pad_vector = pad_offset_scalar * y_axis

                if capture_center is not None:
                    # Past the compliant first-contact window: the box is no
                    # longer in free ballistic flight and its contact-force
                    # dynamics are done settling, so hold a decaying captured
                    # reference instead of chasing a moving prediction
                    # (mirrors capture_center/capture_reference_velocity in
                    # main_dynamic_box_squeeze.py's VELOCITY_MATCH/CAPTURE).
                    capture_reference_velocity = capture_reference_velocity * np.exp(
                        -control_dt / cfg.capture_linear_time_constant_s
                    )
                    capture_center = capture_center + capture_reference_velocity * control_dt
                    target_center = capture_center
                    target_velocity = capture_reference_velocity
                elif (
                    impact.first_contact_time_s is not None
                    or phase in {CatchPhase.PRE_CONTACT, CatchPhase.GRASPING}
                ):
                    # Inside the compliant first-contact window (or already
                    # close enough to have entered PRE_CONTACT): contact
                    # forces may already be acting, so the ballistic
                    # predictor's unforced-motion extrapolation is already
                    # wrong or about to be, but there isn't yet a stable
                    # captured point to freeze onto either. Track the box's
                    # live measured state directly. Switching to this at
                    # PRE_CONTACT rather than waiting for the exact instant
                    # of first contact also avoids a one-step discontinuity
                    # between the ballistic prediction and the live position
                    # right at touch, which was itself injecting a sudden
                    # ~0.1 m Z-target jump into the pad targets. Freezing a
                    # captured reference this early reproduces the exact
                    # "freeze too early" bug fixed this morning in the
                    # box-squeeze track.
                    #
                    # target_velocity is left None here (not the box's own
                    # velocity, and not clipped to maximum_capture_speed --
                    # that clip is for the post-window *arrest* target). The
                    # required-velocity feedforward computed below already
                    # accounts for whatever residual gap remains to this
                    # (now live, not predicted) target in the remaining TTC,
                    # which is what actually pulls the arm in fast enough;
                    # using raw box_velocity here starves that pull the
                    # moment there is still a real gap left to close.
                    target_center = box_position
                    target_velocity = None
                    # remaining_ttc is "time until the box crosses
                    # catch_plane_x" -- once we're this close/already
                    # touching, that clock has served its purpose (or the
                    # box has already crossed the plane, at which point the
                    # formula degenerates and pins to the floor while the
                    # residual gap keeps growing). Use a small fixed
                    # tracking time constant for the required-velocity
                    # divisor instead so the near-field correction stays
                    # meaningful regardless of where the box sits relative
                    # to that plane, and add the box's own (live, true)
                    # velocity as feedforward -- a fixed-time gap-closing
                    # term alone undershoots a target that is itself still
                    # accelerating (falling).
                    tracking_time_constant = 0.05
                    velocity_feedforward_base = box_velocity
                else:
                    target_center = intercept_center
                    target_velocity = None
                    tracking_time_constant = remaining_ttc
                    velocity_feedforward_base = np.zeros(3)

                left_target = target_center + pad_vector
                right_target = target_center - pad_vector
                if impact.in_first_contact_window:
                    # FirstContactForceLimiter.update() (above) only *reports*
                    # impact state -- the actual force cap comes from
                    # relief_distances(), which predicts each pad's
                    # near-future force (measured force + filtered rate *
                    # force_prediction_horizon_s) and, once that prediction
                    # crosses predictive_force_guard_ratio*first_contact_force_limit,
                    # returns an outward offset to relieve it. This call was
                    # missing entirely here (present in
                    # main_dynamic_box_squeeze.py's rotating branch as
                    # `left_target += y_axis * left_relief`) -- first-contact
                    # peak force routinely exceeded first_contact_force_limit
                    # (18 N) by 25-35% (measured 16-24 N, mean ~20.5 N) as a
                    # result, even though the catch itself still succeeded.
                    left_relief, right_relief = limiter.relief_distances(contact, impact)
                    left_target = left_target + y_axis * left_relief
                    right_target = right_target - y_axis * right_relief

                left_ee = get_ee_transform(data, arms["left"])[:3, 3]
                right_ee = get_ee_transform(data, arms["right"])[:3, 3]
                # Feed the MPC the arm's actual measured pose, not the last
                # commanded desired_transforms -- the impedance controller
                # does not track its target instantaneously, so using the
                # commanded pose as "current state" makes the MPC believe
                # it is already closer to the target than it physically is,
                # systematically under-correcting.
                ee_positions = np.concatenate([left_ee, right_ee])
                measured_velocity = (ee_positions - previous_control_positions) / control_dt
                previous_control_positions = ee_positions.copy()

                endpoint_error = 0.5 * (
                    float(np.linalg.norm(left_ee - left_target))
                    + float(np.linalg.norm(right_ee - right_target))
                )
                minimum_endpoint_error = min(minimum_endpoint_error, endpoint_error)
                relative_normal_speed = max(
                    abs(float(np.dot(box_velocity, y_axis))), 1e-6
                )
                impact_command = adaptive_impact_command(
                    cfg,
                    object_mass=object_mass,
                    contact_face_area=4.0 * half_x * half_z,
                    relative_normal_speed=relative_normal_speed,
                )

                old_phase = phase
                both_contact = contact.left.active and contact.right.active
                stable_contact = bool(
                    contact.left.normal_force > config.stable_grasp_force
                    and contact.right.normal_force > config.stable_grasp_force
                )
                if (
                    phase is CatchPhase.INTERCEPT
                    and endpoint_error < config.precontact_distance
                    and remaining_ttc <= cfg.ttc_soften_window_s
                ):
                    # endpoint_error alone is not a proximity signal: it is
                    # measured against intercept_center, a *future* point the
                    # arm can (and does) converge onto long before the box
                    # itself is anywhere nearby. Switching to PRE_CONTACT here
                    # also swaps the tracking target from intercept_center
                    # (continuous, TTC-based, converges onto box_position as
                    # remaining_ttc -> 0) to the box's raw live position --
                    # firing that swap while the box is still ~1 m / most of
                    # a second away snaps the target backward by that same
                    # ~1 m, which the arm then spends the rest of the flight
                    # failing to re-close. Require genuine time-proximity
                    # (mirrors the ttc_soften_window_s gate already used for
                    # impact-stiffness softening) before allowing the switch.
                    phase = CatchPhase.PRE_CONTACT
                elif phase is CatchPhase.PRE_CONTACT and both_contact:
                    phase = CatchPhase.GRASPING
                    bilateral_contact_time = time_s
                elif phase is CatchPhase.GRASPING:
                    hold_timer = hold_timer + control_dt if stable_contact else 0.0
                    if hold_timer >= config.stable_grasp_time:
                        phase = CatchPhase.GRASPED
                        hold_timer = 0.0
                elif phase is CatchPhase.GRASPED:
                    if stable_contact:
                        hold_timer += control_dt
                    else:
                        hold_timer = 0.0
                    if hold_timer >= config.required_hold_s:
                        phase = CatchPhase.SUCCESS
                if impact.emergency:
                    phase = CatchPhase.FAILED
                    failure_reason = "emergency contact force exceeded"
                elif (
                    phase in {CatchPhase.INTERCEPT, CatchPhase.PRE_CONTACT}
                    and box_position[0] < cfg.catch_plane_x - 0.16
                ):
                    phase = CatchPhase.FAILED
                    failure_reason = "box passed the interception workspace"
                if phase is not old_phase:
                    phase_started = time_s

                bimanual_phase = _PHASE_TO_BIMANUAL[phase]
                observation = build_bimanual_observation(
                    object_velocity=prediction.velocity,
                    left_endpoint_error=left_target - left_ee,
                    right_endpoint_error=right_target - right_ee,
                    left_ee_velocity=measured_velocity[:3],
                    right_ee_velocity=measured_velocity[3:],
                    left_force=contact.left.normal_force,
                    right_force=contact.right.normal_force,
                    time_to_contact=remaining_ttc,
                    prediction_confidence=prediction.confidence,
                    phase=bimanual_phase,
                )

                if last_observation is not None and config.online_learning:
                    reward = _reward(
                        previous_endpoint_error=previous_endpoint_error,
                        endpoint_error=endpoint_error,
                        left_force=contact.left.normal_force,
                        right_force=contact.right.normal_force,
                        velocity=command_velocity,
                        phase=phase,
                        force_limit_exceeded=impact.force_limit_exceeded,
                        emergency=impact.emergency,
                    )
                    rollout.add(
                        observation=last_observation,
                        phase=last_phase,
                        ee_positions=last_ee_positions,
                        object_position=last_object_position,
                        object_velocity=last_object_velocity,
                        relative_reference=last_relative_reference,
                        previous_velocity=last_previous_velocity,
                        action=last_action,
                        reward=reward,
                        done=phase in {CatchPhase.SUCCESS, CatchPhase.FAILED},
                    )
                    total_transitions += 1
                    if len(rollout) >= config.rollout_size:
                        bootstrap_value = learner.predict_value(observation)
                        latest_update = learner.update(
                            rollout,
                            online=not config.offline_training,
                            next_value=bootstrap_value,
                        )
                        updates.append(latest_update)
                        rollout.clear()

                relative_reference = right_target - left_target
                if target_velocity is not None:
                    # Captured: target_center/target_velocity already decay
                    # toward a hold point, so feed them directly.
                    object_velocity_feedforward = target_velocity
                else:
                    # Still approaching: the MPC's "velocity" cost term
                    # tracks object_velocity as a feedforward reference.
                    # Feeding it zero (as if the target were already
                    # reached) fights the "object" position cost's pull
                    # toward a still-distant point. Feed the average
                    # closing velocity actually required to arrive on time
                    # instead, so both cost terms agree on approaching fast.
                    # Pure "gap / time" guidance ignores the target's own
                    # motion (undershoots an accelerating box); pure "target
                    # velocity" ignores any residual position error (never
                    # closes a gap that's still open). Combine both: track
                    # velocity_feedforward_base (the box's own live velocity
                    # near-field, zero while still far) plus a proportional
                    # correction for whatever position error remains now.
                    ee_midpoint = 0.5 * (ee_positions[:3] + ee_positions[3:])
                    position_correction = (target_center - ee_midpoint) / max(
                        tracking_time_constant, control_dt
                    )
                    object_velocity_feedforward = velocity_feedforward_base + position_correction
                # Per-step Cartesian velocity exploration noise (~std, see
                # exploration_std's docstring above) has a small risk of
                # destabilizing the delicate compliant hold on any given
                # step -- negligible over the ~80-step 0.3 s window this was
                # tuned against (30/30), but that risk compounds over a much
                # longer hold (a 5 s hold is ~500 steps) and drove success
                # down to ~50% even at the actor's minimum std floor
                # (_LOG_STD_MIN in online_actor_critic.py). A control
                # experiment (rollout_size larger than the episode, so
                # ~zero weight updates fire) still failed 0/10 at the same
                # exploration_std, isolating the cause to the noise itself,
                # not actor drift. Once GRASPED, the ideal action is "hold
                # still" -- there is nothing left to explore that outweighs
                # the risk of injecting noise into an already-successful
                # hold, so stop sampling stochastic actions from that phase
                # onward (still using training=True, hence still collecting
                # rollout/learning signal, up through GRASPING).
                exploring = config.online_learning and phase not in {
                    CatchPhase.GRASPED,
                    CatchPhase.SUCCESS,
                }
                action = learner.act(
                    observation=observation,
                    phase=bimanual_phase,
                    ee_positions=ee_positions,
                    object_position=target_center,
                    object_velocity=object_velocity_feedforward,
                    relative_reference=relative_reference,
                    previous_velocity=previous_velocity,
                    training=exploring,
                )
                last_phase = bimanual_phase
                last_ee_positions = ee_positions
                last_object_position = target_center.copy()
                last_object_velocity = object_velocity_feedforward.copy()
                last_relative_reference = relative_reference
                last_previous_velocity = previous_velocity
                last_action = action
                command_velocity = action.velocity
                previous_velocity = action.mean_velocity
                previous_endpoint_error = endpoint_error
                last_observation = observation

                if impact.first_contact_time_s is None:
                    # remaining_ttc measures time until the box's *center*
                    # crosses catch_plane_x -- a fixed X reference tuned for
                    # the box-squeeze track's own arm placement. In this
                    # scenario, actual Y-face pad contact happens while the
                    # box center is still ~0.15 m short of catch_plane_x (box
                    # center around x~0.45 at first touch vs catch_plane_x
                    # =0.30), so remaining_ttc still reads ~0.10-0.12 s right
                    # up until contact, and the TTC-based ramp below barely
                    # softens before the impact-window branch takes over --
                    # first-contact peak force routinely landed at 20-24 N as
                    # a result. catch_plane_x is used in several other places
                    # (reachability, the "passed the interception workspace"
                    # failure check) tuned around 0.30, so changing it broke
                    # those instead of just fixing softening (verified: 0/20
                    # success at catch_plane_x=0.4). endpoint_error (measured
                    # Cartesian distance to target) is a directly-verified,
                    # accurate proximity signal instead -- take whichever of
                    # the two estimates is softer, so an inaccurate TTC can
                    # only make things safer, never override a genuinely
                    # close-range endpoint_error into staying too stiff.
                    proximity_ratio = float(
                        np.clip(
                            endpoint_error / max(config.stiffness_softening_distance, 1e-6),
                            0.0,
                            1.0,
                        )
                    )
                    tangential_k = min(
                        float(
                            adaptive_stiffness(
                                remaining_ttc,
                                cfg.tangential_stiffness,
                                impact_command.tangential_stiffness,
                                cfg.ttc_soften_window_s,
                            )
                        ),
                        proximity_ratio * cfg.tangential_stiffness
                        + (1.0 - proximity_ratio) * impact_command.tangential_stiffness,
                    )
                    normal_k = min(
                        float(
                            adaptive_stiffness(
                                remaining_ttc,
                                cfg.normal_stiffness,
                                impact_command.normal_stiffness,
                                cfg.ttc_soften_window_s,
                            )
                        ),
                        proximity_ratio * cfg.normal_stiffness
                        + (1.0 - proximity_ratio) * impact_command.normal_stiffness,
                    )
                    rotational_k = min(
                        float(
                            adaptive_stiffness(
                                remaining_ttc,
                                cfg.rotational_stiffness,
                                impact_command.rotational_stiffness,
                                cfg.ttc_soften_window_s,
                            )
                        ),
                        proximity_ratio * cfg.rotational_stiffness
                        + (1.0 - proximity_ratio) * impact_command.rotational_stiffness,
                    )
                elif impact.in_first_contact_window:
                    tangential_k = impact_command.tangential_stiffness
                    normal_k = impact_command.normal_stiffness
                    rotational_k = impact_command.rotational_stiffness
                else:
                    # contact_blend (computed above, shared with the
                    # pair_solref ramp) is 0 right as the window ends and
                    # eases to 1 over hold_contact_transition_s. Ramping the
                    # Cartesian impedance stiffness on the same schedule
                    # instead of stepping it straight to cfg.*_stiffness
                    # matters because D_pos/D_rot are NOT re-derived here --
                    # jumping K from the soft impact_command value to the
                    # full cfg value in one control step (e.g. 45 -> 800
                    # N/m) instantly divides the effective damping ratio by
                    # sqrt(K_new/K_old), so the same D that was
                    # well-damped against the soft K becomes badly
                    # underdamped against the stiff K. That rang the contact
                    # (force oscillating between ~0 and ~25 N) and pumped
                    # enough angular velocity into the box each ring to spin
                    # it out from between the pads within ~150 ms of the
                    # step, even though the residual position error at that
                    # instant was already down to a few millimeters.
                    tangential_k = (
                        1.0 - contact_blend
                    ) * impact_command.tangential_stiffness + contact_blend * cfg.tangential_stiffness
                    normal_k = (
                        1.0 - contact_blend
                    ) * impact_command.normal_stiffness + contact_blend * cfg.normal_stiffness
                    rotational_k = (
                        1.0 - contact_blend
                    ) * impact_command.rotational_stiffness + contact_blend * cfg.rotational_stiffness
                # A perfectly simultaneous bilateral touch is rare -- one pad
                # typically registers a step or two before the other. Rigid
                # rotational impedance fights the box's residual spin from
                # that one-sided touch and pumps angular velocity up sharply
                # before contact is lost entirely (same failure mode found
                # and fixed in the box-squeeze track's main_dynamic_box_squeeze.py
                # earlier this session). Soften it during first contact like
                # the linear axes, instead of leaving it at cfg.rotational_stiffness
                # unconditionally.
                for name in ("left", "right"):
                    impedance[name].K[:] = np.diag(
                        [rotational_k] * 3 + [tangential_k, normal_k, tangential_k]
                    )
                    impedance[name].D[:] = np.diag(
                        [cfg.rotational_damping] * 3
                        + [cfg.tangential_damping, cfg.normal_damping, cfg.tangential_damping]
                    )

                for index, name in enumerate(("left", "right")):
                    base_commanded_position[name] = (
                        ee_positions[3 * index : 3 * (index + 1)]
                        + config.command_lookahead_s
                        * command_velocity[3 * index : 3 * (index + 1)]
                    )
                    desired_transforms[name][:3, 3] = base_commanded_position[name]
                desired_transforms["left"][:3, :3] = np.column_stack(
                    [y_axis, -effective_rotation[:, 0], effective_rotation[:, 2]]
                )
                desired_transforms["right"][:3, :3] = np.column_stack(
                    [-y_axis, effective_rotation[:, 0], effective_rotation[:, 2]]
                )

                rows.append(
                    {
                        "time_s": time_s,
                        "phase": phase.value,
                        "box_x": float(box_position[0]),
                        "box_z": float(box_position[2]),
                        "left_force_n": contact.left.normal_force,
                        "right_force_n": contact.right.normal_force,
                        "box_speed_mps": float(np.linalg.norm(box_velocity)),
                        "box_angular_speed_radps": float(
                            np.linalg.norm(
                                data.qvel[box_dof_address + 3 : box_dof_address + 6]
                            )
                        ),
                        "tangential_k": tangential_k,
                        "normal_k": normal_k,
                        "endpoint_error_m": endpoint_error,
                        "left_err_x": float(left_ee[0] - left_target[0]),
                        "left_err_y": float(left_ee[1] - left_target[1]),
                        "left_err_z": float(left_ee[2] - left_target[2]),
                        "right_err_x": float(right_ee[0] - right_target[0]),
                        "right_err_y": float(right_ee[1] - right_target[1]),
                        "right_err_z": float(right_ee[2] - right_target[2]),
                        "actor_loss": 0.0 if latest_update is None else latest_update.actor_loss,
                        "critic_loss": 0.0 if latest_update is None else latest_update.critic_loss,
                    }
                )

            if impact.in_first_contact_window:
                # Apply force relief every substep, not just every
                # control_stride (10 ms) -- reusing relief_distances() (see
                # note above at the missing-call site) at the same cadence
                # limiter.update() already runs at, so a fast-closing box
                # cannot build a large contact force for up to a full 10 ms
                # before the first corrective outward nudge takes effect.
                left_relief, right_relief = limiter.relief_distances(contact, impact)
                desired_transforms["left"][:3, 3] = (
                    base_commanded_position["left"] + y_axis * left_relief
                )
                desired_transforms["right"][:3, 3] = (
                    base_commanded_position["right"] - y_axis * right_relief
                )
            else:
                desired_transforms["left"][:3, 3] = base_commanded_position["left"]
                desired_transforms["right"][:3, 3] = base_commanded_position["right"]

            for name in ("left", "right"):
                impedance[name].apply(model, data, arms[name], desired_transforms[name])
                data.ctrl[gripper_ids[name]] = FFW_GRIPPERS[name].open_ctrl

            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if phase in {CatchPhase.SUCCESS, CatchPhase.FAILED}:
                break
    finally:
        if viewer is not None:
            viewer.close()

    if phase not in {CatchPhase.SUCCESS, CatchPhase.FAILED}:
        failure_reason = failure_reason or "timeout"
        phase = CatchPhase.FAILED

    if last_observation is not None and config.online_learning:
        rollout.add(
            observation=last_observation,
            phase=last_phase,
            ee_positions=last_ee_positions,
            object_position=last_object_position,
            object_velocity=last_object_velocity,
            relative_reference=last_relative_reference,
            previous_velocity=last_previous_velocity,
            action=last_action,
            reward=0.0,
            done=True,
        )
        total_transitions += 1
    if len(rollout) and config.online_learning:
        latest_update = learner.update(rollout, online=not config.offline_training, next_value=0.0)
        updates.append(latest_update)
        rollout.clear()

    if config.checkpoint_path:
        learner.save(config.checkpoint_path)
    if config.log_path and rows:
        log_path = Path(config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    final_actor = np.concatenate(
        [parameter.detach().cpu().numpy().ravel() for parameter in learner.actor.parameters()]
    )
    final_box_speed = float(np.linalg.norm(data.qvel[box_dof_address : box_dof_address + 3]))
    return BoxCatchSummary(
        success=phase is CatchPhase.SUCCESS,
        final_phase=phase.value,
        failure_reason=failure_reason,
        simulated_time_s=float(data.time),
        first_contact_time_s=limiter.first_contact_time_s,
        first_contact_peak_force_n=float(limiter.peak_first_contact_force),
        bilateral_contact_time_s=bilateral_contact_time,
        hold_time_s=float(hold_timer),
        minimum_endpoint_error_m=float(minimum_endpoint_error),
        final_box_speed_mps=final_box_speed,
        online_updates=learner.update_count,
        total_transitions=total_transitions,
        actor_weight_change_l2=float(np.linalg.norm(final_actor - initial_actor)),
        device=str(learner.device),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-online-learning", action="store_true")
    parser.add_argument(
        "--command-lookahead",
        type=float,
        default=0.3,
        help="seconds the impedance target leads the measured EE pose",
    )
    parser.add_argument(
        "--log", default=str(ROOT / "sweep_results" / "acmpc_box_catch.csv")
    )
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_box_catch(
        AcmpcBoxCatchConfig(
            seed=args.seed,
            device=args.device,
            viewer=args.viewer,
            online_learning=not args.no_online_learning,
            command_lookahead_s=args.command_lookahead,
            log_path=args.log,
            checkpoint_path=args.checkpoint,
        )
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    if not summary.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
