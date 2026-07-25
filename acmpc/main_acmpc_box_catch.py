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
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.clik import build_serial_arm, get_ee_transform
from control.clik.catching import adaptive_stiffness
from control.clik.impedance import (
    CartesianImpedanceConfig,
    CartesianImpedanceController,
    ee_jacobian_world,
)
from control.mpc import (
    ACMPCRolloutBuffer,
    DifferentiableMPCConfig,
    OnlineActorCriticACMPC,
    OnlineActorCriticConfig,
    PPOUpdateSummary,
    build_bimanual_observation,
)
# COST_NAMES/observation_dim are not re-exported from control.mpc's package
# __init__ (only the shared, 5-BimanualPhase-sized API is) -- imported
# directly from the submodule instead of adding a new package-level export,
# to keep this generalization scoped to what box-catch actually needs.
from control.mpc.online_actor_critic import COST_NAMES, observation_dim
from control.mpc.wandb_logger import (
    WandbLogger,
    build_contact_log,
    build_episode_reward_log,
    build_mpc_weight_log,
    build_ppo_update_log,
    init_wandb,
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
from box_squeeze.main_box_squeeze import LEFT_HOME_Q, RIGHT_HOME_Q
from box_squeeze.main_dynamic_box_squeeze import (
    HAND_CAMERA_COLLISION_BIT,
    _disable_duplicate_end_effector_collisions,
)
from robot.ffw_config import FFW_ARMS, FFW_GRIPPERS


SCENE = ROOT / "model/robotis_ffw/scene_ffw_sg2_fixed_base_box_dynamic_squeeze.xml"

_NOMINAL_BOX_MASS = 0.50
_NOMINAL_BOX_FRICTION = 1.20
_NOMINAL_BOX_HALF_SIZE = (0.055, 0.150, 0.055)
_GRAVITY_MPS2 = 9.81
# AC-MPC box-catch's own contact surface (left/right_catch_pad in
# ffw_sg2.xml), separate from left/right_squeeze_pad so box_squeeze and the
# bimanual handle-grasp demo (which use the squeeze pads/ee_site unmodified)
# are unaffected. Positioned further out to clear arm_l/r_link7's mesh,
# which protrudes 3.9cm further toward the box than the squeeze pad's own
# position -- always active (not phase-gated) since it's already correctly
# positioned and compliant-tuned to be the first point of contact.
CATCH_PAD_COLLISION_BIT = 64


class CatchPhase(Enum):
    """Episode state machine.

    INTERCEPT/PRE_IMPACT/CAPTURE/HOLD are the four *control* phases -- each
    has a row in _BOX_CATCH_PHASE_PRIORS and a slot in the actor's phase
    observation/prior blending (see CatchControlPhase below). SUCCESS and
    FAILED are terminal states, not control phases: reaching either ends the
    episode without computing another action (see the `terminal` handling in
    run_box_catch), and neither has a phase-prior row or an observation slot.

    Renamed from the original PRE_CONTACT/GRASPING/GRASPED to match the
    physical phases this scenario actually goes through -- PRE_IMPACT (an
    impending but not yet real touch), CAPTURE (some but not yet stable
    bilateral contact), HOLD (stable bilateral grip) -- and to stop
    overloading "success" as a fifth in-loop phase when it is really the
    terminal outcome of HOLD.
    """

    INTERCEPT = "intercept"
    PRE_IMPACT = "pre_impact"
    CAPTURE = "capture"
    HOLD = "hold"
    SUCCESS = "success"
    FAILED = "failed"


class CatchControlPhase(IntEnum):
    """Index into _BOX_CATCH_PHASE_PRIORS / the actor's phase observation.

    Deliberately a box-catch-local enum (not the shared, 5-member
    BimanualPhase in control/mpc/online_actor_critic.py, which the unrelated
    handle-grasp demo in main_bimanual_acmpc.py depends on by name and by
    count) -- this scenario has no manipulation phase and only 4 control
    phases, and OnlineActorCriticConfig.n_phases/phase_priors below size the
    actor/critic and observation to match this enum's 4 members instead of
    the shared enum's 5.
    """

    INTERCEPT = 0
    PRE_IMPACT = 1
    CAPTURE = 2
    HOLD = 3


N_CONTROL_PHASES = len(CatchControlPhase)

_CATCH_PHASE_TO_CONTROL_INDEX = {
    CatchPhase.INTERCEPT: CatchControlPhase.INTERCEPT,
    CatchPhase.PRE_IMPACT: CatchControlPhase.PRE_IMPACT,
    CatchPhase.CAPTURE: CatchControlPhase.CAPTURE,
    CatchPhase.HOLD: CatchControlPhase.HOLD,
    # SUCCESS/FAILED intentionally absent: both are terminal (see CatchPhase's
    # docstring) and run_box_catch never looks up a control index for them --
    # it breaks out of the control loop before reaching that point.
}

# Per-phase diagnostic init modes (see build_diagnostic_state). Each
# test_* mode places the box directly in that phase's regime -- reusing the
# real scene/impedance/MPC, not a separate simplified model -- so a phase
# can be evaluated repeatedly regardless of whether the earlier phases would
# have succeeded. "full_episode"/"test_intercept" are both the existing,
# unmodified far-away ballistic launch (INTERCEPT start); test_intercept is
# just an explicit name for it, not a different init.
DIAGNOSTIC_MODES = (
    "full_episode",
    "test_intercept",
    "test_pre_impact",
    "test_capture_left_contact",
    "test_capture_right_contact",
    "test_hold",
)
_DIAGNOSTIC_MODE_PHASE = {
    "test_pre_impact": CatchPhase.PRE_IMPACT,
    "test_capture_left_contact": CatchPhase.CAPTURE,
    "test_capture_right_contact": CatchPhase.CAPTURE,
    "test_hold": CatchPhase.HOLD,
}

# object, grasp geometry, compression proxy, velocity feed-forward, command
# smoothness -- indexed by CatchControlPhase. COST_NAMES[2] (in
# control/mpc/online_actor_critic.py) is still literally called "force" for
# backward compatibility with the unrelated handle-grasp demo's checkpoints/
# code (see that module's comments), but it does not track measured or
# predicted contact force -- it pulls the relative left/right separation
# toward a *compressed position* reference, so it is labeled "compression"
# here and in this file's logging (see _log_cost_contributions below).
# Measured contact force (contact.left/right.normal_force) is only used
# below for the actor's observation and for phase-transition/success/safety
# judgment, never as an MPC tracking target.
#
# Row values carried over unchanged from the original 5-row table (dropping
# the MANIPULATION row, which this scenario never used -- box-catch has no
# post-catch manipulation task) -- CatchControlPhase.INTERCEPT/PRE_IMPACT/
# CAPTURE/HOLD map 1:1 onto the original table's INTERCEPT/PRE_CONTACT/
# GRASPING/GRASPED rows. These are starting values, not re-validated for the
# renamed phases' entry/exit conditions changing (see AcmpcBoxCatchConfig's
# comments on precontact_confidence_min/hold_entry_dwell_s below) -- the
# physical scale each column operates on differs (position-error weights vs.
# a velocity-feedforward weight vs. a smoothness/damping weight), so do not
# read relative magnitude across columns as relative importance.
#
# The handle-grasp demo's grasp weight (10) gives a hand-separation gain that
# is far too weak to close a large initial separation error within a ~0.5 s
# ballistic flight (measured: ~0.03 m/s of closing velocity per 0.1 m of
# error at weight 10, vs ~0.5 m/s at weight 250) -- the home pose's natural
# hand separation does not match the box's target pad separation, and this
# scenario has no prepare/ready-pose phase to close that gap in advance the
# way the box-squeeze track's does. Object/velocity/compression/smoothness
# priors are left close to the original INTERCEPT/PRE_CONTACT values.
#
# The "compression" column pulls the relative separation toward a
# *compressed* reference (relative_reference shrunk by grasp_compression),
# while "grasp" pulls it toward the raw, uncompressed contact separation --
# the two compete, and the QP's effective target compression is the
# compression-weighted share of grasp_compression (compression / (grasp +
# compression)) -- so, e.g., grasp=60, compression=12 at
# grasp_compression=0.03 only ever pulls ~16% (~5 mm) of the way to full
# compression, well short of the several-mm-at-hundreds-of-N/m needed for a
# secure friction grip. Once actually touching (CAPTURE onward), there is no
# reason to still weight exact-contact-surface tracking anywhere near
# compression: raised compression to dominate grasp instead.
_BOX_CATCH_PHASE_PRIORS = (
    (30.0, 250.0, 0.05, 4.0, 0.4),   # INTERCEPT
    (30.0, 250.0, 1.5, 3.0, 0.5),    # PRE_IMPACT
    (16.0, 40.0, 55.0, 1.5, 1.0),    # CAPTURE
    (18.0, 20.0, 65.0, 2.0, 1.5),    # HOLD
)


@dataclass
class AcmpcBoxCatchConfig:
    seed: int = 7
    device: str = "auto"
    online_learning: bool = True
    # Distance left_catch_pad/right_catch_pad are mounted further along
    # local-x than left_ee/right_ee (see ffw_sg2.xml) -- added to
    # pad_offset_scalar so the tracked ee point's target moves out by the
    # same amount the catch pad is mounted in, keeping the pad's actual
    # contact point calibrated correctly. Measured directly from forward
    # kinematics at the home posture (mj_forward, LEFT_HOME_Q/RIGHT_HOME_Q):
    # ||site_xpos[left_catch_pad_site] - site_xpos[left_ee]|| = 0.0386 m
    # (and symmetrically for right), almost purely along world y (the grip
    # separation axis) -- not the previous 0.0441, which over-corrected the
    # standoff by 5.5mm relative to the model's actual geometry.
    catch_pad_standoff: float = 0.0386
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
    # Bounds cumulative drift from the engineered prior across all episodes
    # sharing a checkpoint. Without this, a curriculum training loop's
    # online updates (bounded individually by maximum_online_actor_delta,
    # but with no cap on their sum) can slowly walk the actor away from a
    # validated prior into a much worse region -- measured on the "full"
    # curriculum stage: 81%/93% success in episodes 0-40 degrading to ~38%
    # by episodes 74-99 (~1600+ cumulative updates), specifically via
    # INTERCEPT's first-horizon-step velocity weight dropping ~62% below
    # its tuned value and GRASPED's force weight rising ~55% above it.
    maximum_cumulative_actor_delta: Optional[float] = 0.4
    # 3-way ablation switch (see acmpc_box_catch_integration_status.md):
    # False (default) = AdaptiveCostActor, a bounded residual around
    # _BOX_CATCH_PHASE_PRIORS. True = PriorFreeCostActor, cost weights
    # learned from scratch via PPO with no engineered phase prior at all --
    # _BOX_CATCH_PHASE_PRIORS is then unused.
    use_prior_free_actor: bool = False
    # Overridable for ablations (e.g. deliberately degrading the prior to
    # test whether the learned residual actually compensates for a real
    # gap, vs. the default already-tuned table leaving little for it to
    # do). Defaults to the validated table below.
    phase_priors: tuple = ()  # replaced with _BOX_CATCH_PHASE_PRIORS in __post_init__
    # A phase-agnostic single ratio (object, grasp, force, velocity,
    # smoothness -- COST_NAMES order), not a per-phase table -- the plain
    # average of _BOX_CATCH_PHASE_PRIORS's 5 rows per column. A uniform
    # scalar here (all 5 equal) was tried first and never produced enough
    # tracking aggressiveness to close the gap to a fast box even when the
    # box was slowed and moved closer (see PriorFreeCostActor's docstring):
    # equal weights make the smoothness term (resists changing velocity)
    # compete one-for-one with object/grasp tracking. This keeps the
    # ablation "no per-phase engineered table" while breaking that symmetry.
    prior_free_initial_weights: tuple[float, float, float, float, float] = (
        25.2,
        115.6,
        25.71,
        2.7,
        0.98,
    )
    # Amplifies only the contact/hold-related reward terms (see _reward's
    # docstring-comment). 1.0 (default) is a no-op; the prior-free training
    # curriculum raises this since it rarely reaches contact at all early on
    # and needs a stronger signal on the rare occasions it does.
    hold_reward_scale: float = 1.0
    # One-time terminal reward/penalty on reaching SUCCESS/FAILED, on top of
    # the per-step shaping terms in _reward().
    success_reward: float = 50.0
    failure_penalty: float = 50.0
    # How far AdaptiveCostActor's residual can move each cost weight from
    # its phase prior, as a fraction (0.65 = weights range over
    # prior*[0.35, 1.65], before the hard 500.0 clamp). Unused when
    # use_prior_free_actor=True.
    weight_delta_fraction: float = 0.65
    mpc_horizon: int = 8
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
    # d_pre: fallback distance gate for INTERCEPT -> PRE_IMPACT (see
    # precontact_confidence_min below for the primary TTC-based gate). This
    # is compared against the actual measured hand-to-box distance (not
    # endpoint_error, which measures hand-to-*target* and can be small while
    # the box itself is still far away -- see the transition site's
    # comments), so it only fires once the box is genuinely nearby,
    # independent of whether the TTC estimate is trustworthy yet.
    precontact_distance: float = 0.10
    # c_min: minimum BallisticBoxPredictor.confidence (see
    # control/squeeze/ballistic.py -- confidence ramps from 0 to 1 over the
    # predictor's first 4 samples/control-steps, i.e. ~40ms, then stays at
    # 1.0) required before the TTC estimate is trusted for the INTERCEPT ->
    # PRE_IMPACT gate; below this, only the precontact_distance fallback can
    # trigger the transition. Set loosely below the predictor's steady-state
    # value (1.0) rather than at some fraction picked without a physical
    # basis: by the time a real ballistic flight is anywhere near the
    # precontact window (hundreds of ms in), confidence has long since
    # saturated, so this gate is only ever the limiting factor during the
    # first ~40ms of an episode (when it should not be trusted anyway).
    precontact_confidence_min: float = 0.75
    # Separate from precontact_distance (which also gates the INTERCEPT ->
    # PRE_IMPACT phase transition -- widening that reintroduces the
    # too-early-switch bug fixed earlier this session). This one only feeds
    # the pre-contact stiffness-softening ratio below.
    stiffness_softening_distance: float = 0.10
    # F_detect: PRE_IMPACT -> CAPTURE fires once either pad's measured normal
    # force exceeds this (see read_bilateral_pad_contact's `.active`
    # boolean, combined with this threshold, at the transition site). Left
    # at its original (previously unused) default -- small enough that
    # "first measurable touch" is what triggers CAPTURE, matching CAPTURE's
    # definition (some, not yet stable, contact) rather than requiring an
    # already-substantial force.
    contact_detect_force_n: float = 0.05
    # hold_entry_dwell: how long CAPTURE -> HOLD's "both pads >=
    # required_grip_force" condition (see run_box_catch's required_grip_force
    # = strict_grip_force_margin*mg/(2*mu)) must hold continuously before
    # promoting, and (reused, see the HOLD -> CAPTURE demotion comments in
    # run_box_catch) how long it must stay *broken* before demoting back.
    # 0.05 s = 5 control periods at control_dt=0.01s: long enough to not
    # promote/demote off a single noisy contact-force sample (MuJoCo contact
    # forces can read a brief zero or spike on the exact step contact
    # geometry updates), short enough to stay well under the 5 s hold this
    # gates into, so it does not meaningfully delay reaching HOLD. Carried
    # over from the original stable_grasp_time default, whose value was
    # already tuned against this same physical noise floor.
    hold_entry_dwell_s: float = 0.05
    # Time a phase-prior/phase-observation smoothstep blend takes to go from
    # the old phase's value (beta=0) to the new phase's value (beta=1) after
    # a transition -- see _smoothstep and the blending state in
    # run_box_catch. 0.08 s = 8 control periods: long enough that a single
    # discrete phase-prior jump is spread over several QP solves instead of
    # hitting the differentiable MPC as one discontinuous step, short enough
    # that it has fully converged well before the next physically meaningful
    # event in this scenario (the fastest phase, CAPTURE, is gated by
    # hold_entry_dwell_s=0.05s of its own, so an 0.08s blend does not delay
    # the cost table from reflecting a new phase's real requirements for
    # long). This value has not been re-validated against the full seed
    # sweep -- treat it as a reasonable starting point, not a tuned constant.
    phase_blend_time_s: float = 0.08
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
    # Physics-grounded stability criteria. NOTE: despite this being written
    # up above domain_parameters as if "evaluation only", the HOLD-phase
    # branch in run_box_catch actually gates the *official* hold_timer (and
    # therefore success/reward) with strict_stable_contact built from these
    # exact fields -- there is no separate eval-only copy. F_req =
    # strict_grip_force_margin * mg / (2 * mu) is the minimum per-pad normal
    # force for the two-sided friction grip to support the box's weight;
    # the margin is a safety factor above that theoretical minimum.
    strict_grip_force_margin: float = 1.3
    strict_grip_force_max_n: float = 32.0
    strict_box_speed_max_mps: float = 0.05
    # 0.10 (original) let a single held box's natural contact-noise angular
    # jitter reset the 5s HOLD streak: SC3 (seed=7, engineered priors) held
    # median 0.086 rad/s but p90=0.146/p99=0.74, so 0.10 broke on noise, not
    # instability -- max continuous clean streak was only 3.05s. Swept
    # against that same run's logged angular-speed trace: 0.15 is the
    # smallest value that clears 5.0s (yields 5.84s); loosening further past
    # it barely helps (0.20 -> 6.00s, 0.30 -> 6.09s, plateaus ~6.1s), so this
    # is the minimal fix, not a wide-open bound.
    strict_box_angular_speed_max_radps: float = 0.15
    # W&B logging (see control/mpc/wandb_logger.py). Only takes effect when
    # run_box_catch is not given an explicit wandb_logger= (a caller that
    # manages its own run across many episodes, e.g. the curriculum loop,
    # always passes one and this flag is ignored). False is a no-op for
    # every existing caller/test.
    use_wandb: bool = False
    wandb_run_name: Optional[str] = None
    # Control steps between W&B mpc/state/catch log points -- these fire at
    # control-step frequency, so a too-small interval floods the dashboard.
    wandb_log_interval: int = 10
    # Per-phase diagnostic init (see DIAGNOSTIC_MODES/build_diagnostic_state
    # below): overrides the box's initial qpos/qvel and the episode's
    # starting phase so a phase can be tested repeatedly without needing
    # the earlier phases to succeed first. "full_episode" (default) is an
    # exact no-op -- the normal far-away ballistic launch, INTERCEPT start.
    diagnostic_mode: str = "full_episode"

    def __post_init__(self) -> None:
        if self.diagnostic_mode not in DIAGNOSTIC_MODES:
            raise ValueError(
                f"diagnostic_mode must be one of {DIAGNOSTIC_MODES}, got {self.diagnostic_mode!r}"
            )
        if self.squeeze is None:
            self.squeeze = DynamicSideSqueezeConfig(random_seed=self.seed)
        if not self.phase_priors:
            self.phase_priors = _BOX_CATCH_PHASE_PRIORS
        if not self.use_prior_free_actor and len(self.phase_priors) != N_CONTROL_PHASES:
            raise ValueError(
                f"phase_priors must have {N_CONTROL_PHASES} rows (one per "
                f"CatchControlPhase), got {len(self.phase_priors)}"
            )
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
    # Control steps actually run this episode, regardless of
    # online_learning -- unlike total_transitions (only incremented inside
    # the online_learning-gated rollout block, so it stays 0 for a fixed
    # baseline run), this is always a true step count. Use this, not
    # total_transitions, for a global_step counter that must stay
    # monotonic across both fixed-baseline and online-learning episodes.
    control_step_count: int
    actor_weight_change_l2: float
    device: str
    total_reward: float
    mean_reward_per_step: float
    mean_actor_loss: float
    mean_critic_loss: float
    mean_entropy: float
    mean_approximate_kl: float
    strict_success: bool
    strict_hold_time_s: float
    required_grip_force_n: float
    # --- Cost-contribution / phase-blending diagnostics (aggregated over the
    # episode, not logged per-step, to keep this summary -- the thing a
    # curriculum run actually rolls up across episodes -- from being
    # swamped by control-step-frequency noise; the full per-step detail
    # (weight/*, cost_raw/*, cost_weighted/* below) still goes to the CSV
    # log_path when one is given). See _log_cost_contributions's docstring
    # for exactly how raw/weighted costs are computed.
    phase_transition_count: int
    # The only *backward* transition in this state machine (HOLD -> CAPTURE
    # demotion) -- a direct chatter signal, unlike phase_transition_count
    # which also counts normal forward progress.
    hold_to_capture_demotion_count: int
    intercept_dwell_s: float
    pre_impact_dwell_s: float
    capture_dwell_s: float
    hold_dwell_s: float
    # Fraction of control steps where the phase-prior/observation smoothstep
    # blend was still in progress (beta < 1) -- "transition 중인 비율".
    blend_active_fraction: float
    # Fraction of (control step x velocity component) samples where the
    # commanded Cartesian velocity sat at/near the MPC's tanh saturation
    # bound (|v| >= 0.98*mpc_velocity_limit).
    velocity_saturation_fraction: float
    mean_mpc_solve_time_s: float
    # Fraction of (control step x cost dim x horizon step) actor-output
    # weight samples at/near AdaptiveCostActor's hard clamp bounds
    # [1e-3, 500.0] -- a weight pinned at a bound means the residual wanted
    # to move further than delta_fraction/the clamp allowed.
    weight_lower_bound_hit_fraction: float
    weight_upper_bound_hit_fraction: float
    # Mean/std of each cost dimension's raw (unweighted) first-horizon-step
    # residual across the episode -- see _log_cost_contributions.
    mean_residual_object: float
    std_residual_object: float
    mean_residual_grasp: float
    std_residual_grasp: float
    mean_residual_compression: float
    std_residual_compression: float
    mean_residual_velocity: float
    std_residual_velocity: float
    mean_residual_smoothness: float
    std_residual_smoothness: float
    # Mean absolute step-to-step change in the actor's mean output weight
    # (averaged over cost dims and horizon), restricted to steps that stayed
    # within the same control phase -- a cheap proxy for how sensitive the
    # actor's output is to a small observation change, *not* a literal
    # parameter-gradient norm (computing that at rollout time would require
    # a backward pass this loop does not otherwise do; the genuine
    # gradient-derived diagnostics already available are mean_actor_loss/
    # mean_critic_loss/mean_entropy/mean_approximate_kl above, from the PPO
    # updates that did run).
    mean_actor_output_variation: float
    # Fraction of HOLD-phase control steps where each strict_stable_contact
    # sub-condition broke (see the HOLD branch above) -- 0.0 when HOLD was
    # never reached. Distinguishes an insufficient-grip hold break (force)
    # from an unstable-box-motion break (speed/angular) without inventing a
    # new threshold; reuses the same booleans hold_timer/strict_hold_timer
    # already gate on.
    hold_force_violation_fraction: float
    hold_speed_violation_fraction: float
    hold_angular_violation_fraction: float


@dataclass(frozen=True)
class EpisodeFunnel:
    """One episode's funnel stage reach + a single mutually-exclusive failure
    category, derived entirely from BoxCatchSummary fields already computed
    by run_box_catch -- no new success/safety judgment, only naming/grouping
    of existing signals."""

    reached_pre_impact: bool
    first_contact_detected: bool
    impact_safe: bool
    bilateral_contact_achieved: bool
    hold_entered: bool
    stable_hold_completed: bool
    episode_success: bool
    # "" when episode_success -- one of: viewer closed, emergency force,
    # interception miss, excessive first-contact impact, unilateral-contact
    # failure, insufficient grip, contact loss, unstable box motion, timeout.
    # "numerical/solver failure" is not produced: the differentiable MPC
    # solve (torch.linalg.solve) has no solved/failed status to reuse here
    # (unlike the OSQP-based controllers elsewhere in this repo), so
    # detecting it would require inventing a new signal -- flagged as a gap
    # rather than guessed.
    failure_category: str


def compute_episode_funnel(
    summary: BoxCatchSummary, config: AcmpcBoxCatchConfig
) -> EpisodeFunnel:
    reached_pre_impact = summary.pre_impact_dwell_s > 0.0
    first_contact_detected = summary.first_contact_time_s is not None
    impact_safe = summary.first_contact_peak_force_n <= config.squeeze.first_contact_force_limit
    bilateral_contact_achieved = summary.bilateral_contact_time_s is not None
    hold_entered = summary.hold_dwell_s > 0.0
    stable_hold_completed = summary.hold_time_s >= config.required_hold_s
    episode_success = summary.success

    reason = summary.failure_reason.lower()
    if episode_success:
        failure_category = ""
    elif "viewer closed" in reason:
        failure_category = "viewer closed"
    elif "emergency" in reason:
        failure_category = "emergency force"
    elif not reached_pre_impact or not first_contact_detected:
        # Covers both never leaving INTERCEPT and reaching PRE_IMPACT but
        # the box crossing the interception-workspace boundary before any
        # touch (the dominant "full" stage failure mode found this session).
        failure_category = "interception miss"
    elif not impact_safe:
        # Peak first-contact force is a running max over the whole episode
        # (FirstContactForceLimiter never resets it), so an unsafe impact is
        # chronologically the earliest problem regardless of what happens
        # in later phases -- checked before bilateral/hold/timeout below.
        failure_category = "excessive first-contact impact"
    elif not bilateral_contact_achieved:
        failure_category = "unilateral-contact failure"
    elif not hold_entered:
        failure_category = "insufficient grip"
    elif not stable_hold_completed:
        # HOLD was entered but strict_stable_contact never held continuously
        # for required_hold_s -- attribute to whichever of its sub-conditions
        # broke more often (hold_*_violation_fraction, already computed in
        # run_box_catch's HOLD branch).
        force_violation = summary.hold_force_violation_fraction
        motion_violation = max(
            summary.hold_speed_violation_fraction, summary.hold_angular_violation_fraction
        )
        failure_category = (
            "contact loss" if force_violation >= motion_violation else "unstable box motion"
        )
    else:
        # Should not normally trigger once the stages above are exhaustive;
        # kept as an explicit fallback rather than silently mis-attributing.
        failure_category = "timeout"

    return EpisodeFunnel(
        reached_pre_impact=reached_pre_impact,
        first_contact_detected=first_contact_detected,
        impact_safe=impact_safe,
        bilateral_contact_achieved=bilateral_contact_achieved,
        hold_entered=hold_entered,
        stable_hold_completed=stable_hold_completed,
        episode_success=episode_success,
        failure_category=failure_category,
    )


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


# Mirrors AdaptiveCostActor.forward's hard clamp in
# control/mpc/online_actor_critic.py (torch.clamp(weights, min=1e-3,
# max=500.0)) -- not re-exported from that module, so duplicated here only
# for the weight_lower/upper_bound_hit_fraction diagnostic below (item 9).
_WEIGHT_CLAMP_MIN = 1e-3
_WEIGHT_CLAMP_MAX = 500.0

# Logging-only relabel of COST_NAMES[2] ("force") to "compression" (item 7 --
# see the comment above _BOX_CATCH_PHASE_PRIORS for why the underlying dict
# key stays "force"). Used for weight/cost_raw/cost_weighted CSV column
# names (item 9) so logged output is not mislabeled even though the shared
# module's dict key is unchanged for backward compatibility.
_LOG_COST_LABELS = {
    "object": "object",
    "grasp": "grasp",
    "force": "compression",
    "velocity": "velocity",
    "smoothness": "smoothness",
}


def _cost_contributions(
    *,
    weights: dict[str, np.ndarray],
    target_center: np.ndarray,
    ee_midpoint: np.ndarray,
    relative_reference: np.ndarray,
    achieved_relative: np.ndarray,
    grasp_compression: float,
    object_velocity_feedforward: np.ndarray,
    measured_velocity: np.ndarray,
    mean_velocity: np.ndarray,
    previous_mean_velocity: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-cost-dimension weight/raw-residual/weighted-residual, for logging.

    "raw" is each term's *first horizon-step* (k=0) squared-error residual,
    evaluated from already-measured/commanded quantities using the same
    reference construction DifferentiableBimanualMPC.forward uses --
    intentionally NOT the QP's literal horizon-summed internal cost, which
    forward() does not return (changing its 2-tuple return signature would
    break every other caller in this codebase: main_bimanual_acmpc.py, this
    file, and both phase7/phase12 tests). This is therefore an honest,
    first-order approximation of "how far is each term from satisfied right
    now", not a re-derivation of the QP's exact objective value.

    COST_NAMES[2] is still literally "force" (see the comment above
    _BOX_CATCH_PHASE_PRIORS) but is reported under both its raw dict key and
    -- by the caller, in the CSV/summary -- the "compression" label, since it
    tracks a compressed position reference, not force.
    """

    object_delta = target_center - ee_midpoint
    grasp_delta = relative_reference - achieved_relative
    rel_norm = max(float(np.linalg.norm(relative_reference)), 1e-6)
    compressed_reference = relative_reference * (1.0 - grasp_compression / rel_norm)
    compression_delta = compressed_reference - achieved_relative
    velocity_delta = np.concatenate(
        [
            object_velocity_feedforward - measured_velocity[:3],
            object_velocity_feedforward - measured_velocity[3:],
        ]
    )
    smoothness_delta = mean_velocity - previous_mean_velocity

    raw = {
        "object": float(np.dot(object_delta, object_delta)),
        "grasp": float(np.dot(grasp_delta, grasp_delta)),
        "force": float(np.dot(compression_delta, compression_delta)),
        "velocity": float(np.dot(velocity_delta, velocity_delta)) / 2.0,
        "smoothness": float(np.dot(smoothness_delta, smoothness_delta)),
    }
    result: dict[str, dict[str, float]] = {}
    for name in COST_NAMES:
        weight_first_step = float(np.asarray(weights[name]).reshape(-1)[0])
        result[name] = {
            "weight": weight_first_step,
            "raw": raw[name],
            "weighted": weight_first_step * raw[name],
        }
    return result


def build_diagnostic_state(
    mode: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_half_y: float,
) -> tuple[np.ndarray, np.ndarray, CatchPhase]:
    """Box (position, velocity) and starting phase for one diagnostic mode.

    Reads the arms' *current* pose (caller must already have set them to
    home and called mj_forward once) to place the box relative to where
    left/right_catch_pad actually are -- same scene, same pads, no separate
    simplified geometry. "full_episode"/"test_intercept" are not handled
    here; run_box_catch keeps its normal ballistic-launch position/velocity
    for those.
    """

    left_pad_gid = _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_catch_pad")
    right_pad_gid = _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_catch_pad")
    left_pad_pos = data.geom_xpos[left_pad_gid].copy()
    right_pad_pos = data.geom_xpos[right_pad_gid].copy()
    pad_half_thickness = float(model.geom_size[left_pad_gid][0])

    box_x = 0.5 * (left_pad_pos[0] + right_pad_pos[0])
    box_z = 0.5 * (left_pad_pos[2] + right_pad_pos[2])
    # Box-center Y at which left/right pad would each just touch the box
    # surface (left approaches from +y, right from -y -- see Task 3's
    # right-arm measurement in this session; left is the mirror image).
    left_touch_y = left_pad_pos[1] - pad_half_thickness - box_half_y
    right_touch_y = right_pad_pos[1] + pad_half_thickness + box_half_y
    bilateral_center_y = 0.5 * (left_touch_y + right_touch_y)
    compression_margin = 0.003  # 3mm initial overlap so contact reads nonzero from step 0
    single_side_offset = 0.02  # 2cm toward one pad, clear of the other

    if mode == "test_hold":
        box_y = bilateral_center_y
        box_velocity = np.zeros(3)
        phase = CatchPhase.HOLD
    elif mode == "test_capture_left_contact":
        box_y = left_touch_y - compression_margin + single_side_offset
        box_velocity = np.array([0.0, 0.0, -0.2])
        phase = CatchPhase.CAPTURE
    elif mode == "test_capture_right_contact":
        box_y = right_touch_y + compression_margin - single_side_offset
        box_velocity = np.array([0.0, 0.0, -0.2])
        phase = CatchPhase.CAPTURE
    elif mode == "test_pre_impact":
        standoff = 0.08
        box_y = bilateral_center_y
        box_velocity = np.array([0.0, 0.0, -0.5])
        box_z = box_z + standoff  # still above/short of the pads, closing via -z fall
        phase = CatchPhase.PRE_IMPACT
    else:
        raise ValueError(f"build_diagnostic_state does not handle mode {mode!r}")

    return np.array([box_x, box_y, box_z]), box_velocity, phase


def _reward(
    *,
    previous_endpoint_error: float,
    endpoint_error: float,
    left_force: float,
    right_force: float,
    velocity: np.ndarray,
    previous_velocity: np.ndarray,
    phase: CatchPhase,
    required_grip_force: float,
    maximum_grip_force: float,
    force_limit_exceeded: bool,
    emergency: bool,
    success_reward: float = 50.0,
    failure_penalty: float = 50.0,
    hold_reward_scale: float = 1.0,
) -> float:
    """Condition-based reward: contact/hold bonuses key off measured state
    (both-pad contact, grip force inside [required_grip_force,
    maximum_grip_force]) rather than a per-phase lookup table, so the
    contact/bilateral/phase terms this replaced no longer overlap. Grip
    force only rewards being *inside* the physically-required band (see
    `required_grip_force = 1.3*mg/(2*mu)` in run_box_catch) instead of
    scaling up with raw force, which previously gave no disincentive
    against squeezing harder than necessary.
    """
    progress = 25.0 * (previous_endpoint_error - endpoint_error)
    endpoint_penalty = 2.5 * endpoint_error

    both_contact = left_force > 0.35 and right_force > 0.35
    stable_grip = (
        both_contact
        and required_grip_force <= left_force <= maximum_grip_force
        and required_grip_force <= right_force <= maximum_grip_force
    )
    bilateral_bonus = 0.8 if both_contact else 0.0
    hold_bonus = 0.6 if stable_grip else 0.0

    force_band_penalty = 0.0
    if both_contact:
        for force in (left_force, right_force):
            force_band_penalty += max(0.0, required_grip_force - force) ** 2
            force_band_penalty += max(0.0, force - maximum_grip_force) ** 2
    force_band_penalty *= 0.02

    effort_penalty = 0.01 * float(np.dot(velocity, velocity))
    velocity_delta = velocity - previous_velocity
    smoothness_penalty = 0.01 * float(np.dot(velocity_delta, velocity_delta))

    # first_contact_force_limit (18N, "ImpactSafe") is a softer bar than
    # emergency_contact_force (36N, hard episode termination) -- keep both
    # as distinct per-step penalties so training still pushes toward
    # ImpactSafe even though only the emergency case ends the episode.
    safety_penalty = 3.0 if force_limit_exceeded else 0.0
    safety_penalty += 5.0 if emergency else 0.0

    terminal_reward = success_reward if phase is CatchPhase.SUCCESS else 0.0
    terminal_penalty = failure_penalty if phase is CatchPhase.FAILED else 0.0

    # hold_reward_scale (default 1.0, exact no-op) amplifies only the
    # reward *for holding contact* (bilateral_bonus/hold_bonus), not the
    # approach-tracking or safety terms. A from-scratch (no engineered
    # prior) actor rarely reaches contact at all early in training, so
    # whatever reward it does get from a brief, unstable touch is easily
    # washed out by the denser per-step tracking/effort terms that fire on
    # every single step regardless of phase -- amplifying the
    # holding-specific reward gives a stronger, clearer gradient toward
    # "keep this contact" the few times contact is actually reached.
    return float(
        progress
        - endpoint_penalty
        + hold_reward_scale * (bilateral_bonus + hold_bonus)
        - force_band_penalty
        - effort_penalty
        - smoothness_penalty
        - safety_penalty
        + terminal_reward
        - terminal_penalty
    )


def run_box_catch(
    config: Optional[AcmpcBoxCatchConfig] = None,
    *,
    wandb_logger: Optional[WandbLogger] = None,
    global_step_start: int = 0,
) -> BoxCatchSummary:
    config = config or AcmpcBoxCatchConfig()
    # A caller that tracks W&B across many episodes (e.g. the curriculum
    # loop) passes its own logger, already wandb.init'd once, and this
    # function must not create/finish a second run for it -- only a bare
    # run_box_catch(config) call (config.use_wandb=True) self-manages one,
    # scoped to this single episode.
    owns_wandb_logger = wandb_logger is None
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

    weight_mode = "prior-free" if config.use_prior_free_actor else "phase-fixed-residual"
    if wandb_logger is None:
        run_name = config.wandb_run_name or f"{weight_mode}-seed-{config.seed}"
        wandb_logger = init_wandb(
            enabled=config.use_wandb,
            run_name=run_name,
            config={
                "seed": config.seed,
                "control_dt": control_dt,
                "horizon": config.mpc_horizon,
                "weight_mode": weight_mode,
            },
        )
    else:
        wandb_logger.update_config({"control_dt": control_dt})

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
    model.geom_conaffinity[box_geom_id] |= HAND_CAMERA_COLLISION_BIT | CATCH_PAD_COLLISION_BIT | 1
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
    mpc_config = DifferentiableMPCConfig(
        horizon=config.mpc_horizon,
        dt=control_dt,
        velocity_limit=config.mpc_velocity_limit,
        gravity=tuple(float(value) for value in gravity),
        # The MPC's "compression" cost (COST_NAMES[2], still literally named
        # "force" in control/mpc/online_actor_critic.py for backward
        # compatibility -- see this file's comment above _BOX_CATCH_PHASE_
        # PRIORS) pulls the relative separation toward
        # relative_reference*(1 - grasp_compression/|relative_reference|) --
        # a fixed compression *ceiling* no combination of weights can exceed
        # (raising w_compression can only interpolate the solve's effective
        # target closer to this ceiling, never past it). The handle-grasp
        # demo's default (0.008 m) caps real achievable normal force at a
        # few N regardless of phase-prior weights, nowhere near what a
        # 0.5 kg box's own weight needs from friction alone (~10 N/pad, see
        # minimum_symmetric_squeeze_force in control/squeeze/friction.py) to
        # avoid slipping through the pads under gravity during the hold.
        grasp_compression=0.030,
    )
    learner = OnlineActorCriticACMPC(
        mpc_config,
        OnlineActorCriticConfig(
            device=config.device,
            seed=config.seed,
            initial_log_std=float(np.log(max(config.exploration_std, 1e-6))),
            phase_priors=config.phase_priors,
            n_phases=N_CONTROL_PHASES,
            weight_delta_fraction=config.weight_delta_fraction,
            maximum_online_actor_delta=config.maximum_online_actor_delta,
            maximum_cumulative_actor_delta=config.maximum_cumulative_actor_delta,
            use_prior_free_actor=config.use_prior_free_actor,
            prior_free_initial_weights=config.prior_free_initial_weights,
        ),
    )
    if config.checkpoint_path and Path(config.checkpoint_path).exists():
        learner.load(config.checkpoint_path)
    wandb_logger.update_config(
        {
            "algorithm": "acmpc-online-actor-critic",
            "learning_rate": learner.config.actor_lr,
            "critic_learning_rate": learner.config.critic_lr,
            "gamma": learner.config.gamma,
            "gae_lambda": learner.config.gae_lambda,
            "clip_range": learner.config.clip_ratio,
            "entropy_coef": learner.config.entropy_coef,
            "value_coef": learner.config.value_loss_coefficient,
        }
    )
    initial_actor = np.concatenate(
        [parameter.detach().cpu().numpy().ravel() for parameter in learner.actor.parameters()]
    )

    # left_catch_pad/right_catch_pad sit 0.0441m further along local-x than
    # left_ee/right_ee (the tracked control point) -- see the geom comments
    # in ffw_sg2.xml. Since ee itself (not the pad) is what gets driven to
    # pad_offset_scalar, ee's own target must be pushed out by that same
    # 0.0441m so the catch pad (mounted further in) ends up at the intended
    # box-half-y + standoff contact point instead of ee itself landing there
    # and the catch pad overshooting into the box before ee's endpoint_error
    # reaches zero.
    pad_offset_scalar = (
        cfg.box_half_y + cfg.pad_half_thickness + cfg.precontact_gap + config.catch_pad_standoff
    )
    if config.domain_parameters is not None:
        object_mass = config.domain_parameters.mass
        object_friction = config.domain_parameters.friction
        half_x, _, half_z = config.domain_parameters.half_size
    else:
        object_mass = _NOMINAL_BOX_MASS
        object_friction = _NOMINAL_BOX_FRICTION
        half_x, _, half_z = _NOMINAL_BOX_HALF_SIZE

    required_grip_force = (
        config.strict_grip_force_margin
        * object_mass
        * _GRAVITY_MPS2
        / (2.0 * object_friction)
    )

    data.eq_active[fixture_id] = 0
    if config.diagnostic_mode in _DIAGNOSTIC_MODE_PHASE:
        # Reads the arms' current (home) pose to place the box relative to
        # the real catch pads -- see build_diagnostic_state's docstring.
        diagnostic_position, diagnostic_velocity, initial_phase = build_diagnostic_state(
            config.diagnostic_mode, model, data, box_half_y=cfg.box_half_y,
        )
        data.qpos[box_qpos_address : box_qpos_address + 3] = diagnostic_position
        data.qvel[box_dof_address : box_dof_address + 3] = diagnostic_velocity
    else:
        data.qvel[box_dof_address : box_dof_address + 3] = launch_velocity
        initial_phase = CatchPhase.INTERCEPT
    mujoco.mj_forward(model, data)
    predictor.reset(initial_velocity=data.qvel[box_dof_address : box_dof_address + 3].copy())

    if config.diagnostic_mode == "test_hold":
        # The home arm pose alone doesn't reach the box (pads start ~0.43m
        # apart, a nominal box is only ~0.30m wide) -- drive both arms in
        # to grip it first, using the exact same target formula the main
        # loop uses (target_center +/- pad_offset_scalar*y_axis), so HOLD
        # starts from a real, physically-settled bilateral grip instead of
        # an unreachable box placement.
        settle_y_axis = np.array([0.0, 1.0, 0.0])
        settle_center = diagnostic_position
        for name, sign in (("left", 1.0), ("right", -1.0)):
            desired_transforms[name][:3, 3] = settle_center + sign * pad_offset_scalar * settle_y_axis
        for _ in range(int(0.5 / dt)):
            # Pin the box in place while the arms close in -- otherwise it
            # free-falls away from the (fixed-height) target during the
            # settle, since only the arms are meant to move here. Released
            # the instant the settle ends and the main loop begins.
            data.qpos[box_qpos_address : box_qpos_address + 3] = diagnostic_position
            data.qvel[box_dof_address : box_dof_address + 3] = 0.0
            for name in ("left", "right"):
                impedance[name].apply(model, data, arms[name], desired_transforms[name])
            mujoco.mj_step(model, data)

    phase = initial_phase
    phase_started = 0.0
    hold_timer = 0.0
    # HOLD -> CAPTURE demotion debounce (see the HOLD branch below) -- how
    # long the "both pads >= required_grip_force" condition has been
    # continuously broken while in HOLD. Independent of hold_timer, which
    # tracks the *stricter* success criterion and already resets on its own.
    hold_break_timer = 0.0
    strict_hold_timer = 0.0
    max_strict_hold_timer = 0.0
    # Which of strict_stable_contact's three conditions broke, counted only
    # for control steps where phase is HOLD -- reused as-is (not new
    # judgment logic) by the failure-category funnel in
    # main_acmpc_box_catch_curriculum.py to tell "insufficient grip"
    # (force) apart from "unstable box motion" (speed/angular) when a hold
    # streak resets.
    hold_step_count = 0
    hold_force_violation_count = 0
    hold_speed_violation_count = 0
    hold_angular_violation_count = 0
    bilateral_contact_time: Optional[float] = None
    # NOTE: capture_center/capture_reference_velocity/capture_rotation below
    # predate and are unrelated to the new CatchPhase.CAPTURE -- they name
    # the *ballistic* sense of "capture" (freezing a reference at first
    # contact, mirroring main_dynamic_box_squeeze.py), not the phase.
    capture_center: Optional[np.ndarray] = None
    capture_reference_velocity: Optional[np.ndarray] = None
    capture_rotation: Optional[np.ndarray] = None
    failure_reason = ""
    previous_control_positions = np.concatenate(
        [desired_transforms["left"][:3, 3], desired_transforms["right"][:3, 3]]
    )
    previous_velocity = np.zeros(6)
    command_velocity = np.zeros(6)
    previous_command_velocity = np.zeros(6)
    previous_endpoint_error = 0.0
    minimum_endpoint_error = float("inf")

    # --- Phase-prior / phase-observation smoothstep blending state (item 3).
    # Initialized so blend_*_start == blend_*_target at episode start (phase
    # has no prior transition to blend from yet), giving beta's value no
    # effect and zero discontinuity at step 0. _initial_control_index tracks
    # whatever phase was actually injected (INTERCEPT normally, or a
    # diagnostic mode's phase above) so the actor's prior/observation start
    # in agreement with the true physical state instead of always claiming
    # INTERCEPT.
    _initial_control_index = _CATCH_PHASE_TO_CONTROL_INDEX[phase]
    _initial_prior = np.asarray(config.phase_priors[_initial_control_index], dtype=np.float32)
    _initial_onehot = np.zeros(N_CONTROL_PHASES, dtype=np.float32)
    _initial_onehot[_initial_control_index] = 1.0
    blend_prior_start = _initial_prior.copy()
    blend_prior_target = _initial_prior.copy()
    current_blended_prior = _initial_prior.copy()
    blend_onehot_start = _initial_onehot.copy()
    blend_onehot_target = _initial_onehot.copy()
    current_soft_onehot = _initial_onehot.copy()

    rollout = ACMPCRolloutBuffer()
    updates: list[PPOUpdateSummary] = []
    total_transitions = 0
    total_reward = 0.0
    latest_update: Optional[PPOUpdateSummary] = None
    # Task 8: most recently computed lower-controller diagnostics (torque,
    # impedance position error, measured EE velocity) -- one control
    # step stale in the CSV row, same "most recently available" pattern
    # actor_loss/critic_loss below already use for latest_update.
    latest_impedance: dict[str, object] = {"left": None, "right": None}
    last_observation: Optional[np.ndarray] = None
    last_phase: Optional[CatchControlPhase] = None
    last_ee_positions: Optional[np.ndarray] = None
    last_object_position: Optional[np.ndarray] = None
    last_object_velocity: Optional[np.ndarray] = None
    last_relative_reference: Optional[np.ndarray] = None
    last_previous_velocity: Optional[np.ndarray] = None
    last_action = None
    rows: list[dict[str, float | str]] = []

    # --- Episode-level diagnostics accumulated per control step (item 9);
    # reduced to means/fractions once at the end, see BoxCatchSummary.
    phase_transition_count = 0
    hold_to_capture_demotion_count = 0
    phase_dwell_s = {CatchPhase.INTERCEPT: 0.0, CatchPhase.PRE_IMPACT: 0.0,
                      CatchPhase.CAPTURE: 0.0, CatchPhase.HOLD: 0.0}
    blend_active_steps = 0
    control_step_count = 0
    velocity_saturation_samples = 0
    velocity_total_samples = 0
    mpc_solve_time_total_s = 0.0
    weight_lower_hit_samples = 0
    weight_upper_hit_samples = 0
    weight_total_samples = 0
    residual_samples: dict[str, list[float]] = {name: [] for name in COST_NAMES}
    actor_output_variation_sum = 0.0
    actor_output_variation_count = 0
    previous_mean_weights: Optional[np.ndarray] = None
    previous_weights_phase: Optional[CatchControlPhase] = None

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
            contact = read_bilateral_pad_contact(
                model,
                data,
                box_geom_name="dynamic_box_geom",
                left_pad_geom_name="left_catch_pad",
                right_pad_geom_name="right_catch_pad",
            )
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
                    or phase in {CatchPhase.PRE_IMPACT, CatchPhase.CAPTURE}
                ):
                    # Inside the compliant first-contact window (or already
                    # close enough to have entered PRE_IMPACT): contact
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
                if bilateral_contact_time is None and both_contact:
                    # Bilateral (both-pad) contact is now decoupled from the
                    # PRE_IMPACT -> CAPTURE transition (which can fire on a
                    # single-side touch, see below) -- track its first
                    # occurrence independently, wherever it actually happens.
                    bilateral_contact_time = time_s
                # F_L,F_R >= F_req: CAPTURE -> HOLD's promotion bar and (see
                # the HOLD branch) HOLD -> CAPTURE's demotion bar are the
                # same physically-grounded condition (both pads bilaterally
                # contacting and each at/above the friction-grip minimum),
                # not the separate ad hoc `stable_grasp_force` threshold the
                # original code used here.
                both_pads_at_required_force = bool(
                    both_contact
                    and contact.left.normal_force >= required_grip_force
                    and contact.right.normal_force >= required_grip_force
                )
                strict_force_ok = bool(
                    both_contact
                    and required_grip_force <= contact.left.normal_force <= config.strict_grip_force_max_n
                    and required_grip_force <= contact.right.normal_force <= config.strict_grip_force_max_n
                )
                box_angular_velocity = data.qvel[box_dof_address + 3 : box_dof_address + 6]
                strict_speed_ok = float(np.linalg.norm(box_velocity)) <= config.strict_box_speed_max_mps
                strict_angular_ok = (
                    float(np.linalg.norm(box_angular_velocity))
                    <= config.strict_box_angular_speed_max_radps
                )
                strict_stable_contact = strict_force_ok and strict_speed_ok and strict_angular_ok
                strict_hold_timer = (
                    strict_hold_timer + control_dt if strict_stable_contact else 0.0
                )
                max_strict_hold_timer = max(max_strict_hold_timer, strict_hold_timer)
                if phase is CatchPhase.HOLD:
                    hold_step_count += 1
                    if not strict_force_ok:
                        hold_force_violation_count += 1
                    if not strict_speed_ok:
                        hold_speed_violation_count += 1
                    if not strict_angular_ok:
                        hold_angular_violation_count += 1

                # d_hand-box: actual hand-to-box distance (not endpoint_error,
                # which measures hand-to-*target* -- target is intercept_center,
                # a future predicted point the arm converges onto long before
                # the box itself is anywhere nearby, so endpoint_error alone
                # is not a proximity signal to the real box).
                hand_object_distance = 0.5 * (
                    float(np.linalg.norm(left_ee - box_position))
                    + float(np.linalg.norm(right_ee - box_position))
                )
                if phase is CatchPhase.INTERCEPT and (
                    (
                        prediction.confidence >= config.precontact_confidence_min
                        and remaining_ttc <= cfg.ttc_soften_window_s
                    )
                    or hand_object_distance <= config.precontact_distance
                ):
                    # (c_pred >= c_min AND TTC <= T_pre) OR (d_hand-box <=
                    # d_pre). When confidence is low (early in flight, before
                    # BallisticBoxPredictor's velocity estimate has settled --
                    # see control/squeeze/ballistic.py), the first clause is
                    # always False regardless of the TTC value, so the
                    # distance clause is the only path left -- exactly the
                    # "distance as fallback when confidence/TTC are not
                    # trustworthy" behavior asked for, with no extra branching
                    # needed. In the validated primary path, confidence has
                    # already saturated to 1.0 long before TTC<=T_pre becomes
                    # true (saturates within ~40ms of flight start), so this
                    # reduces to the original remaining_ttc<=ttc_soften_window_s
                    # gate. Switching to PRE_IMPACT also swaps the tracking
                    # target from intercept_center (continuous, TTC-based,
                    # converges onto box_position as remaining_ttc -> 0) to
                    # the box's raw live position -- firing that swap while
                    # the box is still far away would snap the target
                    # backward, which the arm would then spend the rest of
                    # the flight failing to re-close. Because the fallback
                    # branch above compares against the *true* box position
                    # (not a future target), it can only trigger once the box
                    # is genuinely close, so this discontinuity risk does not
                    # reappear through that path either.
                    phase = CatchPhase.PRE_IMPACT
                elif phase is CatchPhase.PRE_IMPACT and (
                    (contact.left.active and contact.left.normal_force > config.contact_detect_force_n)
                    or (contact.right.active and contact.right.normal_force > config.contact_detect_force_n)
                ):
                    # F_L > F_detect OR F_R > F_detect: CAPTURE begins on the
                    # *first* one-sided touch (matching CAPTURE's definition:
                    # some, not yet stable, bilateral contact), not on
                    # both_contact as the original PRE_CONTACT -> GRASPING
                    # transition required.
                    phase = CatchPhase.CAPTURE
                elif phase is CatchPhase.CAPTURE:
                    hold_timer = hold_timer + control_dt if both_pads_at_required_force else 0.0
                    if hold_timer >= config.hold_entry_dwell_s:
                        phase = CatchPhase.HOLD
                        hold_timer = 0.0
                        hold_break_timer = 0.0
                elif phase is CatchPhase.HOLD:
                    # Success = ImpactSafe (first-contact peak <=
                    # first_contact_force_limit) AND StableHold_5s (grip
                    # force inside [required_grip_force,
                    # strict_grip_force_max_n], box linear/angular velocity
                    # under strict_box_speed_max_mps/strict_box_angular_
                    # speed_max_radps, all continuously for required_hold_s)
                    # -- unchanged from the original GRASPED -> SUCCESS bar.
                    if strict_stable_contact:
                        hold_timer += control_dt
                    else:
                        hold_timer = 0.0

                    # Demotion (HOLD -> CAPTURE) when the weaker "both pads
                    # holding >= required_grip_force" condition breaks --
                    # chatter-guarded with a *minimum dwell time*: the
                    # condition must stay broken continuously for
                    # hold_entry_dwell_s (the same constant CAPTURE -> HOLD
                    # promotion already uses) before demoting, and promotion
                    # back requires that same dwell of continuous good
                    # contact. A single noisy contact-force sample therefore
                    # cannot flip the phase either direction. Reusing the
                    # existing dwell constant (rather than introducing a
                    # second, separately-tuned exit threshold via hysteresis)
                    # avoids adding another number with no independent
                    # empirical basis.
                    if both_pads_at_required_force:
                        hold_break_timer = 0.0
                    else:
                        hold_break_timer += control_dt
                        if hold_break_timer >= config.hold_entry_dwell_s:
                            phase = CatchPhase.CAPTURE
                            hold_timer = 0.0
                            hold_break_timer = 0.0
                            # The only *backward* transition in this state
                            # machine -- unlike phase_transition_count
                            # (which also counts normal forward progress),
                            # this is a direct, unambiguous chatter signal:
                            # >0 means HOLD was entered and lost at least
                            # once this episode.
                            hold_to_capture_demotion_count += 1

                    if phase is CatchPhase.HOLD and hold_timer >= config.required_hold_s:
                        if impact.force_limit_exceeded:
                            phase = CatchPhase.FAILED
                            failure_reason = (
                                "held stably but first-contact impact exceeded "
                                "the ImpactSafe threshold"
                            )
                        else:
                            phase = CatchPhase.SUCCESS
                # Safety/terminal conditions are hard overrides, never
                # interpolated/blended (item 3's requirement) -- they already
                # were not part of the phase-prior blending mechanism (which
                # only ever touches AdaptiveCostActor's cost-weight table),
                # and FirstContactForceLimiter's own force-relief output
                # (impact.emergency / relief_distances, applied every physics
                # substep below) is computed entirely independently of
                # CatchPhase and is never smoothed either.
                if impact.emergency:
                    phase = CatchPhase.FAILED
                    failure_reason = "emergency contact force exceeded"
                elif (
                    phase in {CatchPhase.INTERCEPT, CatchPhase.PRE_IMPACT}
                    and box_position[0] < cfg.catch_plane_x - 0.16
                ):
                    phase = CatchPhase.FAILED
                    failure_reason = "box passed the interception workspace"

                phase_dwell_s[old_phase] = phase_dwell_s.get(old_phase, 0.0) + control_dt
                if phase is not old_phase:
                    phase_transition_count += 1
                    phase_started = time_s
                    # Snapshot whatever the blend currently evaluates to
                    # (mid-blend or already converged) as the new start,
                    # rather than the old phase's hard table row -- so a
                    # second transition arriving before the first blend
                    # finishes does not reintroduce the discontinuity
                    # blending was meant to remove (item 3's requirement).
                    blend_prior_start = current_blended_prior.copy()
                    blend_onehot_start = current_soft_onehot.copy()
                    if phase in _CATCH_PHASE_TO_CONTROL_INDEX:
                        control_index = int(_CATCH_PHASE_TO_CONTROL_INDEX[phase])
                        blend_prior_target = np.asarray(
                            config.phase_priors[control_index], dtype=np.float32
                        )
                        blend_onehot_target = np.zeros(N_CONTROL_PHASES, dtype=np.float32)
                        blend_onehot_target[control_index] = 1.0
                    # else: phase is now SUCCESS/FAILED (terminal) -- no
                    # control index/prior/observation is needed for it, since
                    # run_box_catch breaks out below before computing another
                    # action from a terminal phase.

                terminal = phase in {CatchPhase.SUCCESS, CatchPhase.FAILED}
                if not terminal:
                    control_index = _CATCH_PHASE_TO_CONTROL_INDEX[phase]
                    blend_s = float(
                        np.clip(
                            (time_s - phase_started) / max(config.phase_blend_time_s, 1e-9),
                            0.0,
                            1.0,
                        )
                    )
                    blend_beta = _smoothstep(blend_s)
                    if blend_beta < 1.0:
                        blend_active_steps += 1
                    current_blended_prior = (
                        (1.0 - blend_beta) * blend_prior_start + blend_beta * blend_prior_target
                    )
                    current_soft_onehot = (
                        (1.0 - blend_beta) * blend_onehot_start + blend_beta * blend_onehot_target
                    )
                    control_step_count += 1

                if not terminal:
                    observation = build_bimanual_observation(
                        object_velocity=prediction.velocity,
                        left_endpoint_error=left_target - left_ee,
                        right_endpoint_error=right_target - right_ee,
                        left_ee_velocity=measured_velocity[:3],
                        right_ee_velocity=measured_velocity[3:],
                        left_force=contact.left.normal_force,
                        right_force=contact.right.normal_force,
                        time_to_contact=remaining_ttc,
                        # Soft (blended) phase encoding, not a hard one-hot,
                        # while a transition's smoothstep blend is still in
                        # progress (item 5) -- current_soft_onehot is the
                        # same beta-interpolated vector used for the cost
                        # prior above, so the observation's own phase signal
                        # and the cost weights it is paired with never
                        # disagree about which phase is "current".
                        phase_encoding=current_soft_onehot,
                        # Measured across box-catch rollouts (see
                        # build_bimanual_observation's docstring): x reaches
                        # -1.35 to -1.47 m/s, z reaches ~3.9 m/s by first
                        # contact, y stays near zero (+-0.025 m/s). The shared
                        # default (0.5 on all axes) saturates x 100% and z 65%
                        # of the time during INTERCEPT/PRE_CONTACT.
                        object_velocity_scale=(1.5, 0.2, 4.0),
                    )
                else:
                    # SUCCESS/FAILED are terminal, not control phases (item
                    # 1) -- no new observation/action is computed for them
                    # (see the `if terminal: break` below). This placeholder
                    # is only ever read as the (masked, irrelevant) GAE
                    # bootstrap value below when the rollout buffer happens
                    # to fill up on exactly this step -- generalized_advantage_
                    # estimate zeroes out a done transition's bootstrap
                    # contribution, so its exact content does not matter.
                    observation = last_observation

                if last_observation is not None and config.online_learning:
                    reward = _reward(
                        previous_endpoint_error=previous_endpoint_error,
                        endpoint_error=endpoint_error,
                        left_force=contact.left.normal_force,
                        right_force=contact.right.normal_force,
                        velocity=command_velocity,
                        previous_velocity=previous_command_velocity,
                        phase=phase,
                        required_grip_force=required_grip_force,
                        maximum_grip_force=config.strict_grip_force_max_n,
                        force_limit_exceeded=impact.force_limit_exceeded,
                        emergency=impact.emergency,
                        success_reward=config.success_reward,
                        failure_penalty=config.failure_penalty,
                        hold_reward_scale=config.hold_reward_scale,
                    )
                    total_reward += reward
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
                        wandb_logger.log(
                            build_ppo_update_log(latest_update),
                            step=global_step_start + control_step_count,
                        )
                        rollout.clear()

                if terminal:
                    # The transition that caused SUCCESS/FAILED was just
                    # recorded above (reward includes _reward's terminal
                    # bonus/penalty, done=True), attributed to the *previous*
                    # action (last_observation/last_action) -- do not compute
                    # a new action from this terminal state (item 2's "성공
                    # 판정 후에는 새로운 action을 계산하지 않고 episode를
                    # 종료"). FAILED is handled the same way for the same
                    # reason: both are terminal, non-control states with no
                    # CatchControlPhase index to build an observation for.
                    last_observation = None
                    break

                previous_command_velocity = command_velocity.copy()

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
                # not actor drift. Once in HOLD, the ideal action is "hold
                # still" -- there is nothing left to explore that outweighs
                # the risk of injecting noise into an already-successful
                # hold, so stop sampling stochastic actions from that phase
                # onward (still using training=True, hence still collecting
                # rollout/learning signal, up through CAPTURE). SUCCESS/
                # FAILED are unreachable here (terminal, see the `if
                # terminal: break` above), so they no longer need mentioning.
                exploring = config.online_learning and phase is not CatchPhase.HOLD
                _act_start = time.perf_counter()
                action = learner.act(
                    observation=observation,
                    phase=control_index,
                    ee_positions=ee_positions,
                    object_position=target_center,
                    object_velocity=object_velocity_feedforward,
                    relative_reference=relative_reference,
                    previous_velocity=previous_velocity,
                    training=exploring,
                    # The smoothstep-blended prior computed above -- see
                    # _resolve_phase_prior in online_actor_critic.py, used
                    # directly instead of a hard per-phase lookup by `phase`.
                    phase_prior=current_blended_prior,
                )
                _act_duration_s = time.perf_counter() - _act_start
                mpc_solve_time_total_s += _act_duration_s

                if control_step_count % config.wandb_log_interval == 0:
                    _wandb_step = global_step_start + control_step_count
                    wandb_logger.log(
                        {
                            **build_mpc_weight_log(
                                final_weights=action.weights,
                                phase_prior=current_blended_prior,
                                phase_id=int(control_index),
                                solver_time_s=_act_duration_s,
                            ),
                            **build_contact_log(contact=contact, box_velocity=box_velocity),
                        },
                        step=_wandb_step,
                    )

                _contributions = _cost_contributions(
                    weights=action.weights,
                    target_center=target_center,
                    ee_midpoint=0.5 * (ee_positions[:3] + ee_positions[3:]),
                    relative_reference=relative_reference,
                    achieved_relative=ee_positions[3:] - ee_positions[:3],
                    grasp_compression=mpc_config.grasp_compression,
                    object_velocity_feedforward=object_velocity_feedforward,
                    measured_velocity=measured_velocity,
                    mean_velocity=action.mean_velocity,
                    previous_mean_velocity=previous_velocity,
                )
                for _name in COST_NAMES:
                    residual_samples[_name].append(_contributions[_name]["raw"])
                    _weight_array = action.weights[_name]
                    weight_total_samples += _weight_array.size
                    weight_lower_hit_samples += int(np.sum(_weight_array <= _WEIGHT_CLAMP_MIN * 1.001))
                    weight_upper_hit_samples += int(np.sum(_weight_array >= _WEIGHT_CLAMP_MAX * 0.999))
                _current_mean_weights = np.array(
                    [float(np.mean(action.weights[_name])) for _name in COST_NAMES]
                )
                if previous_mean_weights is not None and previous_weights_phase == control_index:
                    actor_output_variation_sum += float(
                        np.mean(np.abs(_current_mean_weights - previous_mean_weights))
                    )
                    actor_output_variation_count += 1
                previous_mean_weights = _current_mean_weights
                previous_weights_phase = control_index

                last_phase = control_index
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

                velocity_total_samples += command_velocity.size
                velocity_saturation_samples += int(
                    np.sum(np.abs(command_velocity) >= 0.98 * config.mpc_velocity_limit)
                )

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
                        "blend_beta": blend_beta,
                        # Task 6: transition discontinuity metrics -- reuse
                        # command_velocity/previous_command_velocity/
                        # config.mpc_velocity_limit exactly as the episode
                        # summary's own saturation counter does, and
                        # old_phase/phase exactly as phase_transition_count
                        # does, rather than recomputing anything new.
                        "command_velocity_norm": float(np.linalg.norm(command_velocity)),
                        "command_delta_norm": float(
                            np.linalg.norm(command_velocity - previous_command_velocity)
                        ),
                        "velocity_saturated": bool(
                            np.any(np.abs(command_velocity) >= 0.98 * config.mpc_velocity_limit)
                        ),
                        "phase_transition": bool(phase is not old_phase),
                        # Task 8: lower-controller tracking diagnostics.
                        # impedance_pos_error_norm/torque_norm come from
                        # latest_impedance (one control step stale, same
                        # pattern as actor_loss/critic_loss above -- the
                        # impedance controller applies every physics
                        # substep, not just every control step, so "latest"
                        # is always very recent). measured_ee_velocity/
                        # jacobian_condition/joint_limit_margin are
                        # recomputed fresh here (cheap, no physics/MPC
                        # re-solve) since they don't depend on apply()'s
                        # return value.
                        **{
                            f"{_side}_measured_ee_velocity_norm": float(
                                np.linalg.norm(
                                    ee_jacobian_world(model, data, arms[_side])[3:]
                                    @ data.qvel[arms[_side].qvel_indices]
                                )
                            )
                            for _side in ("left", "right")
                        },
                        **{
                            f"{_side}_impedance_pos_error_norm": (
                                float("nan")
                                if latest_impedance[_side] is None
                                else float(latest_impedance[_side].e_pos_norm)
                            )
                            for _side in ("left", "right")
                        },
                        **{
                            f"{_side}_torque_norm": (
                                float("nan")
                                if latest_impedance[_side] is None
                                else float(np.linalg.norm(latest_impedance[_side].tau))
                            )
                            for _side in ("left", "right")
                        },
                        **{
                            f"{_side}_torque_saturated": bool(
                                latest_impedance[_side] is not None
                                and np.any(
                                    np.abs(latest_impedance[_side].tau)
                                    >= 0.99 * impedance[_side].config.tau_limit
                                )
                            )
                            for _side in ("left", "right")
                        },
                        **{
                            f"{_side}_joint_limit_margin": float(
                                np.min(
                                    np.minimum(
                                        data.qpos[arms[_side].qpos_indices] - arms[_side].ctrl_low,
                                        arms[_side].ctrl_high - data.qpos[arms[_side].qpos_indices],
                                    )
                                )
                            )
                            for _side in ("left", "right")
                        },
                        **{
                            f"{_side}_jacobian_condition": float(
                                np.linalg.cond(ee_jacobian_world(model, data, arms[_side]))
                            )
                            for _side in ("left", "right")
                        },
                        **{
                            f"weight/{_LOG_COST_LABELS[_name]}": _contributions[_name]["weight"]
                            for _name in COST_NAMES
                        },
                        **{
                            f"cost_raw/{_LOG_COST_LABELS[_name]}": _contributions[_name]["raw"]
                            for _name in COST_NAMES
                        },
                        **{
                            f"cost_weighted/{_LOG_COST_LABELS[_name]}": _contributions[_name]["weighted"]
                            for _name in COST_NAMES
                        },
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
                latest_impedance[name] = impedance[name].apply(
                    model, data, arms[name], desired_transforms[name]
                )
                data.ctrl[gripper_ids[name]] = FFW_GRIPPERS[name].open_ctrl

            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            # No post-step "if terminal: break" here (unlike the original) --
            # phase only ever becomes SUCCESS/FAILED inside the control-stride
            # block above, which now breaks immediately once terminal (see
            # "if terminal: break" there), before this mj_step ever runs on a
            # terminal step. Reaching here with a terminal phase is therefore
            # no longer possible.
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
        wandb_logger.log(
            build_ppo_update_log(latest_update),
            step=global_step_start + control_step_count,
        )
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

    episode_final_step = global_step_start + control_step_count
    wandb_logger.log(build_episode_reward_log(total_reward), step=episode_final_step)
    if owns_wandb_logger:
        wandb_logger.finish()

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
        control_step_count=control_step_count,
        actor_weight_change_l2=float(np.linalg.norm(final_actor - initial_actor)),
        device=str(learner.device),
        total_reward=float(total_reward),
        mean_reward_per_step=float(total_reward / total_transitions) if total_transitions else 0.0,
        mean_actor_loss=float(np.mean([u.actor_loss for u in updates])) if updates else 0.0,
        mean_critic_loss=float(np.mean([u.critic_loss for u in updates])) if updates else 0.0,
        mean_entropy=float(np.mean([u.entropy for u in updates])) if updates else 0.0,
        mean_approximate_kl=float(np.mean([u.approximate_kl for u in updates])) if updates else 0.0,
        strict_success=bool(max_strict_hold_timer >= config.required_hold_s),
        strict_hold_time_s=float(max_strict_hold_timer),
        required_grip_force_n=float(required_grip_force),
        phase_transition_count=phase_transition_count,
        hold_to_capture_demotion_count=hold_to_capture_demotion_count,
        intercept_dwell_s=float(phase_dwell_s[CatchPhase.INTERCEPT]),
        pre_impact_dwell_s=float(phase_dwell_s[CatchPhase.PRE_IMPACT]),
        capture_dwell_s=float(phase_dwell_s[CatchPhase.CAPTURE]),
        hold_dwell_s=float(phase_dwell_s[CatchPhase.HOLD]),
        blend_active_fraction=(
            float(blend_active_steps / control_step_count) if control_step_count else 0.0
        ),
        velocity_saturation_fraction=(
            float(velocity_saturation_samples / velocity_total_samples)
            if velocity_total_samples
            else 0.0
        ),
        mean_mpc_solve_time_s=(
            float(mpc_solve_time_total_s / control_step_count) if control_step_count else 0.0
        ),
        weight_lower_bound_hit_fraction=(
            float(weight_lower_hit_samples / weight_total_samples) if weight_total_samples else 0.0
        ),
        weight_upper_bound_hit_fraction=(
            float(weight_upper_hit_samples / weight_total_samples) if weight_total_samples else 0.0
        ),
        mean_residual_object=float(np.mean(residual_samples["object"])) if residual_samples["object"] else 0.0,
        std_residual_object=float(np.std(residual_samples["object"])) if residual_samples["object"] else 0.0,
        mean_residual_grasp=float(np.mean(residual_samples["grasp"])) if residual_samples["grasp"] else 0.0,
        std_residual_grasp=float(np.std(residual_samples["grasp"])) if residual_samples["grasp"] else 0.0,
        mean_residual_compression=float(np.mean(residual_samples["force"])) if residual_samples["force"] else 0.0,
        std_residual_compression=float(np.std(residual_samples["force"])) if residual_samples["force"] else 0.0,
        mean_residual_velocity=float(np.mean(residual_samples["velocity"])) if residual_samples["velocity"] else 0.0,
        std_residual_velocity=float(np.std(residual_samples["velocity"])) if residual_samples["velocity"] else 0.0,
        mean_residual_smoothness=float(np.mean(residual_samples["smoothness"])) if residual_samples["smoothness"] else 0.0,
        std_residual_smoothness=float(np.std(residual_samples["smoothness"])) if residual_samples["smoothness"] else 0.0,
        mean_actor_output_variation=(
            float(actor_output_variation_sum / actor_output_variation_count)
            if actor_output_variation_count
            else 0.0
        ),
        hold_force_violation_fraction=(
            float(hold_force_violation_count / hold_step_count) if hold_step_count else 0.0
        ),
        hold_speed_violation_fraction=(
            float(hold_speed_violation_count / hold_step_count) if hold_step_count else 0.0
        ),
        hold_angular_violation_fraction=(
            float(hold_angular_violation_count / hold_step_count) if hold_step_count else 0.0
        ),
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
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", default=None)
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
            use_wandb=args.use_wandb,
            wandb_run_name=args.wandb_run_name,
        )
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    if not summary.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
