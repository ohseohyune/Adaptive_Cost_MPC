"""Frozen fixed-baseline config for the box-catch AC-MPC pipeline (Task 12
of the "Fixed MPC baseline 성능 최대화" program -- see
fixed_mpc_baseline_tuning_program_status.md).

"Fixed baseline" = phase-scheduled MPC costs with zero learned residual
(`online_learning=False`, `weight_delta_fraction=0.0` -- Task 1 found a
stale checkpoint_path can silently reintroduce a trained residual even with
online_learning=False, so weight_delta_fraction=0.0 is the robust way to
guarantee a true zero-residual run). Frozen here so "what did learning
improve?" can be answered against a known, reproducible reference instead
of whatever the source defaults happen to be at the time -- see
project_goal_and_tracks.

FROZEN_PHASE_PRIORS is _BOX_CATCH_PHASE_PRIORS (the source default) with
the validated fixed-baseline overrides applied: HOLD row's grasp weight
20.0->0.0 and PRE_IMPACT row's velocity weight 3.0->6.0. The source
default now includes the later HOLD object-weight tuning, 18.0->60.0.
Validated across this session's structural fixes (see
fixed_mpc_baseline_tuning_program_status.md's RESOLVED sections):
  1. relative_normal_speed (impact stiffness severity) used total ballistic
     speed, including motion tangent to the pad faces -- fixed to use the
     measured inward pad-to-box speed along each contact normal.
  2. left/right_catch_pad's contact solref was mass-blind (and the intended
     softening ramp targeted the wrong, inactive geom entirely) --
     AcmpcBoxCatchConfig.impact_solref_mass_gain now scales it for
     heavier-than-nominal boxes.

Validated results after the geometry/contact fixes, measured normal-force
deficit feedforward, and installing mass-scaled contact solref before first
contact (50-seed "full" curriculum stage, BASE_SEED=1000,
default_curriculum()[2]):

    controller                  success_rate  unsafe_rate  mean/max peak force
    fixed costs only                 0.620        0.080       12.97N / 21.10N
    + force deficit feedback         0.800        0.080       13.10N / 21.10N
    + pre-contact mass solref         0.840        0.000       12.86N / 16.45N
    + normal-speed/hold tuning        0.860        0.000       13.00N / 16.59N
    + wide-box capture support        0.920        0.000       13.00N / 16.59N

On a disjoint 30-seed holdout (BASE_SEED=9000), force feedback and the
solref timing fix improved success 0.600->0.867 and unsafe_rate
0.067->0.000. The normal-speed/hold tuning preserved 0.867 success, and the
wide-box capture support raised it to 0.967 with 0.000 unsafe while making
bilateral contact and HOLD reach 30/30 (maximum peak force 16.30N).

On a further unseen 100-seed full-stage set (12000-12099), the frozen
candidate reached 0.960 success with 0.000 unsafe, 100/100 HOLD reach, and a
17.26N maximum first-contact peak. All four remaining failures were
unstable-motion episodes.

Usage:
    from acmpc.fixed_baseline_frozen import FROZEN_PHASE_PRIORS
    config = AcmpcBoxCatchConfig(
        online_learning=False, weight_delta_fraction=0.0,
        phase_priors=FROZEN_PHASE_PRIORS,
    )
"""

from __future__ import annotations

# Rows: INTERCEPT, PRE_IMPACT, CAPTURE, HOLD (CatchControlPhase order).
# Columns: object, grasp, force, velocity, smoothness (COST_NAMES order,
# see control/mpc/online_actor_critic.py).
FROZEN_PHASE_PRIORS: tuple[tuple[float, float, float, float, float], ...] = (
    (30.0, 250.0, 0.05, 4.0, 0.4),   # INTERCEPT (unchanged)
    (30.0, 250.0, 1.5, 6.0, 0.5),    # PRE_IMPACT (velocity 3.0 -> 6.0)
    (16.0, 40.0, 55.0, 1.5, 1.0),    # CAPTURE (unchanged)
    (60.0, 0.0, 65.0, 2.0, 1.5),     # HOLD (grasp 20.0 -> 0.0)
)

if __name__ == "__main__":
    # Self-check: the frozen copy must exactly match the validated overrides
    # applied to the live source default.
    from acmpc.box_catch_sweep_train import apply_overrides

    expected = apply_overrides(
        {
            "hold_grasp_weight": 0.0,
            "hold_object_weight": 60.0,
            "pre_impact_velocity_weight": 6.0,
        }
    )
    assert FROZEN_PHASE_PRIORS == expected, (FROZEN_PHASE_PRIORS, expected)
    print("FROZEN_PHASE_PRIORS matches apply_overrides(combined) -- ok")
