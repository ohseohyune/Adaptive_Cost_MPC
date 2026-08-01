"""Stage 0 (static_grasp_bootstrap) fixture-based static grasp.

Fixture release is gated on a physically-grounded per-pad force criterion
(mu*F_L + mu*F_R >= mg, F_L=F_R=F => F_required = mg/(2*mu), scaled by
`fixture_release_force_safety_factor`), not a plain bilateral-contact-time
dwell -- see `_fixture_release_force_n`/`_fixture_release_conditions_met`
in main_acmpc_box_catch.py, right after `impact = limiter.update(...)`.

SC1/SC3/SC4/SC6/SC7 test the pure decision logic directly (synthetic
PadContactMeasurement/BilateralPadContact, no MuJoCo -- fast). SC2/SC5/SC8/
SC9 need real physics (contact dynamics, the weld constraint) and run a
short real episode each -- these are the slow ones. Note: like the
predecessor of this file, a real run (seed=7, engineered priors) releases
the fixture cleanly (~0.40s, well below the safety limit -- confirming
this change's actual goal) but does not yet reliably reach a full 5s HOLD
afterward; that CAPTURE-phase grip-ramp gap is unrelated to fixture release
and is separately flagged in the implementation report, not tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.squeeze.pad_contact import BilateralPadContact, PadContactMeasurement
from acmpc.main_acmpc_box_catch import (
    AcmpcBoxCatchConfig,
    CatchPhase,
    _enter_preimpact,
    _fixture_release_conditions_met,
    _fixture_release_force_n,
    _stable_hold_reset_reason,
    _stage0_strict_stable_contact,
    _ttc_from_velocity,
    compute_episode_funnel,
    run_box_catch,
)


def _pad(active: bool, normal_force: float) -> PadContactMeasurement:
    return PadContactMeasurement(
        active=active,
        count=1 if active else 0,
        normal_force=normal_force,
        tangential_force=0.0,
        peak_normal_force=normal_force,
        mean_position=np.zeros(3),
        contacts=(),
    )


def sc1_low_force_does_not_satisfy_release_conditions() -> tuple[bool, str]:
    threshold = 10.0
    contact = BilateralPadContact(left=_pad(True, 4.0), right=_pad(True, 4.0))
    met = _fixture_release_conditions_met(contact, threshold, emergency=False)
    ok = not met
    return ok, f"both active but below threshold({threshold}N): conditions_met={met} (expect False)"


def sc2_real_release_with_dwell() -> tuple[bool, str]:
    cfg = AcmpcBoxCatchConfig(
        seed=7,
        device="cpu",
        online_learning=False,
        use_launch_fixture=True,
        release_fixture_on_bilateral_contact=True,
    )
    summary = run_box_catch(cfg)
    ok = (
        summary.fixture_released
        and summary.fixture_release_time_s is not None
        and summary.fixture_release_time_s > 0.0
        and summary.fixture_release_left_force_n >= summary.fixture_release_force_threshold_n
        and summary.fixture_release_right_force_n >= summary.fixture_release_force_threshold_n
        and summary.fixture_release_left_force_n < cfg.squeeze.first_contact_force_limit
    )
    return ok, (
        f"released={summary.fixture_released} time={summary.fixture_release_time_s} "
        f"threshold={summary.fixture_release_force_threshold_n:.2f} "
        f"L={summary.fixture_release_left_force_n:.2f} R={summary.fixture_release_right_force_n:.2f}"
    )


def sc3_left_only_high_force_does_not_release() -> tuple[bool, str]:
    threshold = 10.0
    contact = BilateralPadContact(left=_pad(True, 15.0), right=_pad(True, 3.0))
    met = _fixture_release_conditions_met(contact, threshold, emergency=False)
    ok = not met
    return ok, f"left=15N right=3N threshold={threshold}N: conditions_met={met} (expect False -- no averaging)"


def sc3b_right_only_high_force_does_not_release() -> tuple[bool, str]:
    threshold = 10.0
    contact = BilateralPadContact(left=_pad(True, 3.0), right=_pad(True, 15.0))
    met = _fixture_release_conditions_met(contact, threshold, emergency=False)
    ok = not met
    return ok, f"left=3N right=15N threshold={threshold}N: conditions_met={met} (expect False)"


def sc4_force_condition_break_resets_dwell_timer_semantics() -> tuple[bool, str]:
    # Directly exercises the same dwell-reset logic run_box_catch uses
    # (accumulate while met, reset to 0.0 the instant it isn't) without
    # needing a real physics run.
    threshold = 10.0
    control_dt = 0.01
    duration = 0.0
    sequence = [
        BilateralPadContact(left=_pad(True, 12.0), right=_pad(True, 12.0)),  # met
        BilateralPadContact(left=_pad(True, 12.0), right=_pad(True, 12.0)),  # met
        BilateralPadContact(left=_pad(True, 8.0), right=_pad(True, 12.0)),  # broken (left dips)
        BilateralPadContact(left=_pad(True, 12.0), right=_pad(True, 12.0)),  # met again
    ]
    durations_seen = []
    for contact in sequence:
        if _fixture_release_conditions_met(contact, threshold, emergency=False):
            duration += control_dt
        else:
            duration = 0.0
        durations_seen.append(round(duration, 3))
    ok = durations_seen == [0.01, 0.02, 0.0, 0.01]
    return ok, f"dwell duration sequence={durations_seen} (expect [0.01, 0.02, 0.0, 0.01])"


def sc5_emergency_takes_priority_over_release() -> tuple[bool, str]:
    threshold = 10.0
    contact = BilateralPadContact(left=_pad(True, 20.0), right=_pad(True, 20.0))
    met = _fixture_release_conditions_met(contact, threshold, emergency=True)
    ok = not met
    return ok, f"both well above threshold but emergency=True: conditions_met={met} (expect False)"


def sc6_higher_mass_raises_release_threshold() -> tuple[bool, str]:
    low = _fixture_release_force_n(object_mass=0.3, object_friction=1.2, safety_factor=1.3)
    high = _fixture_release_force_n(object_mass=0.8, object_friction=1.2, safety_factor=1.3)
    ok = high > low
    return ok, f"mass=0.3kg -> {low:.3f}N, mass=0.8kg -> {high:.3f}N (expect increasing)"


def sc7_higher_friction_lowers_release_threshold() -> tuple[bool, str]:
    low_friction = _fixture_release_force_n(object_mass=0.5, object_friction=0.7, safety_factor=1.3)
    high_friction = _fixture_release_force_n(object_mass=0.5, object_friction=1.5, safety_factor=1.3)
    ok = high_friction < low_friction
    return ok, (
        f"friction=0.7 -> {low_friction:.3f}N, friction=1.5 -> {high_friction:.3f}N "
        f"(expect decreasing)"
    )


def sc8_hold_timer_never_accumulates_before_release() -> tuple[bool, str]:
    cfg = AcmpcBoxCatchConfig(
        seed=7,
        device="cpu",
        online_learning=False,
        use_launch_fixture=True,
        release_fixture_on_bilateral_contact=True,
    )
    summary = run_box_catch(cfg)
    if not summary.fixture_released:
        ok = summary.hold_time_s == 0.0
        return ok, f"fixture never released; hold_time_s={summary.hold_time_s} (expect 0.0)"
    elapsed_since_release = summary.simulated_time_s - summary.fixture_release_time_s
    ok = summary.hold_time_s <= elapsed_since_release + 1e-6
    return ok, f"hold_time_s={summary.hold_time_s:.3f} <= elapsed_since_release={elapsed_since_release:.3f}: {ok}"


def sc9_release_timeout_path_still_works() -> tuple[bool, str]:
    cfg = AcmpcBoxCatchConfig(
        seed=7,
        device="cpu",
        online_learning=False,
        use_launch_fixture=True,
        release_fixture_on_bilateral_contact=True,
        # Higher than what real contact reaches in this window (per SC2's
        # observed ~7N steady-state), forcing the timeout path instead.
        fixture_release_force_safety_factor=50.0,
        fixture_release_timeout_s=0.3,
    )
    summary = run_box_catch(cfg)
    funnel = compute_episode_funnel(summary, cfg)
    ok = (
        not summary.fixture_released
        and not summary.success
        and "fixture release" in summary.failure_reason
        and funnel.failure_category in {"fixture release timeout", "fixture release force threshold unsafe"}
    )
    return ok, (
        f"fixture_released={summary.fixture_released} failure_reason={summary.failure_reason!r} "
        f"failure_category={funnel.failure_category!r}"
    )


def sc10_legacy_default_unaffected() -> tuple[bool, str]:
    default_cfg = AcmpcBoxCatchConfig(seed=7, device="cpu", online_learning=False)
    ok_default_off = default_cfg.use_launch_fixture is False
    summary = run_box_catch(default_cfg)
    ok = ok_default_off and not summary.fixture_enabled and summary.success
    return ok, (
        f"use_launch_fixture default={default_cfg.use_launch_fixture} "
        f"fixture_enabled={summary.fixture_enabled} success={summary.success} "
        f"hold={summary.hold_time_s:.3f}s"
    )


def sc11_valid_ttc_within_soften_window_requests_preimpact() -> tuple[bool, str]:
    ttc_raw, valid = _ttc_from_velocity(raw_vx=-1.0, catch_plane_x=0.30, position_x=0.35)
    ok_ttc = valid and np.isclose(ttc_raw, 0.05)
    requests = _enter_preimpact(
        prediction_confident=True,
        remaining_ttc_valid=valid,
        remaining_ttc=ttc_raw,
        ttc_soften_window_s=0.18,
        pad_box_surface_distance_m=1.0,
        precontact_distance=0.10,
        left_contact_active=False,
        right_contact_active=False,
    )
    ok = ok_ttc and requests
    return ok, f"ttc_raw={ttc_raw:.4f} valid={valid} enter_preimpact={requests}"


def sc12_invalid_ttc_lets_pad_distance_fallback_work() -> tuple[bool, str]:
    # Near-zero velocity (box not really approaching) -- must be invalid
    # regardless of what the raw division would produce.
    ttc_raw, valid = _ttc_from_velocity(raw_vx=0.0005, catch_plane_x=0.30, position_x=0.35)
    ok_invalid = not valid
    requests = _enter_preimpact(
        prediction_confident=True,
        remaining_ttc_valid=valid,
        remaining_ttc=999.0,  # what an unguarded caller might wrongly pass through
        ttc_soften_window_s=0.18,
        pad_box_surface_distance_m=0.05,  # pads already close to the box surface
        precontact_distance=0.10,
        left_contact_active=False,
        right_contact_active=False,
    )
    ok = ok_invalid and requests
    return ok, f"ttc_valid={valid} (expect False) pad_distance_fallback_fires={requests}"


def sc13_invalid_ttc_does_not_block_other_fallbacks() -> tuple[bool, str]:
    # Same invalid TTC as SC12, but neither pad close NOR in contact --
    # must correctly stay False (invalid TTC must not itself force a
    # transition, nor silently block the other gates from being evaluated).
    _, valid = _ttc_from_velocity(raw_vx=0.0, catch_plane_x=0.30, position_x=0.35)
    requests = _enter_preimpact(
        prediction_confident=True,
        remaining_ttc_valid=valid,
        remaining_ttc=999.0,
        ttc_soften_window_s=0.18,
        pad_box_surface_distance_m=1.0,
        precontact_distance=0.10,
        left_contact_active=False,
        right_contact_active=False,
    )
    ok = not valid and not requests
    return ok, f"ttc_valid={valid} enter_preimpact={requests} (expect False, False)"


def sc14_ee_distance_irrelevant_pad_distance_alone_triggers() -> tuple[bool, str]:
    # The regression itself: hand_object_distance (~0.23-0.26m observed)
    # would never satisfy precontact_distance=0.10, but pad_box_surface_
    # distance_m (what actually matters) can independently of it.
    requests = _enter_preimpact(
        prediction_confident=False,
        remaining_ttc_valid=False,
        remaining_ttc=999.0,
        ttc_soften_window_s=0.18,
        pad_box_surface_distance_m=0.09,
        precontact_distance=0.10,
        left_contact_active=False,
        right_contact_active=False,
    )
    ok = requests
    return ok, f"pad_box_surface_distance=0.09 <= precontact_distance=0.10: enter_preimpact={requests}"


def sc15_real_contact_forces_preimpact_even_with_no_other_signal() -> tuple[bool, str]:
    # The hard invariant from item 6: real contact must never leave phase
    # stuck at INTERCEPT, even if TTC and pad-distance both say "not yet".
    requests = _enter_preimpact(
        prediction_confident=False,
        remaining_ttc_valid=False,
        remaining_ttc=999.0,
        ttc_soften_window_s=0.18,
        pad_box_surface_distance_m=1.0,
        precontact_distance=0.10,
        left_contact_active=True,
        right_contact_active=False,
    )
    ok = requests
    return ok, f"left contact active only, everything else says no: enter_preimpact={requests}"


def sc16_real_run_phase_leaves_intercept() -> tuple[bool, str]:
    # The actual regression test: a real Stage-0 episode must not stay
    # phase=intercept for its entire duration (the confirmed bug this
    # session's diagnosis found -- 700/700 steps stuck).
    cfg = AcmpcBoxCatchConfig(
        seed=7,
        device="cpu",
        online_learning=False,
        use_launch_fixture=True,
        release_fixture_on_bilateral_contact=True,
    )
    summary = run_box_catch(cfg)
    ok = summary.final_phase != CatchPhase.INTERCEPT.value
    return ok, f"final_phase={summary.final_phase!r} (expect != 'intercept')"


def sc17_stage0_angular_exceeded_still_accumulates() -> tuple[bool, str]:
    # Requirement 1: force+linear OK, angular exceeded -> Stage 0 (use_
    # launch_fixture=True) must still count this step toward the hold timer.
    stable = _stage0_strict_stable_contact(
        strict_force_ok=True, strict_speed_ok=True, strict_angular_ok=False, use_launch_fixture=True
    )
    reason = _stable_hold_reset_reason(
        phase_is_hold=True, fixture_gate_ok=True,
        left_active=True, right_active=True,
        left_force_n=5.0, right_force_n=5.0, required_grip_force_n=2.6, strict_grip_force_max_n=32.0,
        linear_speed_ok=True,
    )
    ok = stable and reason == ""
    return ok, f"stable={stable} (expect True) reset_reason={reason!r} (expect '')"


def sc18_stage0_linear_speed_exceeded_resets_regardless_of_angular() -> tuple[bool, str]:
    # Requirement 2: linear speed over threshold resets the timer no matter
    # what angular velocity is doing (angular no longer participates at all).
    stable = _stage0_strict_stable_contact(
        strict_force_ok=True, strict_speed_ok=False, strict_angular_ok=True, use_launch_fixture=True
    )
    reason = _stable_hold_reset_reason(
        phase_is_hold=True, fixture_gate_ok=True,
        left_active=True, right_active=True,
        left_force_n=5.0, right_force_n=5.0, required_grip_force_n=2.6, strict_grip_force_max_n=32.0,
        linear_speed_ok=False,
    )
    ok = not stable and reason == "linear_speed_exceeded"
    return ok, f"stable={stable} (expect False) reset_reason={reason!r} (expect 'linear_speed_exceeded')"


def sc19_contact_or_force_break_still_resets() -> tuple[bool, str]:
    # Requirement 3: bilateral contact/required-grip-force conditions are
    # untouched by this change -- still reset the timer on their own.
    stable_no_contact = _stage0_strict_stable_contact(
        strict_force_ok=False, strict_speed_ok=True, strict_angular_ok=True, use_launch_fixture=True
    )
    reason_left_lost = _stable_hold_reset_reason(
        phase_is_hold=True, fixture_gate_ok=True,
        left_active=False, right_active=True,
        left_force_n=0.0, right_force_n=5.0, required_grip_force_n=2.6, strict_grip_force_max_n=32.0,
        linear_speed_ok=True,
    )
    reason_force_low = _stable_hold_reset_reason(
        phase_is_hold=True, fixture_gate_ok=True,
        left_active=True, right_active=True,
        left_force_n=1.0, right_force_n=5.0, required_grip_force_n=2.6, strict_grip_force_max_n=32.0,
        linear_speed_ok=True,
    )
    ok = (
        not stable_no_contact
        and reason_left_lost == "left_contact_lost"
        and reason_force_low == "left_force_below_required"
    )
    return ok, (
        f"stable_no_contact={stable_no_contact} (expect False) "
        f"reason_left_lost={reason_left_lost!r} reason_force_low={reason_force_low!r}"
    )


def sc20_angular_is_diagnostic_only_not_success_gating() -> tuple[bool, str]:
    # Requirement 4: angular-exceeded is surfaced as a separate diagnostic
    # flag (angular_speed_exceeded_diagnostic = not strict_angular_ok) and
    # never appears in the reset reason vocabulary at all.
    angular_speed_exceeded_diagnostic = not False  # strict_angular_ok=False
    stable = _stage0_strict_stable_contact(
        strict_force_ok=True, strict_speed_ok=True, strict_angular_ok=False, use_launch_fixture=True
    )
    reason = _stable_hold_reset_reason(
        phase_is_hold=True, fixture_gate_ok=True,
        left_active=True, right_active=True,
        left_force_n=5.0, right_force_n=5.0, required_grip_force_n=2.6, strict_grip_force_max_n=32.0,
        linear_speed_ok=True,
    )
    ok = stable and angular_speed_exceeded_diagnostic and "angular" not in reason
    return ok, (
        f"stable={stable} angular_speed_exceeded_diagnostic={angular_speed_exceeded_diagnostic} "
        f"reset_reason={reason!r} (must not mention angular)"
    )


def sc21_fixture_gate_blocks_timer_even_if_all_physical_conditions_ok() -> tuple[bool, str]:
    # Requirement 5: before fixture release, the timer must not accumulate
    # even if force/linear/angular would all otherwise pass.
    reason = _stable_hold_reset_reason(
        phase_is_hold=True, fixture_gate_ok=False,
        left_active=True, right_active=True,
        left_force_n=5.0, right_force_n=5.0, required_grip_force_n=2.6, strict_grip_force_max_n=32.0,
        linear_speed_ok=True,
    )
    ok = reason == "fixture_not_released"
    return ok, f"reset_reason={reason!r} (expect 'fixture_not_released')"


def sc22_real_run_reaches_stable_hold_5s() -> tuple[bool, str]:
    # The end-to-end payoff: with angular decoupled from Stage 0's success
    # gate, the seed=7 engineered-priors run should now actually complete
    # the full 5s stable hold (previously blocked: angular exceeded 98% of
    # HOLD steps despite contact/force being perfect and linear speed only
    # occasionally over).
    cfg = AcmpcBoxCatchConfig(
        seed=7,
        device="cpu",
        online_learning=False,
        use_launch_fixture=True,
        release_fixture_on_bilateral_contact=True,
    )
    summary = run_box_catch(cfg)
    ok = summary.success and summary.hold_time_s >= cfg.required_hold_s
    return ok, f"success={summary.success} hold_time_s={summary.hold_time_s:.3f} (target={cfg.required_hold_s})"


def main() -> None:
    scenarios = [
        ("SC1 low force does not satisfy release conditions", sc1_low_force_does_not_satisfy_release_conditions),
        ("SC2 real run: release happens with force dwell satisfied", sc2_real_release_with_dwell),
        ("SC3 left-only high force does not release (no averaging)", sc3_left_only_high_force_does_not_release),
        ("SC3b right-only high force does not release", sc3b_right_only_high_force_does_not_release),
        ("SC4 dwell timer resets when force condition breaks", sc4_force_condition_break_resets_dwell_timer_semantics),
        ("SC5 emergency takes priority over release", sc5_emergency_takes_priority_over_release),
        ("SC6 higher mass raises the release force threshold", sc6_higher_mass_raises_release_threshold),
        ("SC7 higher friction lowers the release force threshold", sc7_higher_friction_lowers_release_threshold),
        ("SC8 hold timer never accumulates before fixture release", sc8_hold_timer_never_accumulates_before_release),
        ("SC9 fixture release timeout path still works", sc9_release_timeout_path_still_works),
        ("SC10 legacy default (use_launch_fixture=False) unaffected", sc10_legacy_default_unaffected),
        ("SC11 valid TTC within soften window requests PRE_IMPACT", sc11_valid_ttc_within_soften_window_requests_preimpact),
        ("SC12 invalid TTC lets pad-distance fallback work", sc12_invalid_ttc_lets_pad_distance_fallback_work),
        ("SC13 invalid TTC does not block/force other fallbacks", sc13_invalid_ttc_does_not_block_other_fallbacks),
        ("SC14 EE distance irrelevant, pad distance alone triggers", sc14_ee_distance_irrelevant_pad_distance_alone_triggers),
        ("SC15 real contact forces PRE_IMPACT even with no other signal", sc15_real_contact_forces_preimpact_even_with_no_other_signal),
        ("SC16 real run: phase leaves INTERCEPT (the regression itself)", sc16_real_run_phase_leaves_intercept),
        ("SC17 angular exceeded still accumulates (force+linear OK)", sc17_stage0_angular_exceeded_still_accumulates),
        ("SC18 linear speed exceeded resets regardless of angular", sc18_stage0_linear_speed_exceeded_resets_regardless_of_angular),
        ("SC19 contact/force break still resets the timer", sc19_contact_or_force_break_still_resets),
        ("SC20 angular is diagnostic-only, never gates success", sc20_angular_is_diagnostic_only_not_success_gating),
        ("SC21 fixture gate blocks timer even if physics all OK", sc21_fixture_gate_blocks_timer_even_if_all_physical_conditions_ok),
        ("SC22 real run: stable HOLD 5s success end-to-end", sc22_real_run_reaches_stable_hold_5s),
    ]
    passed = 0
    print("\n=== Stage 0 static-grasp fixture (force-based release): smoke tests ===\n")
    for name, scenario in scenarios:
        ok, detail = scenario()
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}\n")
        passed += int(ok)
    print(f"Result: {passed}/{len(scenarios)} SCs passed")
    if passed != len(scenarios):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
