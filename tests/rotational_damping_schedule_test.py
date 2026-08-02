"""Rotational-damping consistency fix for the hold_contact_transition_s
stiffness ramp (see _rotational_damping_schedule in main_acmpc_box_catch.py).

rotational_k ramps impact_command.rotational_stiffness (6.0) -> cfg.
rotational_stiffness (80.0) over hold_contact_transition_s while D was
previously left fixed at cfg.rotational_damping the whole time -- dropping
the damping ratio (zeta ~ D/sqrt(K)) as K rose and sustaining/re-exciting
the box's post-impact residual spin (diagnosed via Stage 1 near_low_speed_
catch evaluation). D(t) = D_hold * sqrt(K(t)/K_hold) keeps zeta pinned at
its K_hold value throughout the ramp. Scoped to that ramp branch only --
the pre-contact TTC-soften branch and the first_contact_window branch keep
D fixed at cfg.rotational_damping unconditionally, unchanged from before.

SC1-SC4 test the pure schedule function directly (no MuJoCo -- fast).
SC5 is a real-physics regression check that Stage 0's fixture-based catch
(which passes through this same ramp) still reaches a full 5s stable hold,
i.e. the general phase-transition/success judgment is unaffected by this
consistency fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.main_acmpc_box_catch import (
    AcmpcBoxCatchConfig,
    _rotational_damping_schedule,
    run_box_catch,
)

K_HOLD = 80.0
D_HOLD = 14.0
K_IMPACT = 6.0


def sc1_equals_hold_damping_at_hold_stiffness():
    # Property check with arbitrary (K_hold, D_hold) pairs, not just the
    # production constants (those are covered end-to-end by SC4/SC5).
    cases = [(80.0, 14.0), (1.0, 1.0), (250.0, 30.0)]
    results = [(_rotational_damping_schedule(k, k, d), d) for k, d in cases]
    ok = all(abs(got - expect) < 1e-9 for got, expect in results)
    print(
        f"[{'PASS' if ok else 'FAIL'}] SC1 rotational_k == rotational_stiffness => "
        f"rotational_d == rotational_damping\n       results={results}"
    )
    assert ok


def sc2_monotonic_in_k():
    ks = [0.0, 1.0, K_IMPACT, 20.0, 50.0, K_HOLD]
    ds = [_rotational_damping_schedule(k, K_HOLD, D_HOLD) for k in ks]
    ok = all(ds[i] <= ds[i + 1] + 1e-12 for i in range(len(ds) - 1))
    print(f"[{'PASS' if ok else 'FAIL'}] SC2 rotational_d monotonically increases with rotational_k\n       ds={ds}")
    assert ok


def sc3_damping_ratio_constant_through_ramp():
    ks = np.linspace(K_IMPACT, K_HOLD, 50)
    zetas = [
        _rotational_damping_schedule(float(k), K_HOLD, D_HOLD) / np.sqrt(float(k)) for k in ks
    ]
    reference = D_HOLD / np.sqrt(K_HOLD)
    ok = all(abs(z - reference) < 1e-9 for z in zetas)
    print(
        f"[{'PASS' if ok else 'FAIL'}] SC3 D/sqrt(K) constant across the ramp\n"
        f"       reference={reference:.6f} max_dev={max(abs(z - reference) for z in zetas):.2e}"
    )
    assert ok


def sc4_converges_exactly_at_transition_end():
    # contact_blend -> 1.0 drives rotational_k -> cfg.rotational_stiffness
    # exactly (the ramp's own linear interpolation, not this function's
    # concern) -- what this function must guarantee is that feeding it that
    # exact K_hold value returns exactly D_hold, with no residual scale
    # error from the sqrt/clip.
    d = _rotational_damping_schedule(K_HOLD, K_HOLD, D_HOLD)
    ok = abs(d - D_HOLD) < 1e-9
    print(
        f"[{'PASS' if ok else 'FAIL'}] SC4 transition converges to exact HOLD K/D\n"
        f"       rotational_k={K_HOLD:.3f} rotational_d={d:.6f} (expect {D_HOLD})"
    )
    assert ok


def sc5_stage0_general_judgment_unchanged():
    cfg = AcmpcBoxCatchConfig(
        seed=7,
        device="cpu",
        online_learning=False,
        use_launch_fixture=True,
        release_fixture_on_bilateral_contact=True,
    )
    summary = run_box_catch(cfg)
    ok = summary.success and summary.hold_time_s >= 5.0
    print(
        f"[{'PASS' if ok else 'FAIL'}] SC5 Stage 0 fixture catch (same ramp branch) still reaches full 5s hold\n"
        f"       success={summary.success} hold_time_s={summary.hold_time_s:.3f} reason={summary.failure_reason!r}"
    )
    assert ok


def main() -> None:
    checks = [
        sc1_equals_hold_damping_at_hold_stiffness,
        sc2_monotonic_in_k,
        sc3_damping_ratio_constant_through_ramp,
        sc4_converges_exactly_at_transition_end,
        sc5_stage0_general_judgment_unchanged,
    ]
    passed = 0
    for check in checks:
        try:
            check()
            passed += 1
        except AssertionError:
            pass
        print()
    print(f"Result: {passed}/{len(checks)} SCs passed")


if __name__ == "__main__":
    main()
