"""Read-only reachability-margin diagnosis for the Stage 1 edge/corner-grasp
bug: was there ever enough lead time for the arms to reach a *corrected*
(geometrically accurate) contact target before real contact happened, and if
so, did the PRE_IMPACT phase gate simply start too late to use it?

Builds entirely on acmpc/diagnose_precontact_grasp_offset.py's recorded
per-substep trace (prediction_position/prediction_velocity/
position_remaining_ttc/y_axis/pad site positions are all already captured
there) -- no new production-code changes, no control-loop changes. The
"corrected target" computed here is diagnostic-only and is never fed to the
controller.

Usage: python3 acmpc/diagnose_reachability_margin.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import (
    STAGE1_SEEDS,
    Row,
    analyze,
    run_episode,
    seed_everything,
)

GRAVITY = np.array([0.0, 0.0, -9.81])
MIN_VALID_CLOSING_SPEED_MPS = 0.02  # floor to stop noise/near-zero speed exploding T_reach_empirical
CLOSING_SPEED_WINDOW_S = 0.08       # trailing window for the empirical closing-speed percentile
CLOSING_SPEED_PERCENTILE = 25       # conservative (p25, not p50) per the task's "p50 또는 p25" guidance
REACHABILITY_SAFETY_MARGIN_S = 0.05  # diagnostic-only threshold, see report
TARGET_STEP_DEDUPE_EPS = 1e-9        # treat identical consecutive corrected targets as "not yet updated"


def _corrected_target(r: Row, pad_offset_scalar: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagnostic-only box-face prediction and both pad targets from it.

    Uses r.position_remaining_ttc (already computed read-only in the main
    loop, never fed to the controller) with the exact same
    p(tau) = p0 + v0*tau + 0.5*g*tau**2 model production code uses.
    """
    tau = max(0.0, float(r.position_remaining_ttc))
    box_pred = r.prediction_position + r.prediction_velocity * tau + 0.5 * GRAVITY * tau**2
    left = box_pred + pad_offset_scalar * r.y_axis
    right = box_pred - pad_offset_scalar * r.y_axis
    return box_pred, left, right


@dataclass
class ReachSample:
    time_s: float
    phase: str
    corrected_ttc: float
    corrected_ttc_valid: bool
    d_left: float
    d_right: float
    d_max: float
    e_left: np.ndarray  # corrected_target_left - left_pad_site_pos
    e_right: np.ndarray
    T_reach_ideal: float
    T_reach_axis: float
    T_reach_empirical: float
    margin_ideal: float
    margin_axis: float
    margin_empirical: float
    corrected_target_left: np.ndarray
    corrected_target_right: np.ndarray


def _closing_speed(e: np.ndarray, v_pad: np.ndarray) -> float:
    # e = p_pad - p_target (pointing away from target); closing speed is
    # positive when v_pad reduces |e|.
    norm = float(np.linalg.norm(e))
    return float(-(e @ v_pad) / (norm + 1e-9))


def build_reach_timeseries(result: dict, out: dict) -> list[ReachSample]:
    dense = result["rec"].dense
    contact_t = out.get("actual_contact_time_s")
    if contact_t is None or not dense:
        return []
    cfg = result["cfg"]
    v_max = float(cfg.mpc_velocity_limit)
    pad_offset_scalar = float(cfg.squeeze.box_half_y + cfg.squeeze.pad_half_thickness)

    pre_rows = [r for r in dense if r.time_s <= contact_t]
    if len(pre_rows) < 2:
        return []

    # instantaneous pad velocities via finite difference between consecutive
    # recorded substeps (physics dt, uniform).
    left_vel = np.zeros((len(pre_rows), 3))
    right_vel = np.zeros((len(pre_rows), 3))
    for i in range(1, len(pre_rows)):
        dt = pre_rows[i].time_s - pre_rows[i - 1].time_s
        if dt > 1e-9:
            left_vel[i] = (pre_rows[i].left_pad_site_pos - pre_rows[i - 1].left_pad_site_pos) / dt
            right_vel[i] = (pre_rows[i].right_pad_site_pos - pre_rows[i - 1].right_pad_site_pos) / dt
    left_vel[0] = left_vel[1] if len(pre_rows) > 1 else left_vel[0]
    right_vel[0] = right_vel[1] if len(pre_rows) > 1 else right_vel[0]

    raw_closing_left = np.zeros(len(pre_rows))
    raw_closing_right = np.zeros(len(pre_rows))
    for i, r in enumerate(pre_rows):
        _, target_left, target_right = _corrected_target(r, pad_offset_scalar)
        e_left = r.left_pad_site_pos - target_left
        e_right = r.right_pad_site_pos - target_right
        raw_closing_left[i] = _closing_speed(e_left, left_vel[i])
        raw_closing_right[i] = _closing_speed(e_right, right_vel[i])

    samples: list[ReachSample] = []
    times = np.array([r.time_s for r in pre_rows])
    for i, r in enumerate(pre_rows):
        _, target_left, target_right = _corrected_target(r, pad_offset_scalar)
        e_left = r.left_pad_site_pos - target_left
        e_right = r.right_pad_site_pos - target_right
        d_left = float(np.linalg.norm(e_left))
        d_right = float(np.linalg.norm(e_right))
        d_max = max(d_left, d_right)

        T_reach_ideal = d_max / v_max if v_max > 1e-9 else float("inf")
        axis_left = float(np.max(np.abs(e_left)) / v_max) if v_max > 1e-9 else float("inf")
        axis_right = float(np.max(np.abs(e_right)) / v_max) if v_max > 1e-9 else float("inf")
        T_reach_axis = max(axis_left, axis_right)

        window_mask = (times <= r.time_s) & (times >= r.time_s - CLOSING_SPEED_WINDOW_S)
        if np.any(window_mask):
            cl_left = float(np.percentile(raw_closing_left[window_mask], CLOSING_SPEED_PERCENTILE))
            cl_right = float(np.percentile(raw_closing_right[window_mask], CLOSING_SPEED_PERCENTILE))
        else:
            cl_left = raw_closing_left[i]
            cl_right = raw_closing_right[i]

        def _t_reach_empirical(d: float, closing_speed: float) -> float:
            if closing_speed <= MIN_VALID_CLOSING_SPEED_MPS:
                return float("inf")
            return d / closing_speed

        T_reach_empirical = max(
            _t_reach_empirical(d_left, cl_left), _t_reach_empirical(d_right, cl_right)
        )

        ttc = float(r.position_remaining_ttc)
        ttc_valid = bool(r.position_remaining_ttc_valid and np.isfinite(ttc) and ttc >= 0.0)

        samples.append(
            ReachSample(
                time_s=r.time_s,
                phase=r.phase,
                corrected_ttc=ttc,
                corrected_ttc_valid=ttc_valid,
                d_left=d_left,
                d_right=d_right,
                d_max=d_max,
                e_left=e_left,
                e_right=e_right,
                T_reach_ideal=T_reach_ideal,
                T_reach_axis=T_reach_axis,
                T_reach_empirical=T_reach_empirical,
                margin_ideal=(ttc - T_reach_ideal) if ttc_valid else float("-inf"),
                margin_axis=(ttc - T_reach_axis) if ttc_valid else float("-inf"),
                margin_empirical=(ttc - T_reach_empirical) if ttc_valid else float("-inf"),
                corrected_target_left=target_left,
                corrected_target_right=target_right,
            )
        )
    return samples


def _first_time(samples: list[ReachSample], pred) -> float | None:
    for s in samples:
        if pred(s):
            return s.time_s
    return None


def _lead(t: float | None, contact_t: float) -> float | None:
    return None if t is None else contact_t - t


def _nearest(samples: list[ReachSample], t: float):
    return min(samples, key=lambda s: abs(s.time_s - t))


def target_stability(samples: list[ReachSample], v_max: float) -> dict:
    # dedupe consecutive identical corrected targets (control-step
    # granularity: the value is recomputed once per control step and held
    # constant across intervening physics substeps, so naive substep-level
    # differencing would be dominated by zeros).
    left_updates = []
    right_updates = []
    prev_left, prev_right, prev_t = None, None, None
    for s in samples:
        if prev_left is None or np.linalg.norm(s.corrected_target_left - prev_left) > TARGET_STEP_DEDUPE_EPS:
            if prev_left is not None:
                dt = s.time_s - prev_t
                if dt > 1e-6:
                    left_updates.append((np.linalg.norm(s.corrected_target_left - prev_left) / dt,
                                          float(np.linalg.norm(s.corrected_target_left - prev_left))))
                    right_updates.append((np.linalg.norm(s.corrected_target_right - prev_right) / dt,
                                           float(np.linalg.norm(s.corrected_target_right - prev_right))))
            prev_left, prev_right, prev_t = s.corrected_target_left, s.corrected_target_right, s.time_s
    if not left_updates:
        return {
            "target_velocity_p95_mps": None,
            "target_max_step_change_m": None,
            "stability": "STABLE",  # never updated / single value -> trivially stable
        }
    vels = [v for v, _ in left_updates] + [v for v, _ in right_updates]
    steps = [d for _, d in left_updates] + [d for _, d in right_updates]
    p95_vel = float(np.percentile(vels, 95))
    return {
        "target_velocity_p95_mps": p95_vel,
        "target_max_step_change_m": float(np.max(steps)),
        "stability": "STABLE" if p95_vel <= v_max else "UNSTABLE",
    }


def diagnose_from_result(seed: int, result: dict, out: dict, samples: list) -> dict:
    if "error" in out or out.get("actual_contact_time_s") is None:
        return {"seed": seed, "error": out.get("error", "no contact")}
    if not samples:
        return {"seed": seed, "error": "no reachability samples"}

    contact_t = out["actual_contact_time_s"]
    cfg = result["cfg"]
    v_max = float(cfg.mpc_velocity_limit)
    stability = target_stability(samples, v_max)

    dense_light_phase = [(lr.time_s, lr.phase) for lr in result["rec"].light]
    pre_impact_entry_t = next((t for t, p in dense_light_phase if p == "pre_impact"), None)

    t_target_valid = _first_time(samples, lambda s: s.corrected_ttc_valid)
    t_ideal_pos = _first_time(samples, lambda s: s.corrected_ttc_valid and s.margin_ideal > 0)
    t_axis_pos = _first_time(samples, lambda s: s.corrected_ttc_valid and s.margin_axis > 0)
    t_emp_pos = _first_time(samples, lambda s: s.corrected_ttc_valid and s.margin_empirical > 0)
    t_emp_safe = _first_time(
        samples, lambda s: s.corrected_ttc_valid and s.margin_empirical >= REACHABILITY_SAFETY_MARGIN_S
    )

    gate_delay = None
    if pre_impact_entry_t is not None and t_emp_safe is not None:
        gate_delay = pre_impact_entry_t - t_emp_safe

    at_gate = _nearest(samples, pre_impact_entry_t) if pre_impact_entry_t is not None else None
    m100 = _nearest(samples, contact_t - 0.10)
    m50 = _nearest(samples, contact_t - 0.05)

    ideal_result = "REACHABLE" if (t_ideal_pos is not None) else "NOT_REACHABLE"
    empirical_result = "REACHABLE" if (t_emp_pos is not None) else "NOT_REACHABLE"

    # classification
    if stability["stability"] == "UNSTABLE":
        diagnosis = "TARGET_UNSTABLE"
    elif t_emp_safe is None:
        # empirical margin never got safely positive at all -- either the
        # target was never even geometrically valid in time (TARGET_VALID_TOO_LATE)
        # or margin stayed negative throughout (also TARGET_VALID_TOO_LATE,
        # since "valid" here means "usably reachable", not just "computable").
        diagnosis = "TARGET_VALID_TOO_LATE"
    elif gate_delay is not None and gate_delay > 0.02:
        diagnosis = "PHASE_GATE_TOO_LATE"
    elif ideal_result == "REACHABLE" and (at_gate is not None and at_gate.margin_empirical < 0):
        diagnosis = "ARM_RESPONSE_TOO_SLOW"
    elif gate_delay is not None and abs(gate_delay) <= 0.02:
        diagnosis = "MIXED"
    else:
        diagnosis = "INCONCLUSIVE"

    return {
        "seed": seed,
        "success": out["success"],
        "actual_first_contact_time_s": contact_t,
        "current_PRE_IMPACT_entry_time_s": pre_impact_entry_t,
        "current_PRE_IMPACT_lead_time_s": _lead(pre_impact_entry_t, contact_t),
        "first_corrected_target_valid_lead_time_s": _lead(t_target_valid, contact_t),
        "first_ideal_margin_positive_lead_time_s": _lead(t_ideal_pos, contact_t),
        "first_axis_margin_positive_lead_time_s": _lead(t_axis_pos, contact_t),
        "first_empirical_margin_positive_lead_time_s": _lead(t_emp_pos, contact_t),
        "first_empirical_margin_above_50ms_lead_time_s": _lead(t_emp_safe, contact_t),
        "gate_delay_after_reachable_s": gate_delay,
        "corrected_target_distance_at_PRE_IMPACT_left_m": at_gate.d_left if at_gate else None,
        "corrected_target_distance_at_PRE_IMPACT_right_m": at_gate.d_right if at_gate else None,
        "T_reach_ideal_at_PRE_IMPACT_s": at_gate.T_reach_ideal if at_gate else None,
        "T_reach_axis_at_PRE_IMPACT_s": at_gate.T_reach_axis if at_gate else None,
        "T_reach_empirical_at_PRE_IMPACT_s": at_gate.T_reach_empirical if at_gate else None,
        "margin_ideal_at_PRE_IMPACT_s": at_gate.margin_ideal if at_gate else None,
        "margin_axis_at_PRE_IMPACT_s": at_gate.margin_axis if at_gate else None,
        "margin_empirical_at_PRE_IMPACT_s": at_gate.margin_empirical if at_gate else None,
        "margin_empirical_100ms_before_contact_s": m100.margin_empirical,
        "margin_empirical_50ms_before_contact_s": m50.margin_empirical,
        "target_velocity_p95_mps": stability["target_velocity_p95_mps"],
        "target_max_step_change_m": stability["target_max_step_change_m"],
        "corrected_target_stability": stability["stability"],
        "ideal_reachability_result": ideal_result,
        "empirical_reachability_result": empirical_result,
        "contact_face_center_error_m": out.get("left_contact_distance_from_face_center_m"),
        "contact_classification": out.get("left_first_contact_classification"),
        "COM_actual_grasp_offset_m": out.get("actual_grasp_center_to_com_offset_m"),
        "diagnosis": diagnosis,
    }


def print_timeseries_table(seed: int, result: dict, out: dict, samples: list, step_s: float = 0.02) -> None:
    contact_t = out["actual_contact_time_s"]
    start_t = max(samples[0].time_s, contact_t - 0.5)
    print(f"  -- seed {seed} timeseries (last {contact_t - start_t:.2f}s before contact, {step_s*1000:.0f}ms steps) --")
    header = (
        f"  {'t-contact':>10} {'phase':>10} {'ttc':>7} {'act_rem':>8} "
        f"{'Tideal':>7} {'Taxis':>7} {'Temp':>8} {'Mideal':>8} {'Maxis':>8} {'Memp':>8} "
        f"{'dL':>6} {'dR':>6}"
    )
    print(header)
    t = start_t
    while t <= contact_t + 1e-9:
        s = _nearest(samples, t)
        act_rem = contact_t - s.time_s
        print(
            f"  {s.time_s - contact_t:+10.3f} {s.phase:>10} {s.corrected_ttc:7.3f} {act_rem:8.3f} "
            f"{s.T_reach_ideal:7.3f} {s.T_reach_axis:7.3f} "
            f"{(s.T_reach_empirical if np.isfinite(s.T_reach_empirical) else -1):8.3f} "
            f"{s.margin_ideal:8.3f} {s.margin_axis:8.3f} "
            f"{(s.margin_empirical if np.isfinite(s.margin_empirical) else -99):8.3f} "
            f"{s.d_left:6.3f} {s.d_right:6.3f}"
        )
        t += step_s


def main() -> None:
    all_out = []
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        seed_everything(seed)
        result = run_episode(seed)
        out = analyze(result)
        samples = build_reach_timeseries(result, out)
        diag = diagnose_from_result(seed, result, out, samples)
        all_out.append(diag)
        for k, v in diag.items():
            print(f"  {k}: {v}")
        if samples:
            print_timeseries_table(seed, result, out, samples)
        print(flush=True)

    print("=== summary ===")
    for d in all_out:
        print(f"  seed={d.get('seed')} diagnosis={d.get('diagnosis')} "
              f"gate_delay_s={d.get('gate_delay_after_reachable_s')} "
              f"stability={d.get('corrected_target_stability')} "
              f"contact_cls={d.get('contact_classification')}")


if __name__ == "__main__":
    main()
