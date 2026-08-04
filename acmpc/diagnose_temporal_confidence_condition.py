"""Deep-dive on the one false positive (seed 96834) vs. the high-invalid
true positive (seed 84433) found by the static-TTC online gate
(diagnose_online_shadow_warning.py), and a search for a causal temporal
condition that separates them without touching oracle contact time.

Reuses the same fixed definitions throughout:
  d_A(t), d_B(t), m(t) = d_B - d_A          at N=20 (diagnose_contact_geometry_common.py)
  online gate: 0.02 <= T_static <= 0.15     (production's own position_remaining_ttc, read-only)
  boundary-risk: m < 0.002 m
  strict TIP: m < 0
  baseline warning: 3 consecutive control steps

Single combined per-seed simulation pass records everything needed (no
separate reruns, no new N=20 calls beyond what run_combined already does).

Read-only / shadow-only: does not touch production target, TTC predictor,
trigger, phase gate, pad orientation, controller gain, or trajectory.

Usage: python3 acmpc/diagnose_temporal_confidence_condition.py
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, _build_stage1_config, seed_everything
from acmpc.diagnose_contact_geometry_common import compute_contact_mode_geometry
from acmpc.diagnose_online_shadow_warning import ground_truth_mode, EXTRA_SEED_RNG_SEED, EXTRA_SEED_COUNT
from acmpc.main_acmpc_box_catch import run_box_catch, _MIN_MEANINGFUL_APPROACH_SPEED_MPS

CONTROL_STEP_S = 0.010
ORACLE_WINDOW_S = (0.02, 0.15)
BOUNDARY_RISK_M = 0.002
FOCUS_SEEDS = [96834, 84433]


# ---------------------------------------------------------------------------
# Single combined recording (adds pad_plane_x / box_vx / invalid-reason
# inputs on top of diagnose_online_shadow_warning.py's Sample)
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    time_s: float
    d_a: float
    d_b: float
    t_static: float
    t_static_valid: bool
    box_vx: float
    pad_plane_x: float
    box_half_x: float


@dataclass
class Recording:
    samples: list = field(default_factory=list)


def run_seed(seed: int) -> tuple[dict, Recording]:
    seed_everything(seed)
    cfg = _build_stage1_config(seed)
    ids = {}
    rec = Recording()
    box_half_x = float(cfg.domain_parameters.half_size[0]) if cfg.domain_parameters is not None else float("nan")

    def on_step(ctx: dict) -> None:
        model, data = ctx["model"], ctx["data"]
        if not ids:
            ids["box_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
            ids["pad_geom"] = {
                s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_catch_pad") for s in ("left", "right")
            }
        contact = ctx["contact"]
        if contact.left.active and contact.right.active:
            return

        box_center = data.geom_xpos[ids["box_geom"]].copy()
        box_rot = data.geom_xmat[ids["box_geom"]].reshape(3, 3).copy()
        box_half = model.geom_size[ids["box_geom"]].copy()
        pad_center = data.geom_xpos[ids["pad_geom"]["left"]].copy()
        pad_rot = data.geom_xmat[ids["pad_geom"]["left"]].reshape(3, 3).copy()
        pad_half = model.geom_size[ids["pad_geom"]["left"]].copy()

        g = compute_contact_mode_geometry(box_center, box_rot, box_half, pad_center, pad_rot, pad_half, "left", resolution=20)

        rec.samples.append(
            Sample(
                time_s=ctx["time_s"], d_a=g.dist_a_m, d_b=g.dist_b_m,
                t_static=float(ctx["position_remaining_ttc"]), t_static_valid=bool(ctx["position_remaining_ttc_valid"]),
                box_vx=float(ctx["prediction_velocity"][0]), pad_plane_x=float(ctx["pad_plane_x"]),
                box_half_x=box_half_x,
            )
        )

    summary = run_box_catch(cfg, step_callback=on_step)
    return {"summary": summary, "cfg": cfg}, rec


# ---------------------------------------------------------------------------
# Part 1: invalid-reason classification
# ---------------------------------------------------------------------------

def classify_invalid_reason(s: Sample) -> str | None:
    if s.t_static_valid:
        return None
    if s.box_vx >= -_MIN_MEANINGFUL_APPROACH_SPEED_MPS:
        return "closing_velocity_sign_invalid"
    # vx is a real approach speed -> ttc_raw is finite; invalid must be from ttc_raw<0
    return "predicted_ttc_negative_or_box_passed_plane"


def part1_invalid_breakdown(seed: int, samples: list[Sample], contact_t: float) -> dict:
    n = len(samples)
    reasons = [classify_invalid_reason(s) for s in samples]
    times = np.array([s.time_s for s in samples])
    valid = np.array([s.t_static_valid for s in samples])

    def frac_invalid(mask):
        sub = ~valid[mask]
        return float(np.mean(sub)) if sub.size else None

    last_030 = (contact_t - times) <= 0.30
    last_020 = (contact_t - times) <= 0.20

    first_valid_idx = np.argmax(valid) if valid.any() else None
    first_valid_t = float(times[first_valid_idx]) if first_valid_idx is not None and valid.any() else None

    gate = valid & (times <= contact_t) & (contact_t - times >= 0)
    ttc = np.array([s.t_static for s in samples])
    gate_open = valid & (ttc >= ORACLE_WINDOW_S[0]) & (ttc <= ORACLE_WINDOW_S[1])
    first_gate_open_t = float(times[np.argmax(gate_open)]) if gate_open.any() else None

    # valid-run lengths
    valid_runs = []
    cur = 0
    for v in valid:
        if v:
            cur += 1
        else:
            if cur > 0:
                valid_runs.append(cur)
            cur = 0
    if cur > 0:
        valid_runs.append(cur)
    dt_est = float(np.median(np.diff(times))) if len(times) > 1 else CONTROL_STEP_S
    longest_valid_run_s = max(valid_runs) * dt_est if valid_runs else 0.0

    # gate-open intervals
    open_intervals = 0
    prev = False
    for v in gate_open:
        if v and not prev:
            open_intervals += 1
        prev = v
    total_open_s = float(np.sum(gate_open)) * dt_est
    chatter = int(np.sum(np.diff(gate_open.astype(int)) != 0))

    reason_counts = {}
    for r in reasons:
        if r is not None:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    return {
        "seed": seed,
        "overall_invalid_fraction": float(1 - np.mean(valid)),
        "last_0.30s_invalid_fraction": frac_invalid(last_030),
        "last_0.20s_invalid_fraction": frac_invalid(last_020),
        "first_valid_ttc_time_s": first_valid_t,
        "first_gate_open_time_s": first_gate_open_t,
        "n_valid_runs": len(valid_runs),
        "longest_valid_run_s": longest_valid_run_s,
        "n_gate_open_intervals": open_intervals,
        "total_gate_open_s": total_open_s,
        "gate_chatter_count": chatter,
        "invalid_reason_counts": reason_counts,
    }


# ---------------------------------------------------------------------------
# Part 2: margin time history inside the gate
# ---------------------------------------------------------------------------

def margin_state(m: float) -> str:
    if m < 0.0:
        return "TIP_EDGE_EXPECTED"
    if m < BOUNDARY_RISK_M:
        return "CONTACT_MODE_BOUNDARY_RISK"
    return "ROBUST_FACE_EXPECTED"


def part2_gate_margin_history(seed: int, samples: list[Sample], contact_t: float) -> dict:
    times = np.array([s.time_s for s in samples])
    d_a = np.array([s.d_a for s in samples])
    d_b = np.array([s.d_b for s in samples])
    m = d_b - d_a
    ttc = np.array([s.t_static for s in samples])
    valid = np.array([s.t_static_valid for s in samples])
    gate_open = valid & (ttc >= ORACLE_WINDOW_S[0]) & (ttc <= ORACLE_WINDOW_S[1])

    dt_est = float(np.median(np.diff(times))) if len(times) > 1 else CONTROL_STEP_S
    idx = np.where(gate_open)[0]
    if idx.size == 0:
        return {"seed": seed, "error": "gate never opens"}

    m_gate = m[idx]
    t_gate = times[idx]

    # longest negative-margin / boundary-risk durations (within gate, using
    # actual elapsed time between consecutive gate samples -- robust to
    # substep-density variation)
    def longest_run_duration(mask_local: np.ndarray) -> float:
        best, cur_start = 0.0, None
        for i in range(len(mask_local)):
            if mask_local[i]:
                if cur_start is None:
                    cur_start = t_gate[i]
                cur_dur = t_gate[i] - cur_start + dt_est
                best = max(best, cur_dur)
            else:
                cur_start = None
        return float(best)

    neg_mask = m_gate < 0.0
    boundary_mask = m_gate < BOUNDARY_RISK_M

    longest_neg_s = longest_run_duration(neg_mask)
    longest_boundary_s = longest_run_duration(boundary_mask)

    # accumulated risk area (trapezoid over gate-open samples only)
    def accumulate_area(values: np.ndarray) -> float:
        area = 0.0
        for i in range(1, len(values)):
            dt = t_gate[i] - t_gate[i - 1]
            if 0 < dt < 0.05:  # skip discontinuous jumps across separate open intervals
                area += 0.5 * (values[i] + values[i - 1]) * dt
        return float(area)

    neg_area = accumulate_area(np.maximum(-m_gate, 0.0))
    boundary_area = accumulate_area(np.maximum(BOUNDARY_RISK_M - m_gate, 0.0))

    # recovery check: after first negative margin, does it return above 2mm?
    recovered = False
    recovery_time_s = None
    if neg_mask.any():
        first_neg_i = int(np.argmax(neg_mask))
        after = m_gate[first_neg_i:]
        after_t = t_gate[first_neg_i:]
        above = np.where(after > BOUNDARY_RISK_M)[0]
        if above.size:
            recovered = True
            recovery_time_s = float(after_t[above[0]] - t_gate[first_neg_i])

    return {
        "seed": seed,
        "first_m<2mm_time_s": float(t_gate[np.argmax(boundary_mask)]) if boundary_mask.any() else None,
        "first_m<0_time_s": float(t_gate[np.argmax(neg_mask)]) if neg_mask.any() else None,
        "longest_negative_margin_duration_s": longest_neg_s,
        "longest_boundary_risk_duration_s": longest_boundary_s,
        "negative_margin_area_m_s": neg_area,
        "boundary_risk_area_m_s": boundary_area,
        "margin_recovers_above_2mm": recovered,
        "recovery_time_s": recovery_time_s,
        "min_margin_m": float(np.min(m_gate)),
        "mean_margin_m": float(np.mean(m_gate)),
        "final_margin_before_gate_close_m": float(m_gate[-1]),
        "gate_open_duration_s": float(t_gate[-1] - t_gate[0]),
    }


# ---------------------------------------------------------------------------
# Part 3/5/6: causal temporal-condition evaluators, applied to all 34 seeds
# ---------------------------------------------------------------------------

def build_seed_arrays(samples: list[Sample]) -> dict:
    times = np.array([s.time_s for s in samples])
    d_a = np.array([s.d_a for s in samples])
    d_b = np.array([s.d_b for s in samples])
    m = d_b - d_a
    ttc = np.array([s.t_static for s in samples])
    valid = np.array([s.t_static_valid for s in samples])
    gate_open = valid & (ttc >= ORACLE_WINDOW_S[0]) & (ttc <= ORACLE_WINDOW_S[1])
    dt_est = float(np.median(np.diff(times))) if len(times) > 1 else CONTROL_STEP_S
    return {"times": times, "m": m, "gate_open": gate_open, "dt_est": dt_est}


def resample_control_grid(arrays: dict, contact_t: float) -> dict:
    times, m, gate_open = arrays["times"], arrays["m"], arrays["gate_open"]
    grid_t = np.arange(times[0], contact_t, CONTROL_STEP_S)
    idxs = np.array([np.argmin(np.abs(times - t)) for t in grid_t])
    return {"grid_t": grid_t, "grid_m": m[idxs], "grid_gate_open": gate_open[idxs]}


def eval_baseline_3step(grid: dict, threshold: float) -> tuple[bool, float | None]:
    m, gate = grid["grid_m"], grid["grid_gate_open"]
    n = len(m)
    consecutive = 0
    for i in range(n):
        if gate[i] and m[i] < threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 3:
            return True, float(grid["grid_t"][i])
    return False, None


def eval_duration(grid: dict, threshold: float, duration_s: float) -> tuple[bool, float | None]:
    m, gate, t = grid["grid_m"], grid["grid_gate_open"], grid["grid_t"]
    start = None
    for i in range(len(m)):
        if gate[i] and m[i] < threshold:
            if start is None:
                start = t[i]
            if t[i] - start >= duration_s:
                return True, float(t[i])
        else:
            start = None
    return False, None


def eval_area(grid: dict, threshold: float, area_th: float) -> tuple[bool, float | None]:
    m, gate, t = grid["grid_m"], grid["grid_gate_open"], grid["grid_t"]
    area = 0.0
    prev_t, prev_v = None, 0.0
    for i in range(len(m)):
        if gate[i]:
            v = max(threshold - m[i], 0.0)
            if prev_t is not None and t[i] - prev_t < 0.05:
                area += 0.5 * (v + prev_v) * (t[i] - prev_t)
            prev_t, prev_v = t[i], v
            if area >= area_th:
                return True, float(t[i])
        else:
            prev_t, prev_v = None, 0.0
    return False, None


def eval_rolling_ratio(grid: dict, threshold: float, window_s: float, ratio_th: float) -> tuple[bool, float | None]:
    m, gate, t = grid["grid_m"], grid["grid_gate_open"], grid["grid_t"]
    for i in range(len(m)):
        if not gate[i]:
            continue
        lo = t[i] - window_s
        mask = (t[: i + 1] >= lo) & gate[: i + 1]
        window_m = m[: i + 1][mask]
        if window_m.size == 0:
            continue
        ratio = float(np.mean(window_m < threshold))
        if ratio >= ratio_th:
            return True, float(t[i])
    return False, None


def eval_hysteresis(grid: dict, enter_th: float, clear_th: float, enter_steps: int = 3) -> tuple[bool, float | None]:
    m, gate, t = grid["grid_m"], grid["grid_gate_open"], grid["grid_t"]
    consecutive = 0
    warned = False
    warn_time = None
    for i in range(len(m)):
        if not gate[i]:
            continue
        if not warned:
            if m[i] < enter_th:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= enter_steps:
                warned = True
                warn_time = float(t[i])
        # once warned, stays warned for this evaluation (we only care about
        # "did it ever warn", clear_th is for future chatter-suppression
        # analysis, not needed to determine ever_warned)
    return warned, warn_time


CANDIDATES = {
    "baseline_3step_m<0": lambda grid: eval_baseline_3step(grid, 0.0),
    "baseline_3step_m<2mm": lambda grid: eval_baseline_3step(grid, BOUNDARY_RISK_M),
    "duration_m<0_20ms": lambda grid: eval_duration(grid, 0.0, 0.020),
    "duration_m<0_30ms": lambda grid: eval_duration(grid, 0.0, 0.030),
    "duration_m<0_50ms": lambda grid: eval_duration(grid, 0.0, 0.050),
    "duration_m<2mm_20ms": lambda grid: eval_duration(grid, BOUNDARY_RISK_M, 0.020),
    "duration_m<2mm_30ms": lambda grid: eval_duration(grid, BOUNDARY_RISK_M, 0.030),
    "duration_m<2mm_50ms": lambda grid: eval_duration(grid, BOUNDARY_RISK_M, 0.050),
    "area_m<0_th1e-4": lambda grid: eval_area(grid, 0.0, 1e-4),
    "area_m<0_th3e-4": lambda grid: eval_area(grid, 0.0, 3e-4),
    "area_m<2mm_th2e-4": lambda grid: eval_area(grid, BOUNDARY_RISK_M, 2e-4),
    "rolling_m<0_50ms_r0.8": lambda grid: eval_rolling_ratio(grid, 0.0, 0.050, 0.8),
    "rolling_m<0_80ms_r0.8": lambda grid: eval_rolling_ratio(grid, 0.0, 0.080, 0.8),
    "rolling_m<0_100ms_r0.6": lambda grid: eval_rolling_ratio(grid, 0.0, 0.100, 0.6),
    "rolling_m<2mm_50ms_r0.8": lambda grid: eval_rolling_ratio(grid, BOUNDARY_RISK_M, 0.050, 0.8),
    "hysteresis_enter0_clear2mm": lambda grid: eval_hysteresis(grid, 0.0, BOUNDARY_RISK_M),
    "hysteresis_enter2mm_clear4mm": lambda grid: eval_hysteresis(grid, BOUNDARY_RISK_M, 0.004),
}


def evaluate_all_candidates(all_traces: dict) -> dict:
    results = {name: [] for name in CANDIDATES}
    for seed, tr in all_traces.items():
        if tr["actual_mode"] == "NO_CONTACT" or tr["contact_t"] is None:
            continue
        grid = resample_control_grid(tr["arrays"], tr["contact_t"])
        for name, fn in CANDIDATES.items():
            warned, warn_t = fn(grid)
            lead = (tr["contact_t"] - warn_t) if warn_t is not None else None
            actual_positive = tr["actual_mode"] == "TIP_EDGE"
            tp = warned and actual_positive
            fp = warned and not actual_positive
            fn_ = (not warned) and actual_positive
            tn = (not warned) and not actual_positive
            results[name].append({"seed": seed, "warned": warned, "lead": lead, "tp": tp, "fp": fp, "fn": fn_, "tn": tn})
    return results


def summarize_candidates(results: dict) -> dict:
    summary = {}
    for name, rows in results.items():
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        tn = sum(r["tn"] for r in rows)
        leads = [r["lead"] for r in rows if r["tp"] and r["lead"] is not None]
        fp_seeds = [r["seed"] for r in rows if r["fp"]]
        fn_seeds = [r["seed"] for r in rows if r["fn"]]
        summary[name] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else None,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else None,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else None,
            "fnr": fn / (fn + tp) if (fn + tp) > 0 else None,
            "lead_mean_s": float(np.mean(leads)) if leads else None,
            "lead_min_s": float(np.min(leads)) if leads else None,
            "lead_p95_s": float(np.percentile(leads, 95)) if leads else None,
            "fp_seeds": fp_seeds,
            "fn_seeds": fn_seeds,
        }
    return summary


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("PART 1+2: focused seed 96834 (FP) vs 84433 (TP) comparison")
    print("=" * 78)
    focus_data = {}
    for seed in FOCUS_SEEDS:
        ctx, rec = run_seed(seed)
        gt = ground_truth_mode(seed)
        contact_t, actual_mode = gt["contact_t"], gt["mode"]
        focus_data[seed] = {"samples": rec.samples, "contact_t": contact_t, "actual_mode": actual_mode}
        print(f"\n=== seed {seed} (actual={actual_mode}, contact_t={contact_t}) ===")
        p1 = part1_invalid_breakdown(seed, rec.samples, contact_t)
        for k, v in p1.items():
            print(f"  {k}: {v}")
        p2 = part2_gate_margin_history(seed, rec.samples, contact_t)
        for k, v in p2.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 78)
    print("PART 4: side-by-side comparison table")
    print("=" * 78)
    for seed in FOCUS_SEEDS:
        d = focus_data[seed]
        print(f"  seed={seed} actual_mode={d['actual_mode']} contact_t={d['contact_t']}")

    print("\n" + "=" * 78)
    print("PART 5+6: full 34-seed re-evaluation of candidate temporal conditions")
    print("=" * 78)
    all_seeds = list(STAGE1_SEEDS)
    rng = random.Random(EXTRA_SEED_RNG_SEED)
    tuning = set(STAGE1_SEEDS)
    extra = []
    while len(extra) < EXTRA_SEED_COUNT:
        cand = rng.randint(10000, 99999)
        if cand not in tuning and cand not in extra:
            extra.append(cand)
    all_seeds += extra

    all_traces = {}
    mode_counts = {"FACE_FACE": 0, "TIP_EDGE": 0, "OTHER_CONTACT": 0, "NO_CONTACT": 0}
    for seed in all_seeds:
        if seed in focus_data:
            samples, contact_t, actual_mode = focus_data[seed]["samples"], focus_data[seed]["contact_t"], focus_data[seed]["actual_mode"]
        else:
            ctx, rec = run_seed(seed)
            gt = ground_truth_mode(seed)
            samples, contact_t, actual_mode = rec.samples, gt["contact_t"], gt["mode"]
        mode_counts[actual_mode] = mode_counts.get(actual_mode, 0) + 1
        if actual_mode == "NO_CONTACT" or contact_t is None or not samples:
            all_traces[seed] = {"actual_mode": "NO_CONTACT", "contact_t": None, "arrays": None}
            continue
        arrays = build_seed_arrays(samples)
        all_traces[seed] = {"actual_mode": actual_mode, "contact_t": contact_t, "arrays": arrays}
        print(f"  processed seed={seed} mode={actual_mode}", flush=True)

    print(f"\n  mode_counts: {mode_counts}")

    results = evaluate_all_candidates(all_traces)
    summary = summarize_candidates(results)
    print("\n=== candidate condition summary ===")
    for name, s in summary.items():
        print(f"  {name}: TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']} "
              f"precision={s['precision']} recall={s['recall']} "
              f"lead_mean={s['lead_mean_s']} lead_min={s['lead_min_s']} lead_p95={s['lead_p95_s']} "
              f"fp_seeds={s['fp_seeds']} fn_seeds={s['fn_seeds']}")

    print("\n" + "=" * 78)
    print("PART 7: runtime (temporal-state update only, reusing existing N20/TTC)")
    print("=" * 78)
    sample_trace = all_traces[FOCUS_SEEDS[0]]
    grid = resample_control_grid(sample_trace["arrays"], sample_trace["contact_t"])
    for name, fn in list(CANDIDATES.items())[:3]:
        durations = []
        for _ in range(50):
            t0 = time.perf_counter()
            fn(grid)
            durations.append(time.perf_counter() - t0)
        d = np.array(durations)
        print(f"  {name}: mean={np.mean(d)*1000:.4f}ms p95={np.percentile(d,95)*1000:.4f}ms max={np.max(d)*1000:.4f}ms")


if __name__ == "__main__":
    main()
