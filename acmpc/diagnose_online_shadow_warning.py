"""Can the refined contact-mode margin be used as a real online shadow
early-warning system -- i.e. using only a *predicted* contact TTC computed
from present-and-past information, instead of the oracle
(t_actual_contact - t) window used to validate it so far?

Single combined per-seed simulation pass (no separate reruns) recording,
every substep:
  - d_A(t), d_B(t), m(t) at N=20 (acmpc/diagnose_contact_geometry_common.py)
  - T_static: production's own position_remaining_ttc (read-only, reused
    exactly as computed -- never modified)
  - box/pad world pose (for the relative-motion and geometry-feature TTC
    estimators, and the rollout estimator)

Four online TTC estimators (A static-plane, B relative-motion, C geometry-
feature min(T_A,T_B), D short-horizon rollout) each independently gate a
3-consecutive-control-step contact-mode warning
(TIP_EDGE_EXPECTED / CONTACT_MODE_BOUNDARY_RISK / ROBUST_FACE_EXPECTED),
using only information available at or before the current step -- the
oracle window is used solely for offline scoring, never inside the gate.

Read-only: does not touch production target, TTC predictor, trigger,
phase gate, pad orientation, controller gain, or trajectory. Shadow
warnings are never wired to any controller.

Usage: python3 acmpc/diagnose_online_shadow_warning.py
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
from acmpc.diagnose_3d_contact_geometry import _box_support_point  # noqa: F401 (kept for parity/reuse)
from acmpc.diagnose_contact_geometry_common import compute_contact_mode_geometry
from acmpc.main_acmpc_box_catch import run_box_catch

CONTROL_STEP_S = 0.010
DERIV_WINDOW = 5
CONSECUTIVE_STEPS_REQUIRED = 3
ORACLE_WINDOW_S = (0.02, 0.15)
MIN_CLOSING_SPEED_MPS = 0.02
BOUNDARY_RISK_M = 0.002
ROLLOUT_HORIZON_S = 0.20
ROLLOUT_STEP_S = 0.010  # coarser than the 5ms suggestion, to keep this tractable this session
EXTRA_SEED_COUNT = 30
EXTRA_SEED_RNG_SEED = 13371337  # distinct from tuning {1000,1017,1005,1001} and the earlier 20-seed
# validation set (random.Random(424242), range 2000-9999)


# ---------------------------------------------------------------------------
# Combined single-pass recording
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    time_s: float
    box_center: np.ndarray
    box_rot: np.ndarray
    box_half: np.ndarray
    pad_center: dict
    pad_rot: dict
    pad_half: dict
    d_a: float
    d_b: float
    m: float
    t_static: float
    t_static_valid: bool


@dataclass
class Recording:
    samples: list = field(default_factory=list)


def run_combined(seed: int) -> tuple[dict, Recording]:
    seed_everything(seed)
    cfg = _build_stage1_config(seed)
    ids = {}
    rec = Recording()

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
        pad_center, pad_rot, pad_half = {}, {}, {}
        for side in ("left", "right"):
            pad_center[side] = data.geom_xpos[ids["pad_geom"][side]].copy()
            pad_rot[side] = data.geom_xmat[ids["pad_geom"][side]].reshape(3, 3).copy()
            pad_half[side] = model.geom_size[ids["pad_geom"][side]].copy()

        g = compute_contact_mode_geometry(box_center, box_rot, box_half, pad_center["left"], pad_rot["left"], pad_half["left"], "left", resolution=20)

        rec.samples.append(
            Sample(
                time_s=ctx["time_s"], box_center=box_center, box_rot=box_rot, box_half=box_half,
                pad_center=pad_center, pad_rot=pad_rot, pad_half=pad_half,
                d_a=g.dist_a_m, d_b=g.dist_b_m, m=g.margin_m,
                t_static=float(ctx["position_remaining_ttc"]), t_static_valid=bool(ctx["position_remaining_ttc_valid"]),
            )
        )

    summary = run_box_catch(cfg, step_callback=on_step)
    return {"summary": summary, "cfg": cfg}, rec


def ground_truth_mode(seed: int) -> dict:
    from acmpc.diagnose_3d_contact_geometry import analyze_snapshot, run_and_snapshot
    snap_result = run_and_snapshot(seed)
    bilateral = None
    for key, snap in snap_result["snapshots"].items():
        if key == "bilateral":
            bilateral = analyze_snapshot(seed, key, snap, snap_result["ids"]["box_geom"])
    if bilateral is None:
        return {"contact_t": None, "mode": "NO_CONTACT"}
    deg = bilateral.get("normal_vs_world_x_deg", 0.0)
    if deg < 20.0:
        mode = "FACE_FACE"
    elif deg > 70.0:
        mode = "TIP_EDGE"
    else:
        mode = "OTHER_CONTACT"
    return {"contact_t": bilateral.get("time_s"), "mode": mode}


# ---------------------------------------------------------------------------
# TTC estimators (causal only)
# ---------------------------------------------------------------------------

def _causal_slope(times: np.ndarray, values: np.ndarray, i: int, window: int) -> float:
    lo = max(0, i - window + 1)
    t, v = times[lo : i + 1], values[lo : i + 1]
    if len(t) < 2:
        return 0.0
    t0 = t - t[-1]
    return float(np.polyfit(t0, v, 1)[0])


def estimator_static(samples: list[Sample], i: int) -> tuple[float | None, bool]:
    s = samples[i]
    return (s.t_static, True) if s.t_static_valid else (None, False)


def estimator_relative(samples: list[Sample], times: np.ndarray, pad_x_left: np.ndarray,
                        box_x: np.ndarray, box_vx: np.ndarray, pad_vx: np.ndarray, i: int) -> tuple[float | None, bool]:
    gap = pad_x_left[i] - box_x[i]
    closing = -(box_vx[i] - pad_vx[i])  # positive when box/pad are closing
    if closing <= MIN_CLOSING_SPEED_MPS:
        return None, False
    t = gap / closing if closing != 0 else None
    if t is None or t <= 0 or not np.isfinite(t):
        return None, False
    return float(t), True


def estimator_geometry(samples: list[Sample], times: np.ndarray, d_a: np.ndarray, d_b: np.ndarray, i: int) -> tuple[float | None, bool]:
    dot_a = _causal_slope(times, d_a, i, DERIV_WINDOW)
    dot_b = _causal_slope(times, d_b, i, DERIV_WINDOW)
    candidates = []
    if d_a[i] > 0 and dot_a < -MIN_CLOSING_SPEED_MPS:
        t_a = -d_a[i] / dot_a
        if t_a > 0 and np.isfinite(t_a):
            candidates.append(t_a)
    if d_b[i] > 0 and dot_b < -MIN_CLOSING_SPEED_MPS:
        t_b = -d_b[i] / dot_b
        if t_b > 0 and np.isfinite(t_b):
            candidates.append(t_b)
    if not candidates:
        return None, False
    return float(min(candidates)), True


def estimator_rollout(samples: list[Sample], times: np.ndarray, i: int,
                       box_vel: np.ndarray, pad_vel_left: np.ndarray, pad_vel_right: np.ndarray,
                       contact_tolerance_m: float = 0.001) -> tuple[float | None, bool]:
    s = samples[i]
    n_steps = int(ROLLOUT_HORIZON_S / ROLLOUT_STEP_S)
    for k in range(1, n_steps + 1):
        tau = k * ROLLOUT_STEP_S
        box_center_f = s.box_center + box_vel * tau
        pad_center_f = {
            "left": s.pad_center["left"] + pad_vel_left * tau,
            "right": s.pad_center["right"] + pad_vel_right * tau,
        }
        g = compute_contact_mode_geometry(
            box_center_f, s.box_rot, s.box_half,
            pad_center_f["left"], s.pad_rot["left"], s.pad_half["left"], "left", resolution=20,
        )
        if min(g.dist_a_m, g.dist_b_m) <= contact_tolerance_m:
            return float(tau), True
    return None, False


# ---------------------------------------------------------------------------
# Online gate + shadow warning pipeline
# ---------------------------------------------------------------------------

def classify_margin(m: float) -> str:
    if m < 0.0:
        return "TIP_EDGE_EXPECTED"
    if m < BOUNDARY_RISK_M:
        return "CONTACT_MODE_BOUNDARY_RISK"
    return "ROBUST_FACE_EXPECTED"


def run_gate_pipeline(times: np.ndarray, m: np.ndarray, ttc: np.ndarray, ttc_valid: np.ndarray,
                       contact_t: float, gate_mode: str, warn_state_target: str) -> dict:
    n = len(times)
    consecutive = 0
    last_state = None
    gate_open = np.zeros(n, dtype=bool)
    warned = np.zeros(n, dtype=bool)
    chatter = 0
    prev_gate = False

    for i in range(n):
        g_open = bool(ttc_valid[i] and ORACLE_WINDOW_S[0] <= ttc[i] <= ORACLE_WINDOW_S[1])
        gate_open[i] = g_open
        if g_open != prev_gate:
            chatter += 1
        prev_gate = g_open

        if g_open:
            state = classify_margin(m[i])
            if state == last_state:
                consecutive += 1
            else:
                consecutive = 1
                last_state = state
            if consecutive >= CONSECUTIVE_STEPS_REQUIRED and state == warn_state_target:
                warned[i] = True
        else:
            if gate_mode == "reset":
                consecutive = 0
                last_state = None
            # pause: leave consecutive/last_state untouched

    warned_idx = np.where(warned)[0]
    ever_warned = warned_idx.size > 0
    first_lead = float(contact_t - times[warned_idx[0]]) if ever_warned else None
    gate_open_times = times[gate_open]
    return {
        "ever_warned": ever_warned,
        "first_warn_lead_s": first_lead,
        "gate_chatter_count": chatter,
        "gate_open_fraction": float(np.mean(gate_open)),
        "gate_first_open_s": float(gate_open_times[0]) if gate_open_times.size else None,
        "gate_last_open_s": float(gate_open_times[-1]) if gate_open_times.size else None,
    }


# ---------------------------------------------------------------------------
# Core 4-seed full comparison
# ---------------------------------------------------------------------------

def analyze_core_seed(seed: int) -> dict:
    ctx, rec = run_combined(seed)
    samples = rec.samples
    gt = ground_truth_mode(seed)
    contact_t, actual_mode = gt["contact_t"], gt["mode"]
    if contact_t is None or not samples:
        return {"seed": seed, "error": "no contact or no samples", "actual_mode": actual_mode}

    times = np.array([s.time_s for s in samples])
    d_a = np.array([s.d_a for s in samples])
    d_b = np.array([s.d_b for s in samples])
    m = d_b - d_a
    box_x = np.array([s.box_center[0] for s in samples])
    pad_x_left = np.array([s.pad_center["left"][0] for s in samples])

    box_vx = np.zeros(len(samples))
    pad_vx = np.zeros(len(samples))
    box_vel_full = np.zeros((len(samples), 3))
    pad_vel_left_full = np.zeros((len(samples), 3))
    pad_vel_right_full = np.zeros((len(samples), 3))
    for i in range(1, len(samples)):
        dt = times[i] - times[i - 1]
        if dt > 1e-9:
            box_vx[i] = (box_x[i] - box_x[i - 1]) / dt
            pad_vx[i] = (pad_x_left[i] - pad_x_left[i - 1]) / dt
            box_vel_full[i] = (samples[i].box_center - samples[i - 1].box_center) / dt
            pad_vel_left_full[i] = (samples[i].pad_center["left"] - samples[i - 1].pad_center["left"]) / dt
            pad_vel_right_full[i] = (samples[i].pad_center["right"] - samples[i - 1].pad_center["right"]) / dt
    box_vx[0], pad_vx[0] = box_vx[1], pad_vx[1]
    box_vel_full[0], pad_vel_left_full[0], pad_vel_right_full[0] = box_vel_full[1], pad_vel_left_full[1], pad_vel_right_full[1]

    n = len(samples)
    ttc_static = np.full(n, np.nan)
    ttc_static_valid = np.zeros(n, dtype=bool)
    ttc_relative = np.full(n, np.nan)
    ttc_relative_valid = np.zeros(n, dtype=bool)
    ttc_geom = np.full(n, np.nan)
    ttc_geom_valid = np.zeros(n, dtype=bool)

    for i in range(n):
        v, ok = estimator_static(samples, i)
        if ok:
            ttc_static[i], ttc_static_valid[i] = v, True
        v, ok = estimator_relative(samples, times, pad_x_left, box_x, box_vx, pad_vx, i)
        if ok:
            ttc_relative[i], ttc_relative_valid[i] = v, True
        v, ok = estimator_geometry(samples, times, d_a, d_b, i)
        if ok:
            ttc_geom[i], ttc_geom_valid[i] = v, True

    # rollout TTC only every 4th sample (~4-6ms substep spacing -> ~20ms), to
    # keep this tractable within the session; interpolate elsewhere via
    # forward-fill so the gate logic still sees a value every substep.
    ttc_rollout = np.full(n, np.nan)
    ttc_rollout_valid = np.zeros(n, dtype=bool)
    rollout_runtimes = []
    for i in range(0, n, 4):
        t0 = time.perf_counter()
        v, ok = estimator_rollout(samples, times, i, box_vel_full[i], pad_vel_left_full[i], pad_vel_right_full[i])
        rollout_runtimes.append(time.perf_counter() - t0)
        if ok:
            ttc_rollout[i], ttc_rollout_valid[i] = v, True
    # forward-fill
    last_v, last_ok = np.nan, False
    for i in range(n):
        if i % 4 == 0:
            last_v, last_ok = ttc_rollout[i], ttc_rollout_valid[i]
        else:
            ttc_rollout[i], ttc_rollout_valid[i] = last_v, last_ok

    # oracle gate (ground truth) for reference
    oracle_ttc = contact_t - times
    oracle_gate_open = (oracle_ttc >= ORACLE_WINDOW_S[0]) & (oracle_ttc <= ORACLE_WINDOW_S[1])

    estimators = {
        "static": (ttc_static, ttc_static_valid),
        "relative": (ttc_relative, ttc_relative_valid),
        "geometry": (ttc_geom, ttc_geom_valid),
        "rollout": (ttc_rollout, ttc_rollout_valid),
        "oracle": (oracle_ttc, np.ones(n, dtype=bool)),
    }

    results = {}
    for est_name, (ttc, valid) in estimators.items():
        for gate_mode in ("reset", "pause"):
            for target, thr_name in (("TIP_EDGE_EXPECTED", "strict_tip"), ("CONTACT_MODE_BOUNDARY_RISK", "boundary_risk")):
                key = f"{est_name}_{gate_mode}_{thr_name}"
                results[key] = run_gate_pipeline(times, m, ttc, valid, contact_t, gate_mode, target)

    # TTC accuracy vs oracle (mean/p95 abs error, only where both defined)
    ttc_errors = {}
    for est_name in ("static", "relative", "geometry", "rollout"):
        ttc, valid = estimators[est_name]
        mask = valid & np.isfinite(oracle_ttc)
        if np.any(mask):
            err = np.abs(ttc[mask] - oracle_ttc[mask])
            ttc_errors[est_name] = {"mean_abs_error_s": float(np.mean(err)), "p95_abs_error_s": float(np.percentile(err, 95))}
        else:
            ttc_errors[est_name] = {"mean_abs_error_s": None, "p95_abs_error_s": None}

    return {
        "seed": seed,
        "actual_mode": actual_mode,
        "contact_t": contact_t,
        "results": results,
        "ttc_errors": ttc_errors,
        "rollout_runtime_mean_s": float(np.mean(rollout_runtimes)) if rollout_runtimes else None,
        "rollout_runtime_p95_s": float(np.percentile(rollout_runtimes, 95)) if rollout_runtimes else None,
        "rollout_runtime_max_s": float(np.max(rollout_runtimes)) if rollout_runtimes else None,
    }


# ---------------------------------------------------------------------------
# Runtime measurement (isolated)
# ---------------------------------------------------------------------------

def measure_runtimes(seed: int, n_calls: int = 200) -> dict:
    ctx, rec = run_combined(seed)
    samples = rec.samples
    if len(samples) < 10:
        return {}
    idxs = np.linspace(5, len(samples) - 1, min(n_calls, len(samples) - 5)).astype(int)

    geom_times = []
    for i in idxs:
        s = samples[i]
        t0 = time.perf_counter()
        compute_contact_mode_geometry(s.box_center, s.box_rot, s.box_half, s.pad_center["left"], s.pad_rot["left"], s.pad_half["left"], "left", resolution=20)
        geom_times.append(time.perf_counter() - t0)

    times = np.array([s.time_s for s in samples])
    d_a = np.array([s.d_a for s in samples])
    d_b = np.array([s.d_b for s in samples])
    deriv_times = []
    for i in idxs:
        t0 = time.perf_counter()
        estimator_geometry(samples, times, d_a, d_b, int(i))
        deriv_times.append(time.perf_counter() - t0)

    full_pipeline_times = []
    for i in idxs:
        t0 = time.perf_counter()
        s = samples[i]
        compute_contact_mode_geometry(s.box_center, s.box_rot, s.box_half, s.pad_center["left"], s.pad_rot["left"], s.pad_half["left"], "left", resolution=20)
        estimator_geometry(samples, times, d_a, d_b, int(i))
        classify_margin(d_b[i] - d_a[i])
        full_pipeline_times.append(time.perf_counter() - t0)

    def stats(arr):
        a = np.array(arr)
        return {"mean_s": float(np.mean(a)), "p95_s": float(np.percentile(a, 95)), "max_s": float(np.max(a))}

    return {
        "geometry_n20": stats(geom_times),
        "derivative_ttc": stats(deriv_times),
        "full_pipeline_no_rollout": stats(full_pipeline_times),
    }


# ---------------------------------------------------------------------------
# Extra-seed generalization (geometry estimator only -- cheapest reliable one)
# ---------------------------------------------------------------------------

def analyze_extra_seed(seed: int) -> dict:
    ctx, rec = run_combined(seed)
    samples = rec.samples
    gt = ground_truth_mode(seed)
    contact_t, actual_mode = gt["contact_t"], gt["mode"]
    if actual_mode == "NO_CONTACT" or contact_t is None or not samples:
        return {"seed": seed, "actual_mode": "NO_CONTACT", "warned": False}

    times = np.array([s.time_s for s in samples])
    d_a = np.array([s.d_a for s in samples])
    d_b = np.array([s.d_b for s in samples])
    m = d_b - d_a
    n = len(samples)
    ttc = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        v, ok = estimator_geometry(samples, times, d_a, d_b, i)
        if ok:
            ttc[i], valid[i] = v, True

    res_strict = run_gate_pipeline(times, m, ttc, valid, contact_t, "pause", "TIP_EDGE_EXPECTED")
    predicted_tip_edge = res_strict["ever_warned"]
    ttc_invalid_fraction = float(1.0 - np.mean(valid))

    return {
        "seed": seed,
        "actual_mode": actual_mode,
        "predicted_tip_edge": predicted_tip_edge,
        "warn_lead_s": res_strict["first_warn_lead_s"],
        "gate_open_fraction": res_strict["gate_open_fraction"],
        "ttc_invalid_fraction": ttc_invalid_fraction,
    }


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("PART A: core 4-seed full estimator comparison")
    print("=" * 78)
    core_results = {}
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        r = analyze_core_seed(seed)
        core_results[seed] = r
        if "error" in r:
            print(f"  {r}")
            continue
        print(f"  actual_mode={r['actual_mode']} contact_t={r['contact_t']:.4f}")
        print(f"  ttc_errors: {r['ttc_errors']}")
        print(f"  rollout_runtime mean/p95/max (s): {r['rollout_runtime_mean_s']} / "
              f"{r['rollout_runtime_p95_s']} / {r['rollout_runtime_max_s']}")
        for key, res in r["results"].items():
            print(f"  {key}: warned={res['ever_warned']} lead={res['first_warn_lead_s']} "
                  f"chatter={res['gate_chatter_count']} gate_open_frac={res['gate_open_fraction']:.3f} "
                  f"gate=[{res['gate_first_open_s']},{res['gate_last_open_s']}]")
        print(flush=True)

    print("=" * 78)
    print("PART B: isolated runtime measurement")
    print("=" * 78)
    runtimes = measure_runtimes(1000)
    for k, v in runtimes.items():
        print(f"  {k}: {v}")
    period_s = CONTROL_STEP_S
    p95_no_rollout = runtimes.get("full_pipeline_no_rollout", {}).get("p95_s")
    if p95_no_rollout is not None:
        if p95_no_rollout < 0.005:
            verdict = "REALTIME_SAFE"
        elif p95_no_rollout < 0.010:
            verdict = "REALTIME_MARGINAL"
        else:
            verdict = "NOT_REALTIME_SAFE"
        print(f"  no-rollout pipeline vs {period_s*1000:.0f}ms period: {verdict}")

    print("=" * 78)
    print(f"PART C: {EXTRA_SEED_COUNT} extra-seed generalization (geometry estimator only)")
    print("=" * 78)
    rng = random.Random(EXTRA_SEED_RNG_SEED)
    tuning_seeds = set(STAGE1_SEEDS)
    extra_seeds = []
    while len(extra_seeds) < EXTRA_SEED_COUNT:
        cand = rng.randint(10000, 99999)
        if cand not in tuning_seeds and cand not in extra_seeds:
            extra_seeds.append(cand)
    print(f"  extra seeds: {extra_seeds}")

    extra_results = []
    for seed in extra_seeds:
        r = analyze_extra_seed(seed)
        extra_results.append(r)
        print(f"  seed={seed} actual={r['actual_mode']} predicted_tip_edge={r.get('predicted_tip_edge')} "
              f"lead={r.get('warn_lead_s')} ttc_invalid_frac={r.get('ttc_invalid_fraction')}", flush=True)

    counts = {"FACE_FACE": 0, "TIP_EDGE": 0, "OTHER_CONTACT": 0, "NO_CONTACT": 0}
    tp = fp = fn = tn = 0
    leads = []
    for r in extra_results:
        counts[r["actual_mode"]] = counts.get(r["actual_mode"], 0) + 1
        if r["actual_mode"] == "NO_CONTACT":
            continue
        predicted = r.get("predicted_tip_edge", False)
        actual_positive = r["actual_mode"] == "TIP_EDGE"
        if predicted and actual_positive:
            tp += 1
            if r.get("warn_lead_s") is not None:
                leads.append(r["warn_lead_s"])
        elif predicted and not actual_positive:
            fp += 1
        elif not predicted and actual_positive:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    print("\n=== extra-seed summary ===")
    print(f"  mode counts: {counts}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  precision={precision} recall={recall}")
    if leads:
        print(f"  lead time mean/min/p95: {np.mean(leads)}/{np.min(leads)}/{np.percentile(leads, 95)}")
    else:
        print("  no true-positive leads recorded (insufficient TIP_EDGE positives)")


if __name__ == "__main__":
    main()
