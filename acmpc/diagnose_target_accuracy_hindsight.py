"""Hindsight accuracy analysis: how close was the corrected (ballistic
prediction-based) target to the *actual* box position at the actual
bilateral-contact moment, at every pre-contact timestep -- and does any
online-available signal (predictor confidence, sample count, target
stability, workspace validity) actually track that accuracy?

This does not touch production control logic. It runs each seed's
*baseline* episode once (no override -- diagnose_precontact_grasp_offset.py
already records the full prediction_position/prediction_velocity/
position_remaining_ttc/confidence/sample-count series every substep), then
reconstructs e_target(t) = corrected_target(t) - p_actual in hindsight,
where p_actual is the box position at the real bilateral-contact time.

corrected_target(t) uses the exact same model as every prior shadow
experiment: prediction.position_after(position_remaining_ttc) at time t.

Usage: python3 acmpc/diagnose_target_accuracy_hindsight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, Row, analyze, run_episode
from acmpc.diagnose_mature_capture_shadow import _in_workspace

GRAVITY = np.array([0.0, 0.0, -9.81])
ERROR_THRESHOLDS_M = [0.01, 0.02, 0.03]
STABILITY_WINDOWS_MS = [20, 30, 50]

# lead times already characterized by diagnose_frozen_target_at_lead.py
# (results quoted from that run's log, reused here rather than rerun --
# box free flight before contact is unaffected by the hand-side override,
# so the *prediction* series in those runs matches baseline's own to
# floating-point precision; only the post-capture control differs).
FROZEN_AT_LEAD_RESULTS = {
    1000: {
        0.40: dict(dist=0.055, gravity=0.197, rotation=21.61, outcome="success"),
        0.30: dict(dist=0.077, gravity=0.251, rotation=62.09, outcome="timeout"),
        0.20: dict(dist=0.077, gravity=0.251, rotation=53.24, outcome="timeout"),
        0.10: dict(dist=0.067, gravity=0.248, rotation=36.00, outcome="success"),
    },
    1017: {
        0.40: dict(dist=0.060, gravity=0.087, rotation=28.66, outcome="success"),
        0.30: dict(dist=0.072, gravity=0.215, rotation=37.66, outcome="success"),
        0.20: dict(dist=0.063, gravity=0.130, rotation=30.70, outcome="success"),
        0.10: dict(dist=0.098, gravity=0.088, rotation=21.42, outcome="success"),
    },
    1005: {
        0.40: dict(dist=0.053, gravity=0.159, rotation=16.49, outcome="success"),
        0.30: dict(dist=0.077, gravity=0.253, rotation=75.10, outcome="timeout"),
        0.20: dict(dist=0.077, gravity=0.251, rotation=64.18, outcome="timeout"),
        0.10: dict(dist=0.074, gravity=0.252, rotation=56.53, outcome="timeout"),
    },
    1001: {
        0.40: dict(dist=None, gravity=None, rotation=None, outcome="workspace_miss"),
        0.30: dict(dist=0.068, gravity=0.280, rotation=68.96, outcome="timeout"),
        0.20: dict(dist=0.067, gravity=0.281, rotation=63.04, outcome="timeout"),
        0.10: dict(dist=0.068, gravity=0.280, rotation=54.28, outcome="timeout"),
    },
}


def corrected_target(row: Row) -> np.ndarray | None:
    if not row.position_remaining_ttc_valid:
        return None
    tau = max(0.0, float(row.position_remaining_ttc))
    return row.prediction_position + row.prediction_velocity * tau + 0.5 * GRAVITY * tau**2


def build_error_series(result: dict, out: dict) -> tuple[list[dict], np.ndarray | None]:
    dense = result["rec"].dense
    contact_t = out.get("bilateral_contact_time_s") or out.get("actual_contact_time_s")
    if contact_t is None or not dense:
        return [], None

    contact_row = min(dense, key=lambda r: abs(r.time_s - contact_t))
    p_actual = contact_row.box_com.copy()

    series = []
    target_history = []  # (time_s, target) for trailing-window spread
    cfg_half = result["cfg"].squeeze
    for row in dense:
        if row.time_s > contact_t:
            break
        target = corrected_target(row)
        if target is None:
            continue
        target_history.append((row.time_s, target))
        while target_history and row.time_s - target_history[0][0] > 0.06:
            target_history.pop(0)

        err_vec = target - p_actual
        spreads = {}
        for w_ms in STABILITY_WINDOWS_MS:
            w_s = w_ms / 1000.0
            pts = [p for t, p in target_history if row.time_s - t <= w_s]
            spreads[w_ms] = float(np.max(np.linalg.norm(np.array(pts) - target, axis=1))) if len(pts) >= 2 else None

        series.append(
            {
                "time_s": row.time_s,
                "lead_time_s": contact_t - row.time_s,
                "error_total_m": float(np.linalg.norm(err_vec)),
                "error_x_m": float(err_vec[0]),
                "error_y_m": float(err_vec[1]),
                "error_z_m": float(err_vec[2]),
                "confidence": row.prediction_confidence,
                "samples": row.predictor_samples,
                "position_remaining_ttc": row.position_remaining_ttc,
                "spread_20ms_m": spreads[20],
                "spread_30ms_m": spreads[30],
                "spread_50ms_m": spreads[50],
                "workspace_ok": _in_workspace(target, cfg_half),
            }
        )
    return series, p_actual


def summarize_seed(seed: int, series: list[dict]) -> dict:
    if not series:
        return {"seed": seed, "error": "no series"}

    errors = np.array([s["error_total_m"] for s in series])
    times = np.array([s["time_s"] for s in series])
    min_idx = int(np.argmin(errors))

    out = {
        "seed": seed,
        "min_error_m": float(errors[min_idx]),
        "min_error_time_s": float(times[min_idx]),
        "min_error_lead_time_s": float(series[min_idx]["lead_time_s"]),
    }

    for thresh in ERROR_THRESHOLDS_M:
        below = np.where(errors <= thresh)[0]
        if below.size == 0:
            out[f"first_time_below_{int(thresh*100)}cm_s"] = None
            out[f"first_lead_time_below_{int(thresh*100)}cm_s"] = None
            out[f"sustained_below_{int(thresh*100)}cm_until_contact"] = False
            continue
        first_idx = int(below[0])
        out[f"first_time_below_{int(thresh*100)}cm_s"] = float(times[first_idx])
        out[f"first_lead_time_below_{int(thresh*100)}cm_s"] = float(series[first_idx]["lead_time_s"])
        # sustained: from first_idx to the end, error never exceeds thresh again
        sustained = bool(np.all(errors[first_idx:] <= thresh))
        out[f"sustained_below_{int(thresh*100)}cm_until_contact"] = sustained
        if not sustained:
            # last contiguous run ending at contact
            last_break = first_idx
            for i in range(first_idx, len(errors)):
                if errors[i] > thresh:
                    last_break = i
            run_start_idx = last_break + 1 if last_break + 1 < len(errors) else None
            out[f"last_sustained_run_start_time_s_below_{int(thresh*100)}cm"] = (
                float(times[run_start_idx]) if run_start_idx is not None else None
            )

    # correlate confidence==1.0 / stability with actual error at that moment
    conf1_idx = next((i for i, s in enumerate(series) if s["confidence"] >= 1.0), None)
    out["error_at_confidence_1_m"] = float(errors[conf1_idx]) if conf1_idx is not None else None
    out["time_at_confidence_1_s"] = float(times[conf1_idx]) if conf1_idx is not None else None

    stable50_idx = next(
        (i for i, s in enumerate(series) if s["spread_50ms_m"] is not None and s["spread_50ms_m"] <= 0.01 and s["confidence"] >= 1.0),
        None,
    )
    out["error_at_confidence+stability50_m"] = float(errors[stable50_idx]) if stable50_idx is not None else None
    out["time_at_confidence+stability50_s"] = float(times[stable50_idx]) if stable50_idx is not None else None

    return out


def main() -> None:
    all_summaries = []
    all_series = {}
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        result = run_episode(seed)
        out = analyze(result)
        series, p_actual = build_error_series(result, out)
        all_series[seed] = series
        summ = summarize_seed(seed, series)
        all_summaries.append(summ)
        print(f"  actual bilateral contact box center (ground truth): {p_actual.tolist() if p_actual is not None else None}")
        print(f"  bilateral_contact_time_s: {out.get('bilateral_contact_time_s')}")
        for k, v in summ.items():
            print(f"  {k}: {v}")

        print("  -- error time series (every 20ms) --")
        print(f"  {'lead_s':>8} {'err_tot':>8} {'err_x':>8} {'err_y':>8} {'err_z':>8} "
              f"{'conf':>5} {'samp':>5} {'ttc':>6} {'sp50':>7} {'ws':>5}")
        last_t = None
        for s in series:
            if last_t is not None and s["time_s"] - last_t < 0.02:
                continue
            last_t = s["time_s"]
            sp50 = s["spread_50ms_m"]
            print(
                f"  {s['lead_time_s']:8.3f} {s['error_total_m']:8.4f} {s['error_x_m']:8.4f} "
                f"{s['error_y_m']:8.4f} {s['error_z_m']:8.4f} {s['confidence']:5.2f} {s['samples']:5d} "
                f"{s['position_remaining_ttc']:6.3f} {(sp50 if sp50 is not None else -1):7.4f} {str(s['workspace_ok']):>5}"
            )
        print(flush=True)

    print("=== lead-time-frozen capture connection (reusing diagnose_frozen_target_at_lead.py results) ===")
    for seed in STAGE1_SEEDS:
        series = all_series[seed]
        if not series:
            continue
        print(f"  -- seed {seed} --")
        for lead, res in FROZEN_AT_LEAD_RESULTS[seed].items():
            # nearest series sample to this lead time
            nearest = min(series, key=lambda s: abs(s["lead_time_s"] - lead))
            print(
                f"    lead={lead:.2f}s: capture_target_error_m={nearest['error_total_m']:.4f} "
                f"(conf={nearest['confidence']:.2f}, samples={nearest['samples']}) -> "
                f"face_dist={res['dist']}, gravity_torque_y={res['gravity']}, "
                f"rotation_deg={res['rotation']}, outcome={res['outcome']}"
            )

    print("\n=== overall classification ===")
    for summ in all_summaries:
        print(f"  seed={summ['seed']} min_error_m={summ.get('min_error_m')} "
              f"first_below_2cm_s={summ.get('first_lead_time_below_2cm_s')} "
              f"sustained_2cm={summ.get('sustained_below_2cm_until_contact')} "
              f"error_at_conf1={summ.get('error_at_confidence_1_m')} "
              f"error_at_conf+stability50={summ.get('error_at_confidence+stability50_m')}")


if __name__ == "__main__":
    main()
