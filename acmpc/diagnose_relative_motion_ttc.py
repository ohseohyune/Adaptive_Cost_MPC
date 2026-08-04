"""Read-only comparison of the current static-plane TTC (assumes the pad
surface is fixed at whatever x it measured at each evaluation instant)
against a relative-motion TTC that also accounts for the pad's own x
velocity.

Static (production, unchanged):
    t_static = (x_pad,0 + h_box,x - x_box,0) / v_box,x
  (this is exactly _ttc_from_velocity(v_box_x, pad_plane_x + h_box_x,
  box_x) already used to compute position_remaining_ttc -- see
  main_acmpc_box_catch.py; not modified here.)

Relative-motion (diagnostic only, this file):
    x_box,0 + v_box,x*t + h_box,x = x_pad,0 + v_pad,x*t
    t_relative = (x_pad,0 - x_box,0 - h_box,x) / (v_box,x - v_pad,x)

Note on sign convention: the equation above was given with "+h_box,x" on
the box side. That is kept as specified in the closing-axis threshold
condition (box's own coordinate + half-width vs. pad position), but to make
t_static a true v_pad,x=0 special case of t_relative -- the natural sanity
check for "compare against the current static TTC" -- h_box,x is added on
the *pad* side of the numerator to match production's established static
formula: t_relative = (x_pad,0 + h_box,x - x_box,0) / (v_box,x - v_pad,x).
This reduces exactly to t_static when v_pad,x=0. Both conventions are
reported below (see "relative_ttc_literal_sign" for the as-given sign) so
the choice is visible rather than silently substituted.

Does not touch production control, the predictor, capture logic, or any
target computation -- purely reads the existing baseline (no
target_override_fn) trace already recorded by
diagnose_precontact_grasp_offset.py's on_step, plus a finite-difference
pad-surface velocity estimate.

Usage: python3 acmpc/diagnose_relative_motion_ttc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, Row, analyze, run_episode

MIN_ABS_DENOMINATOR_MPS = 0.02  # below this, relative TTC is flagged unstable rather than trusted
LEAD_TIME_CHECKPOINTS_S = [0.40, 0.30, 0.20, 0.10]


def _pad_velocities(dense: list[Row]) -> tuple[np.ndarray, np.ndarray]:
    n = len(dense)
    v_left = np.zeros(n)
    v_right = np.zeros(n)
    for i in range(1, n):
        dt = dense[i].time_s - dense[i - 1].time_s
        if dt > 1e-9:
            v_left[i] = (dense[i].left_pad_surface_x - dense[i - 1].left_pad_surface_x) / dt
            v_right[i] = (dense[i].right_pad_surface_x - dense[i - 1].right_pad_surface_x) / dt
    if n > 1:
        v_left[0] = v_left[1]
        v_right[0] = v_right[1]
    return v_left, v_right


def build_series(seed: int) -> tuple[dict, list[dict]]:
    result = run_episode(seed)
    out = analyze(result)
    dense = result["rec"].dense
    cfg = result["cfg"]
    h_box_x = float(cfg.domain_parameters.half_size[0]) if cfg.domain_parameters is not None else float("nan")

    contact_t = out.get("bilateral_contact_time_s") or out.get("actual_contact_time_s")
    left_contact_t = out.get("left_first_contact_time_s")
    right_contact_t = out.get("right_first_contact_time_s")
    if contact_t is None or not dense:
        return out, []

    v_pad_left, v_pad_right = _pad_velocities(dense)

    series = []
    for i, row in enumerate(dense):
        if row.time_s > contact_t:
            break
        if not row.position_remaining_ttc_valid:
            continue
        x_box0 = float(row.prediction_position[0])
        v_box = float(row.prediction_velocity[0])
        if abs(v_box) < 1e-9:
            continue

        for side, x_pad0, v_pad in (
            ("left", row.left_pad_surface_x, v_pad_left[i]),
            ("right", row.right_pad_surface_x, v_pad_right[i]),
        ):
            denom_relative = v_box - v_pad
            t_static = (x_pad0 + h_box_x - x_box0) / v_box
            unstable = abs(denom_relative) < MIN_ABS_DENOMINATOR_MPS
            t_relative = (x_pad0 + h_box_x - x_box0) / denom_relative if not unstable else float("nan")
            t_relative_literal = (
                (x_pad0 - x_box0 - h_box_x) / denom_relative if not unstable else float("nan")
            )

            side_contact_t = left_contact_t if side == "left" else right_contact_t
            actual_remaining = (side_contact_t - row.time_s) if side_contact_t is not None else (contact_t - row.time_s)

            series.append(
                {
                    "seed": seed,
                    "side": side,
                    "time_s": row.time_s,
                    "lead_time_s": contact_t - row.time_s,
                    "x_box0": x_box0,
                    "v_box_x": v_box,
                    "x_pad0": x_pad0,
                    "v_pad_x": float(v_pad),
                    "h_box_x": h_box_x,
                    "denominator_relative": denom_relative,
                    "denominator_unstable": unstable,
                    "t_static": t_static,
                    "t_relative": t_relative,
                    "t_relative_literal_sign": t_relative_literal,
                    "actual_remaining_time_s": actual_remaining,
                    "static_ttc_error_s": t_static - actual_remaining,
                    "relative_ttc_error_s": (t_relative - actual_remaining) if not unstable else float("nan"),
                    "static_x_error_predicted_m": v_box * (t_static - actual_remaining),
                    "relative_x_error_predicted_m": (
                        v_box * (t_relative - actual_remaining) if not unstable else float("nan")
                    ),
                }
            )
    return out, series


def summarize(seed: int, series: list[dict]) -> dict:
    if not series:
        return {"seed": seed, "error": "no series"}

    static_err = np.array([abs(s["static_ttc_error_s"]) for s in series])
    rel_valid = [s for s in series if not s["denominator_unstable"]]
    rel_err = np.array([abs(s["relative_ttc_error_s"]) for s in rel_valid]) if rel_valid else np.array([])

    out = {
        "seed": seed,
        "n_samples": len(series),
        "n_unstable_denominator": sum(1 for s in series if s["denominator_unstable"]),
        "static_ttc_error_mean_s": float(np.mean(static_err)),
        "static_ttc_error_p95_s": float(np.percentile(static_err, 95)),
        "static_ttc_error_final_s": abs(series[-1]["static_ttc_error_s"]),
        "relative_ttc_error_mean_s": float(np.mean(rel_err)) if rel_err.size else None,
        "relative_ttc_error_p95_s": float(np.percentile(rel_err, 95)) if rel_err.size else None,
        "relative_ttc_error_final_s": (
            abs(rel_valid[-1]["relative_ttc_error_s"]) if rel_valid else None
        ),
    }
    if out["relative_ttc_error_mean_s"] is not None and out["static_ttc_error_mean_s"] > 1e-9:
        out["mean_error_reduction_pct"] = 100.0 * (
            out["static_ttc_error_mean_s"] - out["relative_ttc_error_mean_s"]
        ) / out["static_ttc_error_mean_s"]
    else:
        out["mean_error_reduction_pct"] = None

    for lead in LEAD_TIME_CHECKPOINTS_S:
        nearest = min(series, key=lambda s: abs(s["lead_time_s"] - lead))
        out[f"static_ttc_error_at_lead_{lead:.1f}s"] = nearest["static_ttc_error_s"]
        out[f"relative_ttc_error_at_lead_{lead:.1f}s"] = (
            nearest["relative_ttc_error_s"] if not nearest["denominator_unstable"] else None
        )
    return out


def main() -> None:
    all_summaries = []
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        out, series = build_series(seed)
        summ = summarize(seed, series)
        all_summaries.append(summ)
        for k, v in summ.items():
            print(f"  {k}: {v}")

        print("  -- series sample (every 20ms, left pad) --")
        print(f"  {'lead_s':>8} {'x_box0':>8} {'v_box':>8} {'x_pad0':>8} {'v_pad':>8} "
              f"{'denom':>8} {'t_stat':>7} {'t_rel':>7} {'act_rem':>8} {'stat_err':>9} {'rel_err':>9}")
        last_t = None
        for s in [s for s in series if s["side"] == "left"]:
            if last_t is not None and s["time_s"] - last_t < 0.02:
                continue
            last_t = s["time_s"]
            rel_err = s["relative_ttc_error_s"]
            print(
                f"  {s['lead_time_s']:8.3f} {s['x_box0']:8.4f} {s['v_box_x']:8.4f} {s['x_pad0']:8.4f} "
                f"{s['v_pad_x']:8.4f} {s['denominator_relative']:8.4f} {s['t_static']:7.3f} "
                f"{(s['t_relative'] if not s['denominator_unstable'] else float('nan')):7.3f} "
                f"{s['actual_remaining_time_s']:8.3f} {s['static_ttc_error_s']:9.4f} "
                f"{(rel_err if rel_err==rel_err else -99):9.4f}"
            )
        print(flush=True)

    print("=== cross-seed summary ===")
    for s in all_summaries:
        print(f"  seed={s['seed']} static_mean={s.get('static_ttc_error_mean_s')} "
              f"relative_mean={s.get('relative_ttc_error_mean_s')} "
              f"reduction_pct={s.get('mean_error_reduction_pct')} "
              f"n_unstable={s.get('n_unstable_denominator')}/{s.get('n_samples')}")


if __name__ == "__main__":
    main()
