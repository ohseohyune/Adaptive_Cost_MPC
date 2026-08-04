"""Read-only root-cause split for why relative-motion TTC (see
diagnose_relative_motion_ttc.py) helps seeds 1000/1005/1001 but makes seed
1017 worse.

For pad and box separately, compares a constant-velocity (CV) extrapolated
displacement over the *actual* remaining time against the actual measured
displacement:

    dx_CV     = v_x(t) * T_actual
    dx_actual = x(t_contact) - x(t)
    error     = dx_CV - dx_actual

Also checks whether using first-contact time (per side) vs. bilateral-
contact time as "T_actual" changes which error dominates
(CONTACT_EVENT_MISMATCH check).

Does not touch production control, predictor, trigger, or phase-gate code
-- reuses the existing baseline trace (no target_override_fn) plus the
same pad-surface finite-difference velocity/acceleration estimator already
used in diagnose_relative_motion_ttc.py.

Usage: python3 acmpc/diagnose_cv_displacement_error.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, Row, analyze, run_episode
from acmpc.diagnose_relative_motion_ttc import _pad_velocities

CHECKPOINT_LEADS_S = [0.30, 0.20, 0.10, 0.05]


def _finite_diff(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    for i in range(1, len(values)):
        dt = times[i] - times[i - 1]
        if dt > 1e-9:
            out[i] = (values[i] - values[i - 1]) / dt
    if len(values) > 1:
        out[0] = out[1]
    return out


def build(seed: int) -> tuple[dict, list[dict]]:
    result = run_episode(seed)
    out = analyze(result)
    dense: list[Row] = result["rec"].dense
    bilateral_t = out.get("bilateral_contact_time_s") or out.get("actual_contact_time_s")
    left_t = out.get("left_first_contact_time_s")
    right_t = out.get("right_first_contact_time_s")
    if bilateral_t is None or not dense:
        return out, []

    times = np.array([r.time_s for r in dense])
    v_pad_left, v_pad_right = _pad_velocities(dense)
    a_pad_left = _finite_diff(v_pad_left, times)
    a_pad_right = _finite_diff(v_pad_right, times)
    box_vx = np.array([r.prediction_velocity[0] for r in dense])
    box_ax = _finite_diff(box_vx, times)
    box_x = np.array([r.prediction_position[0] for r in dense])
    left_pad_x = np.array([r.left_pad_surface_x for r in dense])
    right_pad_x = np.array([r.right_pad_surface_x for r in dense])

    def _x_at(arr: np.ndarray, t_target: float) -> float:
        idx = int(np.argmin(np.abs(times - t_target)))
        return float(arr[idx])

    box_x_bilateral = _x_at(box_x, bilateral_t)
    left_pad_x_at_left_contact = _x_at(left_pad_x, left_t) if left_t is not None else None
    right_pad_x_at_right_contact = _x_at(right_pad_x, right_t) if right_t is not None else None
    box_x_at_left_contact = _x_at(box_x, left_t) if left_t is not None else None
    box_x_at_right_contact = _x_at(box_x, right_t) if right_t is not None else None

    series = []
    for i, r in enumerate(dense):
        t = r.time_s
        if t > bilateral_t:
            break
        T_bilateral = bilateral_t - t
        if T_bilateral <= 0:
            continue

        for side, pad_x_arr, pad_vx_arr, pad_ax_arr, side_contact_t, pad_x_at_side_contact, box_x_at_side_contact in (
            ("left", left_pad_x, v_pad_left, a_pad_left, left_t, left_pad_x_at_left_contact, box_x_at_left_contact),
            ("right", right_pad_x, v_pad_right, a_pad_right, right_t, right_pad_x_at_right_contact, box_x_at_right_contact),
        ):
            pad_x_t = float(pad_x_arr[i])
            pad_vx_t = float(pad_vx_arr[i])
            pad_ax_t = float(pad_ax_arr[i])
            box_x_t = float(box_x[i])
            box_vx_t = float(box_vx[i])
            box_ax_t = float(box_ax[i])

            # bilateral-referenced (primary)
            dx_pad_cv_bi = pad_vx_t * T_bilateral
            dx_pad_actual_bi = _x_at(pad_x_arr, bilateral_t) - pad_x_t
            dx_box_cv_bi = box_vx_t * T_bilateral
            dx_box_actual_bi = box_x_bilateral - box_x_t

            row = {
                "seed": seed,
                "side": side,
                "time_s": t,
                "lead_time_s": T_bilateral,
                "pad_x": pad_x_t,
                "pad_vx": pad_vx_t,
                "pad_ax": pad_ax_t,
                "box_x": box_x_t,
                "box_vx": box_vx_t,
                "box_ax": box_ax_t,
                "T_actual_bilateral_s": T_bilateral,
                "T_actual_first_contact_s": (side_contact_t - t) if side_contact_t is not None else None,
                "dx_pad_cv_bilateral": dx_pad_cv_bi,
                "dx_pad_actual_bilateral": dx_pad_actual_bi,
                "dx_pad_error_bilateral": dx_pad_cv_bi - dx_pad_actual_bi,
                "dx_box_cv_bilateral": dx_box_cv_bi,
                "dx_box_actual_bilateral": dx_box_actual_bi,
                "dx_box_error_bilateral": dx_box_cv_bi - dx_box_actual_bi,
                "dx_relative_error_bilateral": (dx_box_cv_bi - dx_pad_cv_bi) - (dx_box_actual_bi - dx_pad_actual_bi),
            }

            if side_contact_t is not None and side_contact_t > t:
                T_first = side_contact_t - t
                dx_pad_cv_fc = pad_vx_t * T_first
                dx_pad_actual_fc = pad_x_at_side_contact - pad_x_t
                dx_box_cv_fc = box_vx_t * T_first
                dx_box_actual_fc = box_x_at_side_contact - box_x_t
                row.update(
                    {
                        "dx_pad_cv_first_contact": dx_pad_cv_fc,
                        "dx_pad_actual_first_contact": dx_pad_actual_fc,
                        "dx_pad_error_first_contact": dx_pad_cv_fc - dx_pad_actual_fc,
                        "dx_box_cv_first_contact": dx_box_cv_fc,
                        "dx_box_actual_first_contact": dx_box_actual_fc,
                        "dx_box_error_first_contact": dx_box_cv_fc - dx_box_actual_fc,
                        "dx_relative_error_first_contact": (dx_box_cv_fc - dx_pad_cv_fc)
                        - (dx_box_actual_fc - dx_pad_actual_fc),
                    }
                )
            series.append(row)
    return out, series


def checkpoint_table(seed: int, series: list[dict]) -> list[dict]:
    left_series = [s for s in series if s["side"] == "left"]
    rows = []
    for lead in CHECKPOINT_LEADS_S:
        nearest = min(left_series, key=lambda s: abs(s["lead_time_s"] - lead))
        rows.append(nearest)
    return rows


def main() -> None:
    all_series = {}
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        out, series = build(seed)
        all_series[seed] = series
        rows = checkpoint_table(seed, series)
        for r in rows:
            print(f"  -- lead={r['lead_time_s']:.3f}s (target {CHECKPOINT_LEADS_S}) --")
            for k, v in r.items():
                print(f"    {k}: {v}")
        print(flush=True)

    print("=== cross-seed checkpoint comparison (left pad, bilateral-referenced) ===")
    header = f"  {'seed':>5} {'lead':>6} {'pad_vx':>8} {'pad_ax':>8} {'box_vx':>8} {'box_ax':>8} " \
             f"{'pad_err':>9} {'box_err':>9} {'rel_err':>9}"
    print(header)
    for seed in STAGE1_SEEDS:
        for r in checkpoint_table(seed, all_series[seed]):
            print(
                f"  {seed:>5} {r['lead_time_s']:6.3f} {r['pad_vx']:8.4f} {r['pad_ax']:8.4f} "
                f"{r['box_vx']:8.4f} {r['box_ax']:8.4f} "
                f"{r['dx_pad_error_bilateral']:9.4f} {r['dx_box_error_bilateral']:9.4f} "
                f"{r['dx_relative_error_bilateral']:9.4f}"
            )

    print("\n=== contact-event-mismatch check (bilateral- vs first-contact-referenced error, left pad) ===")
    for seed in STAGE1_SEEDS:
        for r in checkpoint_table(seed, all_series[seed]):
            if "dx_relative_error_first_contact" in r:
                print(
                    f"  seed={seed} lead={r['lead_time_s']:.3f}s "
                    f"rel_err_bilateral={r['dx_relative_error_bilateral']:.4f} "
                    f"rel_err_first_contact={r['dx_relative_error_first_contact']:.4f} "
                    f"T_bilateral={r['T_actual_bilateral_s']:.4f} T_first={r['T_actual_first_contact_s']:.4f}"
                )


if __name__ == "__main__":
    main()
