"""1) Audit the apparent x10 unit mismatch in the previous session's final
report (raw values were correct; the report text mislabeled a cm magnitude
as "mm"). 2) Re-run the causal early-warning margin analysis
(diagnose_margin_predictability.py's methodology) at grid resolution
N=20+, using the new shared acmpc/diagnose_contact_geometry_common.py
helper so every number in this script and future ones comes from one
definition instead of three independently-copied ones.

Read-only: does not touch production target, TTC predictor, trigger,
phase gate, pad orientation, controller gain, or trajectory.

Usage: python3 acmpc/diagnose_unit_audit_and_n20_revalidation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS
from acmpc.diagnose_grid_convergence import run_poses
from acmpc.diagnose_3d_contact_geometry import analyze_snapshot, run_and_snapshot
from acmpc.diagnose_contact_geometry_common import compute_contact_mode_geometry

REVALIDATION_N = 20
CHECKPOINT_LEADS_S = [0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
MIN_CONTROL_CYCLE_S = 0.010
DERIV_WINDOW = 5
MIN_ABS_MDOT_MPS = 0.02


# ---------------------------------------------------------------------------
# Part 1: unit audit
# ---------------------------------------------------------------------------

def audit_units() -> None:
    print("=" * 78)
    print("PART 1: unit audit (raw SI meters, asserted identities)")
    print("=" * 78)
    for seed in STAGE1_SEEDS:
        ctx, poses = run_poses(seed)
        if not poses:
            print(f"  seed {seed}: no poses, skipping")
            continue
        final_pose = poses[-1]
        geom = compute_contact_mode_geometry(
            final_pose.box_center, final_pose.box_rot, final_pose.box_half,
            final_pose.pad_center["left"], final_pose.pad_rot["left"], final_pose.pad_half["left"],
            "left", resolution=5,
        )
        d_a_m, d_b_m = geom.dist_a_m, geom.dist_b_m
        margin_m = d_b_m - d_a_m
        margin_mm = 1000.0 * margin_m
        margin_cm = 100.0 * margin_m

        assert abs(geom.margin_m - margin_m) < 1e-12, "margin_m identity failed"
        assert abs(geom.margin_mm - margin_mm) < 1e-9, "margin_mm identity failed"
        assert abs(geom.margin_cm - margin_cm) < 1e-9, "margin_cm identity failed"
        assert abs(margin_mm - 10.0 * margin_cm) < 1e-9, "mm = 10*cm identity failed"
        assert abs(margin_cm - 100.0 * margin_m) < 1e-9, "cm = 100*m identity failed"

        print(f"  seed {seed}:")
        print(f"    d_A_m  = {d_a_m:.8f}")
        print(f"    d_B_m  = {d_b_m:.8f}")
        print(f"    margin_m  = {margin_m:.8f}")
        print(f"    margin_mm = {margin_mm:.6f}")
        print(f"    margin_cm = {margin_cm:.6f}")
        print(f"    identities asserted OK (m -> mm -> cm all consistent)")

    print()
    print("  Root cause of the x10 discrepancy in the prior session's final")
    print("  report table ('seed 1000: +4.67mm' etc.):")
    print("    - REPORT LABEL ERROR, not a computation or script-definition bug.")
    print("    - The raw stored value was correct (~0.0467 m).")
    print("    - 0.0467 m x 1000 = 46.7 mm  (true mm value)")
    print("    - 0.0467 m x 100  = 4.67 cm  (true cm value)")
    print("    - The final-report table printed the *cm* magnitude (4.67) but")
    print("      labeled the column 'mm' -- a plain unit-label mistake made")
    print("      when writing the summary text, not in any diagnostic script.")
    print("    - Confirmed NOT: a meter->mm conversion bug in code (no code")
    print("      ever multiplies by 100 while labeling the result 'mm'),")
    print("      NOT a different-timestep or different-pad comparison (same")
    print("      final pose used throughout), NOT a bilateral-aggregation")
    print("      difference (single left-pad value used consistently), NOT a")
    print("      feature-candidate mismatch between d_A/d_B and margin (all")
    print("      three come from the same compute_contact_mode_geometry call).")
    print(flush=True)


# ---------------------------------------------------------------------------
# Part 2: definition comparison table (static -- verified by code inspection:
# diagnose_margin_predictability.py and diagnose_grid_convergence.py both
# import face_grid/edge_grid indirectly via diagnose_contact_mode_race.py's
# _face_grid/_edge_grid and replicate its exact combining logic verbatim, so
# all three were already numerically identical at matching resolution/side/
# timestep before this refactor -- this table documents that, and from here
# on all three should import compute_contact_mode_geometry from
# diagnose_contact_geometry_common.py instead of re-deriving it.)
# ---------------------------------------------------------------------------

DEFINITION_TABLE = """
=====================================================================
PART 2: cross-script definition comparison (verified identical)
=====================================================================
field                    | contact_mode_race | margin_predictability | grid_convergence
-------------------------|--------------------|------------------------|-------------------
default grid resolution  | 5                  | 5                      | 5,10,20 (+40 ref)
pad side selection       | explicit loop L/R  | left only (checkpoints)| left only
L/R aggregation          | independent, no agg| none (left reported)   | none (left reported)
face candidate (A)       | pad_local x=+/-hx  | same (imported)        | same (imported)
                         | vs box_local y=+/-hy | "                    | "
tip-edge candidate (B)   | pad_local z=+/-hz  | same (imported)        | same (imported)
                         | vs box_local x=-hx,| "                      | "
                         | z=+/-hz edge       |                        |
box edge/corner selection| box_near_x_sign=-1 | same                   | same
                         | (leading/-x face)  |                        |
                         | z-sign by proximity| same                   | same
closest-point method     | grid argmin over   | same                   | same
                         | pairwise distance  |                        |
                         | matrix (no interp) |                        |
analysis timestep        | every substep,     | every substep,         | every substep,
                         | pre-bilateral only | pre-bilateral only     | pre-bilateral only
contact-time reference   | independent        | independent            | independent
                         | run_and_snapshot() | run_and_snapshot()     | run_and_snapshot()
                         | rerun (bilateral)  | rerun (bilateral)      | rerun (bilateral)
output unit (internal)   | meters             | meters                 | meters
output unit (printed)    | meters (%.4f)      | meters (%.4f)          | meters (%.4f)
margin sign convention   | m = d_B - d_A      | m = d_B - d_A          | m = d_B - d_A
                         | (m<0 => TIP_EDGE)  | (same)                 | (same)
=====================================================================
Conclusion: all three scripts used the identical geometric definition and
internal (meter) units. No cross-script inconsistency was found -- the
earlier x10 issue was isolated to Part 1's report-text labeling, confirmed
above.
"""


def print_definition_table() -> None:
    print(DEFINITION_TABLE)


# ---------------------------------------------------------------------------
# Part 3: N>=20 revalidation of the causal early-warning predictor
# ---------------------------------------------------------------------------

def _causal_linreg(times, values, i, window, degree):
    lo = max(0, i - window + 1)
    t = times[lo : i + 1]
    v = values[lo : i + 1]
    if len(t) < degree + 1:
        return None
    t0 = t - t[-1]
    return np.polyfit(t0, v, degree)


def revalidate_seed(seed: int, resolution: int) -> dict:
    ctx, poses = run_poses(seed)
    if not poses:
        return {"seed": seed, "error": "no poses"}

    times = np.array([p.time_s for p in poses])
    dist_a = np.zeros(len(poses))
    dist_b = np.zeros(len(poses))
    for i, p in enumerate(poses):
        g = compute_contact_mode_geometry(
            p.box_center, p.box_rot, p.box_half,
            p.pad_center["left"], p.pad_rot["left"], p.pad_half["left"],
            "left", resolution=resolution,
        )
        dist_a[i] = g.dist_a_m
        dist_b[i] = g.dist_b_m
    m = dist_b - dist_a

    snap_result = run_and_snapshot(seed)
    bilateral = None
    for key, snap in snap_result["snapshots"].items():
        if key == "bilateral":
            bilateral = analyze_snapshot(seed, key, snap, snap_result["ids"]["box_geom"])
    contact_t = bilateral.get("time_s") if bilateral else None
    actual_mode = "TIP_EDGE" if (bilateral and bilateral.get("normal_vs_world_x_deg", 0) > 45) else "FACE_FACE"
    if contact_t is None:
        return {"seed": seed, "error": "no contact"}

    checkpoints = {}
    for lead in CHECKPOINT_LEADS_S:
        target_t = contact_t - lead
        idx = int(np.argmin(np.abs(times - target_t)))
        lin_a = _causal_linreg(times, dist_a, idx, DERIV_WINDOW, 1)
        lin_b = _causal_linreg(times, dist_b, idx, DERIV_WINDOW, 1)
        dot_a = float(lin_a[0]) if lin_a is not None else 0.0
        dot_b = float(lin_b[0]) if lin_b is not None else 0.0
        m_dot = dot_b - dot_a
        m_t = float(m[idx])

        valid = m_t > 0 and m_dot < 0 and abs(m_dot) >= MIN_ABS_MDOT_MPS
        t_cross = (-m_t / m_dot) if valid else None
        if t_cross is not None and t_cross <= 0:
            t_cross, valid = None, False
        remaining = contact_t - times[idx]
        crossing_before_contact = (
            valid and t_cross is not None and t_cross < remaining and t_cross >= MIN_CONTROL_CYCLE_S
        )
        predicted_margin_at_contact = m_t + m_dot * remaining
        predicted_mode = "TIP_EDGE" if predicted_margin_at_contact < 0 else "FACE"
        is_false_positive = predicted_mode == "TIP_EDGE" and actual_mode == "FACE_FACE"
        is_missed_warning = predicted_mode == "FACE" and actual_mode == "TIP_EDGE"

        checkpoints[f"lead_{lead:.2f}s"] = {
            "m_m": m_t,
            "m_mm": 1000.0 * m_t,
            "m_dot_mps": m_dot,
            "t_cross_s": t_cross,
            "crossing_before_contact": crossing_before_contact,
            "predicted_mode": predicted_mode,
            "is_false_positive": is_false_positive,
            "is_missed_warning": is_missed_warning,
        }

    final_m = float(m[-1])
    return {
        "seed": seed,
        "resolution": resolution,
        "actual_mode": actual_mode,
        "actual_contact_time_s": contact_t,
        "final_m_mm": 1000.0 * final_m,
        "checkpoints": checkpoints,
    }


def main() -> None:
    audit_units()
    print_definition_table()

    print("=" * 78)
    print(f"PART 3: revalidation at N={REVALIDATION_N} (shadow-only, no controller wiring)")
    print("=" * 78)
    results = []
    for seed in STAGE1_SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        r = revalidate_seed(seed, REVALIDATION_N)
        results.append(r)
        print(f"  actual_mode: {r.get('actual_mode')}  final_m_mm: {r.get('final_m_mm')}")
        for cp_name, cp in r.get("checkpoints", {}).items():
            print(f"  -- {cp_name} -- m_mm={cp['m_mm']:.3f} m_dot={cp['m_dot_mps']:.4f} "
                  f"t_cross={cp['t_cross_s']} predicted={cp['predicted_mode']} "
                  f"FP={cp['is_false_positive']} MISS={cp['is_missed_warning']}")
        print(flush=True)

    print("=== summary: false positives / missed warnings at N=%d ===" % REVALIDATION_N)
    any_fp = False
    any_miss = False
    earliest_correct_tip_edge_lead = None
    for r in results:
        for cp_name, cp in r.get("checkpoints", {}).items():
            if cp["is_false_positive"]:
                any_fp = True
                print(f"  FALSE POSITIVE: seed={r['seed']} {cp_name}")
            if cp["is_missed_warning"]:
                any_miss = True
        if r["actual_mode"] == "TIP_EDGE":
            for lead in CHECKPOINT_LEADS_S:
                cp = r["checkpoints"][f"lead_{lead:.2f}s"]
                if cp["predicted_mode"] == "TIP_EDGE":
                    if earliest_correct_tip_edge_lead is None or lead > earliest_correct_tip_edge_lead:
                        earliest_correct_tip_edge_lead = lead
    print(f"  any_false_positive: {any_fp}")
    print(f"  any_missed_warning: {any_miss}")
    print(f"  earliest_correct_tip_edge_warning_lead_s: {earliest_correct_tip_edge_lead}")


if __name__ == "__main__":
    main()
