"""Independent holdout validation of the FROZEN final shadow-warning
candidate, with threshold/window/ratio/resolution completely fixed (no
retuning based on holdout results, per the task constraint).

Frozen candidate (exactly as specified, implemented literally):
    gate_open = static_ttc_valid and 0.02 <= position_remaining_ttc <= 0.15
    if not gate_open: risk_buffer.clear()
    else: risk_buffer.append(margin_m < 0.002); keep only last 5
    warning = gate_open and len(risk_buffer) == 5 and sum(risk_buffer) >= 4

geometry: N=20 (diagnose_contact_geometry_common.py, unchanged)
static TTC: production's own position_remaining_ttc, read-only

Read-only / shadow-only: does not touch production target, TTC predictor,
trigger, phase gate, pad orientation, controller gain, or trajectory.

Usage: python3 acmpc/diagnose_holdout_validation.py
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
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
ORACLE_WINDOW_S = (0.02, 0.15)  # gate bounds, fixed -- ground-truth scoring only
MARGIN_THRESHOLD_M = 0.002
ROLLING_LEN = 5
ROLLING_VOTES_REQUIRED = 4

HOLDOUT_TARGET_SEEDS = 150
HOLDOUT_MIN_SEEDS = 100
HOLDOUT_MIN_TIP_EDGE = 30
HOLDOUT_MIN_FACE_FACE = 50
HOLDOUT_RNG_SEED = 271828182  # distinct from every prior RNG base this session
HOLDOUT_SEED_RANGE = (100_000, 999_999)  # disjoint from every prior seed range used this session


# ---------------------------------------------------------------------------
# Development-set bookkeeping (exact reproduction of every seed already used)
# ---------------------------------------------------------------------------

def development_seeds() -> set[int]:
    seeds = set(STAGE1_SEEDS)
    rng = random.Random(EXTRA_SEED_RNG_SEED)
    extra = []
    while len(extra) < EXTRA_SEED_COUNT:
        c = rng.randint(10000, 99999)
        if c not in seeds and c not in extra:
            extra.append(c)
    seeds.update(extra)
    # explicit named seeds from the task (must already be inside the above set)
    named = {1000, 1017, 1005, 1001, 96834, 84433, 11622, 87052}
    missing = named - seeds
    assert not missing, f"named development seeds not reproduced: {missing}"
    return seeds


def generate_holdout_seeds(n: int, exclude: set[int]) -> list[int]:
    rng = random.Random(HOLDOUT_RNG_SEED)
    lo, hi = HOLDOUT_SEED_RANGE
    out = []
    while len(out) < n:
        c = rng.randint(lo, hi)
        if c not in exclude and c not in out:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Single combined recording (per-step geometry + static TTC + IC snapshot)
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    time_s: float
    d_a: float
    d_b: float
    t_static: float
    t_static_valid: bool
    box_vx: float


@dataclass
class InitialConditions:
    box_half_x: float
    box_half_y: float
    box_half_z: float
    box_mass: float
    box_pos0: np.ndarray
    box_vel0: np.ndarray
    box_angvel0: np.ndarray
    pad_pos0_left: np.ndarray
    pad_pos0_right: np.ndarray


@dataclass
class Recording:
    samples: list = field(default_factory=list)
    ic: InitialConditions | None = None
    loop_iter_times_s: list = field(default_factory=list)


def run_seed(seed: int, measure_loop_time: bool = False) -> tuple[dict, Recording]:
    seed_everything(seed)
    cfg = _build_stage1_config(seed)
    ids = {}
    rec = Recording()
    last_wall = [None]

    def on_step(ctx: dict) -> None:
        model, data = ctx["model"], ctx["data"]
        if not ids:
            ids["box_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
            ids["pad_geom"] = {
                s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_catch_pad") for s in ("left", "right")
            }
            ids["box_body"] = ctx["box_body_id"]

        if measure_loop_time:
            now = time.perf_counter()
            if last_wall[0] is not None:
                rec.loop_iter_times_s.append(now - last_wall[0])
            last_wall[0] = now

        contact = ctx["contact"]
        if contact.left.active and contact.right.active:
            return

        box_center = data.geom_xpos[ids["box_geom"]].copy()
        box_rot = data.geom_xmat[ids["box_geom"]].reshape(3, 3).copy()
        box_half = model.geom_size[ids["box_geom"]].copy()
        pad_center_left = data.geom_xpos[ids["pad_geom"]["left"]].copy()
        pad_rot_left = data.geom_xmat[ids["pad_geom"]["left"]].reshape(3, 3).copy()
        pad_half_left = model.geom_size[ids["pad_geom"]["left"]].copy()

        if rec.ic is None:
            box_dof = ctx["box_dof_address"]
            rec.ic = InitialConditions(
                box_half_x=float(box_half[0]), box_half_y=float(box_half[1]), box_half_z=float(box_half[2]),
                box_mass=float(model.body_mass[ids["box_body"]]),
                box_pos0=box_center.copy(),
                box_vel0=data.qvel[box_dof : box_dof + 3].copy(),
                box_angvel0=data.qvel[box_dof + 3 : box_dof + 6].copy(),
                pad_pos0_left=pad_center_left.copy(),
                pad_pos0_right=data.geom_xpos[ids["pad_geom"]["right"]].copy(),
            )

        g = compute_contact_mode_geometry(
            box_center, box_rot, box_half, pad_center_left, pad_rot_left, pad_half_left, "left", resolution=20
        )

        rec.samples.append(
            Sample(
                time_s=ctx["time_s"], d_a=g.dist_a_m, d_b=g.dist_b_m,
                t_static=float(ctx["position_remaining_ttc"]), t_static_valid=bool(ctx["position_remaining_ttc_valid"]),
                box_vx=float(ctx["prediction_velocity"][0]),
            )
        )

    summary = run_box_catch(cfg, step_callback=on_step)
    return {"summary": summary, "cfg": cfg}, rec


# ---------------------------------------------------------------------------
# Frozen warning pipeline (literal implementation of the pseudocode)
# ---------------------------------------------------------------------------

def run_frozen_pipeline(samples: list[Sample]) -> dict:
    risk_buffer: deque = deque()
    gate_intervals = 0
    prev_gate = False
    gate_total_open_s = 0.0
    chatter = 0
    warning_first_time = None
    warning_intervals = 0
    warning_total_s = 0.0
    prev_warning = False
    dt_est = None
    times = [s.time_s for s in samples]
    if len(times) > 1:
        dt_est = float(np.median(np.diff(times)))
    else:
        dt_est = CONTROL_STEP_S

    for s in samples:
        gate_open = bool(s.t_static_valid and ORACLE_WINDOW_S[0] <= s.t_static <= ORACLE_WINDOW_S[1])
        if gate_open != prev_gate:
            chatter += 1
            if gate_open:
                gate_intervals += 1
        if gate_open:
            gate_total_open_s += dt_est
        prev_gate = gate_open

        if not gate_open:
            risk_buffer.clear()
        else:
            risk_buffer.append(s.d_b - s.d_a < MARGIN_THRESHOLD_M)
            while len(risk_buffer) > ROLLING_LEN:
                risk_buffer.popleft()

        warning = bool(gate_open and len(risk_buffer) == ROLLING_LEN and sum(risk_buffer) >= ROLLING_VOTES_REQUIRED)
        if warning and warning_first_time is None:
            warning_first_time = s.time_s
        if warning != prev_warning:
            if warning:
                warning_intervals += 1
        if warning:
            warning_total_s += dt_est
        prev_warning = warning

    return {
        "warning_first_time_s": warning_first_time,
        "gate_intervals": gate_intervals,
        "gate_total_open_s": gate_total_open_s,
        "gate_chatter": chatter,
        "warning_intervals": warning_intervals,
        "warning_total_s": warning_total_s,
    }


BASELINE_A_M0 = 0.0
BASELINE_B_M2MM = MARGIN_THRESHOLD_M


def run_baseline_3step(samples: list[Sample], threshold: float) -> dict:
    consecutive = 0
    warning_first_time = None
    for s in samples:
        gate_open = bool(s.t_static_valid and ORACLE_WINDOW_S[0] <= s.t_static <= ORACLE_WINDOW_S[1])
        m = s.d_b - s.d_a
        if gate_open and m < threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 3 and warning_first_time is None:
            warning_first_time = s.time_s
    return {"warning_first_time_s": warning_first_time}


# ---------------------------------------------------------------------------
# Ground-truth-based scoring
# ---------------------------------------------------------------------------

def score_seed(warning_first_time: float | None, contact_t: float | None, actual_mode: str) -> str:
    """Returns one of TP, FP, FN, TN, LATE_TP, EXCLUDED."""
    if actual_mode in ("OTHER_CONTACT", "NO_CONTACT"):
        return "EXCLUDED"
    actual_positive = actual_mode == "TIP_EDGE"
    warned = warning_first_time is not None
    if warned and contact_t is not None and warning_first_time > contact_t:
        return "LATE_TP" if actual_positive else "LATE_FP"
    if warned and actual_positive:
        return "TP"
    if warned and not actual_positive:
        return "FP"
    if not warned and actual_positive:
        return "FN"
    return "TN"


def wilson_ci(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
def main() -> None:
    dev_seeds = development_seeds()
    print(f"development seed count: {len(dev_seeds)}")

    holdout_seeds = generate_holdout_seeds(HOLDOUT_TARGET_SEEDS, exclude=dev_seeds)
    print(f"holdout seeds requested: {len(holdout_seeds)}")

    records = {}
    for i, seed in enumerate(holdout_seeds):
        measure = i < 10  # only time the first 10 seeds for loop-runtime stats (cheap sample)
        ctx, rec = run_seed(seed, measure_loop_time=measure)
        gt = ground_truth_mode(seed)
        records[seed] = {"rec": rec, "cfg": ctx["cfg"], "contact_t": gt["contact_t"], "actual_mode": gt["mode"]}
        print(f"  [{i+1}/{len(holdout_seeds)}] seed={seed} mode={gt['mode']} contact_t={gt['contact_t']}", flush=True)

        counts = {}
        for r in records.values():
            counts[r["actual_mode"]] = counts.get(r["actual_mode"], 0) + 1
        n_processed = len(records)
        n_tip = counts.get("TIP_EDGE", 0)
        n_face = counts.get("FACE_FACE", 0)
        if n_processed >= HOLDOUT_MIN_SEEDS and n_tip >= HOLDOUT_MIN_TIP_EDGE and n_face >= HOLDOUT_MIN_FACE_FACE:
            print(f"  stop rule B satisfied at {n_processed} seeds (TIP_EDGE={n_tip}, FACE_FACE={n_face})")
            break
        if n_processed >= len(holdout_seeds):
            break

    print(f"\ntotal holdout seeds processed: {len(records)}")
    mode_counts = {"FACE_FACE": 0, "TIP_EDGE": 0, "OTHER_CONTACT": 0, "NO_CONTACT": 0}
    for r in records.values():
        mode_counts[r["actual_mode"]] = mode_counts.get(r["actual_mode"], 0) + 1
    print(f"mode_counts: {mode_counts}")

    # --- frozen candidate + baselines ---
    outcomes_frozen, outcomes_a, outcomes_b = {}, {}, {}
    leads_tp = []
    failure_seeds = []
    for seed, r in records.items():
        samples, contact_t, actual_mode = r["rec"].samples, r["contact_t"], r["actual_mode"]
        if not samples:
            outcomes_frozen[seed] = "EXCLUDED"
            continue
        pf = run_frozen_pipeline(samples)
        pa = run_baseline_3step(samples, BASELINE_A_M0)
        pb = run_baseline_3step(samples, BASELINE_B_M2MM)

        outcome = score_seed(pf["warning_first_time_s"], contact_t, actual_mode)
        outcomes_frozen[seed] = outcome
        outcomes_a[seed] = score_seed(pa["warning_first_time_s"], contact_t, actual_mode)
        outcomes_b[seed] = score_seed(pb["warning_first_time_s"], contact_t, actual_mode)

        if outcome == "TP" and contact_t is not None and pf["warning_first_time_s"] is not None:
            leads_tp.append(contact_t - pf["warning_first_time_s"])
        if outcome in ("FP", "FN", "LATE_TP", "LATE_FP"):
            failure_seeds.append(seed)

    def confusion(outcomes: dict) -> dict:
        c = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "LATE_TP": 0, "LATE_FP": 0, "EXCLUDED": 0}
        for v in outcomes.values():
            c[v] = c.get(v, 0) + 1
        return c

    c_frozen = confusion(outcomes_frozen)
    c_a = confusion(outcomes_a)
    c_b = confusion(outcomes_b)

    def stats(c: dict) -> dict:
        tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
                "specificity": specificity, "accuracy": accuracy, "f1": f1}

    s_frozen = stats(c_frozen)
    s_a = stats(c_a)
    s_b = stats(c_b)

    print("\n=== confusion matrices ===")
    print(f"  frozen: {c_frozen}  stats={s_frozen}")
    print(f"  baseline_A_m<0: {c_a}  stats={s_a}")
    print(f"  baseline_B_m<2mm: {c_b}  stats={s_b}")

    n_pos = c_frozen["TP"] + c_frozen["FN"] + c_frozen["LATE_TP"]
    n_neg = c_frozen["TN"] + c_frozen["FP"]
    precision_ci = wilson_ci(c_frozen["TP"], c_frozen["TP"] + c_frozen["FP"])
    recall_ci = wilson_ci(c_frozen["TP"], n_pos)
    specificity_ci = wilson_ci(c_frozen["TN"], n_neg)
    print(f"\nn_TIP_EDGE(pos)={n_pos} n_FACE_FACE(neg)={n_neg}")
    print(f"precision_95CI={precision_ci}")
    print(f"recall_95CI={recall_ci}")
    print(f"specificity_95CI={specificity_ci}")

    if leads_tp:
        leads_arr = np.array(leads_tp)
        print(f"\nlead_mean={np.mean(leads_arr)} lead_median={np.median(leads_arr)} "
              f"lead_min={np.min(leads_arr)} lead_p05={np.percentile(leads_arr,5)} "
              f"lead_p95={np.percentile(leads_arr,95)} lead_max={np.max(leads_arr)}")
        for bucket, cond in [
            ("<20ms", leads_arr < 0.020), ("<30ms", leads_arr < 0.030), ("<50ms", leads_arr < 0.050),
            (">=50ms", leads_arr >= 0.050), (">=100ms", leads_arr >= 0.100),
        ]:
            print(f"  lead {bucket}: {int(np.sum(cond))}")

    print(f"\nFP seeds (frozen): {[s for s,v in outcomes_frozen.items() if v=='FP']}")
    print(f"FN seeds (frozen): {[s for s,v in outcomes_frozen.items() if v=='FN']}")
    print(f"LATE_TP seeds (frozen): {[s for s,v in outcomes_frozen.items() if v=='LATE_TP']}")
    print(f"LATE_FP seeds (frozen): {[s for s,v in outcomes_frozen.items() if v=='LATE_FP']}")
    print(f"EXCLUDED (OTHER/NO_CONTACT) seeds: {[s for s,v in outcomes_frozen.items() if v=='EXCLUDED']}")

    # --- failure case detail ---
    print("\n=== failure-case detail ===")
    for seed in failure_seeds:
        r = records[seed]
        samples = r["rec"].samples
        ic = r["rec"].ic
        pf = run_frozen_pipeline(samples)
        d_a = np.array([s.d_a for s in samples])
        d_b = np.array([s.d_b for s in samples])
        m = d_b - d_a
        valid = np.array([s.t_static_valid for s in samples])
        ttc = np.array([s.t_static for s in samples])
        gate = valid & (ttc >= ORACLE_WINDOW_S[0]) & (ttc <= ORACLE_WINDOW_S[1])
        gate_m = m[gate]
        print(f"  seed={seed} outcome={outcomes_frozen[seed]} actual_mode={r['actual_mode']} contact_t={r['contact_t']}")
        print(f"    ic: box_half=({ic.box_half_x:.4f},{ic.box_half_y:.4f},{ic.box_half_z:.4f}) mass={ic.box_mass:.4f} "
              f"box_pos0={ic.box_pos0.tolist()} box_vel0={ic.box_vel0.tolist()} box_angvel0={ic.box_angvel0.tolist()}")
        print(f"    warning_first_time={pf['warning_first_time_s']} gate_intervals={pf['gate_intervals']} "
              f"gate_open_s={pf['gate_total_open_s']:.4f} gate_chatter={pf['gate_chatter']}")
        if gate_m.size:
            print(f"    min_margin_in_gate={np.min(gate_m):.5f} mean_margin_in_gate={np.mean(gate_m):.5f} "
                  f"frac_m<2mm_in_gate={np.mean(gate_m < MARGIN_THRESHOLD_M):.3f}")
        else:
            print("    gate never opened")

    # --- IC distribution summary (lightweight) ---
    print("\n=== holdout initial-condition distribution ===")
    ic_fields = ["box_half_x", "box_half_y", "box_half_z", "box_mass"]
    for field_name in ic_fields:
        vals = [getattr(r["rec"].ic, field_name) for r in records.values() if r["rec"].ic is not None]
        if vals:
            print(f"  {field_name}: min={min(vals):.4f} max={max(vals):.4f} mean={np.mean(vals):.4f} std={np.std(vals):.4f}")
    vel_mags = [float(np.linalg.norm(r["rec"].ic.box_vel0)) for r in records.values() if r["rec"].ic is not None]
    if vel_mags:
        print(f"  box_vel0_norm: min={min(vel_mags):.4f} max={max(vel_mags):.4f} mean={np.mean(vel_mags):.4f} std={np.std(vel_mags):.4f}")

    # --- runtime ---
    print("\n=== runtime ===")
    all_loop_times = []
    for r in records.values():
        all_loop_times.extend(r["rec"].loop_iter_times_s)
    if all_loop_times:
        lt = np.array(all_loop_times) * 1000.0  # ms
        print(f"  loop_iter (substep, includes production step): mean={np.mean(lt):.4f}ms p95={np.percentile(lt,95):.4f}ms "
              f"p99={np.percentile(lt,99):.4f}ms max={np.max(lt):.4f}ms")
        deadline_ms = 10.0
        misses = int(np.sum(lt >= deadline_ms))
        print(f"  10ms deadline misses (substep-level, NOTE: substep dt << control dt so this is not directly "
              f"comparable to a 10ms control-step budget): {misses}/{len(lt)} ({100*misses/len(lt):.2f}%)")

    # N20 geometry itself is unchanged (fixed definition/resolution per this
    # task's constraint), so its cost is reused from the prior isolated
    # measurement (diagnose_online_shadow_warning.py PART B) rather than
    # re-measured here: mean=3.24ms p95=3.40ms max=4.40ms.
    print("  N20 geometry timing: reusing previously measured value (unchanged definition/resolution) "
          "mean=3.24ms p95=3.40ms max=4.40ms (see diagnose_online_shadow_warning.py PART B)")


if __name__ == "__main__":
    main()
