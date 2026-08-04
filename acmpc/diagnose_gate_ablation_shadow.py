"""Shadow ablation: which capture gate (workspace, stability, both, neither)
is actually responsible for the mature-capture trigger being too
conservative for 3/4 seeds while staying safe for seed 1001?

MIN_SAMPLES sweep (diagnose_min_samples_sweep.py) already ruled out sample
count as the bottleneck -- workspace/stability decided capture timing
regardless of the sample floor. This ablation isolates which of those two
gates is doing the (over-)restricting.

Conditions (confidence==1.0 and position_remaining_ttc>0 required in all
four -- those two are cheap/early and were never implicated):
  A. confidence + ttc + workspace only (no stability requirement)
  B. confidence + ttc + stability only (no workspace requirement)
  C. confidence + ttc + workspace + stability (current/baseline gate set)
  D. confidence + ttc only (neither workspace nor stability)

Also records, per seed, the first time each of the four raw gates
(confidence, ttc, workspace, stability) independently becomes true --
these do not depend on which combination is used to trigger capture, since
box free-flight and the predictor sequence are unaffected by the shadow
override before any actual contact happens, so they are computed once per
seed (during condition D's run, which evaluates all four every step) and
reused for the report.

Does not touch production PRE_IMPACT gate/control logic -- shadow only.

Usage: python3 acmpc/diagnose_gate_ablation_shadow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.diagnose_precontact_grasp_offset import STAGE1_SEEDS, _build_stage1_config, analyze, run_episode
from acmpc.diagnose_shadow_precontact_target import summarize_condition
from acmpc.diagnose_mature_capture_shadow import (
    CAPTURE_MIN_CONFIDENCE,
    CAPTURE_STABILITY_THRESHOLD_M,
    CAPTURE_STABILITY_WINDOW_S,
    _in_workspace,
)

CONDITIONS = {
    "A_workspace_only": dict(use_workspace=True, use_stability=False),
    "B_stability_only": dict(use_workspace=False, use_stability=True),
    "C_workspace_and_stability": dict(use_workspace=True, use_stability=True),
    "D_neither": dict(use_workspace=False, use_stability=False),
}


def _make_gated_capture_override(cfg, *, use_workspace: bool, use_stability: bool):
    state = {
        "captured": None,
        "capture_time_s": None,
        "history": [],
        "rejected_workspace_count": 0,
        "first_true": {"confidence": None, "ttc": None, "workspace": None, "stability": None},
    }

    def override_fn(ctx: dict):
        if state["captured"] is not None:
            return state["captured"]
        if not ctx["position_remaining_ttc_valid"]:
            return None
        tau = float(ctx["position_remaining_ttc"])
        t = float(ctx["time_s"])
        ttc_ok = tau > 0.0
        if ttc_ok and state["first_true"]["ttc"] is None:
            state["first_true"]["ttc"] = t
        if not ttc_ok:
            return None

        current = np.asarray(ctx["prediction"].position_after(tau), dtype=float)
        state["history"].append((t, current.copy()))
        while state["history"] and t - state["history"][0][0] > CAPTURE_STABILITY_WINDOW_S:
            state["history"].pop(0)

        confidence_ok = float(ctx["prediction"].confidence) >= CAPTURE_MIN_CONFIDENCE
        if confidence_ok and state["first_true"]["confidence"] is None:
            state["first_true"]["confidence"] = t

        workspace_ok = _in_workspace(current, cfg)
        if not workspace_ok:
            state["rejected_workspace_count"] += 1
        elif state["first_true"]["workspace"] is None:
            state["first_true"]["workspace"] = t

        stability_ok = False
        if state["history"] and (t - state["history"][0][0]) >= CAPTURE_STABILITY_WINDOW_S * 0.8:
            pts = np.array([p for _, p in state["history"]])
            spread = float(np.max(np.linalg.norm(pts - current, axis=1)))
            stability_ok = spread <= CAPTURE_STABILITY_THRESHOLD_M
        if stability_ok and state["first_true"]["stability"] is None:
            state["first_true"]["stability"] = t

        required_ok = confidence_ok and ttc_ok
        if use_workspace:
            required_ok = required_ok and workspace_ok
        if use_stability:
            required_ok = required_ok and stability_ok

        if required_ok:
            state["captured"] = current.copy()
            state["capture_time_s"] = t
            return state["captured"]
        return None

    return override_fn, state


def run_gate_condition(seed: int, use_workspace: bool, use_stability: bool) -> tuple[dict, dict]:
    cfg = _build_stage1_config(seed)
    override_fn, state = _make_gated_capture_override(cfg.squeeze, use_workspace=use_workspace, use_stability=use_stability)
    result = run_episode(seed, target_override_fn=override_fn)
    out = analyze(result)
    out["capture_time_s"] = state["capture_time_s"]
    contact_t = out.get("actual_contact_time_s")
    out["capture_lead_time_s"] = (
        (contact_t - state["capture_time_s"])
        if (contact_t is not None and state["capture_time_s"] is not None)
        else None
    )
    out["rejected_workspace_count"] = state["rejected_workspace_count"]
    out["first_true"] = state["first_true"]
    return out, state


def _seed_verdict(baseline_summary: dict, cond_summary: dict) -> tuple[bool, bool]:
    miss = cond_summary.get("failure_reason") == "box passed the interception workspace"
    lost = bool(cond_summary.get("contact_lost"))
    emergency = cond_summary.get("failure_reason") == "emergency contact force exceeded"
    safe = not miss and not lost and not emergency

    baseline_dist = max(
        baseline_summary.get("left_contact_distance_from_face_center_m") or 0.0,
        baseline_summary.get("right_contact_distance_from_face_center_m") or 0.0,
    )
    cond_dist = max(
        cond_summary.get("left_contact_distance_from_face_center_m") or float("inf"),
        cond_summary.get("right_contact_distance_from_face_center_m") or float("inf"),
    )
    baseline_rot = baseline_summary.get("cumulative_rotation_deg_hold")
    cond_rot = cond_summary.get("cumulative_rotation_deg_hold")
    dist_improved = cond_dist < 0.9 * baseline_dist
    rot_improved = baseline_rot is not None and cond_rot is not None and cond_rot < 0.9 * baseline_rot
    return safe, (dist_improved and rot_improved)


def main() -> None:
    baselines = {seed: summarize_condition({**analyze(run_episode(seed)), "lead_time_s": None}) for seed in STAGE1_SEEDS}

    all_results = {}  # condition -> seed -> (out, safe, improved)
    for cond_name, kwargs in CONDITIONS.items():
        print(f"=== condition {cond_name} ({kwargs}) ===", flush=True)
        all_results[cond_name] = {}
        for seed in STAGE1_SEEDS:
            out, state = run_gate_condition(seed, **kwargs)
            summ = summarize_condition(out)
            summ["capture_time_s"] = out["capture_time_s"]
            summ["capture_lead_time_s"] = out["capture_lead_time_s"]
            summ["rejected_workspace_count"] = out["rejected_workspace_count"]
            summ["first_true_confidence_s"] = out["first_true"]["confidence"]
            summ["first_true_ttc_s"] = out["first_true"]["ttc"]
            summ["first_true_workspace_s"] = out["first_true"]["workspace"]
            summ["first_true_stability_s"] = out["first_true"]["stability"]
            safe, improved = _seed_verdict(baselines[seed], summ)
            summ["safe"] = safe
            summ["improved"] = improved
            all_results[cond_name][seed] = summ
            print(f"  -- seed {seed} --")
            for k, v in summ.items():
                print(f"    {k}: {v}")
        print(flush=True)

    print("=== verdict per condition ===")
    for cond_name in CONDITIONS:
        details = all_results[cond_name]
        all_safe = all(d["safe"] for d in details.values())
        all_improved = all(d["improved"] for d in details.values())
        any_improved = any(d["improved"] for d in details.values())
        print(f"  {cond_name}: all_safe={all_safe} all_improved={all_improved} any_improved={any_improved} "
              f"detail={[(s, details[s]['safe'], details[s]['improved']) for s in STAGE1_SEEDS]}")

    print("\n=== baseline gate summary ===")
    for k, v in baselines.items():
        print(f"  seed {k}: rot={v.get('cumulative_rotation_deg_hold')} "
              f"dist_L={v.get('left_contact_distance_from_face_center_m')}")


if __name__ == "__main__":
    main()
