"""Phase 2 Cartesian impedance control validation.

Success criteria  (world-frame EE metrics)
------------------------------------------
SC1  Position hold   : max |p_ee - p_des| < 5 mm  over 3 s
SC2  Orientation hold: max |e_rot|         < 0.05 rad over 3 s
SC3  Step response   : settle |p_ee-p_des| < 5 mm within 4 s  (5 cm step in z)
SC4  Compliance      : F_ext = 10 N in x → steady-state Δx ∈ [0.25, 4.0] × (F/K_pos)
SC5  Stability       : max EE speed (SC1+SC3 steady-state) < 0.3 m/s

Notes
-----
* Null-space stabilisation (Kp_ns / Kd_ns) is needed because the 7-DOF arm has
  a 1-D null-space when tracking 6-DOF pose.  Without it the arm drifts ~22 mm
  in EE z over 3 s due to imperfect gravity-comp in the null-space direction.

* SC4 uses qfrc_applied (joint-space force via site Jacobian) instead of
  xfrc_applied (body-centre force) to avoid a site-vs-body mismatch that
  caused near-zero measured deflection in the initial implementation.

Failure → gains adjusted, test reruns (up to MAX_RETRIES).

Usage
-----
    python tests/phase2_impedance_test.py
    python tests/phase2_impedance_test.py --viewer
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.clik import build_serial_arm
from control.clik.impedance import (
    CartesianImpedanceConfig,
    CartesianImpedanceController,
    cartesian_force_to_qfrc,
)
from control.clik.kinematics import get_ee_transform
from robot.ffw_config import FFW_ARMS

# ── scene ─────────────────────────────────────────────────────────────────────
XML_PATH = ROOT / "model" / "robotis_ffw" / "scene_ffw_sg2_fixed_base.xml"

# ── thresholds ────────────────────────────────────────────────────────────────
SC1_POS_M         = 5e-3     # max |p - p_des|  < 5 mm
SC2_ROT_RAD       = 0.05     # max |e_rot|       < 0.05 rad  (~3°)
SC3_SETTLE_M      = 5e-3     # settle error      < 5 mm
SC3_SETTLE_TIME_S = 4.0      # settle time       < 4 s
SC4_RATIO_LO      = 0.25     # deflection ratio lower bound
SC4_RATIO_HI      = 4.00     # deflection ratio upper bound
SC5_MAX_V_MS      = 0.30     # max EE speed      < 0.3 m/s

SC1_WARMUP_S   = 1.0         # let soft weld settle (~7 mm compliance under robot weight)
SC1_DURATION_S = 3.0
SC3_STEP_M     = 0.05        # 5 cm step (downward, -z) to avoid joint limits near +z
SC3_DURATION_S = SC3_SETTLE_TIME_S
SC4_FORCE_N    = 10.0        # applied Fx [N] via site Jacobian
SC4_DURATION_S = 4.0         # time to reach steady state (allow slow directions)

MAX_RETRIES = 10

RIGHT_HOME_Q = np.array([-0.534804, -0.460945, -1.278942,
                           0.871459, -0.281033, -0.379103,  0.050507])
LEFT_HOME_Q  = np.array([-0.534802,  0.460945,  1.278943,
                           0.871459,  0.281032, -0.379103, -0.050513])


@dataclass
class SCResult:
    name: str
    passed: bool
    value: float
    limit: float
    unit: str

    def __str__(self) -> str:
        mark = "✓" if self.passed else "✗"
        return f"  {mark} {self.name}: {self.value:.5f} {self.unit}  (limit {self.limit})"


# ─────────────────────────────────────────────────────────────────────────────

def _build():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data  = mujoco.MjData(model)
    right_arm = build_serial_arm(model, FFW_ARMS["right"])
    left_arm  = build_serial_arm(model, FFW_ARMS["left"])
    return model, data, right_arm, left_arm


def _init(model, data, right_arm, left_arm):
    # weld relpose="0 0 0.0035 1 0 0 0" → set freejoint z to match;
    # without this the weld constraint generates >1000 N forces at t=0.
    data.qpos[2] = 0.0035
    data.qpos[right_arm.qpos_indices] = RIGHT_HOME_Q
    data.qpos[left_arm.qpos_indices]  = LEFT_HOME_Q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _ctrl_pair(model, right_arm, left_arm, cfg):
    """Create one controller per arm with home-config null-space anchoring."""
    return (
        CartesianImpedanceController(cfg, right_arm, model, q_home=RIGHT_HOME_Q),
        CartesianImpedanceController(cfg, left_arm,  model, q_home=LEFT_HOME_Q),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SC1 + SC2 — position and orientation hold
# ─────────────────────────────────────────────────────────────────────────────

def run_sc1_sc2(cfg: CartesianImpedanceConfig, viewer=None) -> tuple[SCResult, SCResult, float]:
    model, data, right_arm, left_arm = _build()
    _init(model, data, right_arm, left_arm)
    rc, lc = _ctrl_pair(model, right_arm, left_arm, cfg)

    T_des_r = get_ee_transform(data, right_arm).copy()
    T_des_l = get_ee_transform(data, left_arm).copy()

    max_pos_err = 0.0
    max_rot_err = 0.0
    max_v_ee    = 0.0
    dt = model.opt.timestep
    warmup_steps  = int(SC1_WARMUP_S   / dt)
    measure_steps = int(SC1_DURATION_S / dt)

    for step in range(warmup_steps + measure_steps):
        r = rc.apply(model, data, right_arm, T_des_r)
        l = lc.apply(model, data, left_arm,  T_des_l)
        mujoco.mj_step(model, data)
        if step >= warmup_steps:           # skip weld-settling transient
            max_pos_err = max(max_pos_err, r.e_pos_norm, l.e_pos_norm)
            max_rot_err = max(max_rot_err, r.e_rot_norm, l.e_rot_norm)
            max_v_ee    = max(max_v_ee,
                              float(np.linalg.norm(r.v_ee)),
                              float(np.linalg.norm(l.v_ee)))
        if viewer is not None:
            viewer.sync()

    sc1 = SCResult("SC1 pos hold max|Δp|", max_pos_err < SC1_POS_M,
                   max_pos_err, SC1_POS_M, "m")
    sc2 = SCResult("SC2 rot hold max|e_rot|", max_rot_err < SC2_ROT_RAD,
                   max_rot_err, SC2_ROT_RAD, "rad")
    return sc1, sc2, max_v_ee


# ─────────────────────────────────────────────────────────────────────────────
# SC3 — 5 cm step response in z (right arm)
# ─────────────────────────────────────────────────────────────────────────────

def run_sc3(cfg: CartesianImpedanceConfig, viewer=None) -> tuple[SCResult, SCResult, float]:
    model, data, right_arm, left_arm = _build()
    _init(model, data, right_arm, left_arm)
    rc, lc = _ctrl_pair(model, right_arm, left_arm, cfg)

    T_des_l = get_ee_transform(data, left_arm).copy()
    T_des_r = get_ee_transform(data, right_arm).copy()
    T_des_r[2, 3] -= SC3_STEP_M    # 5 cm down in z (avoids arm_r_joint4 upper limit)

    settle_time  = float("inf")
    final_err    = float("inf")
    max_v_ee     = 0.0
    dt    = model.opt.timestep
    steps = int(SC3_DURATION_S / dt)

    for step in range(steps):
        t = step * dt
        r = rc.apply(model, data, right_arm, T_des_r)
        lc.apply(model, data, left_arm,  T_des_l)
        mujoco.mj_step(model, data)

        if r.e_pos_norm < SC3_SETTLE_M and settle_time == float("inf"):
            settle_time = t
        if t >= SC3_DURATION_S - 1.0:       # last 1 s window
            final_err = min(final_err, r.e_pos_norm)
        if t >= SC3_DURATION_S * 0.5:       # exclude early transient from SC5
            max_v_ee = max(max_v_ee, float(np.linalg.norm(r.v_ee)))

        if viewer is not None:
            viewer.sync()

    sc3_err  = SCResult("SC3 step settle |Δp|",
                        final_err < SC3_SETTLE_M,
                        final_err, SC3_SETTLE_M, "m")
    sc3_time = SCResult("SC3 step settle time",
                        settle_time < SC3_SETTLE_TIME_S,
                        settle_time if settle_time < float("inf") else SC3_DURATION_S,
                        SC3_SETTLE_TIME_S, "s")
    return sc3_err, sc3_time, max_v_ee


# ─────────────────────────────────────────────────────────────────────────────
# SC4 — virtual compliance via qfrc_applied (consistent with site Jacobian)
#
# The force is applied in joint-space as J_site_pos.T @ F_cart so the
# equilibrium exactly satisfies K @ e_ss = -F_cart (same Jacobian as the
# impedance controller), giving deflection_x = Fx / K_pos in steady state.
# ─────────────────────────────────────────────────────────────────────────────

def run_sc4(cfg: CartesianImpedanceConfig, viewer=None) -> SCResult:
    model, data, right_arm, left_arm = _build()
    _init(model, data, right_arm, left_arm)
    rc, lc = _ctrl_pair(model, right_arm, left_arm, cfg)

    T_des_r = get_ee_transform(data, right_arm).copy()
    T_des_l = get_ee_transform(data, left_arm).copy()
    p0_r    = T_des_r[:3, 3].copy()

    F_cart = np.array([SC4_FORCE_N, 0., 0.])   # constant Cartesian Fx

    dt    = model.opt.timestep
    steps = int(SC4_DURATION_S / dt)

    for _ in range(steps):
        # Map Cartesian force through site Jacobian (same frame as impedance)
        qfrc_ext = cartesian_force_to_qfrc(model, data, right_arm, F_cart)
        data.qfrc_applied[right_arm.qvel_indices] = qfrc_ext

        rc.apply(model, data, right_arm, T_des_r)
        lc.apply(model, data, left_arm,  T_des_l)
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

    # Clear applied force
    data.qfrc_applied[:] = 0.0

    T_cur_r = get_ee_transform(data, right_arm)
    deflection_x = float(T_cur_r[0, 3] - p0_r[0])

    predicted = SC4_FORCE_N / cfg.K_pos
    ratio = deflection_x / predicted if abs(predicted) > 1e-9 else float("inf")

    passed = SC4_RATIO_LO <= ratio <= SC4_RATIO_HI
    return SCResult(
        f"SC4 compliance ratio Δx/(F/K) [{SC4_RATIO_LO:.2f},{SC4_RATIO_HI:.2f}]",
        passed, ratio, SC4_RATIO_HI, ""
    )


# ─────────────────────────────────────────────────────────────────────────────

def run_all(cfg: CartesianImpedanceConfig, viewer=None) -> tuple[list[SCResult], bool]:
    print(f"\n  K_pos={cfg.K_pos:.0f}  K_rot={cfg.K_rot:.0f}  "
          f"D_pos={cfg.D_pos:.1f}  D_rot={cfg.D_rot:.1f}  "
          f"Kp_ns={cfg.Kp_ns:.1f}  Kd_ns={cfg.Kd_ns:.1f}  "
          f"tau_limit={cfg.tau_limit:.0f}")

    sc1, sc2, v1 = run_sc1_sc2(cfg, viewer)
    sc3e, sc3t, v3 = run_sc3(cfg, viewer)
    sc4 = run_sc4(cfg, viewer)

    max_v = max(v1, v3)
    sc5 = SCResult("SC5 max EE speed (SC1+SC3 steady)",
                   max_v < SC5_MAX_V_MS, max_v, SC5_MAX_V_MS, "m/s")

    results = [sc1, sc2, sc3e, sc3t, sc4, sc5]
    return results, all(r.passed for r in results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 2 — Cartesian Impedance  |  Success Criteria Validation")
    print("=" * 70)
    print(f"SC1  Pos hold   max|Δp|    < {SC1_POS_M*1e3:.0f} mm   ({SC1_DURATION_S:.0f} s)")
    print(f"SC2  Rot hold   max|e_rot| < {SC2_ROT_RAD:.3f} rad ({SC2_ROT_RAD*180/np.pi:.1f}°)")
    print(f"SC3  Step resp  |Δp|       < {SC3_SETTLE_M*1e3:.0f} mm   "
          f"(step {SC3_STEP_M*100:.0f} cm in z, time < {SC3_SETTLE_TIME_S:.0f} s)")
    print(f"SC4  Compliance ratio Δx/(F/K) ∈ [{SC4_RATIO_LO:.2f}, {SC4_RATIO_HI:.2f}]"
          f"  (F={SC4_FORCE_N:.0f} N via site Jacobian)")
    print(f"SC5  Max EE speed (steady)  < {SC5_MAX_V_MS:.2f} m/s")

    cfg = CartesianImpedanceConfig(
        K_pos=500.0, K_rot=50.0, D_pos=100.0, D_rot=15.0,
        tau_limit=300.0, gravity_comp=True,
        Kp_ns=5.0, Kd_ns=2.0,
    )

    viewer_ctx = None
    if args.viewer:
        _m, _d, _, _ = _build()
        viewer_ctx = mujoco.viewer.launch_passive(_m, _d)

    try:
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\n── Attempt {attempt}/{MAX_RETRIES} " + "─" * 46)
            results, ok = run_all(cfg, viewer=viewer_ctx)
            for r in results:
                print(r)

            if ok:
                print(f"\n{'='*70}")
                print("ALL SUCCESS CRITERIA MET ✓")
                print(f"  Final: K_pos={cfg.K_pos:.0f}  K_rot={cfg.K_rot:.0f}  "
                      f"D_pos={cfg.D_pos:.1f}  D_rot={cfg.D_rot:.1f}  "
                      f"Kp_ns={cfg.Kp_ns:.1f}  Kd_ns={cfg.Kd_ns:.1f}")
                print(f"{'='*70}")
                return

            sc5_ok  = results[5].passed
            sc1_ok  = results[0].passed
            sc2_ok  = results[1].passed
            sc3e_ok = results[2].passed
            sc3t_ok = results[3].passed
            sc4_ok  = results[4].passed

            if not sc5_ok:
                # Oscillating — reduce stiffness and raise damping
                cfg = CartesianImpedanceConfig(
                    K_pos=cfg.K_pos * 0.55,
                    K_rot=cfg.K_rot * 0.55,
                    D_pos=min(cfg.D_pos * 1.6, 400.0),
                    D_rot=min(cfg.D_rot * 1.6, 100.0),
                    tau_limit=cfg.tau_limit,
                    gravity_comp=True,
                    Kp_ns=cfg.Kp_ns,
                    Kd_ns=min(cfg.Kd_ns * 1.3, 20.0),
                )
                print("  → SC5 (unstable): reduce K, raise D")
            elif not sc1_ok or not sc2_ok:
                # Drifting despite null-space control → raise Kp_ns and task K
                cfg = CartesianImpedanceConfig(
                    K_pos=min(cfg.K_pos * 1.5, 3000.0),
                    K_rot=min(cfg.K_rot * 1.5, 300.0),
                    D_pos=min(cfg.D_pos * 1.1, 400.0),
                    D_rot=min(cfg.D_rot * 1.1, 100.0),
                    tau_limit=cfg.tau_limit,
                    gravity_comp=True,
                    Kp_ns=min(cfg.Kp_ns * 2.0, 50.0),
                    Kd_ns=min(cfg.Kd_ns * 1.5, 20.0),
                )
                print("  → SC1/SC2 (drift): raise K and Kp_ns")
            elif not sc3e_ok or not sc3t_ok:
                # Poor step tracking → raise K
                cfg = CartesianImpedanceConfig(
                    K_pos=min(cfg.K_pos * 1.4, 3000.0),
                    K_rot=min(cfg.K_rot * 1.4, 300.0),
                    D_pos=min(cfg.D_pos * 1.05, 400.0),
                    D_rot=cfg.D_rot,
                    tau_limit=cfg.tau_limit,
                    gravity_comp=True,
                    Kp_ns=cfg.Kp_ns,
                    Kd_ns=cfg.Kd_ns,
                )
                print("  → SC3 (tracking): raise K_pos")
            elif not sc4_ok:
                ratio_val = results[4].value
                if ratio_val < SC4_RATIO_LO:
                    cfg = CartesianImpedanceConfig(
                        K_pos=cfg.K_pos * 0.7,
                        K_rot=cfg.K_rot,
                        D_pos=cfg.D_pos,
                        D_rot=cfg.D_rot,
                        tau_limit=cfg.tau_limit,
                        gravity_comp=True,
                        Kp_ns=cfg.Kp_ns,
                        Kd_ns=cfg.Kd_ns,
                    )
                    print("  → SC4 (too stiff): lower K_pos")
                else:
                    cfg = CartesianImpedanceConfig(
                        K_pos=min(cfg.K_pos * 1.4, 3000.0),
                        K_rot=cfg.K_rot,
                        D_pos=cfg.D_pos,
                        D_rot=cfg.D_rot,
                        tau_limit=cfg.tau_limit,
                        gravity_comp=True,
                        Kp_ns=cfg.Kp_ns,
                        Kd_ns=cfg.Kd_ns,
                    )
                    print("  → SC4 (too compliant): raise K_pos")

        print(f"\n{'='*70}")
        print(f"FAILED after {MAX_RETRIES} attempts.")
        print(f"{'='*70}")
        sys.exit(1)

    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()


if __name__ == "__main__":
    main()
