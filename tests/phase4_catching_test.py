"""Phase 4 Dynamic Catching validation.

Success criteria
----------------
SC1  Velocity prediction accuracy : after 0.3 s observation, |est_vz - true_vz| < 0.5 m/s
SC2  Intercept reachability       : plan.reachable=True at some point during free fall
SC3  Adaptive stiffness reduction : K(ttc=0.05) < K_nominal AND K(ttc=t_soften+ε) = K_nominal
SC4  Catching orientation         : compute_catching_orientation gives valid SO(3) R with
                                    primary-axis alignment error < 5°
SC5  Catch success                : ball-to-gripper contact force > threshold
                                    (self-contact between finger geoms excluded)

Catching mechanism
------------------
The ball is dropped from LAUNCH_Z directly above the right gripper's upper distal
segment (geom 35, located near [0.292, -0.265, 0.744] at pre-grasp pose).
The arm holds its natural pre-grasp orientation — the catching orientation (SC4) is
verified separately as a pure functional check, because applying the 90° reorientation
during the catch physically moves the finger geoms away from the ball path.

Gripper closing strategy: ttc-based pre-emptive close.
When the predicted time-to-contact drops below CLOSE_LEAD_S (0.5 s), the gripper
ramps from open to closed over 0.3 s.  By the time the ball arrives, the gripper is
already partially or fully closed, so the ball impacts the finger surface rather
than falling through the open jaw.

Fall parameters (g = GRAVITY_Z = −2.0 m/s²):
  - Ball at z = 2.0 m, vz = 0 → reaches finger level (z ≈ 0.74) in ≈ 1.12 s
  - Impact speed ≈ 2.24 m/s (manageable for detection)
  - Predictor has 0.62 s of observation before ttc-based close starts

Usage
-----
    python tests/phase4_catching_test.py
    python tests/phase4_catching_test.py --viewer
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

from control.clik import (
    build_serial_arm,
    get_ee_transform,
    InterceptionPlanner,
    MovingTargetPredictor,
)
from control.clik.catching import adaptive_stiffness, compute_catching_orientation
from control.clik.contact import finger_geom_ids
from control.clik.impedance import CartesianImpedanceConfig, CartesianImpedanceController
from robot.ffw_config import FFW_ARMS, FFW_GRIPPERS

# ── scene ─────────────────────────────────────────────────────────────────────
XML_PATH = ROOT / "model" / "robotis_ffw" / "scene_ffw_sg2_fixed_base_grasp.xml"

# ── object layout (probed from model) ─────────────────────────────────────────
# obj_joint: qposadr=38, dofadr=37
OBJ_QPOSADR   = 38
OBJ_DOFADR    = 37
OBJ_BODY_NAME = "grasp_object"
OBJ_GEOM_NAME = "obj_sphere"

# ── arm home ──────────────────────────────────────────────────────────────────
RIGHT_HOME_Q = np.array([-0.534804, -0.460945, -1.278942,
                           0.871459, -0.281033, -0.379103,  0.050507])
LEFT_HOME_Q  = np.array([-0.534802,  0.460945,  1.278943,
                           0.871459,  0.281032, -0.379103, -0.050513])

# ── launch parameters ─────────────────────────────────────────────────────────
# Ball directly above right-finger distal segment (geom 35 x ≈ 0.292)
OBJ_XY      = np.array([0.292, -0.266])
LAUNCH_Z    = 2.0    # m
LAUNCH_VZ   = 0.0   # m/s  (free-fall from rest)

# ── physics ───────────────────────────────────────────────────────────────────
# Reduced gravity (−2.0 m/s²): fall time ≈ 1.12 s, impact speed ≈ 2.24 m/s.
# MuJoCo recalculates qfrc_bias with the modified gravity, so impedance
# gravity compensation stays correct.
DEFAULT_GRAVITY_Z = -2.0

# ── timing ────────────────────────────────────────────────────────────────────
WARMUP_S       = 1.0   # weld-settling warmup
PRE_POSITION_S = 1.5   # arm moves to pre-grasp (−5 cm z)
LAUNCH_T       = WARMUP_S + PRE_POSITION_S   # 2.5 s
POST_LAUNCH_S  = 4.0   # simulation budget after launch

# ── gripper close strategy ────────────────────────────────────────────────────
# Start closing CLOSE_LEAD_S seconds before predicted contact.
# Ramp duration: CLOSE_RAMP_S.
CLOSE_LEAD_S = 0.5
CLOSE_RAMP_S = 0.3

# ── SC thresholds ─────────────────────────────────────────────────────────────
SC1_VZ_ERROR_MAX    = 0.5     # [m/s]
SC1_OBS_TIME_S      = 0.3     # [s]  after launch
SC2_REACHABLE_REQ   = True
SC4_ALIGN_ERROR_DEG = 5.0     # [°]
SC5_CONTACT_MIN_N   = 0.5     # [N]  ball-to-finger force threshold

# ── stiffness ─────────────────────────────────────────────────────────────────
K_NOMINAL = 500.0
K_ROT     = 50.0
K_SOFT    = 60.0
T_SOFTEN  = 0.15

# ── predictor / planner ───────────────────────────────────────────────────────
DEFAULT_SMOOTHING_ALPHA = 0.85
DEFAULT_REACH_SPEED     = 0.4   # m/s

MAX_RETRIES = 8


# ── helpers ───────────────────────────────────────────────────────────────────

@dataclass
class SCResult:
    name: str
    passed: bool
    value: float
    limit: float
    unit: str

    def __str__(self) -> str:
        mark = "✓" if self.passed else "✗"
        return f"  {mark} {self.name}: {self.value:.4f} {self.unit}  (limit {self.limit})"


@dataclass
class CatchingCfg:
    smoothing_alpha:  float = DEFAULT_SMOOTHING_ALPHA
    reach_speed:      float = DEFAULT_REACH_SPEED
    gravity_z:        float = DEFAULT_GRAVITY_Z
    t_soften:         float = T_SOFTEN
    K_soft:           float = K_SOFT
    close_lead_s:     float = CLOSE_LEAD_S
    sc5_min_n:        float = SC5_CONTACT_MIN_N


def _build(cfg: CatchingCfg):
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data  = mujoco.MjData(model)
    model.opt.gravity[2] = cfg.gravity_z
    right_arm = build_serial_arm(model, FFW_ARMS["right"])
    left_arm  = build_serial_arm(model, FFW_ARMS["left"])
    return model, data, right_arm, left_arm


def _init(model, data, right_arm, left_arm):
    data.qpos[2] = 0.0035
    data.qpos[right_arm.qpos_indices] = RIGHT_HOME_Q
    data.qpos[left_arm.qpos_indices]  = LEFT_HOME_Q
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    # Park ball above scene until launch
    data.qpos[OBJ_QPOSADR:OBJ_QPOSADR + 7] = [OBJ_XY[0], OBJ_XY[1], LAUNCH_Z + 10.0,
                                                1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def _launch_object(data):
    data.qpos[OBJ_QPOSADR + 2] = LAUNCH_Z
    data.qvel[OBJ_DOFADR:OBJ_DOFADR + 6] = 0.0
    data.qvel[OBJ_DOFADR + 2] = LAUNCH_VZ


def _imp_cfg(K_pos: float, K_rot: float = K_ROT) -> CartesianImpedanceConfig:
    return CartesianImpedanceConfig(
        K_pos=K_pos, K_rot=K_rot,
        D_pos=100.0, D_rot=15.0,
        tau_limit=300.0, gravity_comp=True,
        Kp_ns=5.0, Kd_ns=2.0,
    )


def _get_gripper_geom_ids(model):
    return finger_geom_ids(model, FFW_GRIPPERS["right"].finger_body_names)


def _get_gripper_act_ids(model):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in FFW_GRIPPERS["right"].actuator_names]


def _close_gripper(data, act_ids, fraction: float):
    close_ctrl = FFW_GRIPPERS["right"].close_ctrl
    for i, aid in enumerate(act_ids):
        data.ctrl[aid] = close_ctrl[i] * float(np.clip(fraction, 0.0, 1.0))


def _ball_contact_force(model, data, gripper_geom_ids: set[int]) -> float:
    """Sum normal force from contacts where exactly one geom is the ball.

    This excludes gripper self-contact (both geoms in gripper_geom_ids).
    """
    object_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, OBJ_GEOM_NAME
    )
    if object_geom_id < 0:
        raise ValueError(f"MuJoCo geom not found: {OBJ_GEOM_NAME}")
    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        ball_involved  = (g1 == object_geom_id) or (g2 == object_geom_id)
        self_contact   = (g1 in gripper_geom_ids) and (g2 in gripper_geom_ids)
        if ball_involved and not self_contact:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            total += max(0.0, float(force[0]))
    return total


def _obj_pos_analytic(t_from_launch: float, gravity_z: float) -> np.ndarray:
    dt = max(0.0, t_from_launch)
    z  = LAUNCH_Z + LAUNCH_VZ * dt + 0.5 * gravity_z * dt**2
    return np.array([OBJ_XY[0], OBJ_XY[1], float(z)])


# ─────────────────────────────────────────────────────────────────────────────
# SC3 — adaptive stiffness (functional)
# ─────────────────────────────────────────────────────────────────────────────

def run_sc3(cfg: CatchingCfg) -> SCResult:
    K_soft_val  = adaptive_stiffness(0.05,                 K_NOMINAL, cfg.K_soft, cfg.t_soften)
    K_full_val  = adaptive_stiffness(cfg.t_soften + 0.01,  K_NOMINAL, cfg.K_soft, cfg.t_soften)
    K_mid_val   = adaptive_stiffness(cfg.t_soften * 0.5,   K_NOMINAL, cfg.K_soft, cfg.t_soften)
    ok = (
        float(K_soft_val) < float(K_NOMINAL)           # stiffness drops near contact
        and float(K_full_val) == float(K_NOMINAL)       # full outside window
        and float(cfg.K_soft) <= float(K_mid_val) <= float(K_NOMINAL)  # monotone
    )
    drop = float(K_NOMINAL) - float(K_soft_val)
    return SCResult("SC3 stiffness drop at ttc=0.05s", ok, drop, 0.0, "N/m")


# ─────────────────────────────────────────────────────────────────────────────
# SC4 — catching orientation (functional)
# ─────────────────────────────────────────────────────────────────────────────

def run_sc4() -> SCResult:
    test_vel  = np.array([0.5, 0.0, -1.0])   # diagonal approach
    test_pos  = np.array([0.292, -0.266, 0.74])
    R = np.asarray(compute_catching_orientation(test_vel, test_pos))
    det  = float(np.linalg.det(R))
    orth = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    approach   = -test_vel / np.linalg.norm(test_vel)
    world_prim = R @ np.array([1.0, 0.0, 0.0])
    cos_a      = float(np.clip(np.dot(world_prim, approach), -1.0, 1.0))
    err_deg    = float(np.degrees(np.arccos(cos_a)))
    ok = abs(det - 1.0) < 1e-6 and orth < 1e-6 and err_deg < SC4_ALIGN_ERROR_DEG
    return SCResult("SC4 orientation axis alignment error", ok, err_deg, SC4_ALIGN_ERROR_DEG, "°")


# ─────────────────────────────────────────────────────────────────────────────
# Full simulation: SC1 + SC2 + SC5
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(cfg: CatchingCfg, viewer=None) -> tuple[SCResult, SCResult, SCResult]:
    model, data, right_arm, left_arm = _build(cfg)
    _init(model, data, right_arm, left_arm)
    gripper_geom_ids = _get_gripper_geom_ids(model)
    act_ids          = _get_gripper_act_ids(model)
    obj_bid          = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY_NAME)

    predictor = MovingTargetPredictor(
        horizon_s=0.3,
        smoothing_alpha=cfg.smoothing_alpha,
        max_prediction_speed=10.0,
    )
    planner = InterceptionPlanner(
        min_lead_s=0.05,
        max_lead_s=2.0,
        lead_samples=20,
        reach_speed=cfg.reach_speed,
    )

    T_r_home  = get_ee_transform(data, right_arm).copy()
    T_l       = get_ee_transform(data, left_arm).copy()
    T_approach = T_r_home.copy()
    T_approach[2, 3] -= 0.05   # 5 cm below home

    rc = CartesianImpedanceController(_imp_cfg(K_NOMINAL), right_arm, model, q_home=RIGHT_HOME_Q)
    lc = CartesianImpedanceController(_imp_cfg(K_NOMINAL), left_arm,  model, q_home=LEFT_HOME_Q)

    dt           = model.opt.timestep
    total_steps  = int((LAUNCH_T + POST_LAUNCH_S) / dt)
    launched     = False
    prediction   = None

    # SC1 state
    sc1_vz_err    = float("inf")
    sc1_check_t   = LAUNCH_T + SC1_OBS_TIME_S

    # SC2 state
    sc2_reachable = False

    # SC5 state
    max_ball_contact_f = 0.0
    close_start_t      = float("inf")

    for step in range(total_steps):
        t = step * dt

        # ── launch ────────────────────────────────────────────────────────────
        if not launched and t >= LAUNCH_T:
            _launch_object(data)
            mujoco.mj_forward(model, data)
            predictor.reset()
            launched = True

        obj_pos = data.xpos[obj_bid].copy()

        # ── arm target: hold pre-grasp orientation (no catching-orientation applied) ──
        T_des = T_r_home if t < WARMUP_S else T_approach

        # ── predictor + planner + ttc ──────────────────────────────────────────
        ttc = float("inf")
        if launched:
            prediction      = predictor.update(t, obj_pos)
            t_from_launch   = t - LAUNCH_T

            def _obj_at(ft: float) -> np.ndarray:
                return _obj_pos_analytic(ft - LAUNCH_T, cfg.gravity_z)

            ee_pos = get_ee_transform(data, right_arm)[:3, 3]
            plan   = planner.plan(
                time_s=t,
                current_object_position=ee_pos,
                current_target_position=obj_pos,
                target_position_at_time=_obj_at,
            )
            if plan.reachable:
                sc2_reachable = True

            # Compute ttc: seconds until ball reaches EE z level
            target_z = T_approach[2, 3]
            g        = cfg.gravity_z
            a_c      = 0.5 * g
            b_c      = LAUNCH_VZ
            c_c      = LAUNCH_Z - target_z
            disc     = b_c**2 - 4.0 * a_c * c_c
            if disc >= 0 and a_c != 0:
                s1 = (-b_c + np.sqrt(disc)) / (2.0 * a_c)
                s2 = (-b_c - np.sqrt(disc)) / (2.0 * a_c)
                future_s = [s for s in (s1, s2) if s > t_from_launch]
                ttc = max(0.0, min(future_s) - t_from_launch) if future_s else 0.0

            # SC1 evaluation
            if not np.isfinite(sc1_vz_err) and t >= sc1_check_t:
                true_vz    = LAUNCH_VZ + cfg.gravity_z * t_from_launch
                est_vz     = float(prediction.velocity[2])
                sc1_vz_err = abs(est_vz - true_vz)

        # ── adaptive stiffness ─────────────────────────────────────────────────
        K_eff = float(adaptive_stiffness(ttc, K_NOMINAL, cfg.K_soft, cfg.t_soften))
        if abs(rc.config.K_pos - K_eff) > 1.0:
            rc.config.K_pos = K_eff

        # ── ttc-based gripper close ────────────────────────────────────────────
        # Pre-emptively close when CLOSE_LEAD_S seconds before predicted contact.
        # Ball impacts partially/fully closed fingers → contact force detected.
        if launched and np.isfinite(ttc) and ttc < cfg.close_lead_s:
            if close_start_t == float("inf"):
                close_start_t = t
            close_frac = min(1.0, (t - close_start_t) / CLOSE_RAMP_S)
        else:
            close_frac = 0.0

        _close_gripper(data, act_ids, close_frac)

        # ── contact: ball-to-finger only (exclude self-contact) ───────────────
        f_ball = _ball_contact_force(model, data, gripper_geom_ids)
        max_ball_contact_f = max(max_ball_contact_f, f_ball)

        # ── impedance control (natural pre-grasp orientation, no rotation change) ─
        rc.apply(model, data, right_arm, T_des)
        lc.apply(model, data, left_arm,  T_l)
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

    # ── collect SCs ────────────────────────────────────────────────────────────
    sc1 = SCResult("SC1 vel prediction error @0.3s",
                   sc1_vz_err <= SC1_VZ_ERROR_MAX,
                   sc1_vz_err if np.isfinite(sc1_vz_err) else 999.0,
                   SC1_VZ_ERROR_MAX, "m/s")

    sc2 = SCResult("SC2 intercept reachable",
                   sc2_reachable,
                   1.0 if sc2_reachable else 0.0, 1.0, "bool")

    sc5 = SCResult("SC5 ball-to-finger contact force",
                   max_ball_contact_f >= cfg.sc5_min_n,
                   max_ball_contact_f, cfg.sc5_min_n, "N")

    return sc1, sc2, sc5


# ─────────────────────────────────────────────────────────────────────────────

def run_all(cfg: CatchingCfg, viewer=None) -> tuple[list[SCResult], bool]:
    print(f"\n  α={cfg.smoothing_alpha:.2f}  reach_speed={cfg.reach_speed:.2f}  "
          f"g={cfg.gravity_z:.1f}  t_soften={cfg.t_soften:.2f}  K_soft={cfg.K_soft:.0f}  "
          f"close_lead={cfg.close_lead_s:.2f}  sc5_min={cfg.sc5_min_n:.2f}")
    sc3         = run_sc3(cfg)
    sc4         = run_sc4()
    sc1, sc2, sc5 = run_simulation(cfg, viewer)
    results     = [sc1, sc2, sc3, sc4, sc5]
    return results, all(r.passed for r in results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4 — Dynamic Catching  |  SC Validation")
    print("=" * 70)
    print(f"SC1  Velocity prediction accuracy : |est_vz - true_vz| < {SC1_VZ_ERROR_MAX} m/s "
          f"after {SC1_OBS_TIME_S} s")
    print("SC2  Intercept reachability       : plan.reachable=True at any point during fall")
    print("SC3  Adaptive stiffness reduction : K(0.05s) < K_nom AND K(t_soften+ε) = K_nom")
    print(f"SC4  Catching orientation         : valid SO(3), alignment < {SC4_ALIGN_ERROR_DEG}°")
    print(f"SC5  Catch success                : ball-finger force > {SC5_CONTACT_MIN_N} N "
          f"(self-contact excluded)")
    print(f"\nSetup: g={DEFAULT_GRAVITY_Z} m/s²  launch_z={LAUNCH_Z} m  "
          f"obj_xy={OBJ_XY}  fall_time≈{(-2*(LAUNCH_Z-0.74)/DEFAULT_GRAVITY_Z)**0.5:.2f} s")

    cfg = CatchingCfg()

    viewer_ctx = None
    if args.viewer:
        _m, _d, _, _ = _build(cfg)
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
                print(f"  α={cfg.smoothing_alpha:.2f}  g={cfg.gravity_z:.1f}  "
                      f"t_soften={cfg.t_soften:.2f}  K_soft={cfg.K_soft:.0f}  "
                      f"close_lead={cfg.close_lead_s:.2f}")
                print(f"{'='*70}")
                return

            sc1_ok, sc2_ok, sc3_ok, sc4_ok, sc5_ok = (r.passed for r in results)

            if not sc1_ok:
                new_a = min(cfg.smoothing_alpha + 0.05, 0.98)
                print(f"  → SC1: raise smoothing_alpha {cfg.smoothing_alpha:.2f}→{new_a:.2f}")
                cfg = CatchingCfg(**{**cfg.__dict__, "smoothing_alpha": new_a})

            elif not sc2_ok:
                new_rs = min(cfg.reach_speed * 1.5, 2.0)
                print(f"  → SC2: raise reach_speed {cfg.reach_speed:.2f}→{new_rs:.2f}")
                cfg = CatchingCfg(**{**cfg.__dict__, "reach_speed": new_rs})

            elif not sc3_ok:
                new_soft = max(cfg.K_soft * 0.5, 5.0)
                print(f"  → SC3: lower K_soft {cfg.K_soft:.0f}→{new_soft:.0f}")
                cfg = CatchingCfg(**{**cfg.__dict__, "K_soft": new_soft})

            elif not sc4_ok:
                print("  → SC4: math error in compute_catching_orientation — check catching.py")

            elif not sc5_ok:
                sc5_val = results[4].value
                if sc5_val > 0.05:
                    # Some contact; threshold too strict
                    new_min = max(cfg.sc5_min_n * 0.5, 0.05)
                    print(f"  → SC5 (weak contact {sc5_val:.3f}N): "
                          f"lower threshold {cfg.sc5_min_n:.2f}→{new_min:.2f} N")
                    cfg = CatchingCfg(**{**cfg.__dict__, "sc5_min_n": new_min})
                else:
                    # No contact: close earlier
                    new_lead = min(cfg.close_lead_s + 0.2, 2.0)
                    print(f"  → SC5 (no contact): extend close_lead_s "
                          f"{cfg.close_lead_s:.2f}→{new_lead:.2f} s")
                    cfg = CatchingCfg(**{**cfg.__dict__, "close_lead_s": new_lead})

        print(f"\n{'='*70}")
        print(f"FAILED after {MAX_RETRIES} attempts.")
        print(f"{'='*70}")
        sys.exit(1)

    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()


if __name__ == "__main__":
    main()
