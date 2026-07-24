"""Integrated grasp-and-lift control loop.

Pipeline per simulation step
-----------------------------
    1. Read state    (x_EE, x_obj, contacts)
    2. GraspSM       → GraspState
    3. CartesianMPC  → x_target_next    (where to move, MPC cost ← GraspState)
    4. Impedance     → joint ctrl       (how to get there, K_pos ← GraspState)
    5. Gripper       → open / close     (by GraspState)
    6. mj_step

Scene: scene_ffw_sg2_fixed_base_grasp.xml
Task:  right arm APPROACH → PRE_CONTACT → GRASPING → GRASPED → hold lifted
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.clik import build_serial_arm
from control.clik.catching import adaptive_stiffness
from control.clik.contact import finger_geom_ids, get_contact_forces, total_normal_force
from control.clik.grasp_state_machine import GraspSMConfig, GraspState, GraspStateMachine
from control.clik.impedance import (
    CartesianImpedanceConfig,
    CartesianImpedanceController,
)
from control.clik.kinematics import get_ee_transform
from control.mpc.cartesian_mpc import CartesianMPC, CartesianMPCConfig
from control.mpc.neural_cost_map import pretrained_neural_cost_map
from robot.ffw_config import FFW_ARMS, FFW_GRIPPERS

# ── scene ─────────────────────────────────────────────────────────────────────

SCENE_XML            = "scene_ffw_sg2_fixed_base_grasp.xml"
OBJ_BODY             = "grasp_object"
APPROACH_DZ          = 0.08    # pre-grasp height above object [m]
LIFT_DZ              = 0.10    # target lift height after grasping [m]
USE_NEURAL_COST_MAP  = True    # False → fall back to COST_WEIGHTS lookup table

# ── EE position offset relative to object, per GraspState ────────────────────

_POS_OFFSET: dict[GraspState, np.ndarray] = {
    GraspState.APPROACH:     np.array([0.0, 0.0,  APPROACH_DZ]),
    GraspState.PRE_CONTACT:  np.array([0.0, 0.0,  0.0]),
    GraspState.GRASPING:     np.array([0.0, 0.0,  0.0]),
    GraspState.GRASPED:      np.array([0.0, 0.0,  LIFT_DZ]),
    GraspState.MANIPULATION: np.array([0.0, 0.0,  LIFT_DZ]),
}

# ── desired contact force [N, world frame], per GraspState ────────────────────

_F_DESIRED: dict[GraspState, np.ndarray] = {
    GraspState.APPROACH:     np.zeros(3),
    GraspState.PRE_CONTACT:  np.array([0.0, 0.0, -2.0]),
    GraspState.GRASPING:     np.array([0.0, 0.0, -5.0]),
    GraspState.GRASPED:      np.array([0.0, 0.0, -3.0]),
    GraspState.MANIPULATION: np.array([0.0, 0.0, -3.0]),
}

# ── Cartesian stiffness [N/m] (matches GraspSMConfig.K_pos_*) ────────────────

_K_POS: dict[GraspState, float] = {
    GraspState.APPROACH:     500.0,
    GraspState.PRE_CONTACT:  200.0,
    GraspState.GRASPING:     100.0,
    GraspState.GRASPED:      300.0,
    GraspState.MANIPULATION: 400.0,
}

_CLOSE_STATES = {GraspState.GRASPING, GraspState.GRASPED, GraspState.MANIPULATION}


def main() -> None:
    # ── load model ─────────────────────────────────────────────────────────────
    xml_path = PROJECT_ROOT / "model" / "robotis_ffw" / SCENE_XML
    model    = mujoco.MjModel.from_xml_path(str(xml_path))
    data     = mujoco.MjData(model)
    dt       = float(model.opt.timestep)

    # ── arm / gripper IDs ─────────────────────────────────────────────────────
    arm      = build_serial_arm(model, FFW_ARMS["right"])
    grip_cfg = FFW_GRIPPERS["right"]

    gripper_act_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in grip_cfg.actuator_names
    ])
    gripper_geom_ids = finger_geom_ids(model, grip_cfg.finger_body_names)
    obj_body_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY)

    # ── reset to keyframe 0 (arm home) ────────────────────────────────────────
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    T_init = get_ee_transform(data, arm)
    R_des  = T_init[:3, :3].copy()   # hold initial EE orientation throughout
    q_home = data.qpos[arm.qpos_indices].copy()

    # ── controllers ───────────────────────────────────────────────────────────
    impedance = CartesianImpedanceController(
        config=CartesianImpedanceConfig(K_pos=500.0, K_rot=50.0,
                                         D_pos=80.0,  D_rot=12.0),
        arm=arm,
        model=model,
        q_home=q_home,
    )

    sm = GraspStateMachine(GraspSMConfig())

    if USE_NEURAL_COST_MAP:
        print("Training NeuralCostMap on COST_WEIGHTS schedule…", flush=True)
        cost_map = pretrained_neural_cost_map(
            n_per_state=200, n_epochs=300, lr=2e-3, print_every=0,
        )
        print("NeuralCostMap ready.\n")
    else:
        cost_map = None

    mpc = CartesianMPC(
        config=CartesianMPCConfig(horizon=10, dt=dt, w_reg=0.005, v_max=0.15),
        cost_map=cost_map,
    )

    # ── startup info ─────────────────────────────────────────────────────────
    x_obj_init = data.xpos[obj_body_id].copy()
    print(f"Scene:  {SCENE_XML}")
    print(f"dt={dt*1e3:.1f} ms | horizon=10 | v_max=0.15 m/s")
    print(f"EE init:   {T_init[:3,3].round(4)}")
    print(f"Object:    {x_obj_init.round(4)}")
    print("GraspState transitions will be printed.")
    print("Close the viewer to stop.\n")

    # ── control loop ──────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            t_start = time.perf_counter()

            # 1. read state
            T_cur  = get_ee_transform(data, arm)
            x_ee   = T_cur[:3, 3].copy()
            x_obj  = data.xpos[obj_body_id].copy()

            # 2. grasp state machine
            contacts        = get_contact_forces(model, data, gripper_geom_ids)
            f_contact_total = total_normal_force(contacts)
            state, changed  = sm.update(x_ee, x_obj, contacts, dt, t=data.time)

            if changed:
                print(f"  t={data.time:.3f}s  {sm.history[-1].from_state.value}"
                      f" → {state.value}", flush=True)

            # 3. adaptive stiffness
            #    (static scene: ttc=inf → K = K_nom; catching scene: ttc finite)
            K_nom = _K_POS[state]
            K_pos = float(adaptive_stiffness(
                time_to_contact=float("inf"),   # update to real ttc for catching
                K_nominal=K_nom,
                K_soft=K_nom * 0.3,
                t_soften=0.1,
            ))

            # 4. CartesianMPC → x_target_next
            x_pos_tgt = x_obj + _POS_OFFSET[state]
            f_des     = _F_DESIRED[state]

            mpc_result = mpc.solve(
                grasp_state=state,
                x_ee=x_ee,
                x_target=x_pos_tgt,
                f_desired=f_des,
                x_object=x_obj,
                K_pos=K_pos,
                f_contact_total=f_contact_total,
                ttc=float("inf"),    # static scene; set to real ttc for catching
            )

            # 5. update impedance K_pos; build desired transform T_des
            impedance.K[3, 3] = K_pos
            impedance.K[4, 4] = K_pos
            impedance.K[5, 5] = K_pos
            impedance.config.K_pos = K_pos

            T_des = np.eye(4)
            T_des[:3, :3] = R_des
            T_des[:3,  3] = mpc_result.x_target_next

            # 6. impedance → joint ctrl
            impedance.apply(model, data, arm, T_des)

            # 7. gripper: close in contact states, open otherwise
            if state in _CLOSE_STATES:
                data.ctrl[gripper_act_ids] = grip_cfg.close_ctrl
            else:
                data.ctrl[gripper_act_ids] = grip_cfg.open_ctrl

            # 8. step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # periodic console log
            if step % 200 == 0:
                f_norm = f_contact_total
                e_norm = float(np.linalg.norm(x_ee - x_pos_tgt))
                print(
                    f"t={data.time:6.3f}s | {state.value:12s} | "
                    f"EE=[{x_ee[0]:.3f},{x_ee[1]:.3f},{x_ee[2]:.3f}] | "
                    f"|e|={e_norm:.4f}m | F={f_norm:.2f}N | "
                    f"K={K_pos:.0f}N/m | mpc={mpc_result.status}",
                    flush=True,
                )

            step += 1
            elapsed = time.perf_counter() - t_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
