"""

ID lookup and serial-arm mapping
--------------------------------

get_arm_joint_indices
= 관절 상태를 읽기 위한 주소 찾기

get_arm_actuator_indices
= 관절 명령을 보낼 모터 주소 찾기

build_serial_arm
= 제어에 필요한 주소들을 SerialArm 하나로 묶기

"""

import mujoco
import numpy as np

from control.clik.kinematics import get_ee_transform
from control.clik.types import SerialArm, SerialArmConfig
from control.clik.utils import adjoint, transform_inverse


def _require_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return obj_id


def get_arm_joint_indices(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve joint IDs plus their qpos and qvel addresses.

    For this minimal serial-arm CLIK controller, every controlled joint must be
    a one-DOF hinge or slide joint. That gives one qpos entry and one qvel
    column per controlled joint, which keeps the Jacobian slicing explicit.
    """

    joint_ids = []
    qpos_indices = []
    qvel_indices = []

    one_dof_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }

    for joint_name in joint_names:
        joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in one_dof_types:
            raise ValueError(
                f"{joint_name} is not a one-DOF hinge/slide joint. "
                "This simple serial-arm CLIK helper does not handle ball/free joints."
            )

        joint_ids.append(joint_id)
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))

    return (
        np.asarray(joint_ids, dtype=int),
        np.asarray(qpos_indices, dtype=int),
        np.asarray(qvel_indices, dtype=int),
    )


def get_arm_actuator_indices(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
    actuator_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    """
    Resolve actuator IDs in the same order as joint_names.

    The XML actuator block can list actuators in any order. For example, FFW's
    right arm actuators appear before the left arm actuators, even though the
    left arm joints appear earlier in the body tree. Therefore the mapping is
    resolved by actuator/joint names instead of relying on contiguous IDs.
    """

    if actuator_names is not None:
        if len(actuator_names) != len(joint_names):
            raise ValueError("actuator_names must have the same length as joint_names.")
        actuator_ids = []
        for joint_name, actuator_name in zip(joint_names, actuator_names):
            joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = _require_id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            actuator_joint_id = int(model.actuator_trnid[actuator_id, 0])
            if actuator_joint_id != joint_id:
                raise ValueError(
                    f"{actuator_name} is connected to joint ID {actuator_joint_id}, "
                    f"not {joint_name}."
                )
            actuator_ids.append(actuator_id)
        return np.asarray(actuator_ids, dtype=int)

    actuator_ids = []
    for joint_name in joint_names:
        joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one actuator connected to {joint_name}, "
                f"found {len(matches)}."
            )
        actuator_ids.append(int(matches[0]))

    return np.asarray(actuator_ids, dtype=int)


def build_serial_arm(model: mujoco.MjModel, config: SerialArmConfig) -> SerialArm:
    """Resolve a SerialArmConfig into MuJoCo IDs used by the generic CLIK code."""

    joint_ids, qpos_indices, qvel_indices = get_arm_joint_indices(
        model, config.joint_names
    )
    actuator_ids = get_arm_actuator_indices(
        model, config.joint_names, config.actuator_names
    )

    ee_body_id = _require_id(
        model, mujoco.mjtObj.mjOBJ_BODY, config.end_effector_body_name
    )
    ee_site_id = None
    if config.end_effector_site_name is not None:
        ee_site_id = _require_id(
            model, mujoco.mjtObj.mjOBJ_SITE, config.end_effector_site_name
        )

    ctrl_low = model.actuator_ctrlrange[actuator_ids, 0].copy()
    ctrl_high = model.actuator_ctrlrange[actuator_ids, 1].copy()

    return SerialArm(
        config=config,
        joint_ids=joint_ids,
        actuator_ids=actuator_ids,
        qpos_indices=qpos_indices,
        qvel_indices=qvel_indices,
        ctrl_low=ctrl_low,
        ctrl_high=ctrl_high,
        ee_body_id=ee_body_id,
        ee_site_id=ee_site_id,
    )


def make_arm_home_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
) -> mujoco.MjData:
    """
    Build a temporary state with this arm at theta=0.

    Non-arm coordinates, such as a floating base or lift joint, are preserved so
    the PoE space frame follows the current robot support pose.
    """

    home_data = mujoco.MjData(model)
    home_data.qpos[:] = data.qpos
    home_data.qvel[:] = 0.0
    home_data.qpos[arm.qpos_indices] = 0.0
    mujoco.mj_forward(model, home_data)
    return home_data


def space_screw_axes_from_home(
    model: mujoco.MjModel,
    home_data: mujoco.MjData,
    arm: SerialArm,
) -> list[np.ndarray]:
    """Extract space-frame screw axes S_i from the MuJoCo home state."""

    screw_axes = []
    for joint_id in arm.joint_ids:
        joint_type = int(model.jnt_type[joint_id])
        axis = home_data.xaxis[joint_id].copy()

        if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            anchor = home_data.xanchor[joint_id].copy()
            linear = -np.cross(axis, anchor)
            screw_axis = np.concatenate([axis, linear])
        elif joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
            screw_axis = np.concatenate([np.zeros(3), axis])
        else:
            raise ValueError(
                "PoE Jacobian only supports one-DOF hinge/slide joints; "
                f"joint ID {joint_id} has MuJoCo type {joint_type}."
            )

        screw_axes.append(screw_axis)

    return screw_axes


def body_screw_axes_from_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: SerialArm,
) -> list[np.ndarray]:
    """Compute B_i = Ad_{M^-1} S_i for this arm's current support pose."""

    home_data = make_arm_home_data(model, data, arm)
    home_transform = get_ee_transform(home_data, arm)
    space_axes = space_screw_axes_from_home(model, home_data, arm)
    space_to_body = adjoint(transform_inverse(home_transform))
    return [space_to_body @ screw_axis for screw_axis in space_axes]
