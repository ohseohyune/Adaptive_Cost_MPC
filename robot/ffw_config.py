"""
ROBOTIS FFW dual-arm configuration.

This file is intentionally robot-specific.  The generic CLIK logic lives in
the control/clik package and only consumes these names after they are resolved
into MuJoCo IDs.
"""

from dataclasses import dataclass

import numpy as np

from control.clik import SerialArmConfig


@dataclass
class GripperConfig:
    """Configuration for one parallel-jaw gripper (RH-P12-RN)."""
    name: str
    joint_names: tuple
    actuator_names: tuple
    finger_body_names: tuple  # bodies whose geoms can contact objects
    open_ctrl: np.ndarray     # ctrl for fully open
    close_ctrl: np.ndarray    # ctrl for fully closed


RIGHT_ARM_JOINT_NAMES = tuple(f"arm_r_joint{i}" for i in range(1, 8))
LEFT_ARM_JOINT_NAMES = tuple(f"arm_l_joint{i}" for i in range(1, 8))

RIGHT_ARM_ACTUATOR_NAMES = tuple(f"actuator_arm_r_joint{i}" for i in range(1, 8))
LEFT_ARM_ACTUATOR_NAMES = tuple(f"actuator_arm_l_joint{i}" for i in range(1, 8))


FFW_GRIPPERS = {
    "right": GripperConfig(
        name="right",
        joint_names=("gripper_r_joint1", "gripper_r_joint2",
                     "gripper_r_joint3", "gripper_r_joint4"),
        actuator_names=tuple(f"actuator_gripper_r_joint{i}" for i in range(1, 5)),
        finger_body_names=("gripper_r_rh_p12_rn_r1", "gripper_r_rh_p12_rn_r2",
                           "gripper_r_rh_p12_rn_l1", "gripper_r_rh_p12_rn_l2"),
        open_ctrl=np.zeros(4),
        close_ctrl=np.array([1.1, 1.0, 1.1, 1.0]),
    ),
    "left": GripperConfig(
        name="left",
        joint_names=("gripper_l_joint1", "gripper_l_joint2",
                     "gripper_l_joint3", "gripper_l_joint4"),
        actuator_names=tuple(f"actuator_gripper_l_joint{i}" for i in range(1, 5)),
        finger_body_names=("gripper_l_rh_p12_rn_r1", "gripper_l_rh_p12_rn_r2",
                           "gripper_l_rh_p12_rn_l1", "gripper_l_rh_p12_rn_l2"),
        open_ctrl=np.zeros(4),
        close_ctrl=np.array([1.1, 1.0, 1.1, 1.0]),
    ),
}

FFW_ARMS = {
    "right": SerialArmConfig(
        name="right",
        joint_names=RIGHT_ARM_JOINT_NAMES,
        actuator_names=RIGHT_ARM_ACTUATOR_NAMES,
        end_effector_body_name="arm_r_link7",
        end_effector_site_name="right_ee",
    ),
    "left": SerialArmConfig(
        name="left",
        joint_names=LEFT_ARM_JOINT_NAMES,
        actuator_names=LEFT_ARM_ACTUATOR_NAMES,
        end_effector_body_name="arm_l_link7",
        end_effector_site_name="left_ee",
    ),
}
