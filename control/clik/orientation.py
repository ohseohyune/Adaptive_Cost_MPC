"""Orientation target based on rotation matrices.

"""

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector and reject near-zero directions."""
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def rotation_from_axis_alignment(
    local_primary_axis: np.ndarray,
    world_primary_axis: np.ndarray,
    local_secondary_axis: np.ndarray,
    world_secondary_axis: np.ndarray,
) -> np.ndarray:
    """
    직관적인 방향 지시(축 정렬)로 3차원 회전 행렬(R)을 만드는 함수 

    Build a full rotation matrix from two axis-alignment constraints.

    This is useful for grasp orientation targets such as:
    - gripper forward/palm axis points toward world +x
    - gripper up/finger axis points toward world +z

    The primary axis is matched exactly.  The secondary axis is projected onto
    the plane perpendicular to the primary axis before it is matched, so the
    two constraints define a valid orthonormal frame without Euler angles.

    The returned rotation R maps local end-effector vectors into world vectors:
        v_world = R @ v_local
    """

    local_basis = _basis_from_two_axes(local_primary_axis, local_secondary_axis)
    world_basis = _basis_from_two_axes(world_primary_axis, world_secondary_axis) # 목표물(Target)의 방향
    return world_basis @ local_basis.T


def _basis_from_two_axes(primary_axis: np.ndarray, secondary_axis: np.ndarray) -> np.ndarray:
    """Create an orthonormal basis from two non-collinear axes."""
    primary = normalize(primary_axis)                      # 주 축(Primary Axis) 정규화 -> 가장 중요하게 고정할 첫 번째 축을 단위 벡터로 만듦
    secondary_raw = np.asarray(secondary_axis, dtype=float).reshape(3)
    secondary_projected = secondary_raw - primary * np.dot(primary, secondary_raw)
    secondary = normalize(secondary_projected)             # 보조 축(Secondary Axis) 정규화 -> 두 번째 축을 주 축에 수직이 되도록 투영한 후 단위 벡터로 만듦
    tertiary = np.cross(primary, secondary)                # 세 번째 축(Tertiary Axis) 계산 -> 주 축과 보조 축의 외적을 통해 세 번째 축을 계산하여 직교하는 좌표계를 완성
    return np.column_stack([primary, secondary, tertiary]) # 직교하는 좌표계의 각 축을 열로 가지는 3x3 행렬 반환
