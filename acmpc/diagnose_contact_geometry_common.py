"""Single shared definition of the FACE-candidate / TIP_EDGE-candidate
contact-mode geometry (d_A, d_B, m = d_B - d_A), in raw SI meters
throughout, used by every later diagnostic script instead of each one
reimplementing its own copy.

All internal computation and storage is in meters. Callers convert to
mm/cm only at the point of printing, never internally.

d_A = pad wide face  (pad_local x = +/-pad_half_x, spans y,z)
      <-> box grasp face (box_local y = +/-box_half_y, spans x,z)
d_B = pad tip face    (pad_local z = +/-pad_half_z, spans x,y)
      <-> box edge     (box_local x = -box_half_x, z = +/-box_half_z,
                        varies along y)
m   = d_B - d_A   (m>0: FACE candidate closer; m<0: TIP_EDGE candidate closer)

Pad-face / pad-tip-face sign and box-edge sign are chosen every call by
proximity to the *other* geom's center along the relevant axis -- never a
fixed left/right convention.

Read-only geometry helper -- does not touch production control, target,
predictor, trigger, phase gate, pad orientation, controller gain, or
trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def face_grid(center: np.ndarray, rot: np.ndarray, fixed_axis: int, fixed_value: float,
              span_axes: tuple[int, int], spans: tuple[float, float], n: int) -> np.ndarray:
    a0, a1 = span_axes
    h0, h1 = spans
    u = np.linspace(-h0, h0, n)
    v = np.linspace(-h1, h1, n)
    U, V = np.meshgrid(u, v)
    local = np.zeros((n * n, 3))
    local[:, fixed_axis] = fixed_value
    local[:, a0] = U.ravel()
    local[:, a1] = V.ravel()
    return center + local @ rot.T


def edge_grid(center: np.ndarray, rot: np.ndarray, fixed_axes: tuple[int, int],
              fixed_values: tuple[float, float], span_axis: int, span_half: float, n: int) -> np.ndarray:
    a0, a1 = fixed_axes
    v0, v1 = fixed_values
    u = np.linspace(-span_half, span_half, n)
    local = np.zeros((n, 3))
    local[:, a0] = v0
    local[:, a1] = v1
    local[:, span_axis] = u
    return center + local @ rot.T


@dataclass
class ContactModeGeometry:
    resolution: int
    side: str
    dist_a_m: float
    dist_b_m: float
    margin_m: float
    grid_index_a: tuple[int, int]
    grid_index_b: tuple[int, int]
    pad_x_sign: float
    pad_z_sign: float
    box_z_sign: float
    closest_a_world: np.ndarray
    closest_b_world: np.ndarray

    @property
    def margin_mm(self) -> float:
        return 1000.0 * self.margin_m

    @property
    def margin_cm(self) -> float:
        return 100.0 * self.margin_m


def compute_contact_mode_geometry(
    box_center: np.ndarray, box_rot: np.ndarray, box_half: np.ndarray,
    pad_center: np.ndarray, pad_rot: np.ndarray, pad_half: np.ndarray,
    side: str, resolution: int,
) -> ContactModeGeometry:
    y_sign = 1.0 if side == "left" else -1.0

    face_plus = pad_center + pad_rot[:, 0] * pad_half[0]
    face_minus = pad_center - pad_rot[:, 0] * pad_half[0]
    pad_x_sign = 1.0 if abs(face_plus[0] - box_center[0]) < abs(face_minus[0] - box_center[0]) else -1.0
    pad_wide_face = face_grid(pad_center, pad_rot, 0, pad_x_sign * pad_half[0], (1, 2), (pad_half[1], pad_half[2]), resolution)
    box_grasp_face = face_grid(box_center, box_rot, 1, y_sign * box_half[1], (0, 2), (box_half[0], box_half[2]), resolution)
    diff_a = pad_wide_face[:, None, :] - box_grasp_face[None, :, :]
    dist_a_mat = np.linalg.norm(diff_a, axis=2)
    ia, ja = np.unravel_index(np.argmin(dist_a_mat), dist_a_mat.shape)
    dist_a = float(dist_a_mat[ia, ja])
    closest_a = (pad_wide_face[ia] + box_grasp_face[ja]) / 2.0

    tip_plus = pad_center + pad_rot[:, 2] * pad_half[2]
    tip_minus = pad_center - pad_rot[:, 2] * pad_half[2]
    pad_z_sign = 1.0 if abs(tip_plus[2] - box_center[2]) < abs(tip_minus[2] - box_center[2]) else -1.0
    pad_tip_face = face_grid(pad_center, pad_rot, 2, pad_z_sign * pad_half[2], (0, 1), (pad_half[0], pad_half[1]), resolution)

    box_near_x_sign = -1.0
    box_edge_plus = box_center + box_rot[:, 2] * box_half[2] + box_rot[:, 0] * (box_near_x_sign * box_half[0])
    box_edge_minus = box_center - box_rot[:, 2] * box_half[2] + box_rot[:, 0] * (box_near_x_sign * box_half[0])
    box_z_sign = 1.0 if abs(box_edge_plus[2] - pad_center[2]) < abs(box_edge_minus[2] - pad_center[2]) else -1.0
    box_edge = edge_grid(box_center, box_rot, (0, 2), (box_near_x_sign * box_half[0], box_z_sign * box_half[2]), 1, box_half[1], resolution)
    diff_b = pad_tip_face[:, None, :] - box_edge[None, :, :]
    dist_b_mat = np.linalg.norm(diff_b, axis=2)
    ib, jb = np.unravel_index(np.argmin(dist_b_mat), dist_b_mat.shape)
    dist_b = float(dist_b_mat[ib, jb])
    closest_b = (pad_tip_face[ib] + box_edge[jb]) / 2.0

    margin = dist_b - dist_a
    return ContactModeGeometry(
        resolution=resolution,
        side=side,
        dist_a_m=dist_a,
        dist_b_m=dist_b,
        margin_m=margin,
        grid_index_a=(int(ia), int(ja)),
        grid_index_b=(int(ib), int(jb)),
        pad_x_sign=pad_x_sign,
        pad_z_sign=pad_z_sign,
        box_z_sign=box_z_sign,
        closest_a_world=closest_a,
        closest_b_world=closest_b,
    )
