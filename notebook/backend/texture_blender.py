"""
texture_blender.py — P5: UV Unwrapping & Angle-Weighted Texture Blending
=========================================================================
Chuc nang:
  1. Trai phang UV bang thu vien XAtlas (qua goi xatlas-python).
  2. Hoa tron mau da goc nhin theo trong so Fresnel cos^3(theta):
       weight_i = max(cos(theta_i), 0)^3
     Trong do theta_i la goc giua phap tuyen dinh va tia nhin tu camera i.
  3. Xuat anh kien truc texture (texture atlas PNG) va mesh voi UV coords.
  4. Dong goi ket qua cho export GLB (P6).

Dong goi mau:
  - Mau dinh (vertex colors): dung truc tiep khong can atlas.
  - Mau texture (UV atlas): chinh xac hon, can XAtlas.

Rang buoc:
  - XAtlas: xu ly ~35,000 mat trong ~1.5 giay tren CPU i5-12450HX.
  - Fallback: Neu XAtlas khong kha dung, dung vertex colors truc tiep.
"""

from __future__ import annotations

import warnings
from typing import Optional

import cv2
import numpy as np

from .utils_3d import (
    CameraIntrinsics,
    compute_normals,
    project_points_batch,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEXTURE_SIZE   = 1024   # Kich thuoc texture atlas (pixels)
FRESNEL_POWER  = 3      # So mu Fresnel weight: cos^3(theta)

# ---------------------------------------------------------------------------
# 1. UV UNWRAPPING — XAtlas
# ---------------------------------------------------------------------------

def unwrap_uv_xatlas(
    vertices: np.ndarray,
    faces:    np.ndarray,
) -> tuple:
    """Trai phang UV bang XAtlas.

    XAtlas tao ra bang do UV (atlas) voi min overlap, phu hop cho
    texture baking tren mesh khat dang.

    Args:
        vertices: (V, 3) float32.
        faces:    (F, 3) int32.

    Returns:
        (uv_coords, uv_faces)
        - uv_coords: (V_new, 2) float32 toa do UV [0, 1].
        - uv_faces:  (F, 3) int32 chi so dinh UV.

    Raises:
        ImportError: Neu xatlas khong duoc cai dat.
        RuntimeError: Neu XAtlas that bai.
    """
    try:
        import xatlas
    except ImportError:
        raise ImportError(
            "Can cai dat xatlas: pip install xatlas\n"
            "Hoac: pip install xatlas-python"
        )

    vmapping, uv_faces, uv_coords = xatlas.parametrize(
        positions=vertices.astype(np.float32),
        indices=faces.astype(np.uint32),
    )
    return uv_coords.astype(np.float32), uv_faces.astype(np.int32)


def unwrap_uv_fallback(vertices: np.ndarray, faces: np.ndarray) -> tuple:
    """Fallback UV unwrap khi XAtlas khong kha dung.

    Phuong phap don gian: Spherical projection UV.
    Chat luong thap hon nhung dam bao chay duoc.
    """
    # Center + normalize
    center = vertices.mean(axis=0)
    v_c    = vertices - center
    r = np.linalg.norm(v_c, axis=1, keepdims=True) + 1e-9
    v_n = v_c / r
    u_coord = 0.5 + np.arctan2(v_n[:, 2], v_n[:, 0]) / (2 * np.pi)
    v_coord = 0.5 - np.arcsin(np.clip(v_n[:, 1], -1, 1)) / np.pi
    uv_coords = np.stack([u_coord, v_coord], axis=1).astype(np.float32)
    return uv_coords, faces.astype(np.int32)


# ---------------------------------------------------------------------------
# 2. ANGLE-WEIGHTED COLOR BLENDING — Fresnel cos^3(theta)
# ---------------------------------------------------------------------------

def compute_view_weights(
    vertices:  np.ndarray,
    normals:   np.ndarray,
    Rs:        list,
    ts:        list,
) -> np.ndarray:
    """Tinh trong so Fresnel cos^3(theta) cho moi dinh theo tung view.

    Cong thuc:
        view_dir_i(v) = normalize(cam_pos_i - vertex_v)
        cos_theta_i   = dot(normal_v, view_dir_i(v))
        weight_i(v)   = max(cos_theta_i, 0)^3

    Args:
        vertices: (V, 3) float32.
        normals:  (V, 3) float32 phap tuyen don vi.
        Rs:       Danh sach N ma tran quay (3, 3).
        ts:       Danh sach N vector tinh tien (3,).

    Returns:
        weights: (N, V) float32 — trong so cho moi view va moi dinh.
    """
    V = len(vertices)
    N = len(Rs)
    weights = np.zeros((N, V), dtype=np.float32)

    for i, (Rm, t) in enumerate(zip(Rs, ts)):
        # Vi tri camera trong world: cam_pos = -R^T * t
        cam_pos = -(Rm.T @ np.asarray(t, dtype=np.float64))
        cam_pos = cam_pos.astype(np.float32)

        # Huong nhin tu moi dinh den camera
        view_dirs = cam_pos[None, :] - vertices       # (V, 3)
        norms     = np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-9
        view_dirs /= norms

        # Goc cos
        cos_theta = np.einsum('vd,vd->v', normals, view_dirs)   # (V,)
        cos_theta = np.clip(cos_theta, 0.0, 1.0)

        weights[i] = cos_theta ** FRESNEL_POWER

    return weights


def sample_color_at_vertex(
    vertex:    np.ndarray,
    K:         CameraIntrinsics,
    R:         np.ndarray,
    t:         np.ndarray,
    image_rgb: np.ndarray,
) -> Optional[np.ndarray]:
    """Lay mau mau sac tai toa do pixel tuong ung voi dinh 3D.

    Args:
        vertex:    (3,) float32 toa do dinh.
        K:         CameraIntrinsics da bu tru K'.
        R, t:      Camera pose.
        image_rgb: (H, W, 3) uint8 RGB.

    Returns:
        (3,) float32 mau RGB [0, 1], hoac None neu ngoai canvas.
    """
    Xc    = R @ vertex.astype(np.float64) + t
    depth = Xc[2]
    if depth <= 0:
        return None
    Km  = K.as_matrix()
    uvh = Km @ Xc
    u   = uvh[0] / (uvh[2] + 1e-9)
    v   = uvh[1] / (uvh[2] + 1e-9)
    u_i = int(round(u))
    v_i = int(round(v))
    h, w = image_rgb.shape[:2]
    if 0 <= u_i < w and 0 <= v_i < h:
        return image_rgb[v_i, u_i].astype(np.float32) / 255.0
    return None


def blend_vertex_colors(
    vertices:    np.ndarray,
    normals:     np.ndarray,
    images_rgb:  list,
    K_primes:    list,
    Rs:          list,
    ts:          list,
) -> np.ndarray:
    """Hoa tron mau dinh tu nhieu view bang trong so Fresnel cos^3(theta).

    Thuat toan:
      For each vertex v:
        color_v = sum_i(weight_i(v) * color_i(v)) / sum_i(weight_i(v))

    Args:
        vertices:   (V, 3) float32.
        normals:    (V, 3) float32.
        images_rgb: Danh sach N anh (H, W, 3) uint8 RGB.
        K_primes:   Danh sach N CameraIntrinsics.
        Rs, ts:     Danh sach N poses.

    Returns:
        colors: (V, 3) float32 RGB [0, 1].
    """
    V = len(vertices)
    N = len(images_rgb)
    weights = compute_view_weights(vertices, normals, Rs, ts)   # (N, V)

    color_acc  = np.zeros((V, 3), dtype=np.float64)
    weight_acc = np.zeros(V, dtype=np.float64)

    for i, (img_rgb, K, Rm, t) in enumerate(zip(images_rgb, K_primes, Rs, ts)):
        # Vectorize: chieu tat ca dinh cung mot luc
        uvs, valid = project_points_batch(vertices, K, Rm, t)
        h, w = img_rgb.shape[:2]

        u_int = np.round(uvs[:, 0]).astype(np.int32)
        v_int = np.round(uvs[:, 1]).astype(np.int32)

        in_frame = (
            valid &
            (u_int >= 0) & (u_int < w) &
            (v_int >= 0) & (v_int < h)
        )

        w_i = weights[i]   # (V,)
        active = in_frame & (w_i > 1e-6)

        idx = np.where(active)[0]
        if len(idx) > 0:
            sampled_colors = img_rgb[v_int[idx], u_int[idx]].astype(np.float64) / 255.0
            color_acc[idx]  += w_i[idx, None] * sampled_colors
            weight_acc[idx] += w_i[idx]

    # Trung binh co trong so
    safe_w = np.where(weight_acc > 1e-9, weight_acc, 1.0)
    colors = (color_acc / safe_w[:, None]).astype(np.float32)
    colors = np.clip(colors, 0.0, 1.0)
    return colors


# ---------------------------------------------------------------------------
# 3. PIPELINE P5 — ENTRY POINT
# ---------------------------------------------------------------------------

def apply_texture(
    vertices:    np.ndarray,
    faces:       np.ndarray,
    depth_views: list,
    Rs:          list,
    ts:          list,
    use_xatlas:  bool = True,
) -> dict:
    """Thuc hien toan bo quy trinh P5: UV + Texture Blending.

    Args:
        vertices:    (V, 3) float32.
        faces:       (F, 3) int32.
        depth_views: Ket qua tu estimate_depth_pipeline() (da qua P3, P4).
        Rs, ts:      Camera poses (dong nhat voi P4).
        use_xatlas:  Co dung XAtlas hay fallback.

    Returns:
        dict chua:
          {
            "vertices":    (V, 3) float32,
            "faces":       (F, 3) int32,
            "normals":     (V, 3) float32,
            "vertex_colors": (V, 3) float32,
            "uv_coords":   (V_uv, 2) float32 hoac None,
            "uv_faces":    (F, 3) int32 hoac None,
          }
    """
    images_rgb = [v["canvas_rgb"] for v in depth_views]
    K_primes   = [v["K_prime"]    for v in depth_views]

    # Tinh phap tuyen dinh
    normals = compute_normals(vertices, faces)

    # Blend mau dinh
    vertex_colors = blend_vertex_colors(
        vertices, normals, images_rgb, K_primes, Rs, ts
    )

    # UV Unwrap
    uv_coords, uv_faces = None, None
    if use_xatlas:
        try:
            uv_coords, uv_faces = unwrap_uv_xatlas(vertices, faces)
        except ImportError:
            warnings.warn(
                "XAtlas khong kha dung. Fallback sang Spherical UV.",
                RuntimeWarning,
            )
            uv_coords, uv_faces = unwrap_uv_fallback(vertices, faces)
        except RuntimeError as e:
            warnings.warn(f"XAtlas loi: {e}. Fallback sang Spherical UV.", RuntimeWarning)
            uv_coords, uv_faces = unwrap_uv_fallback(vertices, faces)
    else:
        uv_coords, uv_faces = unwrap_uv_fallback(vertices, faces)

    return {
        "vertices":       vertices,
        "faces":          faces,
        "normals":        normals,
        "vertex_colors":  vertex_colors,
        "uv_coords":      uv_coords,
        "uv_faces":       uv_faces,
    }
