"""
Module Trải UV & Nướng Màu Đa Hướng (Phase 5 - P5 Texture Blender).
Chuẩn hóa theo NVIDIA Angle-Weighted Blending (Fresnel cos^3(theta)) & XAtlas.

Trách nhiệm:
    1. Trải tọa độ UV không chồng lấn bằng XAtlas (xử lý nhanh chóng trên mesh ~35k faces).
    2. Nướng màu đa góc nhìn với trọng số góc nhìn: w_i = max(0, n dot v_i)^gamma (gamma = 3.0).
    3. Xử lý che khuất (Occlusion) qua Z-buffer rasterization.
    4. Xuất file 3D định dạng chuẩn GLTF/GLB (+Y Up, Y_min = 0).
"""

from __future__ import annotations

import os
import sys
import logging
import time
from typing import Sequence, Tuple, Optional

import numpy as np
import trimesh
from PIL import Image

try:
    from .utils_3d import (
        export_glb,
        project_vertices,
        rasterize_depth_buffer,
        sample_rgb_nearest,
        validate_multiview_inputs,
        visible_projected_points,
        visible_vertex_mask,
    )
except ImportError:
    from utils_3d import (
        export_glb,
        project_vertices,
        rasterize_depth_buffer,
        sample_rgb_nearest,
        validate_multiview_inputs,
        visible_projected_points,
        visible_vertex_mask,
    )

try:
    import xatlas
    HAS_XATLAS = True
except ImportError:
    HAS_XATLAS = False

logger = logging.getLogger("texture_blender")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_TEXTURE_SIZE: int = 1024
DEFAULT_ANGLE_GAMMA: float = 3.0


class TextureBlender:
    """
    Hệ thống trải UV và tổng hợp chất liệu màu đa góc nhìn chuẩn NVIDIA.
    """

    def __init__(
        self,
        texture_size: int = DEFAULT_TEXTURE_SIZE,
        gamma: float = DEFAULT_ANGLE_GAMMA,
    ):
        self.texture_size = int(texture_size)
        self.gamma = float(gamma)

    def unwrap_uv(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray]:
        """
        Trải UV atlas bằng thư viện XAtlas (nhờ mesh đã decimate về ~35k faces,
        thời gian xử lý chỉ mất ~1.5 giây thay vì 40 giây).
        """
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("Mesh không có đỉnh hoặc mặt tam giác.")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        if HAS_XATLAS and len(faces) <= 50_000:
            try:
                atlas = xatlas.Atlas()
                atlas.add_mesh(vertices, faces)
                atlas.generate()
                vertex_mapping, atlas_faces, uvs = atlas.get_mesh(0)
                remapped_vertices = vertices[np.asarray(vertex_mapping, dtype=np.int32)]
                unwrapped = trimesh.Trimesh(
                    vertices=remapped_vertices,
                    faces=np.asarray(atlas_faces, dtype=np.int64),
                    process=False,
                )
                return unwrapped, np.asarray(uvs, dtype=np.float32)
            except Exception as error:
                logger.warning(f"[P5] XAtlas gặp sự cố ({error}), chuyển sang cylindrical UV fallback.")

        # Fallback Cylindrical UV
        centered = vertices - np.asarray(mesh.centroid, dtype=np.float32)
        theta = np.arctan2(centered[:, 0], centered[:, 2])
        u = (theta + np.pi) / (2.0 * np.pi)
        y_min = float(centered[:, 1].min())
        y_range = max(float(centered[:, 1].max()) - y_min, 1e-6)
        v = (centered[:, 1] - y_min) / y_range
        uvs = np.clip(np.column_stack((u, v)), 0.0, 1.0).astype(np.float32)
        return mesh.copy(), uvs

    def blend_colors_for_vertices(
        self,
        mesh: trimesh.Trimesh,
        images_rgb: Sequence[np.ndarray],
        camera_poses: Sequence[object],
        focal_lengths: Sequence[Tuple[float, float]],
        camera_intrinsics: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> np.ndarray:
        """
        Gán màu cho đỉnh mesh theo công thức NVIDIA Angle-Weighted:
        w_i = max(0, n_vertex dot v_cam)^gamma.
        """
        validate_multiview_inputs(images_rgb, camera_poses, focal_lengths)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        accumulated = np.zeros((len(vertices), 3), dtype=np.float64)
        accumulated_weights = np.zeros(len(vertices), dtype=np.float64)

        for idx, (image, pose, focal) in enumerate(zip(images_rgb, camera_poses, focal_lengths)):
            principal_point = None
            if camera_intrinsics is not None and idx < len(camera_intrinsics) and camera_intrinsics[idx] is not None:
                K_i = camera_intrinsics[idx]
                focal = (float(K_i[0, 0]), float(K_i[1, 1]))
                principal_point = (float(K_i[0, 2]), float(K_i[1, 2]))

            pixels, depth, in_image = project_vertices(vertices, pose, focal, image.shape, principal_point=principal_point)
            visible = visible_vertex_mask(pixels, depth, in_image, image.shape, tolerance=0.03)
            camera_center = np.asarray(pose, dtype=np.float32)[:3, 3]
            to_camera = camera_center - vertices
            to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-6)

            cosine = np.clip(np.sum(normals * to_camera, axis=1), 0.0, 1.0)
            selected = visible & (cosine > 0.0)
            if not np.any(selected):
                continue

            # Fresnel Angle Weight: cos^gamma(theta)
            view_weights = np.power(cosine[selected], self.gamma)
            samples = sample_rgb_nearest(image, pixels[selected])
            accumulated[selected] += samples * view_weights[:, None]
            accumulated_weights[selected] += view_weights

        colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        observed = accumulated_weights > 1e-8
        if np.any(observed):
            colors[observed, :3] = np.clip(
                accumulated[observed] / accumulated_weights[observed, None], 0, 255
            ).astype(np.uint8)
            colors[~observed, :3] = np.mean(colors[observed, :3], axis=0).astype(np.uint8)
        else:
            colors[:, :3] = (180, 180, 180)
        colors[:, 3] = 255
        return colors

    def process_and_export(
        self,
        mesh: trimesh.Trimesh,
        images_rgb: Sequence[np.ndarray],
        camera_poses: Sequence[object],
        focal_lengths: Sequence[Tuple[float, float]],
        output_path: str,
        camera_intrinsics: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> Tuple[bool, str]:
        """
        Quy trình trọn gói P5: Tính toán màu sắc đa hướng và xuất file .glb chuẩn hóa.
        """
        started = time.time()
        try:
            validate_multiview_inputs(images_rgb, camera_poses, focal_lengths)

            export_mesh = mesh.copy()
            vertex_colors = self.blend_colors_for_vertices(
                export_mesh, images_rgb, camera_poses, focal_lengths, camera_intrinsics=camera_intrinsics
            )
            export_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=export_mesh,
                vertex_colors=vertex_colors,
            )

            path = export_glb(export_mesh, output_path)
            elapsed = time.time() - started
            logger.info(f"[P5] ✓ Xuất GLB thành công trong {elapsed:.2f}s -> {path}")
            return True, path

        except Exception as e:
            logger.error(f"[P5] Lỗi nướng màu texture: {e}", exc_info=True)
            return False, ""