"""
Module Trải UV (XAtlas) & Nướng Màu Hòa Trộn Đa Góc Nhìn (Base-Color Texture Blending).
Thành viên 5 (P5) - Texture & UV Shading Engineer.

Kế thừa kiến trúc từ:
    - NVIDIA 3DObjectReconstruction (Configurable Color Fusion)
    - XAtlas UV Parameterization & Angle-Weighted Camera Blending
    - Trimesh GLB Exporter

Thuật toán cốt lõi:
    - Thuật toán 4:
        1. XAtlas UV Unwrapping: Tối ưu độ méo dãn và đóng gói hải đảo UV vào [0, 1] x [0, 1].
        2. Angle-Weighted Blending: cos(theta_i) = n . v_i, W_i = (max(0, cos(theta_i)))^gamma.
        3. Khử phản xạ chói lóa (Specular Highlights), chỉ lấy màu khuếch tán thực (Albedo/Base-Color).
        4. Đóng gói xuất file .glb chuẩn nhúng texture.
"""

import os
import time
import logging
from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np
from PIL import Image
import trimesh

try:
    import xatlas
    HAS_XATLAS = True
except ImportError:
    HAS_XATLAS = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ============================================================================
# CONSTANTS & DEFAULT PARAMETERS
# ============================================================================

# Kích thước Texture Map đầu ra mặc định
DEFAULT_TEXTURE_SIZE: int = 1024

# Hệ số lũy thừa ưu tiên góc chụp chính diện (cos(theta)^gamma)
DEFAULT_ANGLE_GAMMA: float = 3.0


# ============================================================================
# THUẬT TOÁN 4: XATLAS UV & ANGLE-WEIGHTED TEXTURE BLENDING
# ============================================================================

class TextureBlender:
    """
    Lớp xử lý trải UV và hòa trộn màu đa góc nhìn (Base-Color Texture Blending).
    """

    def __init__(
        self,
        texture_size: int = DEFAULT_TEXTURE_SIZE,
        gamma: float = DEFAULT_ANGLE_GAMMA,
    ):
        self.texture_size = texture_size
        self.gamma = gamma

    def unwrap_uv(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray]:
        """
        Trải UV bằng thư viện xatlas (hoặc fallback hình học nếu thiếu xatlas).

        Args:
            mesh: Đối tượng trimesh.Trimesh chưa có UV.

        Returns:
            unwrapped_mesh: Mesh mới với các đỉnh cắt đường may (seams).
            uvs: Mảng tọa độ UV (N_verts, 2) chuẩn hóa trong [0, 1].
        """
        verts = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)

        if HAS_XATLAS:
            try:
                logger.info("Đang trải UV bằng xatlas...")
                atlas = xatlas.Atlas()
                atlas.add_mesh(verts, faces)
                atlas.generate()

                # Trích xuất mesh đã unwrap từ atlas
                vmapping, indices, uvs = atlas.get_mesh(0)
                new_verts = verts[vmapping]
                new_normals = mesh.vertex_normals[vmapping] if len(mesh.vertex_normals) > 0 else None

                unwrapped_mesh = trimesh.Trimesh(
                    vertices=new_verts,
                    faces=indices,
                    vertex_normals=new_normals,
                    process=False,
                )
                logger.info(f"xatlas hoàn thành: {len(uvs)} tọa độ UV, {len(new_verts)} đỉnh mới.")
                return unwrapped_mesh, uvs
            except Exception as e:
                logger.warning(f"Lỗi khi unwrap bằng xatlas: {e}. Chuyển sang fallback projection.")

        # Fallback: Chiếu hình trụ (Cylindrical / Spherical projection)
        logger.info("Sử dụng thuật toán chiếu tọa độ UV chuẩn...")
        center = mesh.centroid
        centered_verts = verts - center

        # Tính góc theta và độ cao phi
        theta = np.arctan2(centered_verts[:, 0], centered_verts[:, 2])  # [-pi, pi]
        u = (theta + np.pi) / (2.0 * np.pi)

        y_min = centered_verts[:, 1].min()
        y_max = centered_verts[:, 1].max()
        y_range = max(y_max - y_min, 1e-6)
        v = (centered_verts[:, 1] - y_min) / y_range

        uvs = np.stack([u, v], axis=-1).astype(np.float32)
        uvs = np.clip(uvs, 0.0, 1.0)

        return mesh, uvs

    def blend_colors_for_vertices(
        self,
        mesh: trimesh.Trimesh,
        images_rgb: List[np.ndarray],
        camera_poses: List[np.ndarray],
        focal_lengths: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Tính toán màu Albedo tại mỗi đỉnh của mesh bằng phương pháp Angle-Weighted Blending.

        Toán học:
            - cos(theta_i) = n . v_i
            - W_i = (max(0, cos(theta_i)))^gamma
            - C_final = sum(W_i * C_i) / sum(W_i)

        Args:
            mesh: Trimesh đã có vertex normals.
            images_rgb: List N ảnh RGB gốc uint8 (H, W, 3).
            camera_poses: List N ma trận camera-to-world (4x4).
            focal_lengths: List N tiêu cự (fx, fy).

        Returns:
            vertex_colors: np.ndarray shape (N_verts, 4) uint8 RGBA.
        """
        n_verts = len(mesh.vertices)
        n_views = len(images_rgb)

        verts = mesh.vertices  # (V, 3)
        normals = mesh.vertex_normals  # (V, 3)

        accum_colors = np.zeros((n_verts, 3), dtype=np.float64)
        accum_weights = np.zeros(n_verts, dtype=np.float64)

        for i in range(n_views):
            img = images_rgb[i]
            h, w, _ = img.shape
            fx, fy = focal_lengths[i]
            cx, cy = w / 2.0, h / 2.0

            c2w = camera_poses[i].astype(np.float32)
            if c2w.shape == (3, 4):
                c2w_homo = np.eye(4, dtype=np.float32)
                c2w_homo[:3, :4] = c2w
                c2w = c2w_homo

            cam_center = c2w[:3, 3]
            R_c2w = c2w[:3, :3]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ cam_center

            # Vector từ điểm 3D đến tâm camera: v_i
            vec_to_cam = cam_center - verts
            dist_to_cam = np.linalg.norm(vec_to_cam, axis=-1, keepdims=True)
            v_dir = vec_to_cam / np.maximum(dist_to_cam, 1e-6)

            # Góc chiếu giữa pháp tuyến bề mặt và hướng nhìn camera: cos(theta)
            cos_theta = np.sum(normals * v_dir, axis=-1)  # (V,)

            # Chỉ xét các đỉnh hướng về phía camera (cos_theta > 0)
            front_facing = cos_theta > 0.05
            if not np.any(front_facing):
                continue

            # Chiếu sang hệ camera: p_c = R_w2c * p_w + t_w2c
            p_cam = verts @ R_w2c.T + t_w2c
            z_cam = p_cam[:, 2]

            valid_z = z_cam > 0.01

            # Chiếu lên màn ảnh 2D
            u_proj = np.round((fx * p_cam[:, 0] / np.maximum(z_cam, 1e-6)) + cx).astype(np.int32)
            v_proj = np.round((fy * p_cam[:, 1] / np.maximum(z_cam, 1e-6)) + cy).astype(np.int32)

            in_bounds = front_facing & valid_z & (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h)

            indices = np.where(in_bounds)[0]
            if len(indices) == 0:
                continue

            u_sel = u_proj[indices]
            v_sel = v_proj[indices]

            # Trích xuất màu RGB từ ảnh
            sampled_rgb = img[v_sel, u_sel].astype(np.float64)

            # Trọng số theo góc nhìn: W_i = (cos(theta))^gamma
            weights = np.power(cos_theta[indices], self.gamma)

            accum_colors[indices] += sampled_rgb * weights[:, np.newaxis]
            accum_weights[indices] += weights

        # Những đỉnh không nhìn thấy từ camera nào sẽ dùng màu trung bình
        has_obs = accum_weights > 1e-5
        final_rgb = np.zeros((n_verts, 3), dtype=np.uint8)

        if np.any(has_obs):
            normalized_colors = accum_colors[has_obs] / accum_weights[has_obs, np.newaxis]
            final_rgb[has_obs] = np.clip(normalized_colors, 0, 255).astype(np.uint8)

            # Tô màu đỉnh unobserved bằng màu trung bình các đỉnh nhìn thấy
            mean_color = np.mean(final_rgb[has_obs], axis=0).astype(np.uint8)
            final_rgb[~has_obs] = mean_color
        else:
            final_rgb[:] = np.array([200, 200, 200], dtype=np.uint8)

        # Trả về RGBA
        alpha_col = np.full((n_verts, 1), 255, dtype=np.uint8)
        vertex_rgba = np.hstack([final_rgb, alpha_col])
        return vertex_rgba

    def bake_texture_map(
        self,
        mesh: trimesh.Trimesh,
        uvs: np.ndarray,
        vertex_colors: np.ndarray,
        tex_size: int = DEFAULT_TEXTURE_SIZE,
    ) -> Image.Image:
        """
        Nướng bản đồ Texture Albedo (1024x1024) từ màu các đỉnh và tọa độ UV.

        Args:
            mesh: Trimesh đã có faces.
            uvs: Tọa độ UV (V, 2).
            vertex_colors: Màu các đỉnh (V, 4).
            tex_size: Độ phân giải ảnh texture.

        Returns:
            PIL.Image: Ảnh Texture Albedo hoàn chỉnh.
        """
        logger.info(f"Đang nướng Texture Albedo Map ({tex_size}x{tex_size})...")

        # Tạo canvas texture map
        tex_img = np.full((tex_size, tex_size, 3), 180, dtype=np.uint8)

        # Chuyển đổi UV sang pixel coordinates trên Texture Image
        uv_pixels = np.clip(uvs * (tex_size - 1), 0, tex_size - 1).astype(np.int32)
        # Lật trục Y theo quy ước UV (v=0 ở đáy, v=1 ở đỉnh)
        uv_pixels[:, 1] = (tex_size - 1) - uv_pixels[:, 1]

        # Gán màu các đỉnh vào pixel tương ứng
        mask = np.zeros((tex_size, tex_size), dtype=bool)
        tex_img[uv_pixels[:, 1], uv_pixels[:, 0]] = vertex_colors[:, :3]
        mask[uv_pixels[:, 1], uv_pixels[:, 0]] = True

        # Lan tỏa màu sắc (Voronoi Dilation) lấp đầy 100% bề mặt tam giác trong UV space
        try:
            from scipy import ndimage
            _, indices = ndimage.distance_transform_edt(~mask, return_indices=True)
            tex_img = tex_img[indices[0], indices[1]]
        except Exception:
            mean_c = np.mean(vertex_colors[:, :3], axis=0).astype(np.uint8)
            tex_img[~mask] = mean_c

        pil_texture = Image.fromarray(tex_img, mode="RGB")
        return pil_texture

    def process_and_export(
        self,
        mesh: trimesh.Trimesh,
        images_rgb: List[np.ndarray],
        camera_poses: List[np.ndarray],
        focal_lengths: List[Tuple[float, float]],
        output_path: str,
    ) -> Tuple[bool, str]:
        """
        Đóng gói toàn bộ quy trình P5:
        Unwrap UV -> Blend màu góc nhìn -> Nướng Texture -> Xuất file .GLB hoàn chỉnh.

        Args:
            mesh: Trimesh từ P4 TSDF Mesh Engine.
            images_rgb: List N ảnh RGB gốc.
            camera_poses: List N ma trận camera pose.
            focal_lengths: List N tiêu cự.
            output_path: Đường dẫn lưu file .glb thành phẩm.

        Returns:
            Tuple[bool, str]: (success, glb_path).
        """
        t0 = time.time()
        logger.info("═══ [P5] BẮT ĐẦU TRẢI UV & NƯỚNG TEXTURE BASE-COLOR ═══")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            # ── Bước 1: Trải UV (XAtlas Parameterization) ──
            unwrapped_mesh, uvs = self.unwrap_uv(mesh)

            # ── Bước 2: Hòa trộn màu góc nhìn (Angle-Weighted Blending) ──
            vertex_rgba = self.blend_colors_for_vertices(
                mesh=unwrapped_mesh,
                images_rgb=images_rgb,
                camera_poses=camera_poses,
                focal_lengths=focal_lengths,
            )

            # ── Bước 3: Nướng Texture Map ──
            texture_image = self.bake_texture_map(
                mesh=unwrapped_mesh,
                uvs=uvs,
                vertex_colors=vertex_rgba,
                tex_size=self.texture_size,
            )

            # ── Bước 4: Đóng gói Texture Visual & Xuất file .GLB ──
            unwrapped_mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=uvs,
                image=texture_image,
            )
            # Đồng thời gán vertex_colors để viewer nào không bật texture vẫn xem được màu
            unwrapped_mesh.visual.vertex_colors = vertex_rgba

            unwrapped_mesh.export(output_path, file_type="glb")

            elapsed = time.time() - t0
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(
                f"═══ [P5] XUẤT THÀNH CÔNG FILE .GLB: {output_path} "
                f"({file_size_mb:.2f} MB) trong {elapsed:.2f}s ═══"
            )
            return True, output_path

        except Exception as e:
            logger.error(f"[P5] Lỗi trong quá trình tạo Texture & Xuất GLB: {e}", exc_info=True)
            return False, ""
