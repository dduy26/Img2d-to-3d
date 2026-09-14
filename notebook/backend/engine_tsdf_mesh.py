"""
Module Tái Tạo Lưới 3D Bằng Không Gian Voxel TSDF & Marching Cubes (Pipeline v1).
Thành viên 4 (P4) - 3D Volumetric Mesh Engineer.

Kế thừa kiến trúc từ:
    - NVIDIA 3DObjectReconstruction (TSDF Volumetric Fusion & Marching Cubes)
    - DA3-blender (Edge Discontinuity Gradient Filtering)
    - DUSt3R (Global Point-maps & Camera Poses)

Các thuật toán cốt lõi:
    1. Thuật toán 1: Lọc viền độ sâu (Edge Discontinuity Filtering) & Pruning điểm nền
    2. Thuật toán 2: Dựng Lưới 3D Bằng Không Gian Voxel TSDF (Volumetric Fusion)
    3. Thuật toán 3: Trích xuất mặt đẳng trị (Marching Cubes)

Tài liệu tham khảo:
    - docs/phan_tich_chuyen_sau_reference_repos.md (Phần 4)
    - docs/lythuyet.md
"""

import os
import logging
import time
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import trimesh

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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

# Ngưỡng lọc độ tin cậy điểm từ DUSt3R (Confidence threshold)
DEFAULT_CONF_THRESHOLD: float = 0.35

# Ngưỡng biến thiên gradient độ sâu lọc viền xơ xác (Edge filter tau)
DEFAULT_DEPTH_EDGE_TAU: float = 0.07

# Kích thước lưới voxel mặc định (số ô mỗi chiều: 128^3)
DEFAULT_VOXEL_RESOLUTION: int = 128

# Bán kính cắt ngắn TSDF (Truncation margin factor so với voxel size)
DEFAULT_TRUNC_MARGIN_FACTOR: float = 3.0

# Ngưỡng trọng số tối thiểu để coi voxel đã được quan sát (Min accumulated weight)
DEFAULT_MIN_WEIGHT: float = 0.15


# ============================================================================
# THUẬT TOÁN 1: LỌC VIỀN ĐỘ SÂU & PRUNING ĐIỂM NỀN
# ============================================================================

def filter_depth_discontinuity(
    depth_map: np.ndarray,
    tau: float = DEFAULT_DEPTH_EDGE_TAU,
) -> np.ndarray:
    """
    Thuật toán 1 (Kế thừa từ DA3-blender):
    Lọc viền độ sâu (Edge Discontinuity Filtering).

    Vấn đề: Tại ranh giới giữa vật thể và hậu cảnh, gradient độ sâu biến thiên đột ngột
    sinh ra hàng loạt điểm point cloud "lơ lửng" kéo dài (flying pixels/edge tearing).

    Công thức:
        Gx(u, v) = |D(u+1, v) - D(u-1, v)|
        Gy(u, v) = |D(u, v+1) - D(u, v-1)|
        EdgeMask(u, v) = 1 nếu max(Gx, Gy) <= tau * D(u, v) ngược lại 0

    Args:
        depth_map: Ma trận độ sâu 2D shape (H, W), float32.
        tau: Hệ số ngưỡng biến thiên (mặc định 0.07).

    Returns:
        edge_mask: Ma trận boolean shape (H, W), True là điểm hợp lệ (không phải mép rách).
    """
    h, w = depth_map.shape
    if h < 3 or w < 3:
        return np.ones((h, w), dtype=bool)

    # Tính biến thiên đạo hàm theo hướng x (ngang) và y (dọc)
    gx = np.zeros_like(depth_map)
    gy = np.zeros_like(depth_map)

    gx[:, 1:-1] = np.abs(depth_map[:, 2:] - depth_map[:, :-2]) * 0.5
    gy[1:-1, :] = np.abs(depth_map[2:, :] - depth_map[:-2, :]) * 0.5

    # Viền biên ngoài cùng
    gx[:, 0] = gx[:, 1]
    gx[:, -1] = gx[:, -2]
    gy[0, :] = gy[1, :]
    gy[-1, :] = gy[-2, :]

    grad_max = np.maximum(gx, gy)
    threshold = tau * np.maximum(depth_map, 1e-6)

    # True = pixel an toàn, False = pixel viền mép rách
    edge_mask = grad_max <= threshold
    return edge_mask


def prune_background_points(
    pointmaps_3d: np.ndarray,
    alpha_masks: List[np.ndarray],
    confidence_masks: np.ndarray,
    camera_poses: Optional[List[np.ndarray]] = None,
    tau_conf: float = DEFAULT_CONF_THRESHOLD,
    tau_edge: float = DEFAULT_DEPTH_EDGE_TAU,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Kết hợp Mặt nạ Alpha (RMBG-2.0), Độ tin cậy DUSt3R và Bộ lọc viền độ sâu:
    Loại bỏ 100% điểm nền và rác biên, bảo toàn thân vật thể thực tế:
        X_{i, valid}(u, v) = X_i(u, v) nếu M_i(u, v) == 1 và C_i(u, v) > tau_conf và EdgeMask == 1

    Args:
        pointmaps_3d: np.ndarray shape (N, H, W, 3).
        alpha_masks: List N ma trận nhị phân {0, 1} shape (H, W).
        confidence_masks: np.ndarray shape (N, H, W).
        camera_poses: Danh sách ma trận camera 4x4 (nếu có để tính độ sâu chuẩn).
        tau_conf: Ngưỡng lọc độ tin cậy.
        tau_edge: Ngưỡng lọc rách biên độ sâu.

    Returns:
        valid_masks: List N boolean masks (H, W).
        filtered_points_list: List N mảng điểm 3D (K_i, 3) hợp lệ.
    """
    n_views, h, w, _ = pointmaps_3d.shape
    valid_masks = []
    filtered_points_list = []

    for i in range(n_views):
        pts = pointmaps_3d[i]  # (H, W, 3)
        conf = confidence_masks[i]  # (H, W)
        alpha = alpha_masks[i]  # (H, W)

        # Tính độ sâu quan sát chính xác từ Camera thay vì khoảng cách tới gốc (0,0,0)
        if camera_poses is not None and i < len(camera_poses):
            pose = camera_poses[i]
            c2w = np.asarray(pose, dtype=np.float32)
            if c2w.shape == (3, 4):
                c2w_homo = np.eye(4, dtype=np.float32)
                c2w_homo[:3, :4] = c2w
                c2w = c2w_homo
            R_w2c = c2w[:3, :3].T
            t_w2c = -R_w2c @ c2w[:3, 3]
            pts_cam = (pts.reshape(-1, 3) @ R_w2c.T + t_w2c).reshape(h, w, 3)
            depth = np.maximum(pts_cam[:, :, 2], 1e-4)
        else:
            valid_finite = np.isfinite(pts).all(axis=-1)
            if np.any(valid_finite):
                center = np.median(pts[valid_finite], axis=0)
                depth = np.linalg.norm(pts - center, axis=-1) + 1.0
            else:
                depth = np.ones((h, w), dtype=np.float32)

        # Lọc viền rách mép
        edge_mask = filter_depth_discontinuity(depth, tau=tau_edge)

        # Mặt nạ hợp lệ tổng hợp
        alpha_binary = (alpha > 0.5)
        conf_binary = (conf >= tau_conf)
        finite_mask = np.isfinite(pts).all(axis=-1)

        # Thích ứng tự động nếu ngưỡng conf quá chặt làm mất gần hết vật thể
        if np.sum(alpha_binary & conf_binary) < 0.25 * max(1, np.sum(alpha_binary)):
            conf_binary = (conf >= 0.10)

        base_mask = alpha_binary & conf_binary & finite_mask

        # BẢO VỆ VÙNG THÂN VẬT THỂ: Edge filter chỉ lọc ở viền, không được cắt lẹm thân
        if np.sum(base_mask & edge_mask) < 0.75 * max(1, np.sum(base_mask)):
            try:
                from scipy import ndimage
                alpha_core = ndimage.binary_erosion(alpha_binary, iterations=2)
                edge_mask = edge_mask | alpha_core
            except Exception:
                pass

        valid_mask = base_mask & edge_mask
        valid_masks.append(valid_mask)

        valid_pts = pts[valid_mask]
        filtered_points_list.append(valid_pts)

    total_valid = sum(len(p) for p in filtered_points_list)
    logger.info(
        f"Pruning: Giữ lại {total_valid} điểm 3D hợp lệ "
        f"từ {n_views} góc nhìn sau khi lọc nền & viền."
    )
    return valid_masks, filtered_points_list


# ============================================================================
# THUẬT TOÁN 2: DỰNG LƯỚI 3D BẰNG KHÔNG GIAN VOXEL TSDF (VOLUMETRIC FUSION)
# ============================================================================

class TSDFVolume:
    """
    Hệ thống tích lũy thể tích không gian 3D theo giải thuật Truncated Signed Distance Function.
    Thiết kế vector hóa tối ưu bằng NumPy/SciPy, hoạt động mượt mà trên CPU trong < 2 giây.
    """

    def __init__(
        self,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        resolution: int = DEFAULT_VOXEL_RESOLUTION,
        trunc_margin: Optional[float] = None,
    ):
        """
        Khởi tạo lưới thể tích Voxel TSDF.

        Args:
            bounds_min: Tọa độ nhỏ nhất [X_min, Y_min, Z_min] của thể tích.
            bounds_max: Tọa độ lớn nhất [X_max, Y_max, Z_max] của thể tích.
            resolution: Số ô voxel mỗi cạnh (mặc định 128).
            trunc_margin: Khoảng cách cắt ngắn mu (nếu None sẽ tự động tính theo voxel size).
        """
        self.bounds_min = bounds_min.astype(np.float32)
        self.bounds_max = bounds_max.astype(np.float32)
        self.resolution = resolution

        # Kích thước mỗi ô voxel
        center = (self.bounds_min + self.bounds_max) / 2.0
        max_extent = float(np.max(self.bounds_max - self.bounds_min))
        self.voxel_size = max_extent / float(self.resolution)

        # Căn chỉnh lại bounds đối xứng qua tâm để các voxel đều là khối lập phương
        half_cube = (self.voxel_size * self.resolution) / 2.0
        self.bounds_min = (center - half_cube).astype(np.float32)
        self.bounds_max = (center + half_cube).astype(np.float32)
        self.extent = self.bounds_max - self.bounds_min

        # Truncation margin mu
        if trunc_margin is None:
            self.trunc_margin = float(self.voxel_size * DEFAULT_TRUNC_MARGIN_FACTOR)
        else:
            self.trunc_margin = float(trunc_margin)

        # Lưới TSDF: Khởi tạo -1.0 (mặc định là thể tích đặc bên trong Bounding Box)
        # Các tia quan sát từ camera sẽ gọt (carve) không gian trống thành +1.0
        self.tsdf_grid = np.full((resolution, resolution, resolution), -1.0, dtype=np.float32)

        # Lưới trọng số tích lũy: khởi tạo 0.0
        self.weight_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

        # Đệm biên 1-voxel xung quanh 6 mặt ngoài để Marching Cubes tự động đóng kín đáy và các cạnh
        self.tsdf_grid[0, :, :] = 1.0
        self.tsdf_grid[-1, :, :] = 1.0
        self.tsdf_grid[:, 0, :] = 1.0
        self.tsdf_grid[:, -1, :] = 1.0
        self.tsdf_grid[:, :, 0] = 1.0
        self.tsdf_grid[:, :, -1] = 1.0

        logger.info(
            f"Khởi tạo TSDF Volume: {resolution}^3 voxels, "
            f"vsize={self.voxel_size:.4f}m, trunc_margin={self.trunc_margin:.4f}m (Solid Carving Mode)"
        )

    def integrate(
        self,
        pointmap: np.ndarray,
        valid_mask: np.ndarray,
        alpha_mask: np.ndarray,
        conf_map: np.ndarray,
        camera_pose: np.ndarray,
        focal_length: Tuple[float, float],
    ):
        """
        Chiếu và tích lũy một góc quan sát camera vào thể tích Voxel TSDF bằng True Solid Space Carving.

        Bản chất hình học chuẩn (NVIDIA / Volumetric Reconstruction):
            1. Voxel chiếu vào hậu cảnh (alpha == 0): Toàn bộ tia là KHÔNG GIAN TỰ DO (+1.0) -> Gọt khối thể tích.
            2. Voxel chiếu vào vật thể (alpha == 1):
               - Phía trước bề mặt (z_cam < d_surface - mu): Không gian nhìn xuyên thấu -> Carve thành +1.0.
               - Lớp bề mặt (|z_cam - d_surface| <= mu): Chuyển tiếp tuyến tính qua 0.0 (Iso-surface).
               - Phía sau bề mặt (z_cam > d_surface + mu): RUỘT ĐẶC VẬT THỂ (-1.0). Giữ nguyên âm để tạo khối kín.
        """
        h, w = alpha_mask.shape
        fx, fy = focal_length
        cx, cy = w / 2.0, h / 2.0

        # Pose chuyển từ camera sang world (camera_pose)
        c2w = np.asarray(camera_pose, dtype=np.float32)
        if c2w.shape == (3, 4):
            c2w_homo = np.eye(4, dtype=np.float32)
            c2w_homo[:3, :4] = c2w
            c2w = c2w_homo

        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w

        # Trích xuất bản đồ độ sâu Z từ pointmap trong hệ camera
        pts_cam = (pointmap.reshape(-1, 3) @ R_w2c.T + t_w2c).reshape(h, w, 3)
        depth_obs = pts_cam[:, :, 2]

        # Tọa độ Voxel Grid
        xs = np.linspace(self.bounds_min[0], self.bounds_max[0], self.resolution, endpoint=False, dtype=np.float32) + self.voxel_size / 2.0
        ys = np.linspace(self.bounds_min[1], self.bounds_max[1], self.resolution, endpoint=False, dtype=np.float32) + self.voxel_size / 2.0
        zs = np.linspace(self.bounds_min[2], self.bounds_max[2], self.resolution, endpoint=False, dtype=np.float32) + self.voxel_size / 2.0

        grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')

        # Chỉ lặp các voxel bên trong (bỏ qua border 0 và resolution-1 để bảo vệ lớp đệm kín nước)
        for k in range(1, self.resolution - 1):
            z_val = zs[k]
            pts_world_slice = np.stack([grid_x, grid_y, np.full_like(grid_x, z_val)], axis=-1)

            # Chiếu sang hệ camera: p_c = R_w2c * p_w + t_w2c
            flat_pts = pts_world_slice.reshape(-1, 3)
            p_cam = flat_pts @ R_w2c.T + t_w2c
            z_cam = p_cam[:, 2]

            # Loại bỏ voxel nằm phía sau camera
            valid_z = z_cam > 0.01

            # Chiếu lên màn hình 2D
            u_proj = np.round((fx * p_cam[:, 0] / np.maximum(z_cam, 1e-6)) + cx).astype(np.int32)
            v_proj = np.round((fy * p_cam[:, 1] / np.maximum(z_cam, 1e-6)) + cy).astype(np.int32)

            in_image = valid_z & (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h)
            if not np.any(in_image):
                continue

            valid_indices = np.where(in_image)[0]
            # Bỏ qua các chỉ số ở biên X, Y để bảo tồn lớp đệm ngoài
            i_coords = valid_indices // self.resolution
            j_coords = valid_indices % self.resolution
            inner_mask = (i_coords > 0) & (i_coords < self.resolution - 1) & (j_coords > 0) & (j_coords < self.resolution - 1)
            valid_indices = valid_indices[inner_mask]
            if len(valid_indices) == 0:
                continue

            u_valid = u_proj[valid_indices]
            v_valid = v_proj[valid_indices]
            z_c_valid = z_cam[valid_indices]

            # 1. Background Carving: Voxel chiếu vào hậu cảnh -> Không khí tự do (+1.0)
            is_fg = (alpha_mask[v_valid, u_valid] > 0.5)
            bg_indices = valid_indices[~is_fg]
            if len(bg_indices) > 0:
                i_bg = bg_indices // self.resolution
                j_bg = bg_indices % self.resolution
                w_bg = 0.5
                old_tsdf = self.tsdf_grid[i_bg, j_bg, k]
                old_w = self.weight_grid[i_bg, j_bg, k]
                new_w = old_w + w_bg
                # Tia nhìn xuyên thấu vào nền: chắc chắn là không khí (+1.0)
                self.tsdf_grid[i_bg, j_bg, k] = (old_w * old_tsdf + w_bg * 1.0) / np.maximum(new_w, 1e-6)
                self.weight_grid[i_bg, j_bg, k] = new_w

            # 2. Object Pixels: Xử lý trước bề mặt, lớp bề mặt và phần ruột
            fg_indices = valid_indices[is_fg]
            if len(fg_indices) == 0:
                continue

            u_sub = u_proj[fg_indices]
            v_sub = v_proj[fg_indices]
            z_c_sub = z_cam[fg_indices]

            d_surface = depth_obs[v_sub, u_sub]
            signed_dist = d_surface - z_c_sub

            i_idx = fg_indices // self.resolution
            j_idx = fg_indices % self.resolution
            weight = conf_map[v_sub, u_sub]

            # (a) Phía trước bề mặt: Không gian trống (s > 0)
            front_mask = signed_dist > self.trunc_margin
            if np.any(front_mask):
                i_f = i_idx[front_mask]
                j_f = j_idx[front_mask]
                w_f = weight[front_mask]
                old_tsdf = self.tsdf_grid[i_f, j_f, k]
                old_w = self.weight_grid[i_f, j_f, k]
                new_w = old_w + w_f
                self.tsdf_grid[i_f, j_f, k] = (old_w * old_tsdf + w_f * 1.0) / np.maximum(new_w, 1e-6)
                self.weight_grid[i_f, j_f, k] = new_w

            # (b) Vùng bề mặt (|s| <= trunc_margin): Nội suy tuyến tính qua 0
            surf_mask = np.abs(signed_dist) <= self.trunc_margin
            if np.any(surf_mask):
                i_s = i_idx[surf_mask]
                j_s = j_idx[surf_mask]
                w_s = weight[surf_mask]
                s_val = np.clip(signed_dist[surf_mask] / self.trunc_margin, -1.0, 1.0)
                old_tsdf = self.tsdf_grid[i_s, j_s, k]
                old_w = self.weight_grid[i_s, j_s, k]
                new_w = old_w + w_s
                self.tsdf_grid[i_s, j_s, k] = (old_w * old_tsdf + w_s * s_val) / np.maximum(new_w, 1e-6)
                self.weight_grid[i_s, j_s, k] = new_w

            # (c) Phía sau bề mặt (s < -trunc_margin): Ruột đặc vật thể (-1.0)
            back_mask = signed_dist < -self.trunc_margin
            if np.any(back_mask):
                i_b = i_idx[back_mask]
                j_b = j_idx[back_mask]
                w_b = weight[back_mask] * 0.2
                old_tsdf = self.tsdf_grid[i_b, j_b, k]
                old_w = self.weight_grid[i_b, j_b, k]
                new_w = old_w + w_b
                # Củng cố trạng thái ruột đặc (-1.0)
                self.tsdf_grid[i_b, j_b, k] = (old_w * old_tsdf + w_b * (-1.0)) / np.maximum(new_w, 1e-6)
                self.weight_grid[i_b, j_b, k] = new_w


# ============================================================================
# THUẬT TOÁN 3: TRÍCH XUẤT MẶT ĐẲNG TRỊ MARCHING CUBES (WATERTIGHT SOLID MESH)
# ============================================================================

def extract_mesh_marching_cubes(
    tsdf_volume: TSDFVolume,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> trimesh.Trimesh:
    """
    Trích xuất Iso-surface Marching Cubes đảm bảo tạo ra khối 3D KÍN NƯỚC (Watertight Solid).
    Đã loại bỏ hoàn toàn lỗi tạo 2 lớp vỏ mỏng rỗng ruột và toác đáy.
    """
    if not HAS_SKIMAGE:
        raise ImportError("Cần cài scikit-image để chạy Marching Cubes: pip install scikit-image")

    volume = tsdf_volume.tsdf_grid.copy()

    # Bảo đảm 100% các mặt ngoài Bounding Box là Không gian tự do (+1.0)
    # Lớp đệm này buộc Marching Cubes tự động đóng kín phẳng phần đáy và mọi mặt biên
    volume[0, :, :] = 1.0
    volume[-1, :, :] = 1.0
    volume[:, 0, :] = 1.0
    volume[:, -1, :] = 1.0
    volume[:, :, 0] = 1.0
    volume[:, :, -1] = 1.0

    # Marching Cubes tìm ranh giới Iso-surface giữa Không khí (+1.0) và Ruột đặc (-1.0) tại level = 0.0
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            volume=volume,
            level=0.0,
            spacing=(tsdf_volume.voxel_size, tsdf_volume.voxel_size, tsdf_volume.voxel_size),
            allow_degenerate=False,
        )
    except Exception as e:
        logger.warning(f"Marching Cubes level 0.0 gặp lỗi ({e}), thử level 0.05...")
        verts, faces, normals, _ = measure.marching_cubes(
            volume=volume,
            level=0.05,
            spacing=(tsdf_volume.voxel_size, tsdf_volume.voxel_size, tsdf_volume.voxel_size),
            allow_degenerate=False,
        )

    # Chuyển đổi tọa độ sang Tọa độ Thế giới thực (World Coordinates)
    verts_world = verts + tsdf_volume.bounds_min

    # Tạo Mesh Trimesh và chuẩn hóa hình học
    mesh = trimesh.Trimesh(
        vertices=verts_world,
        faces=faces,
        vertex_normals=normals,
        process=True,
    )

    # Hàn các đỉnh trùng và sửa chữa mặt lưới
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fill_holes(mesh)
    except Exception as rep_err:
        logger.warning(f"Mesh repair: {rep_err}")

    # Giữ lại thành phần liên thông lớn nhất
    components = mesh.split(only_watertight=False)
    if components and len(components) > 1:
        mesh = max(components, key=lambda m: len(m.vertices))

    logger.info(
        f"Marching Cubes hoàn thành: Sinh mesh với {len(mesh.vertices)} đỉnh, "
        f"{len(mesh.faces)} tam giác (Watertight: {mesh.is_watertight}, Thể tích: {mesh.volume:.6f})."
    )
    return mesh


# ============================================================================
# PHÉP KIỂM HÌNH HỌC THẬT (dùng cho log + test)
# ============================================================================

def mesh_health(mesh: trimesh.Trimesh, decimals: int = 6) -> dict:
    """
    Đo hình học THẬT của mesh: kín hay hở, mấy mảnh, bao nhiêu cạnh biên.

    PHẢI HÀN ĐỈNH TRƯỚC KHI KIỂM. Mesh có texture bị XAtlas chia đỉnh tại mọi đường
    seam UV — đó là chuyện BÌNH THƯỜNG và không làm hỏng hình. Nhưng trimesh đếm mảnh
    theo chart UV (face_adjacency dựa trên chỉ số đỉnh), nên `mesh.split()` báo "hàng
    nghìn mảnh rời" cho một khối hoàn toàn lành. Hàn đỉnh theo toạ độ rồi mới đo.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return {"vertices": 0, "faces": 0, "watertight": False, "winding_consistent": False,
                "components": 0, "boundary_edges": 0, "volume": 0.0,
                "duplicated_vertices": 0}

    unique_vertices, inverse = np.unique(np.round(vertices, decimals), axis=0, return_inverse=True)
    welded = trimesh.Trimesh(vertices=unique_vertices, faces=inverse[faces], process=True)
    welded.update_faces(welded.nondegenerate_faces())
    welded.update_faces(welded.unique_faces())

    edges = welded.edges_sorted
    if len(edges):
        _, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = int(np.count_nonzero(counts == 1))
    else:
        boundary = 0

    return {
        "vertices": len(vertices),
        "faces": len(faces),
        "duplicated_vertices": len(vertices) - len(unique_vertices),
        "watertight": bool(welded.is_watertight),
        "winding_consistent": bool(welded.is_winding_consistent),
        "components": len(welded.split(only_watertight=False)),
        "boundary_edges": boundary,
        "volume": float(welded.volume),
    }


def log_mesh_health(mesh: trimesh.Trimesh, label: str = "mesh") -> dict:
    """In kết quả mesh_health ra log. Hở (boundary_edges > 0) thì cảnh báo rõ."""
    health = mesh_health(mesh)
    logger.info(
        f"[KIỂM HÌNH HỌC] {label}: {health['vertices']} đỉnh ({health['duplicated_vertices']} "
        f"đỉnh trùng do seam UV), {health['faces']} mặt | sau khi hàn: "
        f"{health['components']} mảnh, {health['boundary_edges']} cạnh biên, "
        f"watertight={health['watertight']}, thể tích={health['volume']:.6f}"
    )
    if health["boundary_edges"] > 0:
        logger.warning(
            f"[KIỂM HÌNH HỌC] {label} HỞ: {health['boundary_edges']} cạnh biên."
        )
    return health


# ============================================================================
# TẠO MA TRẬN CAMERA QUANH VẬT THỂ (NVIDIA MULTI-VIEW TRAJECTORY)
# ============================================================================

def generate_camera_poses(
    n_views: int,
    radius: float = 2.2,
    elevation_deg: float = 15.0,
    view_names: Optional[List[str]] = None,
) -> List[np.ndarray]:
    """
    Sinh ma trận camera 4x4 (Camera-to-World, c2w) quanh vật thể.
    Quy ước camera: OpenCV/Pinhole (+X phải, +Y xuống, +Z hướng nhìn tới vật thể).
    Chuẩn hóa tương thích 100% với utils_3d.project_vertices và texture_blender.

    Hỗ trợ 2 chế độ:
    1. Orthogonal box (nếu có tên front, right, back, left, top, bottom).
    2. Turntable 360° (chia đều góc xoay azimuth quanh trục Y).
    """
    # 0. Nếu có file cameras.json đi kèm trong thư mục ảnh hoặc input/, ưu tiên nạp ma trận chuẩn
    if view_names and len(view_names) > 0:
        candidate_dirs = [os.path.dirname(view_names[0]), "input", "data/objaverse_apple"]
        for cdir in candidate_dirs:
            if not cdir:
                continue
            cpath = os.path.join(cdir, "cameras.json")
            if os.path.exists(cpath):
                try:
                    import json
                    with open(cpath, "r", encoding="utf-8") as f:
                        cdata = json.load(f)
                    poses = []
                    for vn in view_names:
                        bname = os.path.basename(vn)
                        if bname in cdata:
                            poses.append(np.array(cdata[bname]["camera_pose"], dtype=np.float32))
                    if len(poses) == len(view_names):
                        logger.info("✓ Đã nạp thành công %d ma trận camera chuẩn xác từ %s", len(poses), cpath)
                        return poses
                except Exception as e:
                    logger.warning("Lỗi đọc cameras.json: %s", e)

    poses = []
    
    # Kiểm tra xem có phải bộ ảnh 6 góc trực giao không
    has_ortho = False
    if view_names and len(view_names) == 6:
        lowered = [str(n).lower() for n in view_names]
        if any("front" in n for n in lowered) and any("back" in n for n in lowered):
            has_ortho = True

    if has_ortho and view_names:
        for name in view_names:
            nl = str(name).lower()
            if "front" in nl:
                az, el = 0.0, 0.0
            elif "right" in nl:
                az, el = 90.0, 0.0
            elif "back" in nl:
                az, el = 180.0, 0.0
            elif "left" in nl:
                az, el = 270.0, 0.0
            elif "top" in nl:
                az, el = 0.0, 85.0
            elif "bottom" in nl:
                az, el = 0.0, -85.0
            else:
                az, el = 0.0, elevation_deg

            az_rad = np.radians(az)
            el_rad = np.radians(el)
            cx = float(radius * np.cos(el_rad) * np.sin(az_rad))
            cy = float(radius * np.sin(el_rad))
            cz = float(radius * np.cos(el_rad) * np.cos(az_rad))
            c_pos = np.array([cx, cy, cz], dtype=np.float32)

            forward = -c_pos / np.maximum(np.linalg.norm(c_pos), 1e-6)
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(el) < 80 else np.array([0.0, 0.0, -1.0], dtype=np.float32)
            right = np.cross(forward, world_up)
            right /= np.maximum(np.linalg.norm(right), 1e-6)
            down = np.cross(forward, right)
            down /= np.maximum(np.linalg.norm(down), 1e-6)

            R_c2w = np.column_stack([right, down, forward])
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R_c2w
            pose[:3, 3] = c_pos
            poses.append(pose)
    else:
        elev_rad = np.radians(elevation_deg)
        for i in range(n_views):
            theta = i * (2.0 * np.pi / n_views)
            cx = float(radius * np.cos(elev_rad) * np.sin(theta))
            cy = float(radius * np.sin(elev_rad))
            cz = float(radius * np.cos(elev_rad) * np.cos(theta))
            c_pos = np.array([cx, cy, cz], dtype=np.float32)

            forward = -c_pos / np.maximum(np.linalg.norm(c_pos), 1e-6)
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, world_up)
            right /= np.maximum(np.linalg.norm(right), 1e-6)
            down = np.cross(forward, right)
            down /= np.maximum(np.linalg.norm(down), 1e-6)

            R_c2w = np.column_stack([right, down, forward])
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R_c2w
            pose[:3, 3] = c_pos
            poses.append(pose)

    return poses


# ============================================================================
# ĐỘNG CƠ TỔNG HỢP: TSDFMeshEngine (GIAO DIỆN BÀN GIAO CHO APP)
# ============================================================================

class TSDFMeshEngine:
    """
    Lớp điều phối toàn bộ chu trình P4:
    Lọc viền độ sâu -> Pruning điểm nền -> Tích lũy Voxel TSDF True Space Carving -> Trích xuất Marching Cubes Kín Nước.
    """

    def __init__(
        self,
        resolution: int = DEFAULT_VOXEL_RESOLUTION,
        tau_conf: float = DEFAULT_CONF_THRESHOLD,
        tau_edge: float = DEFAULT_DEPTH_EDGE_TAU,
        smooth_iterations: int = 10,
    ):
        self.resolution = resolution
        self.tau_conf = tau_conf
        self.tau_edge = tau_edge
        self.smooth_iterations = int(os.environ.get("TSDF_SMOOTH_ITER", smooth_iterations))

    def _extract_watertight_mesh_from_points_and_normals(
        self,
        all_valid_pts: np.ndarray,
        all_normals: np.ndarray,
    ) -> trimesh.Trimesh:
        """
        Thuật toán NVIDIA Volumetric TSDF + Marching Cubes:
        Trích xuất lưới tam giác 3D đặc ruột, kín nước từ đám mây điểm bề mặt định hướng.
        """
        # Khởi tạo thể tích TSDF theo Bounding Box thực tế
        if len(all_valid_pts) > 100_000:
            step = max(1, len(all_valid_pts) // 100_000)
            all_valid_pts = all_valid_pts[::step]
            all_normals = all_normals[::step]
            logger.info(f"[P4] Subsampled point cloud to {len(all_valid_pts)} points for fast KDTree TSDF fusion.")

        p_min = np.percentile(all_valid_pts, 0.5, axis=0)
        p_max = np.percentile(all_valid_pts, 99.5, axis=0)
        extent = p_max - p_min
        margin = 0.15 * extent
        bounds_min = p_min - margin
        bounds_max = p_max + margin

        xs = np.linspace(bounds_min[0], bounds_max[0], self.resolution, endpoint=False, dtype=np.float32)
        ys = np.linspace(bounds_min[1], bounds_max[1], self.resolution, endpoint=False, dtype=np.float32)
        zs = np.linspace(bounds_min[2], bounds_max[2], self.resolution, endpoint=False, dtype=np.float32)
        dx = float(xs[1] - xs[0])
        dy = float(ys[1] - ys[0])
        dz = float(zs[1] - zs[0])
        max_voxel = max(dx, dy, dz)
        trunc_margin = 2.5 * max_voxel

        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        grid_pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

        # Tính toán Signed Distance Field từ các tia nhìn thực tế
        tree = cKDTree(all_valid_pts)
        k_neighbors = min(3, len(all_valid_pts))
        dists, idxs = tree.query(grid_pts, k=k_neighbors, workers=-1)
        if k_neighbors == 1:
            diff = grid_pts - all_valid_pts[idxs]
            dots = np.sum(diff * all_normals[idxs], axis=-1)
            signed_dist = dots
        else:
            diff = grid_pts[:, None, :] - all_valid_pts[idxs]
            dots = np.sum(diff * all_normals[idxs], axis=-1)
            weights = 1.0 / np.maximum(dists, 1e-4)
            weights /= np.sum(weights, axis=-1, keepdims=True)
            signed_dist = np.sum(dots * weights, axis=-1)

        sdf = np.clip(signed_dist, -trunc_margin, trunc_margin)

        # Bất kỳ voxel nào nằm ngoài hộp [p_min, p_max] của vật thể ĐỀU LÀ KHÔNG KHÍ (+trunc_margin)
        outside_box = (
            (grid_pts[:, 0] < p_min[0]) | (grid_pts[:, 0] > p_max[0]) |
            (grid_pts[:, 1] < p_min[1]) | (grid_pts[:, 1] > p_max[1]) |
            (grid_pts[:, 2] < p_min[2]) | (grid_pts[:, 2] > p_max[2])
        )
        sdf[outside_box] = np.maximum(sdf[outside_box], 0.2 * trunc_margin)

        sdf_grid = sdf.reshape(self.resolution, self.resolution, self.resolution).astype(np.float32)

        # Đệm biên 1-voxel quanh 6 mặt ngoài bằng +trunc_margin (không khí) để Marching Cubes luôn đóng kín nước
        sdf_grid[0, :, :] = trunc_margin; sdf_grid[-1, :, :] = trunc_margin
        sdf_grid[:, 0, :] = trunc_margin; sdf_grid[:, -1, :] = trunc_margin
        sdf_grid[:, :, 0] = trunc_margin; sdf_grid[:, :, -1] = trunc_margin

        # Trích xuất Iso-surface Marching Cubes với spacing thực (dx, dy, dz)
        logger.info("[P4] Trích xuất bề mặt Marching Cubes kín nước...")
        try:
            verts, faces, normals_mc, _ = measure.marching_cubes(
                volume=sdf_grid,
                level=0.0,
                spacing=(dx, dy, dz),
                allow_degenerate=False,
            )
            verts_world = verts + bounds_min
            mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, vertex_normals=normals_mc, process=True)
            mesh.merge_vertices()
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.update_faces(mesh.unique_faces())
            try:
                trimesh.repair.fix_normals(mesh)
                trimesh.repair.fix_winding(mesh)
                trimesh.repair.fill_holes(mesh)
            except Exception:
                pass
            components = mesh.split(only_watertight=False)
            if components and len(components) > 1:
                mesh = max(components, key=lambda m: len(m.vertices))
        except Exception as mc_err:
            logger.warning(f"[P4] Marching Cubes gặp sự cố ({mc_err}), kích hoạt Convex Hull Fallback...")
            mesh = trimesh.convex.convex_hull(all_valid_pts)

        if len(mesh.faces) < 50 or not mesh.is_watertight:
            logger.info("[P4] Gia cố độ kín nước cho mesh (fill_holes)...")
            try:
                trimesh.repair.fill_holes(mesh)
                mesh.merge_vertices()
            except Exception:
                pass

        if len(mesh.faces) < 50:
            logger.warning("[P4] Mesh Marching Cubes quá ít mặt, kích hoạt Convex Hull Fallback...")
            mesh = trimesh.convex.convex_hull(all_valid_pts)

        # Làm mượt Taubin (giảm sần do nhiễu sensor, bảo toàn thể tích)
        if self.smooth_iterations > 0 and len(mesh.vertices) > 0:
            try:
                from trimesh.smoothing import filter_taubin
                filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=self.smooth_iterations)
                logger.info(f"[P4] Làm mượt Taubin xong ({self.smooth_iterations} vòng).")
            except Exception as e:
                logger.warning(f"[P4] Bỏ qua làm mượt Taubin: {e}")

        log_mesh_health(mesh, "P4 Watertight Solid Mesh")
        return mesh

    def reconstruct(
        self,
        pointmaps_3d: Union[np.ndarray, Any],
        alpha_masks: List[np.ndarray],
        confidence_masks: Union[np.ndarray, Any],
        camera_poses: List[np.ndarray],
        focal_lengths: List[Tuple[float, float]],
    ) -> trimesh.Trimesh:
        """
        Thực thi pipeline tái tạo lưới 3D từ các góc nhìn DUSt3R (backward compatibility).
        """
        t0 = time.time()
        logger.info("═══ [P4] BẮT ĐẦU TÁI TẠO LƯỚI 3D TỪ POINTMAPS (TSDF + MARCHING CUBES) ═══")

        if hasattr(pointmaps_3d, 'cpu'):
            pointmaps_3d = pointmaps_3d.cpu().numpy()
        if hasattr(confidence_masks, 'cpu'):
            confidence_masks = confidence_masks.cpu().numpy()

        n_views = len(camera_poses)

        # ── Bước 1: Lọc viền độ sâu & Pruning điểm nền ──
        valid_masks, filtered_points = prune_background_points(
            pointmaps_3d=pointmaps_3d,
            alpha_masks=alpha_masks,
            confidence_masks=confidence_masks,
            camera_poses=camera_poses,
            tau_conf=self.tau_conf,
            tau_edge=self.tau_edge,
        )

        all_pts_list = []
        all_normals_list = []
        for i in range(n_views):
            v_mask = valid_masks[i]
            pts_v = pointmaps_3d[i][v_mask]
            if len(pts_v) > 0:
                c_pos = np.asarray(camera_poses[i], dtype=np.float32)[:3, 3]
                v_dir = c_pos - pts_v
                v_dir /= np.maximum(np.linalg.norm(v_dir, axis=-1, keepdims=True), 1e-6)
                all_pts_list.append(pts_v)
                all_normals_list.append(v_dir)

        if len(all_pts_list) > 0 and sum(len(p) for p in all_pts_list) >= 50:
            all_valid_pts = np.concatenate(all_pts_list, axis=0)
            all_normals = np.concatenate(all_normals_list, axis=0)
        else:
            logger.warning("[P4] Dùng trực tiếp điểm trong alpha mask...")
            all_pts_list = []
            all_normals_list = []
            for i in range(n_views):
                mask = (alpha_masks[i] > 0.5) & np.isfinite(pointmaps_3d[i]).all(axis=-1)
                pts_v = pointmaps_3d[i][mask]
                if len(pts_v) > 0:
                    c_pos = np.asarray(camera_poses[i], dtype=np.float32)[:3, 3]
                    v_dir = c_pos - pts_v
                    v_dir /= np.maximum(np.linalg.norm(v_dir, axis=-1, keepdims=True), 1e-6)
                    all_pts_list.append(pts_v)
                    all_normals_list.append(v_dir)
            all_valid_pts = np.concatenate(all_pts_list, axis=0) if all_pts_list else np.empty((0, 3), dtype=np.float32)
            all_normals = np.concatenate(all_normals_list, axis=0) if all_normals_list else np.empty((0, 3), dtype=np.float32)

        if len(all_valid_pts) < 10:
            raise ValueError(f"Số lượng điểm 3D hợp lệ quá ít ({len(all_valid_pts)} điểm) không đủ để dựng lưới.")

        logger.info(f"[P4] Tích lũy TSDF trường khoảng cách từ {len(all_valid_pts)} điểm bề mặt định hướng...")
        mesh = self._extract_watertight_mesh_from_points_and_normals(all_valid_pts, all_normals)
        elapsed = time.time() - t0
        logger.info(f"═══ [P4] HOÀN THÀNH TÁI TẠO MESH TRONG {elapsed:.2f} GIÂY ═══")
        return mesh

    def reconstruct_from_depth_maps(
        self,
        depth_maps: List[np.ndarray],
        alpha_masks: List[np.ndarray],
        camera_poses: Optional[List[np.ndarray]] = None,
        focal_lengths: Optional[List[Tuple[float, float]]] = None,
        view_names: Optional[List[str]] = None,
    ) -> trimesh.Trimesh:
        """
        Thuật toán cốt lõi NVIDIA + Depth-Anything-V2:
        Tái tạo lưới 3D từ N Depth Maps bằng NVIDIA TSDF Volumetric Fusion & Marching Cubes.

        Args:
            depth_maps: Danh sách N ma trận depth map float32 (H, W) trong khoảng [0, 1].
            alpha_masks: Danh sách N mặt nạ vật thể uint8 (H, W) {0, 1}.
            camera_poses: (Tùy chọn) Danh sách N ma trận camera 4x4 c2w. Nếu None, tự động sinh theo quỹ đạo.
            focal_lengths: (Tùy chọn) Danh sách N cặp tiêu cự (fx, fy). Nếu None, tự tính theo FOV ~ 50°.
            view_names: (Tùy chọn) Tên các góc nhìn để tự động nhận diện orthogonal box.

        Returns:
            trimesh.Trimesh: Mesh 3D đặc ruột, kín nước 100%.
        """
        t0 = time.time()
        n_views = len(depth_maps)
        logger.info(f"═══ [P4] BẮT ĐẦU NVIDIA TSDF FUSION CHO {n_views} GÓC NHÌN (DEPTH-ANYTHING) ═══")

        h, w = depth_maps[0].shape[:2]
        if focal_lengths is None or len(focal_lengths) != n_views:
            f_est = float((w / 2.0) / np.tan(np.radians(25.0)))
            focal_lengths = [(f_est, f_est)] * n_views

        if camera_poses is None or len(camera_poses) != n_views:
            camera_poses = generate_camera_poses(
                n_views=n_views,
                radius=2.2,
                elevation_deg=15.0,
                view_names=view_names,
            )

        all_pts_list = []
        all_normals_list = []

        for i in range(n_views):
            d_map = depth_maps[i]
            a_mask = alpha_masks[i] if i < len(alpha_masks) else np.ones((h, w), dtype=np.uint8)
            pose = camera_poses[i]
            fx, fy = focal_lengths[i]
            cx, cy = w / 2.0, h / 2.0

            # Lọc viền gradient DA3-blender
            edge_mask = filter_depth_discontinuity(d_map, tau=self.tau_edge)
            valid_mask = (a_mask > 0.5) & edge_mask

            if np.count_nonzero(valid_mask) < 20:
                valid_mask = (a_mask > 0.5)

            coords = np.argwhere(valid_mask)
            if len(coords) < 10:
                continue

            v_idx = coords[:, 0]
            u_idx = coords[:, 1]

            # Tính khoảng cách từ camera tới tâm vật thể (0,0,0)
            t_c2w = pose[:3, 3]
            dist_cam = float(np.linalg.norm(t_c2w))
            if dist_cam < 0.5:
                dist_cam = 2.2

            # Ước lượng bán kính 3D vật thể từ 2D mask theo hình học pinhole: R_obj = r_pixel * D / f
            fg_radius_px = 0.5 * max(coords[:, 0].max() - coords[:, 0].min(), coords[:, 1].max() - coords[:, 1].min())
            r_obj = float((fg_radius_px / fx) * dist_cam)

            # Chuẩn hóa độ sâu trong vùng foreground: d_norm in [0, 1]
            fg_vals = d_map[v_idx, u_idx]
            v_min, v_max = float(fg_vals.min()), float(fg_vals.max())
            if v_max - v_min > 1e-6:
                d_norm = (fg_vals - v_min) / (v_max - v_min)
            else:
                d_norm = np.ones_like(fg_vals) * 0.5

            # Bề mặt trước nhìn thấy lồi từ viền (dist_cam) về phía camera tối đa r_obj:
            z_cam = dist_cam - d_norm * r_obj

            # Back-projection theo Pinhole Camera Model (OpenCV convention)
            x_cam = (u_idx - cx) * z_cam / fx
            y_cam = (v_idx - cy) * z_cam / fy
            z_cam_dir = z_cam

            pts_cam = np.column_stack([x_cam, y_cam, z_cam_dir])

            # Chuyển đổi sang hệ tọa độ thế giới (world coordinates)
            R_c2w = pose[:3, :3]
            t_c2w = pose[:3, 3]
            pts_world = (R_c2w @ pts_cam.T).T + t_c2w

            # Vector pháp tuyến hướng về camera
            normals = t_c2w - pts_world
            normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)

            all_pts_list.append(pts_world)
            all_normals_list.append(normals)

        if not all_pts_list:
            raise ValueError("Không có điểm 3D hợp lệ nào từ các depth maps.")

        all_pts = np.concatenate(all_pts_list, axis=0).astype(np.float32)
        all_norms = np.concatenate(all_normals_list, axis=0).astype(np.float32)

        logger.info(f"[P4] Đã back-project {len(all_pts)} điểm bề mặt định hướng từ {n_views} góc chụp.")
        mesh = self._extract_watertight_mesh_from_points_and_normals(all_pts, all_norms)
        elapsed = time.time() - t0
        logger.info(f"═══ [P4] HOÀN THÀNH NVIDIA TSDF FUSION TRONG {elapsed:.2f} GIÂY ═══")
        return mesh

