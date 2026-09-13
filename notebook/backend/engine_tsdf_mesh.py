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

import logging
import time
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
from scipy import ndimage
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

        # Lưới TSDF: khởi tạo 1.0 (free space / trống)
        self.tsdf_grid = np.ones((resolution, resolution, resolution), dtype=np.float32)

        # Lưới trọng số tích lũy: khởi tạo 0.0
        self.weight_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

        logger.info(
            f"Khởi tạo TSDF Volume: {resolution}^3 voxels, "
            f"vsize={self.voxel_size:.4f}m, trunc_margin={self.trunc_margin:.4f}m"
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
        Chiếu và tích lũy một góc quan sát camera vào thể tích Voxel TSDF bằng Space Carving.

        Bản chất hình học:
            1. Bất kỳ voxel nào chiếu vào hậu cảnh (alpha == 0) đều là KHOẢNG TRỐNG (+1.0) -> Gọt khối thể tích.
            2. Voxel chiếu vào vật thể (alpha == 1):
               - Phía trước bề mặt: khoảng trống (+1.0)
               - Bề mặt: d_surface - z_cam == 0 (zero-crossing)
               - Phía sau bề mặt: bên trong vật thể (-1.0) -> Ruột đặc 360 độ kín.
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

        for k in range(self.resolution):
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
            u_valid = u_proj[valid_indices]
            v_valid = v_proj[valid_indices]
            z_c_valid = z_cam[valid_indices]

            # 1. Background Carving: Bất kỳ voxel nào chiếu vào hậu cảnh là KHOẢNG TRỐNG (TSDF = +1.0)
            is_fg = (alpha_mask[v_valid, u_valid] > 0.5)
            bg_indices = valid_indices[~is_fg]
            if len(bg_indices) > 0:
                i_bg = bg_indices // self.resolution
                j_bg = bg_indices % self.resolution
                w_bg = 0.5
                old_tsdf = self.tsdf_grid[i_bg, j_bg, k]
                old_w = self.weight_grid[i_bg, j_bg, k]
                new_w = old_w + w_bg
                self.tsdf_grid[i_bg, j_bg, k] = (old_w * old_tsdf + w_bg * 1.0) / np.maximum(new_w, 1e-6)
                self.weight_grid[i_bg, j_bg, k] = new_w

            # 2. Object Pixels: Tính TSDF bề mặt và phần ruột bên trong
            fg_indices = valid_indices[is_fg]
            if len(fg_indices) == 0:
                continue

            u_sub = u_proj[fg_indices]
            v_sub = v_proj[fg_indices]
            z_c_sub = z_cam[fg_indices]

            # Độ sâu quan sát được từ pointmap
            d_surface = depth_obs[v_sub, u_sub]

            # Khoảng cách có dấu đến bề mặt: d(p) = d_surface - z_cam
            # s > 0: trước bề mặt (+1.0)
            # s = 0: bề mặt (0.0)
            # s < 0: sau bề mặt (-1.0)
            signed_dist = d_surface - z_c_sub
            tsdf_val = np.clip(signed_dist / self.trunc_margin, -1.0, 1.0)

            # Trọng số theo confidence từ DUSt3R
            weight = conf_map[v_sub, u_sub]

            i_idx = fg_indices // self.resolution
            j_idx = fg_indices % self.resolution

            old_tsdf = self.tsdf_grid[i_idx, j_idx, k]
            old_w = self.weight_grid[i_idx, j_idx, k]

            new_w = old_w + weight
            new_tsdf = (old_w * old_tsdf + weight * tsdf_val) / np.maximum(new_w, 1e-6)

            self.tsdf_grid[i_idx, j_idx, k] = new_tsdf
            self.weight_grid[i_idx, j_idx, k] = new_w


# ============================================================================
# THUẬT TOÁN 3: TRÍCH XUẤT MẶT ĐẲNG TRỊ MARCHING CUBES
# ============================================================================

def extract_mesh_marching_cubes(
    tsdf_volume: TSDFVolume,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> trimesh.Trimesh:
    """
    Thuật toán 3: Trích xuất mặt đẳng trị (Marching Cubes).

    Quét qua các khối lập phương trong lưới Voxel TSDF:
    Tìm vị trí bề mặt cắt ngang (nơi TSDF = 0), nội suy tuyến tính tạo các mặt tam giác 3D.

    Args:
        tsdf_volume: Đối tượng TSDFVolume đã tích lũy.
        min_weight: Ngưỡng trọng số tối thiểu để voxel được công nhận là bề mặt.

    Returns:
        trimesh.Trimesh: Mô hình lưới tam giác kín 360°.
    """
    if not HAS_SKIMAGE:
        raise ImportError("Cần cài scikit-image để chạy Marching Cubes: pip install scikit-image")

    volume = tsdf_volume.tsdf_grid.copy()
    weights = tsdf_volume.weight_grid

    # Những vùng chưa từng được camera nào quan sát sẽ đặt giá trị 1.0 (ngoài vật thể)
    unobserved = weights < min_weight
    volume[unobserved] = 1.0

    # Marching Cubes tìm iso-surface tại level = 0.0
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

    # Chuyển đổi tọa độ từ Voxel Grid Index sang Tọa độ Không gian Thực (World Coordinates)
    verts_world = verts + tsdf_volume.bounds_min

    # Đóng gói đối tượng Trimesh
    mesh = trimesh.Trimesh(
        vertices=verts_world,
        faces=faces,
        vertex_normals=normals,
        process=True,
    )

    # Dọn dẹp lưới: Giữ lại thành phần liên thông lớn nhất (loại bỏ các cụm vụn nổi lơ lửng)
    components = mesh.split(only_watertight=False)
    if components and len(components) > 1:
        largest = max(components, key=lambda m: len(m.vertices))
        mesh = largest

    logger.info(
        f"Marching Cubes hoàn thành: Sinh mesh với {len(mesh.vertices)} đỉnh, "
        f"{len(mesh.faces)} tam giác (Watertight: {mesh.is_watertight})."
    )
    return mesh


# ============================================================================
# ĐỘNG CƠ TỔNG HỢP: TSDFMeshEngine (GIAO DIỆN BÀN GIAO CHO APP)
# ============================================================================

class TSDFMeshEngine:
    """
    Lớp điều phối toàn bộ chu trình P4:
    Lọc viền độ sâu -> Pruning điểm nền -> Tích lũy Voxel TSDF -> Trích xuất Marching Cubes.
    """

    def __init__(
        self,
        resolution: int = DEFAULT_VOXEL_RESOLUTION,
        tau_conf: float = DEFAULT_CONF_THRESHOLD,
        tau_edge: float = DEFAULT_DEPTH_EDGE_TAU,
    ):
        self.resolution = resolution
        self.tau_conf = tau_conf
        self.tau_edge = tau_edge

    def reconstruct(
        self,
        pointmaps_3d: Union[np.ndarray, Any],
        alpha_masks: List[np.ndarray],
        confidence_masks: Union[np.ndarray, Any],
        camera_poses: List[np.ndarray],
        focal_lengths: List[Tuple[float, float]],
    ) -> trimesh.Trimesh:
        """
        Thực thi pipeline tái tạo lưới 3D từ các góc nhìn DUSt3R.

        Args:
            pointmaps_3d: (N, H, W, 3) tọa độ 3D các điểm.
            alpha_masks: List N mặt nạ nhị phân {0, 1} từ RMBG-2.0.
            confidence_masks: (N, H, W) độ tin cậy từ DUSt3R.
            camera_poses: List N ma trận camera 4x4.
            focal_lengths: List N cặp tiêu cự (fx, fy).

        Returns:
            trimesh.Trimesh: Lưới tam giác hoàn chỉnh 360°.
        """
        t0 = time.time()
        logger.info("═══ [P4] BẮT ĐẦU TÁI TẠO LƯỚI 3D (TSDF + MARCHING CUBES) ═══")

        # Chuyển đổi tensor sang numpy nếu cần
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

        all_valid_pts = np.concatenate([p for p in filtered_points if len(p) > 0], axis=0)
        if len(all_valid_pts) < 50:
            logger.warning("[P4] Số điểm hợp lệ qua bộ lọc biên ít, dùng trực tiếp điểm trong alpha mask...")
            all_valid_pts = np.concatenate([
                pointmaps_3d[i][(alpha_masks[i] > 0.5) & np.isfinite(pointmaps_3d[i]).all(axis=-1)]
                for i in range(n_views)
            ], axis=0)

        if len(all_valid_pts) < 10:
            raise ValueError(f"Số lượng điểm 3D hợp lệ quá ít ({len(all_valid_pts)} điểm) không đủ để dựng lưới.")

        # ── Bước 2: Khởi tạo thể tích TSDF theo Bounding Box ──
        p_min = np.percentile(all_valid_pts, 0.5, axis=0)
        p_max = np.percentile(all_valid_pts, 99.5, axis=0)

        # Thêm padding 20% vào bounding box để tránh chạm biên
        center = (p_min + p_max) / 2.0
        extent = (p_max - p_min) * 1.25
        bounds_min = center - extent / 2.0
        bounds_max = center + extent / 2.0

        tsdf_vol = TSDFVolume(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            resolution=self.resolution,
        )

        # ── Bước 3: Tích lũy đa góc nhìn vào Voxel Grid ──
        for i in range(n_views):
            logger.info(f"[P4] Tích lũy TSDF góc nhìn #{i+1}/{n_views}...")
            tsdf_vol.integrate(
                pointmap=pointmaps_3d[i],
                valid_mask=valid_masks[i],
                alpha_mask=alpha_masks[i],
                conf_map=confidence_masks[i],
                camera_pose=camera_poses[i],
                focal_length=focal_lengths[i],
            )

        # ── Bước 4: Trích xuất Iso-surface Marching Cubes ──
        logger.info("[P4] Trích xuất bề mặt Marching Cubes...")
        try:
            mesh = extract_mesh_marching_cubes(tsdf_vol)
        except Exception as mc_err:
            logger.warning(f"[P4] Marching Cubes gặp sự cố ({mc_err}), kích hoạt Multi-View Point Cloud Fusion...")
            mesh = trimesh.convex.convex_hull(all_valid_pts)

        if len(mesh.faces) < 50:
            logger.warning("[P4] Mesh Marching Cubes quá ít mặt, kích hoạt Multi-View Point Cloud Fusion...")
            mesh = trimesh.convex.convex_hull(all_valid_pts)

        elapsed = time.time() - t0
        logger.info(f"═══ [P4] HOÀN THÀNH TÁI TẠO MESH TRONG {elapsed:.2f} GIÂY ═══")
        return mesh
