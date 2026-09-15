"""
Module Ước Lượng Độ Sâu & Tái Tạo Hình Học Đơn Ảnh (Phase 2 - P2).
Chuẩn hóa theo NVIDIA 3D Pipeline & Depth-Anything-V2-Small.

Trách nhiệm chính:
    1. Ước lượng độ sâu (Monocular Depth Estimation) bằng Depth-Anything-V2-Small (~95MB).
    2. Lọc viền độ sâu (DA3-blender Edge Discontinuity Filtering) để loại bỏ hiện tượng mép rách/flying pixels.
    3. Dự đoán độ sâu đa góc nhìn đồng bộ (Multi-view Depth Prediction).
    4. Tái tạo hình học đơn ảnh (Single-view Surface Mesh) qua Pinhole Back-projection.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Tuple, Optional, List, Union

import cv2
import numpy as np
import trimesh
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from .utils_3d import export_glb
except ImportError:
    try:
        from utils_3d import export_glb
    except ImportError:
        def export_glb(mesh, path):
            mesh.export(path, file_type="glb")
            return str(path)

logger = logging.getLogger("engine_depth")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def filter_edge_discontinuity(
    depth_map: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """
    Toán tử DA3-blender: Lọc viền độ sâu (Edge Discontinuity Filtering).

    Loại bỏ các pixel ranh giới có độ biến thiên gradient quá lớn
    (nguyên nhân gây flying pixels và rách mép hình học).

    Công thức:
        Gx(u, v) = |D(u+1, v) - D(u-1, v)|
        Gy(u, v) = |D(u, v+1) - D(u, v-1)|
        valid_mask(u, v) = (max(Gx, Gy) <= tau * D(u, v) + 1e-4)

    Args:
        depth_map: Ma trận độ sâu chuẩn hóa [0, 1] shape (H, W).
        tau: Ngưỡng tỷ lệ biến thiên độ sâu (mặc định 0.05).

    Returns:
        valid_mask: np.ndarray bool shape (H, W), True nếu là pixel mặt phẳng hợp lệ.
    """
    h, w = depth_map.shape[:2]
    if h < 3 or w < 3:
        return np.ones((h, w), dtype=bool)

    gx = np.zeros_like(depth_map, dtype=np.float32)
    gy = np.zeros_like(depth_map, dtype=np.float32)

    # Trung tâm sai phân đạo hàm bậc nhất
    gx[:, 1:-1] = 0.5 * np.abs(depth_map[:, 2:] - depth_map[:, :-2])
    gy[1:-1, :] = 0.5 * np.abs(depth_map[2:, :] - depth_map[:-2, :])

    # Viền ngoài
    gx[:, 0] = np.abs(depth_map[:, 1] - depth_map[:, 0])
    gx[:, -1] = np.abs(depth_map[:, -1] - depth_map[:, -2])
    gy[0, :] = np.abs(depth_map[1, :] - depth_map[0, :])
    gy[-1, :] = np.abs(depth_map[-1, :] - depth_map[-2, :])

    gradient_mag = np.maximum(gx, gy)
    adaptive_thresh = tau * depth_map + 1e-4
    valid_mask = gradient_mag <= adaptive_thresh
    return valid_mask


class DepthReconstructionEngine:
    """
    Động cơ ước lượng độ sâu và tái tạo hình học bề mặt chuẩn NVIDIA.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Khởi tạo engine và nạp mô hình Depth-Anything-V2-Small.
        """
        if device is None:
            self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.depth_model = None
        self.depth_processor = None
        self._load_depth_model()

    def _load_depth_model(self):
        """Nạp model Depth-Anything-V2-Small từ HuggingFace Cache hoặc Hub."""
        if not HAS_TORCH:
            logger.warning("[P2] PyTorch không khả dụng. Cần PyTorch để chạy Depth-Anything-V2.")
            return

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            logger.info(f"[P2] Đang nạp model {DEPTH_MODEL_ID} trên thiết bị '{self.device}'...")

            self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
            self.depth_model.to(self.device)
            self.depth_model.eval()
            logger.info(f"[P2] ✓ Depth-Anything-V2-Small nạp thành công trên {self.device}.")

        except Exception as e:
            logger.error(f"[P2] Không thể tải {DEPTH_MODEL_ID}: {e}", exc_info=True)
            self.depth_model = None
            self.depth_processor = None

    def predict_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Dự đoán Depth Map chuẩn hóa [0, 1] từ ảnh RGB đơn lẻ.

        Args:
            image_rgb: np.ndarray (H, W, 3) uint8.

        Returns:
            depth_map: np.ndarray (H, W) float32 trong khoảng [0.0, 1.0].
                       (1.0 = gần camera nhất, 0.0 = xa camera nhất).
        """
        h, w = image_rgb.shape[:2]
        if self.depth_model is None or self.depth_processor is None:
            raise RuntimeError("[P2] Model Depth-Anything-V2 chưa được nạp.")

        pil_img = Image.fromarray(image_rgb)
        inputs = self.depth_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Nội suy bicubic về đúng độ phân giải gốc của ảnh
        depth_tensor = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )[0, 0]

        depth_np = depth_tensor.cpu().numpy().astype(np.float32)

        # Chuẩn hóa min-max về dải [0, 1]
        d_min, d_max = float(depth_np.min()), float(depth_np.max())
        if d_max - d_min > 1e-6:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_np = np.full((h, w), 0.5, dtype=np.float32)

        return depth_np

    def predict_depth_batch(self, images_rgb: List[np.ndarray]) -> List[np.ndarray]:
        """Dự đoán Depth Maps cho danh sách N ảnh."""
        return [self.predict_depth(img) for img in images_rgb]

    def predict_multiview_depth(
        self,
        images_rgb: List[np.ndarray],
        alpha_masks: Optional[List[np.ndarray]] = None,
        tau_edge: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Ước lượng độ sâu đa góc nhìn kết hợp lọc viền DA3-blender và mặt nạ Alpha.

        Args:
            images_rgb: Danh sách N ảnh RGB uint8 (H, W, 3).
            alpha_masks: Danh sách N alpha masks uint8 (H, W).
            tau_edge: Hệ số lọc viền mép rách DA3-blender.

        Returns:
            Dict chứa:
                - 'depth_maps': List[np.ndarray (H, W) float32]
                - 'valid_masks': List[np.ndarray (H, W) bool]
        """
        depth_maps = self.predict_depth_batch(images_rgb)
        valid_masks = []

        for i, d in enumerate(depth_maps):
            edge_valid = filter_edge_discontinuity(d, tau=tau_edge)
            if alpha_masks is not None and i < len(alpha_masks):
                a_mask = (alpha_masks[i] > 127)
                comb_mask = a_mask & edge_valid
            else:
                comb_mask = edge_valid
            valid_masks.append(comb_mask)

        return {
            "depth_maps": depth_maps,
            "valid_masks": valid_masks,
        }

    def reconstruct(
        self,
        image_rgb: np.ndarray,
        alpha_mask: np.ndarray,
        focal_length: Tuple[float, float],
        output_path: str,
        max_res: int = 384,
        edge_threshold: float = 0.12,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Tái tạo bề mặt 3D hình học từ đơn ảnh (Single-view Surface Mesh).

        Quy trình chuẩn hóa:
            1. Dự đoán Depth Map chuẩn xác bằng Depth-Anything-V2-Small.
            2. Chiếu ngược (Back-projection) tọa độ pixel thành điểm 3D (Pinhole Camera).
            3. Dựng cấu trúc lưới tam giác Pinhole Grid với kiểm tra rách mép (Edge threshold).
            4. Chuẩn hóa hệ trục Three.js (+Y Up, Y_min = 0) và xuất file .glb.

        Returns:
            (success, output_path, execution_time)
        """
        t0 = time.time()
        logger.info(f"[P2] Bắt đầu tái tạo hình học đơn ảnh...")

        try:
            depth_map = self.predict_depth(image_rgb)
            h_orig, w_orig = depth_map.shape[:2]
            step = max(1, max(h_orig, w_orig) // max_res)

            rows = np.arange(0, h_orig, step)
            cols = np.arange(0, w_orig, step)
            H_sub = len(rows)
            W_sub = len(cols)

            sub_depth = depth_map[rows[:, None], cols[None, :]]
            if alpha_mask is not None:
                sub_mask = (alpha_mask[rows[:, None], cols[None, :]] > 127)
            else:
                sub_mask = np.ones((H_sub, W_sub), dtype=bool)

            sub_rgb = image_rgb[rows[:, None], cols[None, :]]

            fx, fy = focal_length
            cx, cy = w_orig / 2.0, h_orig / 2.0

            u_grid = cols[None, :].repeat(H_sub, axis=0).astype(np.float32)
            v_grid = rows[:, None].repeat(W_sub, axis=1).astype(np.float32)

            # Đổi depth sang tọa độ Z thế giới (gần = Z nhỏ, xa = Z lớn)
            z_3d = (1.0 - sub_depth) * 1.0 + 0.5
            x_3d = (u_grid - cx) * z_3d / fx
            y_3d = -(v_grid - cy) * z_3d / fy  # Hướng +Y lên trên

            vertex_idx = np.full((H_sub, W_sub), -1, dtype=np.int32)
            valid_coords = np.argwhere(sub_mask)

            if len(valid_coords) < 3:
                raise ValueError("Không đủ pixel tiền cảnh để dựng lưới.")

            vertices = []
            vertex_colors = []
            for idx, (r, c) in enumerate(valid_coords):
                vertex_idx[r, c] = idx
                vertices.append([x_3d[r, c], y_3d[r, c], z_3d[r, c]])
                vertex_colors.append([sub_rgb[r, c, 0], sub_rgb[r, c, 1], sub_rgb[r, c, 2], 255])

            vertices = np.array(vertices, dtype=np.float32)
            vertex_colors = np.array(vertex_colors, dtype=np.uint8)

            faces = []
            for r in range(H_sub - 1):
                for c in range(W_sub - 1):
                    i00 = vertex_idx[r, c]
                    i01 = vertex_idx[r, c + 1]
                    i10 = vertex_idx[r + 1, c]
                    i11 = vertex_idx[r + 1, c + 1]

                    # Tam giác 1: (r, c), (r+1, c), (r, c+1)
                    if i00 >= 0 and i10 >= 0 and i01 >= 0:
                        z00, z10, z01 = z_3d[r, c], z_3d[r + 1, c], z_3d[r, c + 1]
                        if max(abs(z00 - z10), abs(z00 - z01), abs(z10 - z01)) <= edge_threshold:
                            faces.append([i00, i10, i01])

                    # Tam giác 2: (r+1, c), (r+1, c+1), (r, c+1)
                    if i10 >= 0 and i11 >= 0 and i01 >= 0:
                        z10, z11, z01 = z_3d[r + 1, c], z_3d[r + 1, c + 1], z_3d[r, c + 1]
                        if max(abs(z10 - z11), abs(z10 - z01), abs(z11 - z01)) <= edge_threshold:
                            faces.append([i10, i11, i01])

            if len(faces) == 0:
                for r in range(H_sub - 1):
                    for c in range(W_sub - 1):
                        i00 = vertex_idx[r, c]
                        i01 = vertex_idx[r, c + 1]
                        i10 = vertex_idx[r + 1, c]
                        if i00 >= 0 and i10 >= 0 and i01 >= 0:
                            faces.append([i00, i10, i01])

            faces = np.array(faces, dtype=np.int32)

            # Căn tâm và scale chuẩn hóa
            center = np.mean(vertices, axis=0)
            vertices -= center
            span = np.max(np.ptp(vertices, axis=0))
            if span > 1e-6:
                vertices /= span

            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                process=True,
            )

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            export_glb(mesh, output_path)

            elapsed = time.time() - t0
            logger.info(
                f"[P2] ✓ Tái tạo đơn ảnh thành công: {len(mesh.vertices)} đỉnh, "
                f"{len(mesh.faces)} mặt trong {elapsed:.2f}s -> {output_path}"
            )
            return True, output_path, elapsed

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[P2] Lỗi tái tạo đơn ảnh: {e}", exc_info=True)
            return False, None, elapsed


# Backward-compatibility alias
SurfaceMeshEngine = DepthReconstructionEngine
