"""
Module Kiểm Soát Chất Lượng Đa Tầng (Phase 3 - P3 Quality Gate).
Chuẩn hóa theo NVIDIA 3D Pipeline.

Trách nhiệm:
    - Tầng 1 (Cosine Angle Verification): Kiểm tra góc chênh lệch giữa các camera để phát hiện
      trùng lặp (duplicate views, delta_theta < 5 độ).
    - Tầng 2 (Co-visibility & Angular Coverage): Phân tích đồ thị quan sát liên thông (NetworkX),
      đảm bảo góc bao phủ tối thiểu >= 45 độ quanh vật thể.
    - Tầng 3 (Confidence & Silhouette Validity): Kiểm tra tính hợp lệ của Alpha Mask và biến thiên độ sâu.
    - Cơ chế cứu hộ (Fail-safe Fallback): Khi multi-view không đạt chuẩn, tự động đề xuất
      chuyển sang chế độ đơn ảnh (Single-view Depth Engine) dựa trên ảnh neo (Anchor View #0).
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger("quality_gate")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class QualityGate:
    """
    Hệ thống kiểm soát chất lượng dữ liệu trước khi tái tạo lưới 3D.
    """

    def __init__(
        self,
        min_angle_deg: float = 5.0,
        max_angle_deg: float = 120.0,
        min_total_span_deg: float = 45.0,
        min_foreground_ratio: float = 0.01,
        max_foreground_ratio: float = 0.95,
    ):
        self.min_angle_deg = min_angle_deg
        self.max_angle_deg = max_angle_deg
        self.min_total_span_deg = min_total_span_deg
        self.min_foreground_ratio = min_foreground_ratio
        self.max_foreground_ratio = max_foreground_ratio

    def evaluate_camera_poses(
        self,
        camera_poses: List[np.ndarray],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Tầng 1 & Tầng 2: Kiểm tra ma trận tư thế camera.
        """
        n = len(camera_poses)
        if n < 2:
            return True, "Chỉ có 1 camera, bỏ qua kiểm tra góc đa chiều.", {"num_views": n}

        # Trích xuất vector hướng nhìn (forward vector)
        forward_vectors = []
        for p in camera_poses:
            # OpenCV convention: +Z là hướng nhìn tới vật thể
            z_dir = p[:3, 2]
            norm = float(np.linalg.norm(z_dir))
            forward_vectors.append(z_dir / max(norm, 1e-6))

        # Tầng 1: Kiểm tra góc chênh lệch từng cặp camera (Cosine Angle Verification)
        min_detected_angle = 360.0
        max_detected_angle = 0.0
        duplicate_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                cos_theta = float(np.dot(forward_vectors[i], forward_vectors[j]))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = float(np.degrees(np.arccos(cos_theta)))
                min_detected_angle = min(min_detected_angle, angle)
                max_detected_angle = max(max_detected_angle, angle)

                if angle < self.min_angle_deg:
                    duplicate_pairs.append((i, j, angle))

        if duplicate_pairs:
            logger.warning(
                f"[P3] Cảnh báo góc trùng lặp: các cặp ảnh {duplicate_pairs} "
                f"có delta_theta < {self.min_angle_deg}°."
            )

        # Tầng 2: Phân tích đồ thị quan sát chung (Co-visibility Graph)
        if HAS_NETWORKX:
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for i in range(n):
                for j in range(i + 1, n):
                    cos_theta = float(np.dot(forward_vectors[i], forward_vectors[j]))
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle = float(np.degrees(np.arccos(cos_theta)))
                    if angle <= self.max_angle_deg:
                        G.add_edge(i, j, weight=angle)

            is_connected = nx.is_connected(G)
            components = nx.number_connected_components(G)
        else:
            is_connected = True
            components = 1

        metrics = {
            "num_views": n,
            "min_angle": round(min_detected_angle, 2),
            "max_angle": round(max_detected_angle, 2),
            "duplicate_pairs": duplicate_pairs,
            "is_connected": is_connected,
            "components": components,
        }

        if max_detected_angle < self.min_total_span_deg:
            return False, (
                f"Góc bao phủ quá hẹp ({max_detected_angle:.1f}° < {self.min_total_span_deg}°). "
                f"Các ảnh gần như chụp cùng 1 hướng, nguy cơ gây bẹt hình học."
            ), metrics

        if not is_connected:
            return False, (
                f"Đồ thị camera bị đứt đoạn ({components} thành phần rời rạc). "
                f"Khoảng cách góc giữa các cụm chụp vượt quá {self.max_angle_deg}°."
            ), metrics

        return True, "Kiểm tra camera poses đạt chuẩn NVIDIA.", metrics

    def evaluate_silhouettes_and_depth(
        self,
        alpha_masks: List[np.ndarray],
        depth_maps: Optional[List[np.ndarray]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Tầng 3: Kiểm tra tính hợp lệ của Silhouette Masks và Depth Maps.
        """
        ratios = []
        for idx, mask in enumerate(alpha_masks):
            h, w = mask.shape[:2]
            fg_count = int(np.count_nonzero(mask > 127))
            ratio = fg_count / float(h * w)
            ratios.append(ratio)

            if ratio < self.min_foreground_ratio:
                return False, (
                    f"Ảnh #{idx} không phát hiện thấy vật thể "
                    f"(diện tích tiền cảnh {ratio*100:.2f}% < {self.min_foreground_ratio*100}%)."
                ), {"foreground_ratios": ratios}

            if ratio > self.max_foreground_ratio:
                return False, (
                    f"Ảnh #{idx} mặt nạ tiền cảnh chiếm gần như toàn bộ khung hình "
                    f"({ratio*100:.2f}% > {self.max_foreground_ratio*100}%), khả năng tách nền lỗi."
                ), {"foreground_ratios": ratios}

        # Kiểm tra độ biến thiên depth maps
        depth_variances = []
        if depth_maps is not None:
            for idx, d in enumerate(depth_maps):
                std_d = float(np.std(d))
                depth_variances.append(std_d)
                if std_d < 1e-4:
                    return False, (
                        f"Depth map #{idx} không có độ biến thiên hình học (std={std_d:.6f})."
                    ), {"depth_variances": depth_variances}

        return True, "Silhouette và Depth hợp lệ.", {
            "foreground_ratios": [round(r, 4) for r in ratios],
            "depth_variances": [round(v, 4) for v in depth_variances] if depth_variances else [],
        }

    def evaluate(
        self,
        camera_poses: Optional[List[np.ndarray]] = None,
        alpha_masks: Optional[List[np.ndarray]] = None,
        depth_maps: Optional[List[np.ndarray]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Đánh giá tổng hợp toàn bộ chuỗi dữ liệu đầu vào.

        Returns:
            passed: bool
            reason: str
            info: dict chứa chi tiết metrics và chỉ dẫn fallback_to_single_view.
        """
        info = {
            "fallback_to_single_view": False,
            "best_view_index": 0,
        }

        # 1. Kiểm tra camera poses nếu có
        if camera_poses is not None and len(camera_poses) >= 2:
            cam_ok, cam_reason, cam_metrics = self.evaluate_camera_poses(camera_poses)
            info.update(cam_metrics)
            if not cam_ok:
                info["fallback_to_single_view"] = True
                return False, cam_reason, info

        # 2. Kiểm tra silhouettes và depth nếu có
        if alpha_masks is not None and len(alpha_masks) > 0:
            mask_ok, mask_reason, mask_metrics = self.evaluate_silhouettes_and_depth(
                alpha_masks=alpha_masks,
                depth_maps=depth_maps,
            )
            info.update(mask_metrics)
            if not mask_ok:
                info["fallback_to_single_view"] = True
                return False, mask_reason, info

        return True, "Quality passed: Dữ liệu đạt chuẩn NVIDIA 3D Pipeline.", info