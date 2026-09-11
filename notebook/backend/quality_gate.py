import numpy as np

class QualityGate:
    def __init__(self, min_confidence=0.45, max_loss=2.5, min_angle_deg=5.0, max_angle_deg=120.0):
        # Thiết lập các hằng số làm ngưỡng đánh giá
        self.min_confidence = min_confidence
        self.max_loss = max_loss
        self.min_angle = min_angle_deg
        self.max_angle = max_angle_deg

    def _calculate_angle_between_poses(self, poses):
        """Tính toán góc chênh lệch giữa các camera pose (giả định có 2 pose chính)"""
        if len(poses) < 2:
            return 0.0
        
        # Trích xuất vector hướng nhìn (forward vector) từ ma trận rotation 3x3
        # Giả định trục Z (cột thứ 3) là hướng nhìn của camera
        z_dir_1 = poses[0][:3, 2]
        z_dir_2 = poses[1][:3, 2]
        
        # Tính góc bằng Cosine Similarity
        cos_theta = np.dot(z_dir_1, z_dir_2) / (np.linalg.norm(z_dir_1) * np.linalg.norm(z_dir_2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        return angle

    def evaluate(self, poses, confidence_map, ba_loss):
        """
        Đánh giá chất lượng tái tạo 3D.
        Trả về: (is_valid: bool, reason: str)
        """
        # 1. Kiểm tra Confidence (Độ tự tin của mô hình matching)
        mean_conf = np.mean(confidence_map)
        if mean_conf < self.min_confidence:
            return False, f"Low confidence: {mean_conf:.2f} < {self.min_confidence}"

        # 2. Kiểm tra Bundle Adjustment Loss (Độ lệch sau tối ưu)
        if ba_loss > self.max_loss:
            return False, f"High loss: {ba_loss:.2f} > {self.max_loss}"

        # 3. Kiểm tra Camera Overlap (Độ phủ của góc chụp)
        if poses is not None and len(poses) >= 2:
            angle = self._calculate_angle_between_poses(poses)
            if angle < self.min_angle or angle > self.max_angle:
                return False, f"Bad overlap angle: {angle:.2f} degrees"

        # Vượt qua toàn bộ bài test
        return True, "Quality passed"   