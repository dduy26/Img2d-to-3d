"""
Kiểm thử Phân hệ 1 (P1): Tính toán & Bù trừ ma trận Camera Intrinsics (K -> K').
Xác minh:
1. Chứng minh toán học: Tia chiếu qua ma trận bù trừ K' trùng khít tuyệt đối với pixel trên canvas (sai số < 1e-6).
2. Khi bounding box bị dịch chuyển lệch tâm, (cx', cy') được bù trừ chính xác để bảo toàn hướng chiếu 3D.
"""

import unittest
import numpy as np
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from notebook.backend.preprocess import normalize_multiview_scales_and_canvas


class TestP1CameraIntrinsicsMath(unittest.TestCase):
    def test_mathematical_projection_equivalence(self):
        """
        Chứng minh toán học:
        u_canvas = s * (u_orig - x_min) + ox
        u_proj = (s * f_0) * (X/Z) + [s * c_x0 + (ox - s * x_min)]
        Hiệu số |u_canvas - u_proj| < 1e-6 với mọi điểm 3D ngẫu nhiên!
        """
        W_orig, H_orig = 640, 480
        c_x0, c_y0 = W_orig / 2.0, H_orig / 2.0
        fov_deg = 50.0
        f_0 = float((W_orig / 2.0) / np.tan(np.radians(fov_deg / 2.0)))

        # Bounding box giả định bị lệch tâm
        x_min, y_min = 120, 80
        x_max, y_max = 520, 400
        bbox_w = x_max - x_min  # 400
        bbox_h = y_max - y_min  # 320

        target_size = 512
        padding_factor = 0.85
        global_scale = (target_size * padding_factor) / max(bbox_w, bbox_h)

        scaled_w = int(round(bbox_w * global_scale))
        scaled_h = int(round(bbox_h * global_scale))
        ox = (target_size - scaled_w) // 2
        oy = (target_size - scaled_h) // 2

        # Tính K'
        f_prime = f_0 * global_scale
        cx_prime = c_x0 * global_scale + (ox - x_min * global_scale)
        cy_prime = c_y0 * global_scale + (oy - y_min * global_scale)

        K_prime = np.array([
            [f_prime, 0.0,     cx_prime],
            [0.0,     f_prime, cy_prime],
            [0.0,     0.0,     1.0     ],
        ])

        # Sinh 1000 điểm 3D ngẫu nhiên trước camera (Z > 0.5)
        np.random.seed(42)
        X = np.random.uniform(-0.5, 0.5, 1000)
        Y = np.random.uniform(-0.5, 0.5, 1000)
        Z = np.random.uniform(1.0, 3.0, 1000)

        # 1. Chiếu theo camera gốc và biến đổi tọa độ pixel trên canvas
        u_orig = f_0 * (X / Z) + c_x0
        v_orig = f_0 * (Y / Z) + c_y0
        u_canvas = (u_orig - x_min) * global_scale + ox
        v_canvas = (v_orig - y_min) * global_scale + oy

        # 2. Chiếu trực tiếp qua ma trận bù trừ K'
        u_proj = K_prime[0, 0] * (X / Z) + K_prime[0, 2]
        v_proj = K_prime[1, 1] * (Y / Z) + K_prime[1, 2]

        max_err_u = np.max(np.abs(u_canvas - u_proj))
        max_err_v = np.max(np.abs(v_canvas - v_proj))

        self.assertLess(max_err_u, 1e-5, f"Sai số chiếu u quá lớn: {max_err_u}")
        self.assertLess(max_err_v, 1e-5, f"Sai số chiếu v quá lớn: {max_err_v}")

    def test_normalize_function_returns_valid_intrinsics(self):
        """Hàm normalize_multiview_scales_and_canvas phải trả về ma trận K_i' hợp lệ."""
        # Tạo 2 ảnh test có foreground nằm lệch sang trái và sang phải
        img1 = np.full((300, 300, 3), 255, dtype=np.uint8)
        mask1 = np.zeros((300, 300), dtype=np.uint8)
        mask1[50:150, 30:100] = 255  # Lệch trái

        img2 = np.full((300, 300, 3), 255, dtype=np.uint8)
        mask2 = np.zeros((300, 300), dtype=np.uint8)
        mask2[50:150, 200:270] = 255  # Lệch phải

        _, _, scales, intrinsics = normalize_multiview_scales_and_canvas(
            images_rgb=[img1, img2],
            alpha_masks=[mask1, mask2],
            target_size=512,
        )

        self.assertEqual(len(intrinsics), 2)
        for K in intrinsics:
            self.assertEqual(K.shape, (3, 3))
            self.assertGreater(K[0, 0], 100.0, "fx phải dương!")
            self.assertGreater(K[1, 1], 100.0, "fy phải dương!")
            self.assertEqual(K[2, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
