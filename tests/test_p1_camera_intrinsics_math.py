"""
test_p1_camera_intrinsics_math.py — Kiem thu toan hoc K -> K' P1
"""

import unittest
import numpy as np
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# API moi
from notebook.backend.utils_3d import (
    CameraIntrinsics,
    compensate_intrinsics,
    compute_uniform_scale_params,
    project_points_batch,
    round_to_16,
)
from notebook.backend.preprocess import place_on_canvas


class TestP1CameraIntrinsicsMath(unittest.TestCase):
    def test_mathematical_projection_equivalence(self):
        """
        Chung minh toan hoc:
        Chieu qua K' phai cho ket qua giong chieu qua K goc + bien doi pixel.
        Sai so |u_canvas - u_proj| < 1e-5 voi moi diem 3D ngau nhien.
        """
        W_orig, H_orig = 640, 480
        c_x0, c_y0 = W_orig / 2.0, H_orig / 2.0
        fov_deg = 50.0
        f_0 = float((W_orig / 2.0) / np.tan(np.radians(fov_deg / 2.0)))

        # Uniform Scale va letterbox
        target_size = 512
        scale, new_w, new_h, ox, oy = compute_uniform_scale_params(W_orig, H_orig, target_size)

        # Tinh K'
        f_prime   = f_0 * scale
        cx_prime  = c_x0 * scale + ox
        cy_prime  = c_y0 * scale + oy

        K_prime_mat = np.array([
            [f_prime, 0.0,      cx_prime],
            [0.0,     f_prime,  cy_prime],
            [0.0,     0.0,      1.0     ],
        ])

        # Sinh 1000 diem 3D ngau nhien truoc camera (Z > 0.5)
        rng = np.random.default_rng(42)
        Xw = rng.uniform(-0.5, 0.5, 1000)
        Yw = rng.uniform(-0.5, 0.5, 1000)
        Zw = rng.uniform(1.0, 3.0, 1000)

        # 1. Chieu theo camera goc va bien doi toa do pixel tren canvas
        u_orig   = f_0 * (Xw / Zw) + c_x0
        v_orig   = f_0 * (Yw / Zw) + c_y0
        u_canvas = u_orig * scale + ox
        v_canvas = v_orig * scale + oy

        # 2. Chieu truc tiep qua ma tran bu tru K'
        u_proj = K_prime_mat[0, 0] * (Xw / Zw) + K_prime_mat[0, 2]
        v_proj = K_prime_mat[1, 1] * (Yw / Zw) + K_prime_mat[1, 2]

        max_err_u = np.max(np.abs(u_canvas - u_proj))
        max_err_v = np.max(np.abs(v_canvas - v_proj))

        self.assertLess(max_err_u, 1e-5, f"Sai so chieu u qua lon: {max_err_u}")
        self.assertLess(max_err_v, 1e-5, f"Sai so chieu v qua lon: {max_err_v}")

    def test_compensate_intrinsics_returns_valid_K_prime(self):
        """compensate_intrinsics phai tra ve K' hop le voi fx, fy > 0."""
        for orig_w, orig_h in [(640, 480), (800, 1200), (400, 400), (1920, 1080)]:
            K = CameraIntrinsics.default_ortho(orig_w, orig_h)
            K_prime, scale = compensate_intrinsics(K, orig_w, orig_h, 512)
            self.assertGreater(K_prime.fx, 0.0, f"fx phai duong cho {orig_w}x{orig_h}")
            self.assertGreater(K_prime.fy, 0.0, f"fy phai duong cho {orig_w}x{orig_h}")
            self.assertGreater(scale, 0.0, "scale phai duong")
            # K_prime.cx phai trong khoang hop ly (0 den canvas_size)
            self.assertGreater(K_prime.cx, 0.0)
            self.assertLess(K_prime.cx, 512.0 * 2)

    def test_round_to_16_correctness(self):
        """round_to_16 phai tra ve boi so cua 16."""
        test_vals = [100, 256, 300, 480, 512, 640, 768, 1000]
        for v in test_vals:
            result = round_to_16(v)
            self.assertEqual(result % 16, 0, f"round_to_16({v}) = {result} khong chia het cho 16!")

    def test_project_points_batch_consistency(self):
        """Chieu hang loat phai cho ket qua giong chieu tung diem."""
        K = CameraIntrinsics(fx=300, fy=300, cx=256, cy=256, width=512, height=512)
        R = np.eye(3)
        t = np.array([0.0, 0.0, 3.0])
        pts = np.array([[0.1, 0.2, 0], [-0.3, 0.1, 0], [0.0, -0.2, 0]], dtype=np.float64)

        uvs, valid = project_points_batch(pts, K, R, t)
        self.assertEqual(uvs.shape, (3, 2))
        self.assertTrue(valid.all())
        # Kiem tra tinh nhat quan
        for i, pt in enumerate(pts):
            Xc = R @ pt + t
            expected_u = K.fx * Xc[0] / Xc[2] + K.cx
            expected_v = K.fy * Xc[1] / Xc[2] + K.cy
            self.assertAlmostEqual(uvs[i, 0], expected_u, places=4)
            self.assertAlmostEqual(uvs[i, 1], expected_v, places=4)


if __name__ == "__main__":
    unittest.main()
