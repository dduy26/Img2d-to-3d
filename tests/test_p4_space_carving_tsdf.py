"""
Kiểm thử Phân hệ 4 (P4): Tái tạo lưới TSDF thể tích & True Silhouette Space Carving.
Xác minh:
1. Lưới 3D tạo ra đạt chuẩn Kín nước (Watertight: True, Boundary edges: 0, Components: 1).
2. Tích hợp trực tiếp ma trận camera_intrinsics K' từ P1 để chiếu tia chính xác.
3. Không bị lỗi non-manifold, không bị thủng lỗ, thể tích dương.
"""

import unittest
import numpy as np
import trimesh
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from notebook.backend.engine_tsdf_mesh import TSDFMeshEngine, mesh_health


class TestP4SpaceCarvingTSDF(unittest.TestCase):
    def test_watertight_manifold_reconstruction(self):
        """Kiểm tra tạo lưới kín nước từ 4 góc nhìn vuông góc (Front, Right, Back, Left)."""
        target_size = 256
        masks = []
        depth_maps = []
        intrinsics = []

        # Tạo vật thể hình cầu r=50 tại tâm (128, 128)
        Y, X = np.ogrid[:target_size, :target_size]
        dist_sq = (X - 128)**2 + (Y - 128)**2
        sphere_mask = (dist_sq <= 50**2).astype(np.uint8) * 255

        # Sinh 4 view giống nhau đối với hình cầu
        for i in range(4):
            masks.append(sphere_mask.copy())
            d = np.full((target_size, target_size), 2.2, dtype=np.float32)
            d[sphere_mask > 127] = 2.2 - 0.20 * np.sqrt(np.maximum(0, 1.0 - dist_sq[sphere_mask > 127]/(50**2)))
            depth_maps.append(d)

            # Giả định ma trận K' từ P1
            f = 274.0
            K = np.array([
                [f,   0.0, 128.0],
                [0.0, f,   128.0],
                [0.0, 0.0, 1.0  ],
            ], dtype=np.float32)
            intrinsics.append(K)

        engine = TSDFMeshEngine(resolution=64, target_faces=10000)
        mesh = engine.reconstruct_from_depth_maps(
            depth_maps=depth_maps,
            alpha_masks=masks,
            camera_intrinsics=intrinsics,
            view_names=["front.png", "right.png", "back.png", "left.png"],
        )

        health = mesh_health(mesh)
        self.assertTrue(health["watertight"], "Lưới phải kín nước 100%!")
        self.assertEqual(health["boundary_edges"], 0, "Lưới không được có cạnh biên hở!")
        self.assertEqual(health["components"], 1, "Lưới phải là 1 khối liên thông duy nhất!")
        self.assertGreater(health["volume"], 0.0, "Thể tích phải lớn hơn 0!")
        self.assertGreater(len(mesh.faces), 500, "Số lượng mặt phải đủ chi tiết!")


if __name__ == "__main__":
    unittest.main()
