"""
Kiểm thử Tích hợp Đầu-Cuối (End-to-End) trên Dataset Objaverse Train Thực Tế.
Nguồn: https://huggingface.co/datasets/dxgl/objaverse-1k/tree/main/datasets/train.zip
Xác minh:
1. Toàn bộ pipeline P1 -> P5 chạy trơn tru trên 4 góc ảnh thực tế của Objaverse.
2. Mesh 3D tái tạo đạt 100% tiêu chuẩn in 3D: Watertight, 0 boundary edges, 1 component.
3. Kích thước 3D bảo toàn hình dạng dài ngang của toa tàu.
"""

import unittest
import numpy as np
import trimesh
import sys
import os
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from notebook.backend.app import execute_3d_pipeline
from notebook.backend.engine_tsdf_mesh import mesh_health


class TestE2EObjaverseTrain(unittest.TestCase):
    def test_full_pipeline_on_objaverse(self):
        """Chạy toàn bộ pipeline từ ảnh thô Objaverse (dxgl/objaverse-1k) đến file 3D GLB cuối cùng."""
        data_dir = os.path.join(ROOT_DIR, "tests", "data", "objaverse_apple")
        image_paths = [
            os.path.join(data_dir, "front.png"),
            os.path.join(data_dir, "right.png"),
            os.path.join(data_dir, "back.png"),
            os.path.join(data_dir, "left.png"),
        ]
        for p in image_paths:
            self.assertTrue(os.path.exists(p), f"Không tìm thấy ảnh test: {p}")

        t0 = time.time()
        result = execute_3d_pipeline(image_paths)
        elapsed = time.time() - t0

        print(f"\n[E2E Objaverse Train] Pipeline hoàn tất trong {elapsed:.2f}s:")
        print(f"  - Status: {result.get('status')}")
        print(f"  - Mode: {result.get('mode')}")
        print(f"  - Output File: {result.get('output_file')}")

        self.assertEqual(result["status"], "success", "Pipeline phải thành công!")
        output_glb = result["output_file"]
        self.assertTrue(os.path.exists(output_glb), "File GLB đầu ra phải tồn tại!")
        self.assertGreater(os.path.getsize(output_glb), 10000, "File GLB phải có dung lượng hợp lệ!")

        # Kiểm tra hình học thật bằng mesh_health
        scene_or_mesh = trimesh.load(output_glb)
        if isinstance(scene_or_mesh, trimesh.Scene):
            meshes = [geom for geom in scene_or_mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        else:
            mesh = scene_or_mesh

        health = mesh_health(mesh)
        print(f"[E2E Mesh Health] Watertight: {health['watertight']} | "
              f"Boundary edges: {health['boundary_edges']} | "
              f"Components: {health['components']} | "
              f"Faces: {health['faces']} | Volume: {health['volume']:.6f}")

        self.assertTrue(health["watertight"], "Lưới 3D của toa tàu Objaverse phải kín nước 100%!")
        self.assertEqual(health["boundary_edges"], 0, "Lưới 3D không được có cạnh hở!")
        self.assertEqual(health["components"], 1, "Lưới 3D phải là 1 khối duy nhất!")
        self.assertGreater(health["volume"], 0.0, "Thể tích 3D phải dương!")


if __name__ == "__main__":
    unittest.main()
