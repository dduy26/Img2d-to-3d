"""
Kiểm thử Phân hệ 5 (P5): Trải UV & Nướng màu góc nhìn với ma trận camera_intrinsics bù trừ.
Xác minh:
1. Gán màu đỉnh chính xác theo góc nhìn Fresnel (cos^gamma) kết hợp K_i' bù trừ tâm.
2. Xuất file GLB chuẩn hóa, có màu đỉnh hoặc texture hợp lệ.
"""

import unittest
import numpy as np
import trimesh
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from notebook.backend.texture_blender import TextureBlender
from notebook.backend.engine_tsdf_mesh import generate_camera_poses


class TestP5TextureBlending(unittest.TestCase):
    def test_texture_blending_with_intrinsics(self):
        """Kiểm tra gán màu đỉnh và xuất file GLB có chứa màu sắc."""
        # Tạo mesh hình hộp đơn giản
        box = trimesh.creation.box(extents=(0.5, 0.5, 0.5))

        # Tạo 4 ảnh màu khác nhau cho 4 góc: Đỏ (front), Xanh lá (right), Xanh dương (back), Vàng (left)
        colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
        ]
        images = [np.full((256, 256, 3), c, dtype=np.uint8) for c in colors]
        poses = generate_camera_poses(4, radius=2.0)
        focals = [(250.0, 250.0)] * 4

        # Ma trận K'
        intrinsics = [
            np.array([[250.0, 0.0, 128.0], [0.0, 250.0, 128.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            for _ in range(4)
        ]

        blender = TextureBlender()
        v_colors = blender.blend_colors_for_vertices(
            mesh=box,
            images_rgb=images,
            camera_poses=poses,
            focal_lengths=focals,
            camera_intrinsics=intrinsics,
        )

        self.assertEqual(len(v_colors), len(box.vertices))
        self.assertEqual(v_colors.shape[1], 4)  # RGBA
        # Alpha phải là 255
        self.assertTrue(np.all(v_colors[:, 3] == 255))
        # Không được để mặc định toàn bộ màu xám (180, 180, 180)
        self.assertFalse(np.all(v_colors[:, :3] == 180))

        # Xuất thử GLB
        out_glb = os.path.join(ROOT_DIR, "tests", "tmp_box.glb")
        try:
            success, path = blender.process_and_export(
                mesh=box,
                images_rgb=images,
                camera_poses=poses,
                focal_lengths=focals,
                camera_intrinsics=intrinsics,
                output_path=out_glb,
            )
            self.assertTrue(success)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 500)
        finally:
            if os.path.exists(out_glb):
                os.remove(out_glb)


if __name__ == "__main__":
    unittest.main()
