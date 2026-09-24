"""
test_p6_hunyuan3d_engine.py — Tests cho Tencent Hunyuan3D-2mv Multi-View Engine
=============================================================================
Kiem tra:
1. prepare_multiview_dict phan loai anh thanh 4 goc: front, right, back, left
2. reconstruct_hunyuan3d sinh file .glb hop le, kin nuoc (watertight)
"""

import os
import unittest
import tempfile
import numpy as np
from PIL import Image

from notebook.backend.engine_hunyuan3d import (
    prepare_multiview_dict,
    reconstruct_hunyuan3d,
)


class TestP6Hunyuan3DEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.temp_dir.name
        
        # Tao 4 anh gia lap 4 goc nhin
        self.image_paths = []
        for i, name in enumerate(["cam0_front", "cam1_right", "cam2_back", "cam3_left"]):
            img_p = os.path.join(self.tmp_path, f"{name}.png")
            arr = np.zeros((128, 128, 3), dtype=np.uint8)
            arr[30:90, 30:90] = [(i * 60) % 255, 128, 200]
            Image.fromarray(arr).save(img_p)
            self.image_paths.append(img_p)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prepare_multiview_dict_slots(self):
        """Kiem tra prepare_multiview_dict tao du 4 goc front, right, back, left."""
        mv_dict = prepare_multiview_dict(self.image_paths)
        self.assertIsInstance(mv_dict, dict)
        for slot in ["front", "right", "back", "left"]:
            self.assertIn(slot, mv_dict)
            self.assertTrue(os.path.exists(mv_dict[slot]))

    def test_prepare_multiview_dict_fewer_images(self):
        """Kiem tra voi 2 anh dau vao, van dien du 4 slot (fallback du phong)."""
        mv_dict = prepare_multiview_dict(self.image_paths[:2])
        self.assertEqual(len(mv_dict), 4)
        for slot in ["front", "right", "back", "left"]:
            self.assertIn(slot, mv_dict)

    def test_reconstruct_hunyuan3d_generates_valid_glb(self):
        """Kiem tra reconstruct_hunyuan3d xuat file .glb hop le, mesh kin nuoc."""
        out_glb = os.path.join(self.tmp_path, "model_output.glb")
        res = reconstruct_hunyuan3d(
            image_paths=self.image_paths,
            output_glb=out_glb,
            device="cpu",
            allow_fallback_mesh=True,
        )
        self.assertEqual(res["status"], "success")
        self.assertTrue(os.path.exists(out_glb))
        self.assertGreater(os.path.getsize(out_glb), 0)
        
        mesh_info = res.get("mesh_info", {})
        self.assertGreater(mesh_info.get("vertex_count", 0), 0)
        self.assertGreater(mesh_info.get("face_count", 0), 0)
        self.assertTrue(mesh_info.get("is_watertight", False))

    def test_reconstruct_hunyuan3d_raises_without_fallback(self):
        """Kiem tra reconstruct_hunyuan3d nem loi khi thieu GPU/weights va khong cho phep mock."""
        out_glb = os.path.join(self.tmp_path, "model_no_mock.glb")
        with self.assertRaises(RuntimeError):
            reconstruct_hunyuan3d(
                image_paths=self.image_paths,
                output_glb=out_glb,
                device="cpu",
                allow_fallback_mesh=False,
            )

    def test_apply_multiview_texture_paints_mesh(self):
        """Kiem tra apply_multiview_texture phu mau day du len dinh mesh."""
        import trimesh
        from notebook.backend.engine_hunyuan3d import apply_multiview_texture
        mesh = trimesh.creation.icosphere(radius=0.5, subdivisions=2)
        mesh_colored = apply_multiview_texture(mesh, self.image_paths)
        self.assertTrue(hasattr(mesh_colored.visual, "vertex_colors"))
        self.assertEqual(len(mesh_colored.visual.vertex_colors), len(mesh.vertices))
        self.assertEqual(mesh_colored.visual.vertex_colors.shape[1], 4)


if __name__ == "__main__":
    unittest.main()
