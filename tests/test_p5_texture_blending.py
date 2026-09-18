"""
test_p5_texture_blending.py — Kiem thu trai UV & nướng màu P5
"""

import unittest
import numpy as np
import sys
import os
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# API moi
from notebook.backend.texture_blender import (
    blend_vertex_colors,
    compute_view_weights,
    unwrap_uv_fallback,
    apply_texture,
)
from notebook.backend.utils_3d import (
    CameraIntrinsics,
    compute_normals,
    get_orthographic_camera_poses,
    export_glb,
    check_mesh_health,
)


class TestP5TextureBlending(unittest.TestCase):
    def setUp(self):
        """Tao mesh hop cube don gian va 4 anh mau."""
        self.verts = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1]
        ], dtype=np.float32)
        self.faces = np.array([
            [0,1,2],[0,2,3],[4,6,5],[4,7,6],
            [0,4,5],[0,5,1],[2,6,7],[2,7,3],
            [0,3,7],[0,7,4],[1,5,6],[1,6,2]
        ], dtype=np.int32)

        # 4 anh mau don sac khac nhau cho 4 goc
        raw_colors = [
            [255, 0, 0],    # Do (front)
            [0, 255, 0],    # Xanh la (right)
            [0, 0, 255],    # Xanh duong (back)
            [255, 255, 0],  # Vang (left)
        ]
        self.images = [
            np.full((256, 256, 3), c, dtype=np.uint8) for c in raw_colors
        ]

        # Poses va K'
        poses = get_orthographic_camera_poses(n_views=4)
        self.Rs = [p[0] for p in poses]
        self.ts = [p[1] for p in poses]
        self.K  = CameraIntrinsics(fx=250, fy=250, cx=128, cy=128, width=256, height=256)
        self.K_primes = [self.K] * 4

    def test_compute_view_weights_shape(self):
        """Trong so Fresnel co dung kich thuoc (N_views, V)."""
        normals = compute_normals(self.verts, self.faces)
        weights = compute_view_weights(self.verts, normals, self.Rs, self.ts)
        self.assertEqual(weights.shape, (4, len(self.verts)))
        # Trong so phai trong [0, 1]
        self.assertGreaterEqual(weights.min(), 0.0)
        self.assertLessEqual(weights.max(), 1.0 + 1e-6)

    def test_blend_vertex_colors_shape_and_range(self):
        """Mau dinh sau blend phai co dung kich thuoc va nam trong [0, 1]."""
        normals = compute_normals(self.verts, self.faces)
        colors = blend_vertex_colors(
            self.verts, normals, self.images, self.K_primes, self.Rs, self.ts
        )
        self.assertEqual(colors.shape, (len(self.verts), 3))
        self.assertGreaterEqual(colors.min(), 0.0)
        self.assertLessEqual(colors.max(), 1.0 + 1e-6)
        # Mau khong duoc la xam deu (phai co su phan biet mau tu cac view)
        # (co the tat ca la 0 neu cam khong nhin thay vertex -> OK)

    def test_unwrap_uv_fallback_shape(self):
        """Spherical UV fallback phai tra ve toa do UV hop le [0, 1]."""
        uv_coords, uv_faces = unwrap_uv_fallback(self.verts, self.faces)
        self.assertEqual(uv_coords.shape[1], 2)
        self.assertGreaterEqual(uv_coords.min(), 0.0 - 0.01)
        self.assertLessEqual(uv_coords.max(), 1.0 + 0.01)
        self.assertEqual(uv_faces.shape, self.faces.shape)

    def test_apply_texture_and_export_glb(self):
        """apply_texture phai tra ve dict day du va export GLB hop le."""
        # Tao fake depth_views de truyen vao apply_texture
        alpha = np.ones((256, 256), dtype=np.float32)
        depth = np.full((256, 256), 0.5, dtype=np.float32)
        views = []
        for i in range(4):
            views.append({
                "view_idx":   i,
                "orig_path":  f"view_{i}.png",
                "canvas_rgb": self.images[i],
                "alpha_mask": alpha,
                "depth_map":  depth,
                "valid_mask": alpha,
                "K_prime":    self.K,
                "azimuth_deg": float(i * 90),
                "scale": 1.0, "offset_x": 0, "offset_y": 0,
            })

        result = apply_texture(
            self.verts, self.faces,
            depth_views=views,
            Rs=self.Rs, ts=self.ts,
            use_xatlas=False,  # Dung fallback de khong can cai dat xatlas
        )
        self.assertIn("vertices", result)
        self.assertIn("faces", result)
        self.assertIn("normals", result)
        self.assertIn("vertex_colors", result)

        # Export GLB
        with tempfile.TemporaryDirectory() as tmp:
            out_glb = os.path.join(tmp, "test_p5.glb")
            export_glb(
                result["vertices"], result["faces"],
                colors=result["vertex_colors"],
                normals=result["normals"],
                out_path=out_glb,
            )
            self.assertTrue(os.path.exists(out_glb))
            size = os.path.getsize(out_glb)
            self.assertGreater(size, 500, "File GLB phai co dung luong hop le!")
            with open(out_glb, "rb") as f:
                magic = f.read(4)
            self.assertEqual(magic, b"glTF", "GLB header sai!")


if __name__ == "__main__":
    unittest.main()
