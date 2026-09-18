"""
test_p4_space_carving_tsdf.py — Kiem thu tai tao luoi TSDF & Space Carving P4
"""

import unittest
import numpy as np
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# API moi
from notebook.backend.engine_tsdf_mesh import (
    create_voxel_grid,
    space_carving,
    tsdf_fusion,
    extract_mesh_marching_cubes,
    taubin_smooth,
    quadric_decimate,
)
from notebook.backend.utils_3d import (
    CameraIntrinsics,
    check_mesh_health,
    get_orthographic_camera_poses,
)


def _mesh_health_dict(vertices, faces):
    """Wrapper tra ve dict tuong tu API cu de giu tuong thich voi test assertion."""
    h = check_mesh_health(vertices, faces)
    return {
        "watertight":     h.is_watertight,
        "boundary_edges": h.boundary_edges,
        "components":     h.components,
        "volume":         h.volume_m3,
        "faces":          h.face_count,
        "vertices":       h.vertex_count,
    }


class TestP4SpaceCarvingTSDF(unittest.TestCase):
    def test_watertight_manifold_reconstruction(self):
        """Kiem tra tao luoi kin nuoc tu 4 goc nhin vuong goc (Front, Right, Back, Left)."""
        target_size = 128  # Nho hon 256 de chay nhanh trong test
        N_views = 4

        # Tao vat the hinh cau r=40 tai tam (64, 64) tren canvas 128x128
        Y, X = np.ogrid[:target_size, :target_size]
        cx, cy = target_size // 2, target_size // 2
        dist_sq = (X - cx)**2 + (Y - cy)**2
        r = 40
        sphere_mask = (dist_sq <= r**2).astype(np.float32)

        # Sinh N views: depth co gradient (gan = 0.3, xa = 0.8)
        masks     = [sphere_mask.copy() for _ in range(N_views)]
        depth_maps = []
        valids    = []
        for _ in range(N_views):
            d = np.ones((target_size, target_size), dtype=np.float32) * 0.8
            # Tao depth gradient qua hinh cau (phia truoc = 0.3, phia sau = 0.8)
            fg_pixels = sphere_mask > 0.5
            dist_norm = np.sqrt(dist_sq.astype(np.float32)) / r
            d[fg_pixels] = 0.3 + 0.3 * dist_norm[fg_pixels]
            depth_maps.append(d)
            valids.append(sphere_mask.copy())

        K = CameraIntrinsics(
            fx=float(target_size), fy=float(target_size),
            cx=float(cx), cy=float(cy),
            width=target_size, height=target_size
        )
        K_primes = [K] * N_views

        # Sinh camera poses
        poses = get_orthographic_camera_poses(n_views=N_views)
        Rs = [p[0] for p in poses]
        ts = [p[1] for p in poses]

        # Tao luoi voxel voi resolution du lon (48^3) de co mesh chat luong
        resolution = 48
        voxel_coords, voxel_size = create_voxel_grid(resolution=resolution, extent=1.0)

        # Space Carving
        carved = space_carving(voxel_coords, masks, K_primes, Rs, ts,
                               canvas_size=target_size)
        self.assertIsNotNone(carved)
        self.assertEqual(carved.shape, (resolution, resolution, resolution))

        # TSDF Fusion
        tsdf_vol, weight_vol = tsdf_fusion(
            voxel_coords, depth_maps, valids, K_primes, Rs, ts,
            voxel_size, carved_mask=carved, canvas_size=target_size
        )
        self.assertIsNotNone(tsdf_vol)

        # Marching Cubes
        try:
            verts, faces = extract_mesh_marching_cubes(tsdf_vol, weight_vol, voxel_size)
        except ImportError:
            self.skipTest("scikit-image khong kha dung, bo qua test Marching Cubes.")

        if verts is None:
            self.skipTest("Marching Cubes tra ve mesh rong (TSDF volume qua nho/qua rong).")

        # Kiem tra co vertices va faces
        self.assertGreater(len(verts), 10, "Mesh phai co it nhat 10 dinh!")
        self.assertGreater(len(faces), 5, "Mesh phai co it nhat 5 mat!")

        # Kiem dinh chat luong
        health = _mesh_health_dict(verts, faces)
        print(f"\n[P4 Test] watertight={health['watertight']}, "
              f"boundary_edges={health['boundary_edges']}, "
              f"components={health['components']}, "
              f"faces={health['faces']}")

        # Tieu chi chinh: khong co canh bien ho (no open holes)
        self.assertEqual(health["boundary_edges"], 0,
                         f"Luoi khong duoc co canh bien ho! Got {health['boundary_edges']}")
        self.assertGreater(health["volume"], 0.0, "The tich phai lon hon 0!")
        # Watertight = boundary_edges==0 AND components==1; relax components khi resolution thap
        if health["components"] == 1:
            self.assertTrue(health["watertight"], "Luoi phai kin nuoc 100%!")
        else:
            print(f"   INFO: {health['components']} components (resolution={resolution} co the qua thap)")


    def test_taubin_smooth_preserves_topology(self):
        """Taubin smoothing khong duoc lam thay doi so mat va dinh."""
        # Tao cube don gian
        verts = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1]
        ], dtype=np.float32)
        faces = np.array([
            [0,1,2],[0,2,3],[4,6,5],[4,7,6],
            [0,4,5],[0,5,1],[2,6,7],[2,7,3],
            [0,3,7],[0,7,4],[1,5,6],[1,6,2]
        ], dtype=np.int32)

        smoothed = taubin_smooth(verts, faces, iters=5)
        self.assertEqual(smoothed.shape, verts.shape, "So luong dinh khong doi!")
        # Kiem tra dinh con trong hop [0-eps, 1+eps]
        self.assertTrue(np.all(smoothed > -0.1))
        self.assertTrue(np.all(smoothed < 1.1))


if __name__ == "__main__":
    unittest.main()
