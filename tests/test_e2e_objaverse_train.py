"""
test_e2e_objaverse_train.py — Kiem thu tich hop E2E tren Dataset Objaverse Train
"""

import unittest
import numpy as np
import sys
import os
import time
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# API moi
from notebook.backend.preprocess      import preprocess_images
from notebook.backend.quality_gate    import run_quality_gate
from notebook.backend.engine_tsdf_mesh import (
    create_voxel_grid, space_carving, tsdf_fusion,
    extract_mesh_marching_cubes, taubin_smooth, quadric_decimate,
)
from notebook.backend.texture_blender import apply_texture
from notebook.backend.utils_3d import (
    check_mesh_health,
    export_glb,
    compute_normals,
    get_orthographic_camera_poses,
)


def _simple_pipeline_no_ai(image_paths: list, output_path: str) -> dict:
    """Pipeline E2E don gian khong can GPU (dung fake depth).

    Thay the cho execute_3d_pipeline (API cu) de test dat hang chi voi CPU.
    Dung do sau dong nhat (flat depth) de kiem tra Space Carving la chinh.
    """
    # P1: Tien xu ly
    preprocessed = preprocess_images(image_paths, normalize_exposure_flag=False)
    if not preprocessed:
        return {"status": "error", "error": "No images"}

    # P3: Quality Gate (khong can depth model)
    # Tao fake depth views tu preprocessed
    fake_depth_views = []
    for v in preprocessed:
        alpha = v["alpha_mask"]
        depth = np.where(alpha > 0.5, 0.5, 0.0).astype(np.float32)
        fv = dict(v)
        fv["depth_map"]  = depth
        fv["valid_mask"] = alpha
        fake_depth_views.append(fv)

    gate = run_quality_gate(fake_depth_views)
    active_views = gate.valid_views if gate.mode == "multi_view" else [gate.anchor_view]

    N = len(active_views)
    resolution = 32  # Nho de chay nhanh

    # P4: TSDF Fusion
    alphas    = [v["alpha_mask"]  for v in active_views]
    depths    = [v["depth_map"]   for v in active_views]
    valids    = [v["valid_mask"]  for v in active_views]
    K_primes  = [v["K_prime"]     for v in active_views]
    canvas_sz = active_views[0]["canvas_rgb"].shape[1]

    poses = get_orthographic_camera_poses(n_views=N)
    Rs = [p[0] for p in poses]
    ts = [p[1] for p in poses]

    voxel_coords, voxel_size = create_voxel_grid(resolution=resolution)
    carved = space_carving(voxel_coords, alphas, K_primes, Rs, ts, canvas_size=canvas_sz)
    tsdf_vol, weight_vol = tsdf_fusion(
        voxel_coords, depths, valids, K_primes, Rs, ts, voxel_size,
        carved_mask=carved, canvas_size=canvas_sz
    )

    try:
        from skimage.measure import marching_cubes
        verts, faces = extract_mesh_marching_cubes(tsdf_vol, weight_vol, voxel_size)
    except ImportError:
        return {"status": "error", "error": "scikit-image khong kha dung"}

    if verts is None:
        return {"status": "error", "error": "Marching Cubes ra mesh rong"}

    # Smooth nhe
    verts = taubin_smooth(verts, faces, iters=3)

    # P5: Texture (don gian)
    result_tex = apply_texture(verts, faces, active_views, Rs, ts, use_xatlas=False)

    # Export GLB
    export_glb(
        result_tex["vertices"], result_tex["faces"],
        colors=result_tex["vertex_colors"],
        normals=result_tex["normals"],
        out_path=output_path,
    )

    health = check_mesh_health(result_tex["vertices"], result_tex["faces"])
    return {
        "status": "success",
        "output_file": output_path,
        "mode": gate.mode,
        "watertight": health.is_watertight,
        "boundary_edges": health.boundary_edges,
        "components": health.components,
        "faces": health.face_count,
        "volume": health.volume_m3,
    }


class TestE2EObjaverseTrain(unittest.TestCase):
    def test_full_pipeline_no_gpu_on_objaverse(self):
        """Chay toan bo pipeline (khong GPU) tren anh Objaverse apple."""
        data_dir = os.path.join(ROOT_DIR, "tests", "data", "objaverse_apple")
        if not os.path.exists(data_dir):
            self.skipTest("Dataset Objaverse apple chua san sang!")

        image_paths = [
            os.path.join(data_dir, "front.png"),
            os.path.join(data_dir, "right.png"),
            os.path.join(data_dir, "back.png"),
            os.path.join(data_dir, "left.png"),
        ]
        for p in image_paths:
            if not os.path.exists(p):
                self.skipTest(f"Thieu anh: {p}")

        with tempfile.TemporaryDirectory() as tmp:
            out_glb = os.path.join(tmp, "e2e_apple.glb")
            t0 = time.time()
            result = _simple_pipeline_no_ai(image_paths, out_glb)
            elapsed = time.time() - t0

        print(f"\n[E2E Test] Pipeline hoan tat trong {elapsed:.2f}s")
        print(f"  Status:   {result.get('status')}")
        print(f"  Mode:     {result.get('mode')}")
        print(f"  Faces:    {result.get('faces')}")
        print(f"  WT:       {result.get('watertight')}")
        print(f"  Boundary: {result.get('boundary_edges')}")

        self.assertEqual(result["status"], "success", f"Pipeline that bai: {result.get('error')}")
        self.assertEqual(result["boundary_edges"], 0, "Luoi khong duoc co canh bien ho!")
        self.assertEqual(result["components"], 1, "Luoi phai la 1 khoi lien thong!")
        self.assertTrue(result["watertight"], "Luoi 3D phai kin nuoc 100%!")
        self.assertGreater(result["volume"], 0.0, "The tich phai duong!")

    def test_full_pipeline_no_gpu_on_objaverse_train(self):
        """Chay pipeline tren bo du lieu objaverse_train."""
        data_dir = os.path.join(ROOT_DIR, "tests", "data", "objaverse_train", "images")
        if not os.path.exists(data_dir):
            self.skipTest("Dataset Objaverse train chua san sang!")

        image_paths = [
            os.path.join(data_dir, "front.png"),
            os.path.join(data_dir, "right.png"),
            os.path.join(data_dir, "back.png"),
            os.path.join(data_dir, "left.png"),
        ]
        for p in image_paths:
            if not os.path.exists(p):
                self.skipTest(f"Thieu anh: {p}")

        with tempfile.TemporaryDirectory() as tmp:
            out_glb = os.path.join(tmp, "e2e_train.glb")
            result = _simple_pipeline_no_ai(image_paths, out_glb)

        self.assertEqual(result["status"], "success", f"Pipeline that bai: {result.get('error')}")
        self.assertEqual(result["boundary_edges"], 0)
        self.assertTrue(result["watertight"])


if __name__ == "__main__":
    unittest.main()
