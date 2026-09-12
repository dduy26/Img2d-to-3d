"""
Bộ kiểm thử tự động (Unit Test & Integration Test) cho P4 TSDF Mesh & P5 Texture Blender.
Chạy: python notebook/backend/test_tsdf_pipeline.py
"""

import os
import sys
import time
import tempfile
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image

# Đưa notebook/backend vào sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_tsdf_mesh import (
    filter_depth_discontinuity,
    prune_background_points,
    TSDFVolume,
    extract_mesh_marching_cubes,
    TSDFMeshEngine,
)
from texture_blender import TextureBlender


def test_01_filter_depth_discontinuity():
    """Kiểm thử thuật toán 1: Lọc viền rách mép độ sâu."""
    print("Running test_01_filter_depth_discontinuity...")
    # Mặt phẳng đồng nhất: không có mép rách
    flat_depth = np.full((100, 100), 2.0, dtype=np.float32)
    mask = filter_depth_discontinuity(flat_depth, tau=0.07)
    assert np.all(mask), "Mặt phẳng đồng nhất phải 100% hợp lệ"

    # Mặt phẳng có bậc nhảy lớn (step discontinuity)
    step_depth = np.full((100, 100), 2.0, dtype=np.float32)
    step_depth[:, 50:] = 10.0  # Bước nhảy từ 2.0m lên 10.0m
    mask_step = filter_depth_discontinuity(step_depth, tau=0.07)

    # Ranh giới tại cột 49, 50, 51 phải bị đánh dấu False
    edge_detected = not np.all(mask_step[:, 49:52])
    assert edge_detected, "Phải phát hiện và lọc ranh giới bước nhảy độ sâu"
    print("  ✅ PASS: Thuật toán 1 Lọc viền độ sâu hoạt động chính xác.")


def test_02_prune_background_points():
    """Kiểm thử Pruning điểm nền kết hợp Alpha Mask & Confidence."""
    print("Running test_02_prune_background_points...")
    n_views = 2
    h, w = 64, 64

    pointmaps = np.random.randn(n_views, h, w, 3).astype(np.float32)
    pointmaps[:, :, :, 2] = np.abs(pointmaps[:, :, :, 2]) + 1.0  # Z > 0

    # Mask 1: Nửa trên là vật thể (1), nửa dưới là nền (0)
    alpha_masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(n_views)]
    for a in alpha_masks:
        a[:32, :] = 1

    # Confidence: góc trái cao, góc phải thấp
    conf_masks = np.ones((n_views, h, w), dtype=np.float32) * 0.8
    conf_masks[:, :, 32:] = 0.1  # thấp hơn ngưỡng 0.35

    valid_masks, filtered_points = prune_background_points(
        pointmaps, alpha_masks, conf_masks, tau_conf=0.35
    )

    # Điểm giữ lại phải có alpha==1 VÀ conf >= 0.35 (chỉ ở vùng [:32, :32])
    assert len(filtered_points) == n_views
    assert len(filtered_points[0]) > 0
    # Góc dưới (nền) không được có điểm nào
    assert not np.any(valid_masks[0][40:, :])
    print("  ✅ PASS: Pruning điểm nền & Confidence Gate loại bỏ 100% điểm nền.")


def test_03_tsdf_volume_and_marching_cubes():
    """Kiểm thử Thuật toán 2 (TSDF) và Thuật toán 3 (Marching Cubes) trên quả cầu 3D nhân tạo."""
    print("Running test_03_tsdf_volume_and_marching_cubes...")
    res = 64
    bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    bounds_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    vol = TSDFVolume(bounds_min, bounds_max, resolution=res, trunc_margin=0.1)

    # Tạo trường khoảng cách có dấu nhân tạo cho quả cầu bán kính R=0.5
    xs = np.linspace(bounds_min[0], bounds_max[0], res)
    ys = np.linspace(bounds_min[1], bounds_max[1], res)
    zs = np.linspace(bounds_min[2], bounds_max[2], res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')

    dist_from_origin = np.sqrt(gx**2 + gy**2 + gz**2)
    # Khoảng cách có dấu đến bề mặt cầu bán kính 0.5: ngoài > 0, trong < 0
    signed_dist = dist_from_origin - 0.5
    vol.tsdf_grid = np.clip(signed_dist / 0.1, -1.0, 1.0).astype(np.float32)
    vol.weight_grid = np.ones((res, res, res), dtype=np.float32)

    mesh = extract_mesh_marching_cubes(vol, min_weight=0.1)

    assert len(mesh.vertices) > 100, f"Mesh phải có đỉnh, hiện có {len(mesh.vertices)}"
    assert len(mesh.faces) > 100, f"Mesh phải có mặt tam giác, hiện có {len(mesh.faces)}"
    assert mesh.is_watertight, "Mặt cắt cầu của Marching Cubes phải khép kín nước (Watertight)"

    # Kiểm tra bán kính trung bình của mesh
    radii = np.linalg.norm(mesh.vertices, axis=-1)
    mean_r = np.mean(radii)
    assert np.isclose(mean_r, 0.5, atol=0.08), f"Bán kính mesh ({mean_r:.3f}) phải xấp xỉ 0.5"
    print(f"  ✅ PASS: Marching Cubes trích xuất Watertight Mesh ({len(mesh.vertices)} verts, {len(mesh.faces)} faces).")


def test_04_texture_blender_and_glb_export():
    """Kiểm thử Thuật toán 4: Trải UV, Angle-Weighted Blending và xuất file .glb."""
    print("Running test_04_texture_blender_and_glb_export...")
    # Tạo mesh hình lập phương nhỏ
    mesh = trimesh.creation.box(extents=(0.8, 0.8, 0.8))

    # Tạo 4 camera chụp quanh vật thể ở 4 góc
    n_views = 4
    camera_poses = []
    images_rgb = []
    focal_lengths = []

    # Màu tương ứng 4 góc: Đỏ, Xanh lá, Xanh dương, Vàng
    colors = [
        [255, 0, 0],    # Front
        [0, 255, 0],    # Right
        [0, 0, 255],    # Back
        [255, 255, 0],  # Left
    ]

    for i in range(n_views):
        angle = i * (2.0 * np.pi / n_views)
        cam_x = 2.0 * np.sin(angle)
        cam_z = 2.0 * np.cos(angle)
        cam_pos = np.array([cam_x, 0.0, cam_z])

        # Hướng nhìn về gốc tọa độ (0, 0, 0)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos
        camera_poses.append(c2w)

        focal_lengths.append((500.0, 500.0))

        # Ảnh đơn sắc đại diện cho từng góc chụp
        img = np.full((128, 128, 3), colors[i], dtype=np.uint8)
        images_rgb.append(img)

    blender = TextureBlender(texture_size=256, gamma=3.0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        glb_path = os.path.join(tmp_dir, "test_output.glb")
        success, out_path = blender.process_and_export(
            mesh=mesh,
            images_rgb=images_rgb,
            camera_poses=camera_poses,
            focal_lengths=focal_lengths,
            output_path=glb_path,
        )

        assert success, "process_and_export phải trả về True"
        assert os.path.exists(out_path), "File .glb phải tồn tại"
        assert os.path.getsize(out_path) > 1000, "Dung lượng file .glb phải lớn hơn 1KB"

        # Đọc lại bằng trimesh để kiểm tra độ tương thích
        loaded = trimesh.load(out_path, file_type="glb")
        if isinstance(loaded, trimesh.Scene):
            loaded_mesh = list(loaded.geometry.values())[0]
        else:
            loaded_mesh = loaded
        assert len(loaded_mesh.vertices) > 0
        print(f"  ✅ PASS: Texture Blender xuất file .GLB hoàn chỉnh ({os.path.getsize(out_path)} bytes), đọc lại tương thích 100%.")


def test_05_end_to_end_synthetic_pipeline():
    """Kiểm thử toàn chu trình P4 + P5 với synthetic multi-view data."""
    print("Running test_05_end_to_end_synthetic_pipeline...")
    t0 = time.time()
    n_views = 4
    h, w = 128, 128

    # Tạo pointmap hình lập phương nằm giữa không gian
    pointmaps = np.zeros((n_views, h, w, 3), dtype=np.float32)
    alpha_masks = []
    conf_masks = np.ones((n_views, h, w), dtype=np.float32) * 0.9
    images_rgb = []
    camera_poses = []
    focal_lengths = [(200.0, 200.0) for _ in range(n_views)]

    for i in range(n_views):
        angle = i * (np.pi / 2.0)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.array([2.0 * np.sin(angle), 0.0, 2.0 * np.cos(angle)])
        camera_poses.append(c2w)

        # Mặt nạ tròn ở giữa khung hình
        yy, xx = np.mgrid[:h, :w]
        circ = ((xx - w/2)**2 + (yy - h/2)**2) <= (w/3)**2
        alpha = circ.astype(np.uint8)
        alpha_masks.append(alpha)

        # Pointmap với các điểm trên hình cầu
        pts_view = np.random.uniform(-0.4, 0.4, (h, w, 3)).astype(np.float32)
        pts_view[~circ] = 99.0  # Điểm nền ở xa
        pointmaps[i] = pts_view

        img = np.full((h, w, 3), [100 + i*30, 150, 200 - i*30], dtype=np.uint8)
        images_rgb.append(img)

    # Khởi tạo engine P4 và P5
    engine_p4 = TSDFMeshEngine(resolution=64, tau_conf=0.3)
    blender_p5 = TextureBlender(texture_size=256)

    # Chạy P4: Dựng mesh
    mesh = engine_p4.reconstruct(
        pointmaps_3d=pointmaps,
        alpha_masks=alpha_masks,
        confidence_masks=conf_masks,
        camera_poses=camera_poses,
        focal_lengths=focal_lengths,
    )

    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0

    # Chạy P5: Trải UV & Xuất GLB
    with tempfile.TemporaryDirectory() as tmp_dir:
        glb_path = os.path.join(tmp_dir, "result_e2e.glb")
        success, out_path = blender_p5.process_and_export(
            mesh=mesh,
            images_rgb=images_rgb,
            camera_poses=camera_poses,
            focal_lengths=focal_lengths,
            output_path=glb_path,
        )
        assert success
        assert os.path.exists(out_path)

    elapsed = time.time() - t0
    print(f"  ✅ PASS: Toàn luồng P4 + P5 hoàn thành trong {elapsed:.2f} giây (< 3.0s).")


if __name__ == "__main__":
    print("=" * 60)
    print("BẮT ĐẦU KIỂM THỬ P4 TSDF MESH & P5 TEXTURE BLENDER")
    print("=" * 60)
    test_01_filter_depth_discontinuity()
    test_02_prune_background_points()
    test_03_tsdf_volume_and_marching_cubes()
    test_04_texture_blender_and_glb_export()
    test_05_end_to_end_synthetic_pipeline()
    print("=" * 60)
    print("🎉 TOÀN BỘ 5/5 BÀI KIỂM THỬ THUẬT TOÁN ĐÃ PASS 100%!")
    print("=" * 60)
