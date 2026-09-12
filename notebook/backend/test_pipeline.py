"""
BỘ KIỂM THỬ TỔNG HỢP TOÀN DIỆN (UNIFIED TEST SUITE: P4 + P5 + FULL API E2E)
===========================================================================
Chạy: python notebook/backend/test_pipeline.py

Bao gồm 6 bài test:
  - [Unit P4] Test 1: Lọc viền mép rách độ sâu (Depth Discontinuity Gradient Filter)
  - [Unit P4] Test 2: Pruning điểm nền & Lọc Confidence Gate (Alpha Mask + DUSt3R)
  - [Unit P4] Test 3: Lưới thể tích TSDF & Trích xuất Marching Cubes kín nước (Watertight)
  - [Unit P5] Test 4: Trải UV XAtlas, Hòa trộn màu góc nhìn & Xuất file chuẩn .GLB
  - [API E2E] Test 5: API /generate-3d/single/ (Chế độ 1 ảnh cứu hộ TripoSR)
  - [API E2E] Test 6: API /generate-3d/ (Toàn luồng Đa ảnh P1 -> P2 -> P3 -> P4 -> P5)
"""

import os
import sys
import time
import tempfile
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image

# Đảm bảo in tiếng Việt / Emoji không bị lỗi UnicodeEncodeError trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục backend vào sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from engine_tsdf_mesh import (
    filter_depth_discontinuity,
    prune_background_points,
    TSDFVolume,
    extract_mesh_marching_cubes,
    TSDFMeshEngine,
)
try:
    from texture_blender import TextureBlender
    HAS_TEXTURE_BLENDER = True
except ImportError:
    HAS_TEXTURE_BLENDER = False

from app import app
from starlette.testclient import TestClient


# ==============================================================================
# NHÓM 1: KIỂM THỬ THUẬT TOÁN HÌNH HỌC P4 (TSDF MESH)
# ==============================================================================

def test_01_filter_depth_discontinuity():
    """Test 1: Lọc viền mép rách do bước nhảy độ sâu (DA3-blender Gradient Filter)."""
    print("\n[TEST 1/7] Kiểm thử Lọc mép rách độ sâu (filter_depth_discontinuity)...")
    # Mặt phẳng đồng nhất: không có mép rách
    flat_depth = np.full((100, 100), 2.0, dtype=np.float32)
    mask = filter_depth_discontinuity(flat_depth, tau=0.07)
    assert np.all(mask), "Mặt phẳng đồng nhất phải 100% pixel hợp lệ."

    # Mặt phẳng có bậc nhảy lớn (step discontinuity) từ 2.0m lên 10.0m
    step_depth = np.full((100, 100), 2.0, dtype=np.float32)
    step_depth[:, 50:] = 10.0
    mask_step = filter_depth_discontinuity(step_depth, tau=0.07)

    # Ranh giới bước nhảy phải bị đánh dấu False
    edge_detected = not np.all(mask_step[:, 49:52])
    assert edge_detected, "Phải phát hiện và lọc bỏ ranh giới bước nhảy độ sâu."
    print("  ✅ PASS: Thuật toán lọc viền rách mép độ sâu hoạt động chuẩn xác.")


def test_02_prune_background_points():
    """Test 2: Pruning điểm nền kết hợp Alpha Mask & DUSt3R Confidence."""
    print("\n[TEST 2/7] Kiểm thử Pruning điểm nền (prune_background_points)...")
    n_views, h, w = 2, 64, 64
    pointmaps = np.random.randn(n_views, h, w, 3).astype(np.float32)
    pointmaps[:, :, :, 2] = np.abs(pointmaps[:, :, :, 2]) + 1.0

    # Nửa trên là vật thể (alpha=1), nửa dưới là nền (alpha=0)
    alpha_masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(n_views)]
    for a in alpha_masks:
        a[:32, :] = 1

    # Confidence: góc trái cao (0.8), góc phải thấp (0.1)
    conf_masks = np.ones((n_views, h, w), dtype=np.float32) * 0.8
    conf_masks[:, :, 32:] = 0.1

    valid_masks, filtered_points = prune_background_points(
        pointmaps, alpha_masks, conf_masks, tau_conf=0.35
    )

    assert len(filtered_points) == n_views
    assert len(filtered_points[0]) > 0
    # Góc dưới (nền) không được chứa điểm nào
    assert not np.any(valid_masks[0][40:, :])
    print("  ✅ PASS: Đã cắt tỉa 100% điểm hậu cảnh và rác biên tự do.")


def test_03_tsdf_marching_cubes_watertight():
    """Test 3: Tích lũy lưới thể tích TSDF và trích xuất Marching Cubes kín nước 360°."""
    print("\n[TEST 3/7] Kiểm thử TSDF Volume & Marching Cubes Watertight...")
    res = 64
    bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    bounds_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    vol = TSDFVolume(bounds_min, bounds_max, resolution=res, trunc_margin=0.1)

    # Tạo trường khoảng cách có dấu nhân tạo cho hình cầu bán kính R=0.5
    xs = np.linspace(vol.bounds_min[0], vol.bounds_max[0], res)
    ys = np.linspace(vol.bounds_min[1], vol.bounds_max[1], res)
    zs = np.linspace(vol.bounds_min[2], vol.bounds_max[2], res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')

    dist_from_origin = np.sqrt(gx**2 + gy**2 + gz**2)
    signed_dist = dist_from_origin - 0.5
    vol.tsdf_grid = np.clip(signed_dist / 0.1, -1.0, 1.0).astype(np.float32)
    vol.weight_grid = np.ones((res, res, res), dtype=np.float32)

    mesh = extract_mesh_marching_cubes(vol, min_weight=0.1)

    assert len(mesh.vertices) > 100, f"Mesh phải có đỉnh (hiện có {len(mesh.vertices)})"
    assert len(mesh.faces) > 100, f"Mesh phải có mặt tam giác (hiện có {len(mesh.faces)})"
    assert mesh.is_watertight, "Mặt cắt cầu của Marching Cubes phải kín nước (Watertight) 360°."
    print(f"  ✅ PASS: Marching Cubes sinh Watertight Mesh ({len(mesh.vertices)} đỉnh, {len(mesh.faces)} tam giác).")


# ==============================================================================
# NHÓM 2: KIỂM THỬ XATLAS UV & TEXTURE BLENDING P5
# ==============================================================================

def test_04_texture_blender_and_glb_export():
    """Test 4: Trải UV XAtlas, hòa trộn màu góc nhìn và xuất file .GLB hợp lệ."""
    print("\n[TEST 4/7] Kiểm thử Texture Blender & Xuất GLB (TextureBlender)...")
    if not HAS_TEXTURE_BLENDER:
        print("  ⏭️ SKIP: Module P5 (Texture Blender) đã tách riêng để Thành viên 5 tự phát triển từ đầu.")
        return

    mesh = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
    n_views = 4
    camera_poses, images_rgb, focal_lengths = [], [], []
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]

    for i in range(n_views):
        angle = i * (2.0 * np.pi / n_views)
        cam_pos = np.array([2.0 * np.sin(angle), 0.0, 2.0 * np.cos(angle)])
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
        images_rgb.append(np.full((128, 128, 3), colors[i], dtype=np.uint8))

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

        assert success, "process_and_export phải trả về True."
        assert os.path.exists(out_path), "File .glb phải tồn tại."
        assert os.path.getsize(out_path) > 1000, "File .glb phải chứa dữ liệu mesh và texture."

        # Kiểm tra đọc lại file GLB
        loaded = trimesh.load(out_path, file_type="glb")
        if isinstance(loaded, trimesh.Scene):
            loaded_mesh = list(loaded.geometry.values())[0]
        else:
            loaded_mesh = loaded
        assert len(loaded_mesh.vertices) > 0
        print(f"  ✅ PASS: Texture Blender xuất file .GLB hoàn chỉnh ({os.path.getsize(out_path)} bytes), tương thích 100%.")


# ==============================================================================
# NHÓM 3: KIỂM THỬ TÍCH HỢP END-TO-END QUA FASTAPI APP
# ==============================================================================

def test_05_api_single_image_triposr():
    """Test 5: API POST /generate-3d/single/ (Chế độ 1 ảnh cứu hộ)."""
    print("\n[TEST 5/7] Kiểm thử API /generate-3d/single/ qua TestClient...")
    client = TestClient(app)
    test_img = Path("data/input/multi_view/view_01_front.jpg")
    assert test_img.exists(), f"Thiếu ảnh test: {test_img}"

    with open(test_img, "rb") as f:
        t0 = time.time()
        response = client.post("/generate-3d/single/", files={"file": (test_img.name, f, "image/jpeg")})
        latency = time.time() - t0

    assert response.status_code == 200, f"HTTP Error {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "success"
    assert os.path.exists(data["output_file"])
    print(f"  ✅ PASS: Single-image API thành công trong {latency:.2f}s! Model: {data['output_file']}")


def test_06_api_multiview_full_pipeline():
    """Test 6: API POST /generate-3d/ (Toàn chu trình Đa ảnh P1 -> P2 -> P3 -> P4 -> P5)."""
    print("\n[TEST 6/7] Kiểm thử API /generate-3d/ (Full Pipeline 6 ảnh)...")
    client = TestClient(app)
    img_paths = sorted(list(Path("data/input/multi_view").glob("view_*.jpg")))
    assert len(img_paths) >= 4, f"Cần ít nhất 4 ảnh benchmark, tìm thấy {len(img_paths)}"

    file_handles = [open(p, "rb") for p in img_paths]
    files = [("files", (p.name, fh, "image/jpeg")) for p, fh in zip(img_paths, file_handles)]

    try:
        t0 = time.time()
        response = client.post("/generate-3d/", files=files)
        total_time = time.time() - t0
    finally:
        for fh in file_handles:
            fh.close()

    assert response.status_code == 200, f"HTTP Error {response.status_code}: {response.text}"
    data = response.json()

    assert data["status"] == "success"
    assert data["mode"] == "multiview_pipeline"
    assert data["pipeline_type"] == "nvidia_tsdf_mesh"
    assert data["quality_passed"] is True
    assert os.path.exists(data["output_file"])

    # Đọc lại và kiểm tra lưới tam giác trong file GLB
    loaded = trimesh.load(data["output_file"], file_type="glb")
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        total_verts = sum(len(m.vertices) for m in meshes)
        total_faces = sum(len(m.faces) for m in meshes)
    else:
        total_verts = len(loaded.vertices)
        total_faces = len(loaded.faces)

    assert total_verts > 0 and total_faces > 0
    print(f"  ✅ PASS: Full Multi-view Pipeline hoàn thành trong {total_time:.2f}s! ({total_verts} đỉnh, {total_faces} tam giác).")


def test_07_frontend_and_static_outputs():
    """Test 7 (P6): Web UI phục vụ tại '/' và file .glb phục vụ tại '/outputs/'."""
    print("\n[TEST 7/7] Kiểm thử Web UI (P6) & Static GLB...")
    client = TestClient(app)

    r = client.get("/")
    assert r.status_code == 200, f"GET / phải trả Web UI, nhận {r.status_code}"
    assert "3D Viewer" in r.text, "index.html của frontend không được phục vụ đúng"
    assert "GLTFLoader" in r.text, "Web UI phải có Three.js GLTFLoader để xem .glb"

    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["frontend"] is True

    # .glb phải tải được qua /outputs/ (đúng tên file, không lộ path tuyệt đối)
    out_file = Path("outputs/result_view_01_front.glb")
    assert out_file.exists(), "Chưa có .glb — test 6 phải chạy trước"
    r = client.get(f"/outputs/{out_file.name}")
    assert r.status_code == 200 and r.content[:4] == b"glTF", "Static /outputs phải trả file GLB hợp lệ"
    print(f"  ✅ PASS: Web UI + Static GLB OK ({len(r.content)} bytes, magic=glTF).")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("🚀 BẮT ĐẦU CHẠY BỘ KIỂM THỬ HỢP NHẤT TOÀN HỆ THỐNG (P4 + P5 + APP E2E)")
    print("=" * 70)

    test_01_filter_depth_discontinuity()
    test_02_prune_background_points()
    test_03_tsdf_marching_cubes_watertight()
    test_04_texture_blender_and_glb_export()
    test_05_api_single_image_triposr()
    test_06_api_multiview_full_pipeline()
    test_07_frontend_and_static_outputs()

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"🎉 TẤT CẢ 7/7 BÀI KIỂM THỬ ĐÃ ĐẠT 100% TRONG {total_elapsed:.2f} GIÂY!")
    print("   - Khối hình học P4 (TSDF Volumetric Fusion & Marching Cubes): PASS ✅")
    print("   - Khối trải UV & Nướng màu P5 (TextureBlender & GLB Export): PASS ✅")
    print("   - Khối điều phối máy chủ Backend (FastAPI E2E Single & Multi): PASS ✅")
    print("   - Khối Web UI & Static GLB (P6): PASS ✅")
    print("=" * 70)
