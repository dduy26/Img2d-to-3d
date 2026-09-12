"""
Script kiểm thử End-to-End toàn bộ hệ thống qua FastAPI TestClient.
Kiểm thử cả 2 chế độ:
  1. Single-image: 1 ảnh -> Fallback/Direct
  2. Multi-view: 6 ảnh benchmark -> P1 -> P2 -> P3 -> P4 TSDF Mesh -> P5 Texture Blender -> GLB
"""

import os
import sys
import time
from pathlib import Path
from starlette.testclient import TestClient
import trimesh

# Đưa notebook/backend vào sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

client = TestClient(app)

def test_api_single_image():
    """Kiểm thử API tải lên 1 ảnh duy nhất."""
    print("\n" + "=" * 60)
    print("TEST 1: API /generate-3d/single/ (Chế độ 1 ảnh)")
    print("=" * 60)

    test_img = Path("data/input/multi_view/view_01_front.jpg")
    assert test_img.exists(), f"Không tìm thấy ảnh {test_img}"

    with open(test_img, "rb") as f:
        files = {"file": (test_img.name, f, "image/jpeg")}
        t0 = time.time()
        response = client.post("/generate-3d/single/", files=files)
        latency = time.time() - t0

    assert response.status_code == 200, f"Lỗi HTTP {response.status_code}: {response.text}"
    data = response.json()
    print("Response JSON:", data)
    assert data["status"] == "success"
    assert "output_file" in data
    assert os.path.exists(data["output_file"])
    assert os.path.getsize(data["output_file"]) > 0
    print(f"✅ PASS: Single-image API thành công trong {latency:.2f}s! File: {data['output_file']}")


def test_api_multiview_pipeline():
    """Kiểm thử API tải lên nhiều ảnh (Full Pipeline P1 -> P2 -> P3 -> P4 -> P5)."""
    print("\n" + "=" * 60)
    print("TEST 2: API /generate-3d/ (Full Multi-view Pipeline)")
    print("=" * 60)

    img_paths = sorted(list(Path("data/input/multi_view").glob("view_*.jpg")))
    assert len(img_paths) >= 4, f"Cần ít nhất 4 ảnh, tìm thấy {len(img_paths)}"
    print(f"Gửi {len(img_paths)} ảnh benchmark vào hệ thống...")

    files = []
    file_handles = []
    try:
        for p in img_paths:
            fh = open(p, "rb")
            file_handles.append(fh)
            files.append(("files", (p.name, fh, "image/jpeg")))

        t0 = time.time()
        response = client.post("/generate-3d/", files=files)
        total_latency = time.time() - t0

    finally:
        for fh in file_handles:
            fh.close()

    assert response.status_code == 200, f"Lỗi HTTP {response.status_code}: {response.text}"
    data = response.json()
    print("Response JSON:", data)

    assert data["status"] == "success"
    assert data["mode"] == "multiview_pipeline"
    assert data["num_input_images"] == len(img_paths)
    assert os.path.exists(data["output_file"])

    file_size_kb = os.path.getsize(data["output_file"]) / 1024
    print(f"File output: {data['output_file']} ({file_size_kb:.1f} KB)")
    assert file_size_kb > 1.0, "Dung lượng file .glb phải lớn hơn 1KB"

    # Kiểm tra độ hợp lệ của file .glb xuất ra
    loaded = trimesh.load(data["output_file"], file_type="glb")
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        assert len(meshes) > 0, "File GLB Scene phải chứa ít nhất 1 mesh"
        total_verts = sum(len(m.vertices) for m in meshes)
        total_faces = sum(len(m.faces) for m in meshes)
    else:
        total_verts = len(loaded.vertices)
        total_faces = len(loaded.faces)

    print(f"Kiểm tra cấu trúc 3D Mesh: {total_verts} đỉnh, {total_faces} tam giác.")
    assert total_verts > 0 and total_faces > 0

    print(f"✅ PASS: Full Multi-view Pipeline hoàn thành thành công trong {total_latency:.2f}s!")


if __name__ == "__main__":
    test_api_single_image()
    test_api_multiview_pipeline()
    print("\n" + "=" * 60)
    print("🎉 TOÀN BỘ CÁC BÀI TEST API END-TO-END ĐÃ ĐẠT 100%!")
    print("=" * 60)
