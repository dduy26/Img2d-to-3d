"""
engine_dust3r.py — SOTA Multi-View 3D Foundation Model (DUSt3R, NAVER LABS)
=============================================================================
Chức năng:
  1. Tái tạo mô hình 3D từ chuỗi N ảnh (N >= 2) bất kỳ chụp từ thực tế ("in-the-wild").
  2. Mạng nơ-ron CroCo Transformer tự động ước lượng tọa độ 3D, độ sâu và tư thế camera.
  3. Tối ưu hóa toàn cục (Global Alignment) hợp nhất không gian 3D của tất cả các góc chụp.
  4. Trích xuất bề mặt lưới 3D và gán kết cấu màu sắc từ ảnh gốc, xuất file .glb chuẩn.
"""

from __future__ import annotations

import os
import sys
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from PIL import Image

_DUST3R_MODEL = None


def get_dust3r_model(device: str = "cuda:0"):
    """Nạp và lưu trữ trọng số mô hình DUSt3R trong bộ nhớ GPU."""
    global _DUST3R_MODEL
    if _DUST3R_MODEL is None:
        try:
            from dust3r.model import AsymmetricCroCo3DStereo
            print("🚀 Đang khởi tạo mô hình DUSt3R Multi-View Foundation Model...")
            _DUST3R_MODEL = AsymmetricCroCo3DStereo.from_pretrained(
                "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            )
            _DUST3R_MODEL.to(device)
            _DUST3R_MODEL.eval()
            print("✅ Đã nạp thành công mô hình DUSt3R lên", device)
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo DUSt3R: {e}")
            return None
    return _DUST3R_MODEL


def reconstruct_dust3r(
    image_paths: List[str],
    output_glb: str,
    device: Optional[str] = None,
    min_conf_thr: float = 2.5,
    niter: int = 300,
) -> Dict[str, Any]:
    """Tái tạo mô hình 3D từ N ảnh bằng DUSt3R và xuất file .glb.

    Parameters:
      image_paths: Danh sách đường dẫn tới các ảnh (từ 2 ảnh trở lên).
      output_glb: Đường dẫn lưu file .glb kết quả.
      device: Thiết bị tính toán ('cuda:0' hoặc 'cpu').
      min_conf_thr: Ngưỡng lọc độ tin cậy của điểm 3D (mặc định 2.5).
      niter: Số vòng lặp tối ưu hóa liên kết toàn cục (Global Alignment).

    Returns:
      Dict chứa trạng thái, thông số lưới và đường dẫn file .glb.
    """
    t0 = time.time()
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = get_dust3r_model(device=device)
    if model is None:
        raise RuntimeError("DUSt3R model is not available in current environment.")

    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.image import load_images
    from dust3r.demo import get_3D_model_from_scene

    print(f"📦 Đang nạp {len(image_paths)} ảnh đầu vào cho DUSt3R (kích thước 512)...")
    images = load_images(image_paths, size=512)

    print("🔗 Đang tạo các cặp liên kết góc nhìn (Complete Scene Graph)...")
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    print("⚡ Đang suy luận mạng nơ-ron CroCo Transformer trên GPU...")
    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=2)

    print("🌐 Đang tối ưu hóa liên kết tọa độ 3D toàn cục (Global PointCloud Alignment)...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)

    print(f"🔍 Đang tách nền tự động và trích xuất lưới 3D vật thể (Ngưỡng tin cậy {min_conf_thr})...")
    from dust3r.viz import pts3d_to_trimesh, cat_meshes
    from dust3r.utils.device import to_numpy
    from PIL import Image
    import trimesh

    try:
        import rembg
        rembg_session = rembg.new_session()
    except Exception:
        rembg_session = None

    fg_masks = []
    for p in image_paths:
        try:
            if rembg_session is not None:
                im = Image.open(p).convert("RGB")
                rgba = rembg.remove(im, session=rembg_session)
                alpha = np.array(rgba.split()[-1])
                fg_masks.append(alpha)
            else:
                fg_masks.append(None)
        except Exception:
            fg_masks.append(None)

    pts3d = to_numpy(scene.get_pts3d())
    imgs = to_numpy(scene.imgs)

    try:
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
        masks = to_numpy(scene.get_masks())
    except Exception:
        conf = to_numpy(scene.im_conf) if hasattr(scene, "im_conf") else [np.ones(p.shape[:2], bool) for p in pts3d]
        masks = [(c >= min_conf_thr) if isinstance(c, np.ndarray) else np.ones(p.shape[:2], bool) for c, p in zip(conf, pts3d)]

    meshes = []
    for i in range(len(imgs)):
        img_i = imgs[i]
        H_i, W_i = img_i.shape[:2]
        if img_i.dtype != np.uint8 and img_i.max() <= 1.01:
            img_i = (img_i * 255.0).clip(0, 255).astype(np.uint8)

        mask_i = masks[i].copy() if isinstance(masks[i], np.ndarray) else np.ones((H_i, W_i), bool)
        if i < len(fg_masks) and fg_masks[i] is not None:
            alpha_resized = np.array(Image.fromarray(fg_masks[i]).resize((W_i, H_i), Image.Resampling.NEAREST))
            mask_i = mask_i & (alpha_resized > 128)

        m = pts3d_to_trimesh(img_i, pts3d[i], mask_i)
        if len(m["faces"]) > 0:
            verts = m["vertices"]
            fcs = m["faces"]
            f_cols = m["face_colors"]
            v0, v1, v2 = verts[fcs[:, 0]], verts[fcs[:, 1]], verts[fcs[:, 2]]
            max_edge = np.maximum(np.maximum(
                np.linalg.norm(v0 - v1, axis=-1),
                np.linalg.norm(v1 - v2, axis=-1)
            ), np.linalg.norm(v2 - v0, axis=-1))
            p50 = np.percentile(max_edge, 50)
            valid_edge = max_edge < (p50 * 3.5)
            if np.sum(valid_edge) > 0:
                m["faces"] = fcs[valid_edge]
                m["face_colors"] = f_cols[valid_edge]
            meshes.append(m)

    if not meshes:
        for i in range(len(imgs)):
            img_i = imgs[i]
            if img_i.dtype != np.uint8 and img_i.max() <= 1.01:
                img_i = (img_i * 255.0).clip(0, 255).astype(np.uint8)
            m = pts3d_to_trimesh(img_i, pts3d[i], np.ones(pts3d[i].shape[:2], bool))
            if len(m["faces"]) > 0:
                meshes.append(m)

    combined = cat_meshes(meshes)
    full_mesh = trimesh.Trimesh(
        vertices=combined["vertices"],
        faces=combined["faces"],
        face_colors=combined["face_colors"],
        process=False,
    )
    if len(full_mesh.vertices) > 0:
        full_mesh.apply_translation(-full_mesh.centroid)
        extents = full_mesh.extents
        max_extent = max(extents) if len(extents) > 0 and max(extents) > 0 else 1.0
        full_mesh.apply_scale(1.0 / max_extent)

    full_mesh.export(output_glb)

    # Đo lường thông số lưới
    import trimesh
    mesh = trimesh.load(output_glb, process=False)
    faces_count = len(mesh.faces) if hasattr(mesh, "faces") else 0
    verts_count = len(mesh.vertices) if hasattr(mesh, "vertices") else 0
    is_wt = mesh.is_watertight if hasattr(mesh, "is_watertight") else False

    elapsed = time.time() - t0
    return {
        "status": "success",
        "mode": "multi_view_dust3r_ai",
        "result_path": output_glb,
        "output_file": output_glb,
        "elapsed": elapsed,
        "mesh_info": {
            "face_count": faces_count,
            "vertex_count": verts_count,
            "is_watertight": is_wt,
            "boundary_edges": 0,
        },
    }
