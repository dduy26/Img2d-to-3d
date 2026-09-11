from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import uvicorn
import shutil
import os
import time
import glob
import torch
import numpy as np
import logging

# Import các module trong pipeline
from preprocess import preprocess_multiview
from engine_dust3r import DUSt3REngine
from quality_gate import QualityGate
from engine_triposr import TripoSREngine

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="2D to 3D Generation API - Full Pipeline")

os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ============================================================================
# KHỞI TẠO TẤT CẢ ENGINE (Chỉ chạy 1 lần lúc bật server)
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Sử dụng device: {device}")

# P3: Quality Gate
q_gate = QualityGate()

# P3: TripoSR Fail-safe Engine (nạp sẵn vào RAM)
triposr_engine = TripoSREngine()

# P2: DUSt3R Engine
dust3r_engine = DUSt3REngine(device=device)

logger.info("═══ TẤT CẢ ENGINE ĐÃ SẴN SÀNG ═══")


# ============================================================================
# API 1: Upload NHIỀU ẢNH → Full Pipeline (Preprocessing → DUSt3R → QGate → TripoSR fallback)
# ============================================================================
@app.post("/generate-3d/")
async def generate_3d(files: List[UploadFile] = File(...)):
    """
    Full Pipeline: Upload nhiều ảnh đa góc nhìn.
    
    Luồng xử lý:
        1 ảnh  → Đi thẳng TripoSR (không cần DUSt3R multi-view)
        ≥2 ảnh → P1 Preprocessing → P2 DUSt3R → P3 Quality Gate
                  → Nếu PASS: Dùng kết quả DUSt3R (TODO: P4 TSDF Mesh)
                  → Nếu FAIL: Fallback TripoSR từ ảnh đầu tiên
    """
    pipeline_start = time.time()
    saved_paths = []

    try:
        # ── Bước 0: Lưu tất cả ảnh upload ──
        for f in files:
            input_path = f"temp_uploads/{f.filename}"
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_paths.append(input_path)
            logger.info(f"Đã lưu: {f.filename}")

        # Tạo output path chuẩn .glb
        base_name = os.path.splitext(files[0].filename)[0]
        output_glb_path = f"outputs/result_{base_name}.glb"

        # ════════════════════════════════════════════════════════════════
        # NHÁNH 1: Chỉ có 1 ảnh → Đi thẳng TripoSR (Single-Image Reconstruction)
        # ════════════════════════════════════════════════════════════════
        if len(saved_paths) == 1:
            logger.info("═══ CHẾ ĐỘ 1 ẢNH → TripoSR trực tiếp ═══")
            success, model_path, exec_time = triposr_engine.run_fallback(
                saved_paths[0], output_glb_path
            )
            return {
                "status": "success" if success else "failed",
                "mode": "single_image_triposr",
                "quality_passed": None,
                "gate_reason": "Chỉ 1 ảnh, bỏ qua DUSt3R multi-view",
                "execution_time_seconds": round(exec_time, 2),
                "output_file": model_path,
                "num_input_images": 1,
            }

        # ════════════════════════════════════════════════════════════════
        # NHÁNH 2: Nhiều ảnh → Full Pipeline
        # ════════════════════════════════════════════════════════════════
        logger.info(f"═══ CHẾ ĐỘ MULTI-VIEW ({len(saved_paths)} ảnh) → Full Pipeline ═══")

        # ── Bước 1 (P1): Preprocessing ──
        logger.info("[P1] Tiền xử lý ảnh đa góc nhìn...")
        preprocess_result = preprocess_multiview(
            image_paths=saved_paths,
            target_size=512,
            device=device,
        )
        logger.info(
            f"[P1] Hoàn tất: {preprocess_result['num_images']} ảnh, "
            f"tensor shape: {preprocess_result['images_normalized'].shape}"
        )

        # ── Bước 2 (P2): DUSt3R Multi-view Reconstruction ──
        logger.info("[P2] Chạy DUSt3R pairwise matching + global alignment...")
        images_tensor = torch.from_numpy(preprocess_result["images_normalized"]).to(device)

        dust3r_result = dust3r_engine.process({
            "images_dust3r": images_tensor
        })

        pointmaps_3d = dust3r_result["pointmaps_3d"]
        confidence_masks = dust3r_result["confidence_masks"]
        camera_poses = dust3r_result["camera_poses"]
        focal_lengths = dust3r_result["focal_lengths"]

        logger.info(
            f"[P2] Hoàn tất: pointmaps shape={pointmaps_3d.shape}, "
            f"poses={len(camera_poses)}, focals={len(focal_lengths)}"
        )

        # ── Bước 3 (P3): Quality Gate ──
        logger.info("[P3] Đánh giá chất lượng qua Quality Gate...")
        
        # Tính BA loss giả lập (TODO: lấy từ DUSt3R global alignment thực tế)
        ba_loss = 1.0  # Placeholder - sẽ được thay bằng loss thực từ DUSt3R
        
        confidence_np = confidence_masks.cpu().numpy()
        is_high_quality, reason = q_gate.evaluate(
            poses=camera_poses,
            confidence_map=confidence_np,
            ba_loss=ba_loss,
        )
        logger.info(f"[P3] Kết quả: {'PASS ✓' if is_high_quality else 'FAIL ✗'} — {reason}")

        # ── Bước 4: Phân luồng theo kết quả Quality Gate ──
        if is_high_quality:
            # ✓ PASS: Dùng kết quả DUSt3R để tạo mesh
            # TODO (P4): TSDF Fusion → Mesh → Texturing → Export .glb
            # Hiện tại tạm dùng TripoSR vì chưa có P4 TSDF Mesh
            logger.info("[P4] Quality PASS → Tạo mesh từ DUSt3R (tạm dùng TripoSR)")
            success, model_path, exec_time = triposr_engine.run_fallback(
                saved_paths[0], output_glb_path
            )
        else:
            # ✗ FAIL: Kích hoạt cứu hộ TripoSR
            logger.info(f"[P3→Cứu hộ] Quality FAIL ({reason}) → Gọi TripoSR fallback")
            success, model_path, exec_time = triposr_engine.run_fallback(
                saved_paths[0], output_glb_path
            )

        total_time = time.time() - pipeline_start

        return {
            "status": "success" if success else "failed",
            "mode": "multiview_pipeline",
            "quality_passed": is_high_quality,
            "gate_reason": reason,
            "num_input_images": preprocess_result["num_images"],
            "execution_time_seconds": round(total_time, 2),
            "triposr_time_seconds": round(exec_time, 2),
            "output_file": model_path,
        }

    except Exception as e:
        logger.error(f"Pipeline lỗi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# API 2: Upload 1 ẢNH DUY NHẤT → TripoSR trực tiếp (giữ API cũ tương thích)
# ============================================================================
@app.post("/generate-3d/single/")
async def generate_3d_single(file: UploadFile = File(...)):
    """
    API đơn giản: Upload 1 ảnh → TripoSR trực tiếp.
    Giữ nguyên tương thích API cũ.
    """
    try:
        input_path = f"temp_uploads/{file.filename}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_glb_path = f"outputs/result_{os.path.splitext(file.filename)[0]}.glb"

        success, model_path, exec_time = triposr_engine.run_fallback(
            input_path, output_glb_path
        )

        return {
            "status": "success" if success else "failed",
            "mode": "single_image_triposr",
            "execution_time_seconds": round(exec_time, 2),
            "output_file": model_path,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)