from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn
import shutil
import os
import time
import glob
import numpy as np
import logging
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import các module trong pipeline
from preprocess import preprocess_multiview
from engine_dust3r import DUSt3REngine
from quality_gate import QualityGate
from engine_triposr import TripoSREngine
from engine_tsdf_mesh import TSDFMeshEngine
from texture_blender import TextureBlender

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="2D to 3D Generation API - Full Pipeline")

os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── P6: Phục vụ file .glb tĩnh (frontend tự fetch từ /outputs/<file>.glb) ──
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
FRONTEND_INDEX = os.path.join(FRONTEND_DIR, "index.html")

# ============================================================================
# KHỞI TẠO TẤT CẢ ENGINE (Chỉ chạy 1 lần lúc bật server)
# ============================================================================
device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
logger.info(f"Sử dụng device: {device}")

# P3: Quality Gate
q_gate = QualityGate()

# P3: TripoSR Fail-safe Engine (nạp sẵn vào RAM)
triposr_engine = TripoSREngine()

# P2: DUSt3R Engine
dust3r_engine = DUSt3REngine(device=device)

# P4: TSDF Volumetric Mesh Engine (NVIDIA reference TSDF + Marching Cubes)
tsdf_engine = TSDFMeshEngine(resolution=128)

# P5: XAtlas UV Parameterization & Base-Color Texture Blender
texture_blender = TextureBlender()

logger.info("═══ TẤT CẢ ENGINE P1-P5 ĐÃ SẴN SÀNG ═══")


# ============================================================================
# API 0: Giao diện Web (P6) + Health check
# ============================================================================
@app.get("/", include_in_schema=False)
async def frontend():
    """Phục vụ giao diện Web UI (Three.js) tại gốc máy chủ."""
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    return {"message": "Frontend chưa được cài. Mở /docs để dùng Swagger UI."}


@app.get("/api/health")
async def health():
    """Health check cho Colab/tunnel."""
    return {
        "status": "ok",
        "device": device,
        "frontend": os.path.exists(FRONTEND_INDEX),
    }



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
        if HAS_TORCH:
            images_tensor = torch.from_numpy(preprocess_result["images_normalized"]).to(device)
        else:
            images_tensor = preprocess_result["images_normalized"]

        dust3r_result = dust3r_engine.process({
            "images_dust3r": images_tensor,
            "image_paths": saved_paths,   # P2 bản thật (DUSt3R) nạp ảnh từ đây
        })

        pointmaps_3d = dust3r_result["pointmaps_3d"]
        confidence_masks = dust3r_result["confidence_masks"]
        camera_poses = dust3r_result["camera_poses"]
        focal_lengths = dust3r_result["focal_lengths"]

        # P2 bản thật chạy DUSt3R ở geometry riêng (cạnh dài 512, crop bội số 16) ->
        # căn lại alpha mask & ảnh RGB về đúng geometry đó, nếu không thì mask lệch
        # từng pixel so với pointmap và P4 sẽ đắp TSDF sai chỗ.
        geom = dust3r_result.get("geometry")
        if geom is not None:
            target = (int(geom[1]), int(geom[0]))          # PIL size = (W, H)
            preprocess_result["alpha_masks"] = [
                np.asarray(Image.fromarray(np.asarray(m, dtype=np.uint8)).resize(target, Image.NEAREST))
                for m in preprocess_result["alpha_masks"]
            ]
            preprocess_result["images_rgb"] = [
                np.asarray(Image.fromarray(np.asarray(a, dtype=np.uint8)).resize(target, Image.BILINEAR))
                for a in preprocess_result["images_rgb"]
            ]
            logger.info(f"[P2->P4/P5] Đã căn alpha mask & ảnh RGB về geometry {geom}")

        logger.info(
            f"[P2] Hoàn tất: pointmaps shape={pointmaps_3d.shape}, "
            f"poses={len(camera_poses)}, focals={len(focal_lengths)}"
        )

        # ── Bước 3 (P3): Quality Gate ──
        logger.info("[P3] Đánh giá chất lượng qua Quality Gate...")
        
        # BA loss thật lấy từ DUSt3R global alignment (chế độ mock trả 1.0)
        ba_loss = dust3r_result.get("ba_loss", 1.0)
        
        if hasattr(confidence_masks, 'cpu'):
            confidence_np = confidence_masks.cpu().numpy()
        else:
            confidence_np = np.asarray(confidence_masks)

        is_high_quality, reason = q_gate.evaluate(
            poses=camera_poses,
            confidence_map=confidence_np,
            ba_loss=ba_loss,
        )
        logger.info(f"[P3] Kết quả: {'PASS ✓' if is_high_quality else 'FAIL ✗'} — {reason}")

        # ── Bước 4: Phân luồng theo kết quả Quality Gate ──
        if is_high_quality:
            # ✓ PASS: Chạy luồng NVIDIA P4 TSDF Mesh & P5 Texture Blender
            logger.info("[P4] Quality PASS → Dựng Mesh TSDF 360°...")
            mesh = tsdf_engine.reconstruct(
                pointmaps_3d=pointmaps_3d,
                alpha_masks=preprocess_result["alpha_masks"],
                confidence_masks=confidence_masks,
                camera_poses=camera_poses,
                focal_lengths=focal_lengths,
            )

            logger.info("[P5] Trải UV XAtlas & Nướng màu Base-Color Texture vào Mesh...")
            success, model_path = texture_blender.process_and_export(
                mesh=mesh,
                images_rgb=preprocess_result["images_rgb"],
                camera_poses=camera_poses,
                focal_lengths=focal_lengths,
                output_path=output_glb_path,
            )
            # Fallback an toàn nếu nướng texture gặp sự cố: xuất mesh thô trực tiếp
            if not success or not os.path.exists(output_glb_path):
                logger.warning("[P5] Nướng texture không thành công, fallback xuất mesh thô của P4.")
                mesh.export(output_glb_path, file_type="glb")
                success = os.path.exists(output_glb_path)
                model_path = output_glb_path

            pipeline_type = "nvidia_tsdf_mesh"
        else:
            # ✗ FAIL: Kích hoạt cứu hộ TripoSR
            logger.info(f"[P3→Cứu hộ] Quality FAIL ({reason}) → Gọi TripoSR fallback")
            success, model_path, _ = triposr_engine.run_fallback(
                saved_paths[0], output_glb_path
            )
            pipeline_type = "triposr_fallback"

        total_time = time.time() - pipeline_start

        return {
            "status": "success" if success else "failed",
            "mode": "multiview_pipeline",
            "pipeline_type": pipeline_type,
            "dust3r_backend": dust3r_result.get("backend", "mock"),
            "quality_passed": is_high_quality,
            "gate_reason": reason,
            "num_input_images": preprocess_result["num_images"],
            "execution_time_seconds": round(total_time, 2),
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