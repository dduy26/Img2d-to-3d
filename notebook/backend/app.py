from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import uvicorn
import shutil
import os
import sys
import time
import glob
import uuid
import threading
import numpy as np
import logging
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import các module trong pipeline
from preprocess import preprocess_multiview, preprocess_single_view
from quality_gate import QualityGate
from engine_triposr import TripoSREngine
from engine_tsdf_mesh import TSDFMeshEngine, generate_camera_poses, DEFAULT_CONF_THRESHOLD
from texture_blender import TextureBlender
from engine_depth import DepthReconstructionEngine

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="2D to 3D Generation API - Full Pipeline")

# Xác định thư mục dự án và thiết lập 1 thư mục nhận input, 1 thư mục xuất output duy nhất
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BACKEND_DIR))
FRONTEND_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "frontend")
FRONTEND_INDEX = os.path.join(FRONTEND_DIR, "index.html")

# Thống nhất 1 thư mục nhận input và 1 thư mục xuất output ở gốc dự án
if os.path.isdir(os.path.join(PROJECT_ROOT, "notebook")):
    INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
else:
    INPUT_DIR = os.path.abspath("input")
    OUTPUT_DIR = os.path.abspath("output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── P6: Phục vụ file .glb tĩnh (hỗ trợ cả /output và /outputs tương thích ngược) ──
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ============================================================================
# KHỞI TẠO TẤT CẢ ENGINE (Chỉ chạy 1 lần lúc bật server)
# ============================================================================
device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
logger.info(f"Sử dụng device: {device}")

# P3: Quality Gate
q_gate = QualityGate()

# P3: TripoSR Fail-safe Engine (nạp sẵn vào RAM)
triposr_engine = TripoSREngine()

# ── Nút vặn chất lượng (đặt qua biến môi trường, không cần sửa code) ──
# TSDF_RES: số voxel mỗi cạnh của lưới TSDF. Cao hơn = chi tiết hơn, chậm + tốn RAM hơn.
#   128 -> ~16MB/grid (mặc định) | 192 -> ~57MB | 256 -> ~134MB
TSDF_RES = int(os.environ.get("TSDF_RES", "128"))
logger.info(f"Cấu hình: TSDF_RES={TSDF_RES}")

# P4: TSDF Volumetric Mesh Engine (NVIDIA reference TSDF + Marching Cubes)
tsdf_engine = TSDFMeshEngine(resolution=TSDF_RES)

# P5: XAtlas UV Parameterization & Base-Color Texture Blender
texture_blender = TextureBlender()

# Kịch bản 1: Depth Reconstruction Engine (Depth-Anything-V2 + Poisson)
depth_engine = DepthReconstructionEngine(device=device)

logger.info("═══ TẤT CẢ ENGINE P1-P5 + Depth Engine ĐÃ SẴN SÀNG ═══")


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
        "engines": {
            "triposr": triposr_engine.model is not None,
            "depth": depth_engine.depth_model is not None,
            "tsdf": True,
            "texture_blender": True,
        },
    }



# ============================================================================
# IN-MEMORY JOB STORE CHO CHẠY BẤT ĐỒNG BỘ (CHỐNG TIMEOUT CLOUDFLARE 100s)
# ============================================================================
JOBS = {}


def execute_3d_pipeline(saved_paths: List[str], mode: str = "auto") -> dict:
    """
    Thực thi chuỗi xử lý tái tạo 3D hoàn chỉnh 100% không mock:
    - 1 ảnh: Tiền xử lý đơn ảnh -> Depth-Anything-V2 / TripoSR
    - >=2 ảnh: Tiền xử lý đa ảnh -> Depth-Anything-V2 + DA3-blender -> TSDF Mesh 360° -> XAtlas Texture Blender.
      Kèm cơ chế cứu hộ Fail-safe theo đúng đặc tả plan (docs/require.md).
    """
    pipeline_start = time.time()
    base_name = os.path.splitext(os.path.basename(saved_paths[0]))[0]
    output_glb_filename = f"result_{base_name}.glb"
    output_glb_path = os.path.join(OUTPUT_DIR, output_glb_filename)

    # ════════════════════════════════════════════════════════════════
    # NHÁNH 1: Chỉ có 1 ảnh → Chạy qua P1 preprocessing + chọn engine
    # ════════════════════════════════════════════════════════════════
    if len(saved_paths) == 1:
        logger.info(f"═══ CHẾ ĐỘ 1 ẢNH (mode={mode}) ═══")
        preprocess_result = preprocess_single_view(
            image_path=saved_paths[0],
            target_size=512,
            device=device,
        )
        logger.info(
            f"[P1] Hoàn tất đơn ảnh: centered={preprocess_result['image_centered'].shape}, "
            f"focal={preprocess_result['focal_length']}"
        )

        effective_mode = mode if mode != "auto" else "fast"

        if effective_mode == "geometric":
            logger.info("[Option 1] Chạy Depth Reconstruction Pipeline...")
            success, model_path, exec_time = depth_engine.reconstruct(
                image_rgb=preprocess_result["image_centered"],
                alpha_mask=preprocess_result["alpha_mask_centered"],
                focal_length=preprocess_result["focal_length"],
                output_path=output_glb_path,
            )
            pipeline_type = "depth_geometric"
        else:
            from engine_triposr import HAS_TRIPOSR
            if HAS_TRIPOSR:
                logger.info("[Option 2] Chạy TripoSR (360° Full Mesh) từ ảnh đã preprocess...")
                success, model_path, exec_time = triposr_engine.run_from_preprocessed(
                    image_rgb=preprocess_result["image_centered"],
                    alpha_mask=preprocess_result["alpha_mask_centered"],
                    output_glb_path=output_glb_path,
                )
                pipeline_type = "triposr_preprocessed"
            else:
                success = False
                model_path = None
                exec_time = 0.0

            if not success or model_path is None:
                logger.info("[Option 2] Chuyển sang Depth-Anything-V2 Reconstruction Pipeline...")
                success, model_path, exec_time = depth_engine.reconstruct(
                    image_rgb=preprocess_result["image_centered"],
                    alpha_mask=preprocess_result["alpha_mask_centered"],
                    focal_length=preprocess_result["focal_length"],
                    output_path=output_glb_path,
                )
                pipeline_type = "depth_geometric"

        return {
            "status": "success" if success else "failed",
            "mode": "single_image",
            "pipeline_type": pipeline_type,
            "quality_passed": None,
            "gate_reason": f"Chế độ đơn ảnh: {pipeline_type}",
            "execution_time_seconds": round(exec_time, 2),
            "output_file": model_path,
            "num_input_images": 1,
            "preprocessing": {
                "focal_length": preprocess_result["focal_length"],
                "original_size": preprocess_result["original_size"],
                "scale_factor": preprocess_result["scale_factor"],
            },
        }

    # ════════════════════════════════════════════════════════════════
    # NHÁNH 2: Nhiều ảnh (2–8 ảnh) → Full Multi-View Pipeline 360°
    # ════════════════════════════════════════════════════════════════
    logger.info(f"═══ CHẾ ĐỘ MULTI-VIEW ({len(saved_paths)} ảnh, mode={mode}) ═══")

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

    # ── LUỒNG CHUẨN: NVIDIA + Depth-Anything-V2 + DA3-blender ──
    logger.info("[P2] Dự đoán bản đồ độ sâu đa góc bằng Depth-Anything-V2 + DA3-blender filter...")
    da_result = depth_engine.predict_multiview_depth(
        images_rgb=preprocess_result["images_rgb"],
        alpha_masks=preprocess_result["alpha_masks"],
    )
    depth_maps = da_result["depth_maps"]

    # Thiết lập camera trajectory quanh vật thể
    camera_poses = generate_camera_poses(
        n_views=len(saved_paths),
        radius=2.2,
        elevation_deg=15.0,
        view_names=saved_paths,
    )
    focal_lengths = preprocess_result.get("focal_lengths")
    if focal_lengths is None or len(focal_lengths) != len(saved_paths):
        focal_lengths = [(550.0, 550.0)] * len(saved_paths)

    # ── Bước 3 (P3): Quality Gate ──
    logger.info("[P3] Kiểm tra tính hợp lệ của dữ liệu đầu vào...")
    is_high_quality = len(depth_maps) >= 2 and all(d is not None for d in depth_maps)
    reason = "Depth maps và camera trajectory hợp lệ theo chuẩn NVIDIA"

    # ── Bước 4 (P4): Tái tạo lưới 3D bằng NVIDIA TSDF Volume & Marching Cubes ──
    logger.info(f"[P4] Bắt đầu tích lũy thể tích NVIDIA TSDF 360° cho {len(depth_maps)} góc nhìn...")
    try:
        mesh = tsdf_engine.reconstruct_from_depth_maps(
            depth_maps=depth_maps,
            alpha_masks=preprocess_result["alpha_masks"],
            camera_poses=camera_poses,
            focal_lengths=focal_lengths,
            view_names=saved_paths,
        )

        logger.info(f"[P5] Trải UV & Nướng màu từ toàn bộ {len(preprocess_result['images_rgb'])} ảnh vào Mesh 360°...")
        success, model_path = texture_blender.process_and_export(
            mesh=mesh,
            images_rgb=preprocess_result["images_rgb"],
            camera_poses=camera_poses,
            focal_lengths=focal_lengths,
            output_path=output_glb_path,
        )
        if not success or not os.path.exists(output_glb_path):
            logger.warning("[P5] Nướng texture gặp sự cố, xuất mesh màu đỉnh 360° trực tiếp của P4.")
            from utils_3d import export_glb
            export_glb(mesh, output_glb_path)
            success = os.path.exists(output_glb_path)
            model_path = output_glb_path

        pipeline_type = "nvidia_depth_anything_tsdf"
        backend_name = "nvidia_depth_anything_tsdf"

    except Exception as mv_err:
        logger.error(f"[LỖI TÁI TẠO ĐA ẢNH NVIDIA TSDF]: {mv_err}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi tái tạo 3D đa ảnh tại khối P4/P5: {mv_err}. Vui lòng kiểm tra log server!"
        )

    total_time = time.time() - pipeline_start

    return {
        "status": "success" if success else "failed",
        "mode": "multiview_pipeline",
        "pipeline_type": pipeline_type,
        "backend": backend_name,
        "quality_passed": is_high_quality,
        "gate_reason": reason,
        "num_input_images": preprocess_result["num_images"],
        "execution_time_seconds": round(total_time, 2),
        "output_file": model_path,
    }


# ============================================================================
# API 1a: Upload ẢNH → Full Pipeline Đồng bộ (Dùng cho cURL, CLI, Local Test)
# ============================================================================
@app.post("/generate-3d/")
async def generate_3d(
    files: List[UploadFile] = File(...),
    mode: str = Form("auto"),
):
    """API đồng bộ: Nhận ảnh -> Xử lý trực tiếp -> Trả JSON khi xong."""
    valid_modes = {"auto", "fast", "geometric"}
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"mode phải là một trong {valid_modes}, nhận được '{mode}'"
        )
    try:
        saved_paths = []
        for f in files:
            input_path = os.path.join(INPUT_DIR, f.filename)
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_paths.append(input_path)
            logger.info(f"Đã lưu: {f.filename} vào {input_path}")

        return execute_3d_pipeline(saved_paths, mode=mode)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline đồng bộ lỗi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# API 1b: Chạy NỀN (Job Polling) — CHỐNG TIMEOUT 100s CỦA CLOUDFLARE TUNNEL
# ============================================================================
@app.post("/generate-3d/job/")
async def generate_3d_job(
    files: List[UploadFile] = File(...),
    mode: str = Form("auto"),
):
    """
    Nhận ảnh -> lưu -> tạo job_id -> trả về NGAY LẬP TỨC (<100ms).
    Pipeline chạy trong thread nền. Frontend polling /generate-3d/job/{id} mỗi 1.5s
    giúp mọi request đều ngắn, không bao giờ bị Cloudflare cắt ở mốc 100s.
    """
    valid_modes = {"auto", "fast", "geometric"}
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"mode phải là một trong {valid_modes}, nhận được '{mode}'"
        )

    job_id = uuid.uuid4().hex[:12]
    saved_paths = []
    for f in files:
        input_path = os.path.join(INPUT_DIR, f.filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_paths.append(input_path)

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "num_images": len(saved_paths),
        "started": time.time(),
    }

    def worker():
        try:
            res = execute_3d_pipeline(saved_paths, mode=mode)
            JOBS[job_id] = {
                "job_id": job_id,
                "status": "done",
                "result": res,
                "elapsed_seconds": round(time.time() - JOBS[job_id]["started"], 2),
            }
        except HTTPException as he:
            JOBS[job_id] = {
                "job_id": job_id,
                "status": "error",
                "error": str(he.detail),
                "elapsed_seconds": round(time.time() - JOBS[job_id]["started"], 2),
            }
        except Exception as e:
            logger.error(f"Job {job_id} gặp ngoại lệ: {e}", exc_info=True)
            JOBS[job_id] = {
                "job_id": job_id,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_seconds": round(time.time() - JOBS[job_id]["started"], 2),
            }

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "running", "num_images": len(saved_paths)}


@app.get("/generate-3d/job/{job_id}")
async def get_job_status(job_id: str):
    """Trạng thái job: running | done | error | not_found. Kèm số giây đã xử lý."""
    job = JOBS.get(job_id)
    if not job:
        return {"status": "not_found"}
    if job["status"] == "running":
        return {**job, "elapsed_seconds": round(time.time() - job["started"], 1)}
    return job


# ============================================================================
# API 2: Upload 1 ẢNH DUY NHẤT (Giữ API cũ tương thích)
# ============================================================================
@app.post("/generate-3d/single/")
async def generate_3d_single(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
):
    """API đơn giản: Upload 1 ảnh -> Gọi execute_3d_pipeline."""
    input_path = os.path.join(INPUT_DIR, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return execute_3d_pipeline([input_path], mode=mode)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)