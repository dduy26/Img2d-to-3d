"""
Ứng Dụng FastAPI Điều Phối Chuỗi Tái Tạo 3D (Phase 6 - P6 Cloud & Web API).
Chuẩn hóa theo NVIDIA 3D Pipeline.

Tích hợp trọn vẹn 5 phân hệ:
    P1 Preprocessing: Tách nền, Histogram matching DALI, Phân loại góc nhìn.
    P2 Depth/Geometry: Depth-Anything-V2-Small + DA3-blender gradient filter.
    P3 Quality Gate: Kiểm định 3 tầng (Cosine angle, Co-visibility graph, Silhouette coverage).
    P4 TSDF Mesh: True Multi-View Silhouette Space Carving, Marching Cubes, Quadric Decimation.
    P5 Texture Blender: Fresnel Angle-Weighted Blending (cos^3 theta), PBR GLB export.

Tính năng phục vụ:
    - Bất đồng bộ qua Job Store (/generate-3d/job/ & /generate-3d/job/{id}) chống timeout 100s Cloudflare.
    - Đồng bộ (/generate-3d/) cho cURL / kiểm thử tự động.
    - Cơ chế cứu hộ (Fail-safe Fallback) khi multi-view gặp rủi ro -> Tự động chuyển sang Single-view Anchor View.
"""

import os
import sys
import time
import uuid
import shutil
import logging
import threading
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import các phân hệ trong backend
try:
    from .preprocess import preprocess_multiview, preprocess_single_view
    from .engine_depth import DepthReconstructionEngine
    from .quality_gate import QualityGate
    from .engine_tsdf_mesh import TSDFMeshEngine, generate_camera_poses
    from .texture_blender import TextureBlender
    from .utils_3d import export_glb
except ImportError:
    from preprocess import preprocess_multiview, preprocess_single_view
    from engine_depth import DepthReconstructionEngine
    from quality_gate import QualityGate
    from engine_tsdf_mesh import TSDFMeshEngine, generate_camera_poses
    from texture_blender import TextureBlender
    from utils_3d import export_glb

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app_backend")

app = FastAPI(title="NVIDIA 3D Reconstruction Pipeline API", version="2.0.0")

# Thư mục dự án, input và output thống nhất
PROJECT_ROOT = os.path.dirname(os.path.dirname(BACKEND_DIR))
FRONTEND_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "frontend")
FRONTEND_INDEX = os.path.join(FRONTEND_DIR, "index.html")

INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount thư mục tĩnh phục vụ file 3D .glb
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ============================================================================
# KHỞI TẠO CÁC ENGINE (Chỉ tải mô hình 1 lần)
# ============================================================================
device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
logger.info(f"[Server] Khởi tạo hệ thống trên thiết bị: {device}")

quality_gate = QualityGate()
depth_engine = DepthReconstructionEngine(device=device)
tsdf_engine = TSDFMeshEngine()
texture_blender = TextureBlender()

logger.info("[Server] ✓ Toàn bộ 5 phân hệ P1-P5 chuẩn NVIDIA đã sẵn sàng phục vụ.")


# ============================================================================
# IN-MEMORY JOB STORE CHO XỬ LÝ NỀN BẤT ĐỒNG BỘ
# ============================================================================
JOBS = {}


def execute_3d_pipeline(saved_paths: List[str], mode: str = "auto") -> dict:
    """
    Điều phối luồng xử lý 2D-to-3D không mock:
    - 1 ảnh: P1 Preprocessing -> P2 Depth Single-view Reconstruction.
    - >=2 ảnh: P1 Multi-view Preprocessing -> P2 Depth-Anything-V2 -> P3 Quality Gate
               -> P4 True Space Carving TSDF Mesh -> P5 Texture Blender.
               Kèm cơ chế cứu hộ Fail-safe nếu Quality Gate cảnh báo rủi ro.
    """
    t_start = time.time()
    base_name = os.path.splitext(os.path.basename(saved_paths[0]))[0]
    output_glb_filename = f"model_{base_name}_{uuid.uuid4().hex[:6]}.glb"
    output_glb_path = os.path.join(OUTPUT_DIR, output_glb_filename)

    # ────────────────────────────────────────────────────────────────
    # NHÁNH 1: Chế độ đơn ảnh (Single-view)
    # ────────────────────────────────────────────────────────────────
    if len(saved_paths) == 1:
        logger.info(f"═══ [PIPELINE] XỬ LÝ CHẾ ĐỘ ĐƠN ẢNH: {saved_paths[0]} ═══")
        prep = preprocess_single_view(saved_paths[0], target_size=512)

        success, model_path, exec_time = depth_engine.reconstruct(
            image_rgb=prep["image_centered"],
            alpha_mask=prep["alpha_mask_centered"],
            focal_length=prep["focal_length"],
            output_path=output_glb_path,
        )

        glb_url = f"/output/{output_glb_filename}" if success else None
        return {
            "status": "success" if success else "failed",
            "mode": "single_view",
            "pipeline": "Depth-Anything-V2-Small (Pinhole Grid)",
            "execution_time_seconds": round(exec_time, 2),
            "output_file": model_path,
            "output_url": glb_url,
            "num_input_images": 1,
        }

    # ────────────────────────────────────────────────────────────────
    # NHÁNH 2: Chế độ đa ảnh (Multi-view 360°)
    # ────────────────────────────────────────────────────────────────
    logger.info(f"═══ [PIPELINE] XỬ LÝ CHẾ ĐỘ MULTI-VIEW ({len(saved_paths)} ảnh) ═══")

    # 1. P1 Preprocessing
    prep = preprocess_multiview(saved_paths, target_size=512)
    images_rgb = prep["images_rgb"]
    alpha_masks = prep["alpha_masks"]
    viewpoint_assignments = prep.get("viewpoint_assignments")
    focal_lengths = prep.get("focal_lengths")

    # 2. P2 Depth Prediction
    depth_res = depth_engine.predict_multiview_depth(images_rgb=images_rgb, alpha_masks=alpha_masks)
    depth_maps = depth_res["depth_maps"]

    # 3. Camera Poses
    camera_poses = generate_camera_poses(
        n_views=len(images_rgb),
        radius=2.2,
        elevation_deg=15.0,
        view_names=saved_paths,
        viewpoint_assignments=viewpoint_assignments,
    )

    # 4. P3 Quality Gate & Fail-safe Decision
    q_passed, q_reason, q_info = quality_gate.evaluate(
        camera_poses=camera_poses,
        alpha_masks=alpha_masks,
        depth_maps=depth_maps,
    )
    logger.info(f"[P3 Quality Gate] Kết quả: passed={q_passed}, lý do: {q_reason}")

    # Nếu Quality Gate cảnh báo lỗi nghiêm trọng -> Kích hoạt Fail-safe fallback sang ảnh neo #0
    if not q_passed and q_info.get("fallback_to_single_view", False):
        anchor_idx = q_info.get("best_view_index", 0)
        logger.warning(
            f"[P3] KÍCH HOẠT CƠ CHẾ CỨU HỘ: Dữ liệu multi-view không đạt chuẩn ({q_reason}). "
            f"Tự động chuyển sang tái tạo đơn ảnh chất lượng cao từ ảnh neo #{anchor_idx}!"
        )
        single_prep = preprocess_single_view(saved_paths[anchor_idx], target_size=512)
        success, model_path, exec_time = depth_engine.reconstruct(
            image_rgb=single_prep["image_centered"],
            alpha_mask=single_prep["alpha_mask_centered"],
            focal_length=single_prep["focal_length"],
            output_path=output_glb_path,
        )
        glb_url = f"/output/{output_glb_filename}" if success else None
        return {
            "status": "success" if success else "failed",
            "mode": "fallback_single_view",
            "pipeline": "Quality-Gate Fail-safe -> Depth-Anything-V2",
            "quality_passed": False,
            "quality_reason": q_reason,
            "execution_time_seconds": round(time.time() - t_start, 2),
            "output_file": model_path,
            "output_url": glb_url,
            "num_input_images": len(saved_paths),
        }

    # 5. P4 True Space Carving TSDF Mesh
    try:
        mesh = tsdf_engine.reconstruct_from_depth_maps(
            depth_maps=depth_maps,
            alpha_masks=alpha_masks,
            camera_poses=camera_poses,
            focal_lengths=focal_lengths,
            view_names=saved_paths,
            viewpoint_assignments=viewpoint_assignments,
        )
    except Exception as e:
        logger.error(f"[P4] Lỗi TSDF Space Carving: {e}. Kích hoạt Fallback sang Single-view...", exc_info=True)
        single_prep = preprocess_single_view(saved_paths[0], target_size=512)
        success, model_path, _ = depth_engine.reconstruct(
            image_rgb=single_prep["image_centered"],
            alpha_mask=single_prep["alpha_mask_centered"],
            focal_length=single_prep["focal_length"],
            output_path=output_glb_path,
        )
        glb_url = f"/output/{output_glb_filename}" if success else None
        return {
            "status": "success" if success else "failed",
            "mode": "fallback_single_view",
            "pipeline": "P4-Failure Fallback -> Depth-Anything-V2",
            "execution_time_seconds": round(time.time() - t_start, 2),
            "output_file": model_path,
            "output_url": glb_url,
            "num_input_images": len(saved_paths),
        }

    # 6. P5 Texture Blender (Fresnel Angle-Weighted Blending)
    p5_success, model_path = texture_blender.process_and_export(
        mesh=mesh,
        images_rgb=images_rgb,
        camera_poses=camera_poses,
        focal_lengths=focal_lengths,
        output_path=output_glb_path,
    )

    if not p5_success or not os.path.exists(output_glb_path):
        logger.warning("[P5] Nướng texture gặp sự cố, xuất mesh màu đỉnh P4 trực tiếp.")
        export_glb(mesh, output_glb_path)
        p5_success = os.path.exists(output_glb_path)
        model_path = output_glb_path

    total_time = time.time() - t_start
    glb_url = f"/output/{output_glb_filename}" if p5_success else None

    return {
        "status": "success" if p5_success else "failed",
        "mode": "multiview_360",
        "pipeline": "NVIDIA True Space Carving TSDF + Angle-Weighted Blending",
        "quality_passed": True,
        "quality_reason": q_reason,
        "execution_time_seconds": round(total_time, 2),
        "output_file": model_path,
        "output_url": glb_url,
        "num_input_images": len(saved_paths),
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", include_in_schema=False)
async def frontend():
    """Phục vụ giao diện Web UI (Three.js)."""
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    return {"message": "Frontend chưa được cài đặt. Mở /docs để dùng Swagger API."}


@app.get("/api/health")
async def health():
    """Kiểm tra trạng thái sức khỏe của dịch vụ và các mô hình."""
    return {
        "status": "ok",
        "device": device,
        "frontend": os.path.exists(FRONTEND_INDEX),
        "engines": {
            "depth_anything_v2": depth_engine.depth_model is not None,
            "quality_gate": True,
            "tsdf_space_carving": True,
            "texture_blender": True,
        },
    }


@app.post("/generate-3d/")
async def generate_3d(
    files: List[UploadFile] = File(...),
    mode: str = Form("auto"),
):
    """
    API đồng bộ: Nhận danh sách ảnh -> Thực thi pipeline -> Trả về kết quả JSON.
    """
    try:
        saved_paths = []
        for f in files:
            path = os.path.join(INPUT_DIR, f.filename)
            with open(path, "wb") as buf:
                shutil.copyfileobj(f.file, buf)
            saved_paths.append(path)

        return execute_3d_pipeline(saved_paths, mode=mode)
    except Exception as e:
        logger.error(f"Lỗi generate_3d: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-3d/job/")
async def generate_3d_job(
    files: List[UploadFile] = File(...),
    mode: str = Form("auto"),
):
    """
    API bất đồng bộ: Nhận ảnh -> Tạo job_id -> Trả response ngay (<100ms).
    Giải quyết triệt để lỗi ngắt kết nối 100 giây của Cloudflare Tunnel.
    """
    job_id = uuid.uuid4().hex[:12]
    saved_paths = []
    for f in files:
        path = os.path.join(INPUT_DIR, f.filename)
        with open(path, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
        saved_paths.append(path)

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
        except Exception as e:
            logger.error(f"Job {job_id} lỗi: {e}", exc_info=True)
            JOBS[job_id] = {
                "job_id": job_id,
                "status": "error",
                "error": str(e),
                "elapsed_seconds": round(time.time() - JOBS[job_id]["started"], 2),
            }

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "running", "num_images": len(saved_paths)}


@app.get("/generate-3d/job/{job_id}")
async def get_job_status(job_id: str):
    """Kiểm tra tiến độ job (running | done | error | not_found)."""
    job = JOBS.get(job_id)
    if not job:
        return {"status": "not_found"}
    if job["status"] == "running":
        return {**job, "elapsed_seconds": round(time.time() - job["started"], 1)}
    return job


@app.post("/generate-3d/single/")
async def generate_3d_single(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
):
    """API đơn ảnh tương thích ngược."""
    path = os.path.join(INPUT_DIR, file.filename)
    with open(path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    return execute_3d_pipeline([path], mode=mode)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)