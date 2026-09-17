"""
app.py — P6: FastAPI Server & Asynchronous Job Queue
=====================================================
Chuc nang:
  1. Endpoint POST /reconstruct: Nhan N file anh, dua vao hang doi xu ly.
  2. Endpoint GET  /status/{job_id}: Kiem tra trang thai xu ly.
  3. Endpoint GET  /download/{job_id}: Tai file .glb ket qua.
  4. Endpoint GET  /: Phuc vu giao dien Web UI (index.html).
  5. Background worker chay async, mien nhiem timeout 100s cua Cloudflare.

Kien truc:
  - FastAPI + asyncio.Queue (khong can Redis/Celery cho local).
  - Job co vong doi: PENDING -> PROCESSING -> DONE / ERROR.
  - Ket qua .glb luu tam vao output/ va phuc vu download.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
import traceback
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Internal pipeline imports
from .preprocess      import preprocess_images
from .engine_depth    import estimate_depth_pipeline
from .quality_gate    import run_quality_gate
from .engine_tsdf_mesh import reconstruct_mesh
from .texture_blender import apply_texture
from .utils_3d        import (
    check_mesh_health,
    compute_normals,
    export_glb,
    get_orthographic_camera_poses,
)

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent.parent   # notebook/
OUTPUT_DIR  = BASE_DIR.parent / "output"
FRONTEND_DIR = BASE_DIR / "frontend"
TEMP_DIR    = Path(tempfile.gettempdir()) / "imgtomodel_jobs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE_MB = 50
MAX_FILES          = 8

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ImgToModel 3D Reconstruction API",
    description="Chuyen doi anh 2D sang mo hinh 3D .glb (GLTF 2.0 Binary)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ---------------------------------------------------------------------------
# Job Store (in-memory)
# ---------------------------------------------------------------------------

jobs: dict = {}   # job_id -> {"status", "created_at", "result_path", "error", "mode"}

def new_job() -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status":     "PENDING",
        "created_at": time.time(),
        "result_path": None,
        "error":      None,
        "mode":       None,
        "mesh_info":  None,
    }
    return job_id

# ---------------------------------------------------------------------------
# Core Pipeline Execution
# ---------------------------------------------------------------------------

def execute_3d_pipeline(job_id_or_paths, image_paths: list = None) -> dict:
    """Chay toan bo pipeline 3D cho mot job hoac danh sach anh.

    Ho tro 2 cach goi:
      1. execute_3d_pipeline(image_paths) -> tra ve dict ket qua (dung trong Colab / script)
      2. execute_3d_pipeline(job_id, image_paths) -> cap nhat jobs[job_id] va tra ve dict

    Thu tu:
      P1 -> P2 -> P3 (Quality Gate) -> P4 (TSDF) -> P5 (Texture) -> GLB Export.
    """
    if image_paths is None:
        if isinstance(job_id_or_paths, (list, tuple)):
            job_id = new_job()
            image_paths = list(job_id_or_paths)
        else:
            job_id = str(job_id_or_paths)
            image_paths = []
    else:
        job_id = str(job_id_or_paths)
        if job_id not in jobs:
            jobs[job_id] = {
                "status":     "PENDING",
                "created_at": time.time(),
                "result_path": None,
                "error":      None,
                "mode":       None,
                "mesh_info":  None,
            }

    try:
        jobs[job_id]["status"] = "PROCESSING"
        out_glb = str(OUTPUT_DIR / f"{job_id}.glb")

        # UU TIEN: Neu N >= 2 va co GPU CUDA -> Chay DUSt3R Multi-View AI Engine
        import torch
        if len(image_paths) >= 2 and torch.cuda.is_available():
            try:
                from .engine_dust3r import reconstruct_dust3r
                dust3r_res = reconstruct_dust3r(image_paths, out_glb, device="cuda:0")
                jobs[job_id]["status"] = "DONE"
                jobs[job_id]["result_path"] = out_glb
                jobs[job_id]["mode"] = "multi_view_dust3r"
                jobs[job_id]["mesh_info"] = dust3r_res.get("mesh_info", {})
                return {
                    "status": "success",
                    "output_file": out_glb,
                    "result_path": out_glb,
                    "mode": "multi_view_dust3r",
                    "pipeline": "DUSt3R Multi-View AI Engine",
                    "mesh_info": dust3r_res.get("mesh_info", {}),
                }
            except Exception as d_err:
                print(f"⚠️ DUSt3R gap loi: {d_err}. Chuyen sang fallback...")

        # P1: Tien xu ly
        preprocessed = preprocess_images(image_paths)

        # P2: Uoc luong do sau
        depth_views = estimate_depth_pipeline(preprocessed)

        # P3: Quality Gate
        gate_result = run_quality_gate(depth_views)
        jobs[job_id]["mode"] = gate_result.mode

        if gate_result.mode == "multi_view":
            active_views = gate_result.valid_views
        else:
            # Single-view fallback: chi dung anchor
            active_views = [gate_result.anchor_view]

        # P4: Reconstruct Mesh
        vertices, faces, mesh_health = reconstruct_mesh(active_views)
        jobs[job_id]["mesh_info"] = {
            "vertex_count":   mesh_health.vertex_count,
            "face_count":     mesh_health.face_count,
            "is_watertight":  mesh_health.is_watertight,
            "boundary_edges": mesh_health.boundary_edges,
            "components":     mesh_health.components,
            "euler_number":   mesh_health.euler_number,
        }

        # P5: Texture
        N = len(active_views)
        poses = get_orthographic_camera_poses(n_views=N)
        Rs = [p[0] for p in poses]
        ts = [p[1] for p in poses]

        texture_result = apply_texture(vertices, faces, active_views, Rs, ts)

        # Export GLB
        normals = texture_result["normals"]
        colors  = texture_result["vertex_colors"]
        out_glb = str(OUTPUT_DIR / f"{job_id}.glb")
        export_glb(vertices, faces, colors=colors, normals=normals, out_path=out_glb)

        jobs[job_id]["status"]      = "DONE"
        jobs[job_id]["result_path"] = out_glb

    except Exception as e:
        jobs[job_id]["status"] = "ERROR"
        jobs[job_id]["error"]  = str(e) + "\n" + traceback.format_exc()

    info = jobs[job_id]
    return {
        "status":      "success" if info["status"] == "DONE" else "error",
        "output_file": info.get("result_path"),
        "result_path": info.get("result_path"),
        "mode":        info.get("mode"),
        "pipeline":    "P1-P5 Full TSDF Mesh",
        "mesh_info":   info.get("mesh_info"),
        "error":       info.get("error"),
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, summary="Web UI")
async def serve_ui():
    """Phuc vu giao dien Web UI (Three.js viewer)."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="vi">
<head><meta charset="UTF-8"><title>ImgToModel</title></head>
<body>
  <h1>ImgToModel API</h1>
  <p>Frontend chua duoc build. Vui long tao notebook/frontend/index.html.</p>
  <p><a href="/docs">API Docs (Swagger UI)</a></p>
</body>
</html>
""")


@app.post("/reconstruct", summary="Bat dau tai tao 3D tu anh")
async def reconstruct(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="1-8 file anh (JPEG/PNG)"),
):
    """Nhan N file anh va bat dau qua trinh tai tao 3D.

    Returns:
        {"job_id": "...", "status": "PENDING"}
    """
    # Kiem tra so luong
    if len(files) < 1:
        raise HTTPException(400, "Vui long tai len it nhat 1 anh.")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Toi da {MAX_FILES} anh moi lan.")

    # Luu file tam thoi
    job_id  = new_job()
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for f in files:
        # Kiem tra dinh dang
        suffix = Path(f.filename or "img.jpg").suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
            raise HTTPException(400, f"Dinh dang khong ho tro: {suffix}")
        dest = job_dir / f"{len(saved_paths):04d}{suffix}"
        content = await f.read()
        if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(400, f"File {f.filename} qua lon (>{MAX_UPLOAD_SIZE_MB}MB).")
        dest.write_bytes(content)
        saved_paths.append(str(dest))

    # Chay pipeline trong background task
    background_tasks.add_task(execute_3d_pipeline, job_id, saved_paths)

    return JSONResponse({
        "job_id": job_id,
        "status": "PENDING",
        "message": f"Da nhan {len(saved_paths)} anh. Pipeline dang chay.",
    })


@app.get("/status/{job_id}", summary="Kiem tra trang thai job")
async def get_status(job_id: str):
    """Tra ve trang thai hien tai cua job.

    Status values: PENDING | PROCESSING | DONE | ERROR
    """
    if job_id not in jobs:
        raise HTTPException(404, f"Khong tim thay job: {job_id}")
    job = jobs[job_id]
    resp = {
        "job_id":    job_id,
        "status":    job["status"],
        "mode":      job.get("mode"),
        "mesh_info": job.get("mesh_info"),
        "elapsed_s": round(time.time() - job["created_at"], 2),
    }
    if job["status"] == "ERROR":
        resp["error"] = job["error"]
    if job["status"] == "DONE":
        resp["download_url"] = f"/download/{job_id}"
    return JSONResponse(resp)


@app.get("/download/{job_id}", summary="Tai file .glb ket qua")
async def download_result(job_id: str):
    """Tra ve file .glb da tai tao."""
    if job_id not in jobs:
        raise HTTPException(404, f"Khong tim thay job: {job_id}")
    job = jobs[job_id]
    if job["status"] != "DONE":
        raise HTTPException(400, f"Job chua hoan thanh. Trang thai: {job['status']}")
    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(500, "File ket qua khong ton tai.")
    return FileResponse(
        path=result_path,
        media_type="model/gltf-binary",
        filename=f"model_{job_id[:8]}.glb",
    )


@app.get("/api/health", summary="Health check (API alias)")
@app.get("/health", summary="Health check")
async def health_check():
    """Kiem tra server co dang hoat dong khong."""
    return {"status": "ok", "jobs_count": len(jobs)}


# ---------------------------------------------------------------------------
# Chay thu cuc bo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "notebook.backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
