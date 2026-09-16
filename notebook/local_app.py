"""
local_app.py — Local Gradio UI & Edge Preprocessing Client
==========================================================
Chức năng:
  1. Giao diện Web Gradio trực quan chạy tại máy cục bộ (Local Edge).
  2. Tích hợp Module Tiền xử lý tại Local:
       - Tách nền tự động bằng Alpha Matting / Lab Chroma Otsu / RMBG.
       - Chuẩn hóa kích thước giữ nguyên tỉ lệ (Aspect Ratio) trên canvas 512x512.
  3. Kết nối Cloud Hybrid (Google Colab T4 qua Cloudflare Tunnel / Ngrok):
       - Gửi ảnh sạch đã tách nền lên API Colab.
       - Nhận file 3D định dạng .glb duy nhất và hiển thị xoay 360 độ trực tiếp.
  4. Hỗ trợ Local Engine Fallback: Tự động chạy offline trên máy nếu không có Cloud.
"""

from __future__ import annotations

import os
import sys
import time
import io
import json
import tempfile
import requests
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image
import gradio as gr

# Thêm đường dẫn project root vào sys.path
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from notebook.backend.preprocess import (
    load_image_safe,
    compute_alpha_mask,
    place_on_canvas,
    ensure_rgb,
    CANVAS_SIZE,
)
from notebook.backend.app import execute_3d_pipeline, jobs, new_job

OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. HÀM TIỀN XỬ LÝ CỤC BỘ (LOCAL PREPROCESSING)
# ---------------------------------------------------------------------------

def process_local_images(image_files) -> Tuple[List[Image.Image], str]:
    """Tách nền và chuẩn hóa danh sách ảnh đầu vào thành canvas 512x512 trong suốt."""
    if not image_files:
        return [], "⚠️ Vui lòng tải lên ít nhất 1 ảnh."

    if not isinstance(image_files, list):
        image_files = [image_files]

    processed_pil_images = []
    log_messages = []

    t0 = time.time()
    for idx, item in enumerate(image_files):
        # Gradio có thể truyền path dạng str hoặc object có thuộc tính .name
        path = item.name if hasattr(item, "name") else str(item)
        raw = load_image_safe(path)
        rgb = ensure_rgb(raw)

        # Tách nền tự động
        alpha = compute_alpha_mask(raw)

        # Chuẩn hóa kích thước giữ tỉ lệ trên canvas 512x512
        canvas_rgb, canvas_alpha, K_prime, scale, off_x, off_y = place_on_canvas(
            rgb, alpha, canvas_size=CANVAS_SIZE
        )

        # Ghép thành RGBA để hiển thị nền trong suốt
        alpha_u8 = (canvas_alpha * 255.0).clip(0, 255).astype(np.uint8)
        rgba = np.dstack([canvas_rgb, alpha_u8])
        pil_img = Image.fromarray(rgba, mode="RGBA")
        processed_pil_images.append(pil_img)

        fg_percent = (canvas_alpha > 0.5).mean() * 100.0
        log_messages.append(
            f"• Ảnh {idx + 1} ({Path(path).name}): Tỉ lệ co giãn {scale:.3f}, Foreground {fg_percent:.2f}%"
        )

    elapsed = time.time() - t0
    status_text = (
        f"✅ Tiền xử lý hoàn tất {len(processed_pil_images)} ảnh trong {elapsed:.2f} giây!\n"
        + "\n".join(log_messages)
    )
    return processed_pil_images, status_text


# ---------------------------------------------------------------------------
# 2. HÀM ĐIỀU PHỐI TẠO MÔ HÌNH 3D (HYBRID DISPATCHER)
# ---------------------------------------------------------------------------

def generate_3d_model(
    image_files,
    cloud_url: str,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], str]:
    """Tạo mô hình 3D .glb từ ảnh qua Cloud Colab API hoặc Local Fallback."""
    if not image_files:
        return None, "⚠️ Vui lòng tải lên ít nhất 1 ảnh trước khi tạo 3D!"

    if not isinstance(image_files, list):
        image_files = [image_files]

    progress(0.1, desc="Đang tiền xử lý ảnh...")
    # Bước 1: Lưu ảnh sạch đã tiền xử lý vào thư mục tạm
    temp_dir = Path(tempfile.mkdtemp(prefix="local_3d_"))
    saved_clean_paths = []

    for idx, item in enumerate(image_files):
        path = item.name if hasattr(item, "name") else str(item)
        raw = load_image_safe(path)
        rgb = ensure_rgb(raw)
        alpha = compute_alpha_mask(raw)
        c_rgb, c_alpha, _, _, _, _ = place_on_canvas(rgb, alpha, canvas_size=CANVAS_SIZE)
        alpha_u8 = (c_alpha * 255.0).clip(0, 255).astype(np.uint8)
        rgba = np.dstack([c_rgb, alpha_u8])

        save_path = temp_dir / f"view_{idx:02d}.png"
        Image.fromarray(rgba, mode="RGBA").save(save_path)
        saved_clean_paths.append(str(save_path))

    cloud_endpoint = cloud_url.strip().rstrip("/")

    # Bước 2: Kiểm tra xem có gửi lên Cloud Colab API không
    if cloud_endpoint and not cloud_endpoint.startswith("http://127.0.0.1") and not cloud_endpoint.startswith("http://localhost"):
        progress(0.3, desc=f"Đang gửi request lên Cloud Colab ({cloud_endpoint})...")
        t0 = time.time()
        try:
            # Gửi Multipart files lên Colab API
            files_payload = []
            for p in saved_clean_paths:
                files_payload.append(("files", (Path(p).name, open(p, "rb"), "image/png")))

            reconstruct_url = f"{cloud_endpoint}/reconstruct"
            resp = requests.post(reconstruct_url, files=files_payload, timeout=30)
            if resp.status_code != 200:
                return None, f"❌ Lỗi từ Colab Server ({resp.status_code}): {resp.text}"

            job_data = resp.json()
            job_id = job_data.get("job_id")
            if not job_id:
                return None, f"❌ Không nhận được job_id từ Cloud: {job_data}"

            # Polling kiểm tra trạng thái
            progress(0.5, desc="Cloud T4 đang xử lý mô hình 3D...")
            poll_url = f"{cloud_endpoint}/status/{job_id}"
            max_wait_seconds = 180
            start_poll = time.time()

            while time.time() - start_poll < max_wait_seconds:
                poll_resp = requests.get(poll_url, timeout=10)
                if poll_resp.status_code == 200:
                    status_data = poll_resp.json()
                    st = status_data.get("status")
                    if st == "DONE":
                        break
                    elif st == "ERROR":
                        return None, f"❌ Lỗi suy luận trên Cloud: {status_data.get('error')}"
                time.sleep(2)
            else:
                return None, "❌ Quá thời gian chờ (Timeout 180s) từ Cloud Colab!"

            # Tải file .glb về
            progress(0.85, desc="Đang tải file .glb về máy cục bộ...")
            download_url = f"{cloud_endpoint}/download/{job_id}"
            dl_resp = requests.get(download_url, timeout=60)
            if dl_resp.status_code != 200:
                return None, f"❌ Lỗi khi tải kết quả .glb: {dl_resp.status_code}"

            local_glb = OUTPUT_DIR / f"cloud_model_{job_id[:8]}.glb"
            local_glb.write_bytes(dl_resp.content)

            elapsed = time.time() - t0
            mesh_info = status_data.get("mesh_info", {})
            info_str = (
                f"🎉 TÁI TẠO 3D THÀNH CÔNG TỪ CLOUD COLAB!\n"
                f"• Tổng thời gian: {elapsed:.2f} giây\n"
                f"• Chế độ: {status_data.get('mode')}\n"
                f"• Số mặt tam giác (Faces): {mesh_info.get('face_count', 'N/A')}\n"
                f"• Kín nước (Watertight): {mesh_info.get('is_watertight', 'N/A')}\n"
                f"• File xuất: {local_glb.name}"
            )
            progress(1.0, desc="Hoàn tất!")
            return str(local_glb), info_str

        except Exception as e:
            return None, f"❌ Lỗi kết nối tới Cloud Colab: {e}\n(Vui lòng kiểm tra lại URL Cloudflare Tunnel)"

    # Bước 3: Nếu không có Cloud URL hoặc là localhost -> Chạy Local Pipeline Engine
    progress(0.3, desc="Đang khởi chạy Local 3D Reconstruction Pipeline...")
    t0 = time.time()
    try:
        job_id = new_job()
        execute_3d_pipeline(job_id, saved_clean_paths)
        job_info = jobs.get(job_id, {})

        if job_info.get("status") == "DONE":
            out_glb = job_info.get("result_path")
            mesh_info = job_info.get("mesh_info", {})
            elapsed = time.time() - t0
            info_str = (
                f"🎉 TÁI TẠO 3D CỤC BỘ (LOCAL ENGINE) THÀNH CÔNG!\n"
                f"• Thời gian thực thi: {elapsed:.2f} giây\n"
                f"• Chế độ: {job_info.get('mode')}\n"
                f"• Số mặt tam giác (Faces): {mesh_info.get('face_count', 'N/A'):,}\n"
                f"• Số đỉnh (Vertices): {mesh_info.get('vertex_count', 'N/A'):,}\n"
                f"• Kín nước (Watertight): {mesh_info.get('is_watertight', 'N/A')}\n"
                f"• Cạnh biên hở (Open Edges): {mesh_info.get('boundary_edges', 0)}\n"
                f"• File lưu tại: {out_glb}"
            )
            progress(1.0, desc="Hoàn tất!")
            return str(out_glb), info_str
        else:
            return None, f"❌ Lỗi trong Local Pipeline: {job_info.get('error')}"

    except Exception as e:
        return None, f"❌ Lỗi ngoài dự kiến: {e}"


# ---------------------------------------------------------------------------
# 3. GIAO DIỆN GRADIO UI HIỆN ĐẠI
# ---------------------------------------------------------------------------

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.hero-header {
    text-align: center;
    padding: 20px 0 10px 0;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 20px;
}
.hero-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.hero-header p {
    color: #94a3b8;
    font-size: 1.05rem;
}
"""

with gr.Blocks(title="2D to 3D AI Studio — Hybrid Client") as demo:
    with gr.Column(elem_classes=["hero-header"]):
        gr.Markdown(
            """
            # 🌟 2D to 3D AI Studio — Hybrid Architecture
            ### Tiền xử lý tại Local (Edge) ⇄ Dựng hình 3D siêu tốc trên Cloud (Google Colab T4)
            """
        )

    with gr.Row():
        # Cột bên trái: Upload & Cấu hình
        with gr.Column(scale=5):
            gr.Markdown("### 📥 1. Tải Lên Ảnh Đầu Vào")
            image_input = gr.File(
                label="Chọn 1 hoặc nhiều ảnh vật thể (.png / .jpg / .webp)",
                file_count="multiple",
                file_types=["image"],
            )

            gr.Markdown("### 🌐 2. Kết Nối Cloud Server (Tùy chọn)")
            cloud_url_input = gr.Textbox(
                label="Địa chỉ Colab Tunnel URL (Cloudflare / Localtunnel)",
                placeholder="Ví dụ: https://xxxx.trycloudflare.com (Để trống để chạy thuần Local)",
                value="",
            )

            with gr.Row():
                btn_preprocess = gr.Button("🔍 1. Tiền Xử Lý & Tách Nền (Local)", variant="secondary")
                btn_generate = gr.Button("🚀 2. Tạo Mô Hình 3D (.glb)", variant="primary")

            status_output = gr.Textbox(
                label="📋 Nhật Ký Hoạt Động & Thông Số Kỹ Thuật",
                lines=8,
                interactive=False,
            )

        # Cột bên phải: Hiển thị Preview & Model 3D Viewer
        with gr.Column(scale=6):
            gr.Markdown("### 🖼️ Ảnh Sạch Sau Khi Tách Nền (Canvas 512x512)")
            preview_gallery = gr.Gallery(
                label="Preview ảnh tiền xử lý",
                columns=3,
                height=220,
                object_fit="contain",
            )

            gr.Markdown("### 🧊 Khung Nhìn Tương Tác 3D (Interactive 3D Viewer)")
            model3d_viewer = gr.Model3D(
                label="Mô hình 3D .glb hoàn thiện (Kéo chuột để xoay 360°, cuộn để phóng to)",
                height=420,
                clear_color=[0.12, 0.15, 0.2, 1.0],
            )

    # Ràng buộc sự kiện
    btn_preprocess.click(
        fn=process_local_images,
        inputs=[image_input],
        outputs=[preview_gallery, status_output],
    )

    btn_generate.click(
        fn=generate_3d_model,
        inputs=[image_input, cloud_url_input],
        outputs=[model3d_viewer, status_output],
    )


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ĐANG KHỞI CHẠY GIAO DIỆN LOCAL GRADIO CLIENT...")
    print("👉 Mở trình duyệt tại: http://127.0.0.1:7860")
    print("=" * 70)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft(), css=custom_css)
