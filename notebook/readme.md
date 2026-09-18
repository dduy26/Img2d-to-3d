# Hệ Thống Tái Tạo Mô Hình 3D Từ Ảnh 2D (2D to 3D Dual-Engine Studio)

Hệ thống tái tạo mô hình 3D nguyên khối hoàn chỉnh từ ảnh 2D (đơn ảnh hoặc 2–8 ảnh đa góc 360°) sử dụng kiến trúc kết hợp các mô hình Deep Learning SOTA:
1. **⚡ Luồng 1 (Single-View - 1 ảnh):** **TripoSR** (ViT + Triplane NeRF + Marching Cubes) sinh mesh 3D kín nước 100% (Watertight) trong **~1.5 giây**.
2. **🌐 Luồng 2 (Multi-View - $N \ge 2$ ảnh):** **Tencent Hunyuan3D-2mv** (DiT Flow Matching Pipeline) kết hợp bộ gán góc Hungarian Viewpoint Assignment (P1) sinh lưới 3D đặc kín nước 100% chuẩn CAD/Game asset, phủ màu chân thực từ ảnh chụp thực tế bằng **Multi-View Texture Blender** (Fresnel $\cos^3\theta$ + Z-buffer Occlusion culling).

---

## 1. Cấu trúc thư mục

Kiến trúc thư mục được tổ chức tinh gọn và chuẩn hoá:

```text
Img2d-to-3d/
├── input/                # Thư mục nhận ảnh đầu vào
├── output/               # Thư mục xuất file mô hình 3D .glb
└── notebook/
    ├── frontend/         # Giao diện Web 3D Viewer (Three.js CDN + HTML5)
    │   └── index.html
    ├── demo_colab.ipynb  # Sổ tay chạy trọn gói trên Google Colab (T4 GPU)
    ├── local_app.py      # Trình khách Gradio UI chạy cục bộ
    └── backend/
        ├── app.py               # Máy chủ FastAPI (API Điều phối & Job Polling)
        ├── preprocess.py        # P1: Tách nền, vá lỗ PET/highlights, nhận diện mặt (CLIP/HOG)
        ├── engine_hunyuan3d.py  # P2: Tencent Hunyuan3D-2mv DiT Multi-View Pipeline
        ├── engine_depth.py      # P2: Depth-Anything-V2 dự đoán độ sâu đa góc & đơn ảnh
        ├── quality_gate.py      # P3: Kiểm định chất lượng góc chụp & bao phủ camera
        ├── engine_tsdf_mesh.py  # P4: Volumetric TSDF Space Carving & Marching Cubes 360°
        ├── texture_blender.py   # P5: Trải phẳng UV & nướng màu bề mặt chân thực
        └── utils_3d.py          # Xuất định dạng GLB chuẩn PBR / Vertex Colors
```

---

## 2. Yêu cầu hệ thống & Cài đặt

- **Ngôn ngữ:** Python 3.10 – 3.13.
- **Phần cứng:** Chạy tối ưu trên GPU NVIDIA (Google Colab T4 15GB VRAM) và hỗ trợ chạy trên CPU / GPU cục bộ.

**Cài đặt các thư viện cần thiết:**

```bash
pip install fastapi uvicorn python-multipart trimesh rembg onnxruntime
pip install transformers xatlas roma einops safetensors matplotlib tqdm fast-simplification scipy
```

*(Trên Google Colab GPU T4, notebook `demo_colab.ipynb` tự động cài đặt `hy3dgen` và nạp mô hình `tencent/Hunyuan3D-2mv`).*

---

## 3. Luồng hoạt động (Workflow P1 → P6)

Hệ thống hoạt động theo quy trình 6 phân hệ khép kín:

1. **P1 Preprocessing (`preprocess.py`):**
   - Tách nền vật thể bằng Rembg / RMBG-2.0.
   - Thuật toán `refine_alpha_mask` tự động lấp kín lỗ thủng phản xạ trên chai nhựa trong suốt / kim loại bóng.
   - Phân loại mặt ảnh (`classify_viewpoints`) bằng CLIP ViT Zero-shot kết hợp HOG Bilateral Gradient Symmetry, giải thuật gán cặp Hungarian (`linear_sum_assignment`) gán đúng góc thực $0^\circ, 90^\circ, 180^\circ, 270^\circ$ mà không bị ghép loạn góc.
2. **P2 3D Geometry Generation:**
   - **Đơn ảnh:** `TripoSR` sinh mesh 3D kín nước trong 1.5s.
   - **Đa ảnh ($N \ge 2$):** `Tencent Hunyuan3D-2mv` DiT Flow Matching Pipeline sinh khối 3D chuẩn CAD/Game asset có đầy đủ mặt đế và lòng vật thể.
   - **Dự phòng Local:** `Depth-Anything-V2` dự đoán độ sâu đa góc.
3. **P3 Quality Gate (`quality_gate.py`):**
   - Kiểm tra độ bao phủ góc chụp, tránh hiện tượng chụp trùng 1 góc hoặc thiếu góc đối xứng; tự động fallback về luồng đơn ảnh nếu bộ ảnh đa góc không đạt chuẩn.
4. **P4 Volumetric TSDF Mesh (`engine_tsdf_mesh.py`):**
   - Không gian voxel thích ứng hình dáng vật thể (Bounding Box thích ứng vật thể thon cao/dẹt).
   - Chiếu chùm tia ngược Space Carving loại bỏ voxel thừa ngoài silhouette.
   - Marching Cubes nặn lưới đa giác 3D đặc, kín nước (Watertight Solid Mesh).
5. **P5 Multi-View Texture Blender (`texture_blender.py`):**
   - Chiếu chùm tia màu từ toàn bộ các góc chụp thực tế lên lưới 3D.
   - Sử dụng trọng số Fresnel $\cos^3\theta$ kết hợp bộ đệm độ sâu Z-buffer triệt tiêu che khuất và bóng chói, nướng màu sắc rực rỡ vào file `.glb`.
6. **P6 Web App & Cloud (`app.py`, `frontend/index.html`, `demo_colab.ipynb`):**
   - API bất đồng bộ với Job Polling chống timeout 100s của Cloudflare Tunnel.
   - Trình hiển thị Three.js xoay lật 360°, hỗ trợ tải file `.glb`.

---

## 4. Hướng dẫn chạy & Nghiệm thu

### Bước 1: Khởi động máy chủ Local

```bash
cd notebook/backend
python -m uvicorn app:app --reload --port 8000
```
*Đợi thông báo:* `Application startup complete.`

### Bước 2: Thao tác Web UI
1. Truy cập trình duyệt: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) (hoặc Swagger UI: `http://127.0.0.1:8000/docs`).
2. Kéo thả 1 ảnh (Đơn ảnh) hoặc 2–8 ảnh các mặt của vật thể.
3. Nhấn **Tạo mô hình 3D**.
4. Hoặc khởi chạy giao diện Gradio:
   ```bash
   python notebook/local_app.py
   ```

### Bước 3: Nghiệm thu kết quả
- Mô hình 3D được tự động tải về hoặc xem trực tiếp trên Three.js viewer.
- File kết quả được lưu tại thư mục `output/<tên_file>.glb`.
