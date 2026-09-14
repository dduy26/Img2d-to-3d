# Hệ Thống Tái Tạo Mô Hình 3D Từ Ảnh 2D (2D to 3D Generation Pipeline)

Hệ thống tái tạo mô hình 3D hoàn chỉnh từ ảnh 2D (đơn ảnh hoặc 2–8 ảnh đa góc 360°) sử dụng kiến trúc kết hợp Deep Learning hiện đại (Depth-Anything-V2, CLIP ViT / HOG Bilateral Symmetry, RMBG-2.0 / Rembg) và Hình học vi sai (Volumetric TSDF Space Carving & Marching Cubes, XAtlas UV Blending).

---

## 1. Cấu trúc thư mục

Kiến trúc thư mục được tổ chức tinh gọn và chuẩn hoá:

```text
Img2d-to-3d/
├── input/                # Thư mục duy nhất nhận ảnh đầu vào (user upload hoặc đặt ảnh)
├── output/               # Thư mục duy nhất xuất file mô hình 3D .glb
└── notebook/
    ├── frontend/         # Giao diện Web 3D Viewer (Three.js + Tailwind CSS)
    │   └── index.html
    └── backend/
        ├── app.py               # Máy chủ FastAPI (API Điều phối & Job Polling)
        ├── preprocess.py        # P1: Tách nền, vá lỗ PET/highlights, nhận diện mặt (CLIP/HOG)
        ├── engine_depth.py      # P2: Depth-Anything-V2 dự đoán độ sâu đa góc & đơn ảnh
        ├── quality_gate.py      # P3: Kiểm định chất lượng góc chụp & bao phủ camera
        ├── engine_tsdf_mesh.py  # P4: Volumetric TSDF Space Carving & Marching Cubes 360°
        ├── texture_blender.py   # P5: Trải phẳng UV & nướng màu bề mặt chân thực
        └── utils_3d.py          # Xuất định dạng GLB chuẩn PBR
```

---

## 2. Yêu cầu hệ thống & Cài đặt

- **Ngôn ngữ:** Python 3.10 – 3.13 (Thuần Python / PyTorch, **KHÔNG** yêu cầu trình biên dịch C++ hay `torchmcubes`).
- **Phần cứng:** Chạy tốt trên cả GPU NVIDIA (CUDA) và CPU fallback.

**Cài đặt các thư viện cần thiết:**

```bash
pip install fastapi uvicorn python-multipart trimesh rembg onnxruntime
pip install transformers xatlas roma einops safetensors matplotlib tqdm fast-simplification scipy
```

*(Các trọng số Deep Learning siêu nhẹ: Depth-Anything-V2 ~95MB, RMBG-2.0 / Rembg ~160MB, CLIP ~150MB được tự động nạp từ Hugging Face trong lần chạy đầu tiên).*

---

## 3. Luồng hoạt động (Workflow P1 → P6)

Hệ thống hoạt động theo quy trình 6 phân hệ khép kín:

1. **P1 Preprocessing (`preprocess.py`):**
   - Tách nền vật thể bằng Rembg / RMBG-2.0.
   - Thuật toán `refine_alpha_mask` tự động lấp kín lỗ thủng phản xạ trên chai nhựa trong suốt / kim loại bóng.
   - Phân loại mặt ảnh (`classify_viewpoints`) bằng CLIP ViT Zero-shot kết hợp HOG Bilateral Gradient Symmetry, giải thuật gán cặp Hungarian (`linear_sum_assignment`) gán đúng góc thực $0^\circ, 90^\circ, 180^\circ, 270^\circ$ mà không bị ghép loạn góc.
2. **P2 Depth Estimation (`engine_depth.py`):**
   - Dự đoán bản đồ độ sâu sắc nét từng góc bằng `Depth-Anything-V2`.
   - Lọc nhiễu nền bằng mask P1.
3. **P3 Quality Gate (`quality_gate.py`):**
   - Kiểm tra độ bao phủ góc chụp, tránh hiện tượng chụp trùng 1 góc hoặc thiếu góc đối xứng.
4. **P4 Volumetric TSDF Mesh (`engine_tsdf_mesh.py`):**
   - Không gian voxel thích ứng hình dáng vật thể (Bounding Box thích ứng vật thể thon cao/dẹt).
   - Chiếu chùm tia ngược Space Carving loại bỏ voxel thừa ngoài silhouette.
   - Marching Cubes nặn lưới đa giác 3D đặc, kín nước (Watertight Solid Mesh).
5. **P5 Texture Baking (`texture_blender.py`):**
   - Trải phẳng UV bằng XAtlas và pha trộn màu từ ảnh chụp thực tế vào bề mặt 3D.
6. **P6 Web App & Cloud (`app.py` & `frontend/index.html`):**
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

### Bước 3: Nghiệm thu kết quả
- Mô hình 3D được tự động tải về hoặc xem trực tiếp trên Three.js viewer.
- File kết quả được lưu tại thư mục `output/<tên_file>.glb`.
