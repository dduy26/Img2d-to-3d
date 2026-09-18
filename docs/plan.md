# KẾ HOẠCH TỔNG THỂ DỰ ÁN 2D → 3D (MASTER PLAN v2 - DUAL-ENGINE ARCHITECTURE)

> **Phương pháp luận:** Chuẩn hóa theo **Quy trình chuẩn 7 bước** tại [docs/flow.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/flow.md).  
> **Kiến trúc tham chiếu:** Kết hợp tinh hoa từ 3 chuẩn công nghiệp:
> 1. `NVIDIA/3DObjectReconstruction`: TSDF Volumetric Fusion, Marching Cubes Watertight, Configurable Color Fusion ($\cos^3\theta$).
> 2. `xy-gao/DA3-blender`: Edge Discontinuity Gradient Filtering ($\nabla D \le \tau D$), chống hiện tượng rách viền/màng nhện.
> 3. `DepthAnything/Depth-Anything-V2`: Foundation Model ước lượng chiều sâu độ nét cao.
> **Môi trường thực thi mục tiêu:** Google Colab Free (NVIDIA T4 15GB VRAM, RAM 12GB, 2 vCPU, Timeout 90 phút) & Môi trường Local PC.

---

## 📌 PHẦN I: PHÂN TÍCH THỰC TẾ & BẢO TOÀN NGUYÊN LÝ TOÁN HỌC

Hệ thống được thiết kế để vượt qua 4 điểm nghẽn cốt tử:
1. **Quang học vs. Điểm bám (Optical vs. Feature Matching):** Không xóa nền sớm trên ảnh RGB gốc; giữ nguyên bối cảnh cho khâu trích xuất đặc trưng và nướng Albedo Texture; chỉ dùng Alpha Mask làm bộ lọc thể tích trong không gian 3D.
2. **Bảo toàn hình học Epipolar:** Khi đưa ảnh vào khung vuông chuẩn, áp dụng **Global Uniform Scale** $s$ đồng nhất trên toàn bộ $N$ ảnh. Khi có độ dời tâm $(\Delta x_i, \Delta y_i)$, **bắt buộc bù trừ vào ma trận Camera Intrinsics $K_i \to K_i'$** để tia chiếu không bị cắt cụt đối với vật thể bất đối xứng.
3. **Linh hoạt tư thế Camera $[R \mid T]$:** Hỗ trợ cả 2 chế độ: DUSt3R tự giải Pose tự do trên GPU Colab, và Turntable Rig có khóa phẳng đáy (Ground Plane Anchor) trên CPU Local.
4. **Hình học kín nước (Watertight Manifold 100%):** TSDF Space Carving + Marching Cubes với lớp đệm không khí, bảo đảm 0 cạnh hở (Boundary Edges = 0), 1 khối duy nhất, tương thích tuyệt đối mọi phần mềm Slicer in 3D (BambuStudio, Cura, Prusa).

---

## 🧭 BẢNG ĐIỀU HƯỚNG TIẾN ĐỘ 7 BƯỚC (PROJECT ROADMAP)

| Bước | Tên giai đoạn | Trọng tâm công việc | Trạng thái | Sản phẩm bàn giao (Deliverables) |
| :---: | :--- | :--- | :---: | :--- |
| **B1** | **Phân tích yêu cầu (Refine Requirement)** | Khóa chặt phạm vi (In/Out-of-scope), mục tiêu và ràng buộc | ✅ Hoàn thành | [docs/require.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/require.md) |
| **B2** | **Hiểu về dữ liệu (Data Understanding)** | Phân loại dữ liệu ảnh thực tế & Objaverse-1k, chuẩn hóa benchmark | ✅ Hoàn thành | Dữ liệu kiểm thử trong `input/` và scratch |
| **B3** | **Xác định tính năng (Feature Definition)** | Đặc tả tính năng P1 (Tiền xử lý), P2 (Depth/Pose), P3 (Quality Gate), P4 (TSDF Mesh), P5 (Texture), P6 (Web UI) | ✅ Hoàn thành | Bảng đặc tả tính năng F1.x → F6.x |
| **B4** | **Giải pháp Kỹ thuật (Technical Solution)** | Chuẩn hóa toàn bộ Pipeline theo triết lý NVIDIA 3D & Dual-Engine | ✅ Hoàn thành | [docs/nvidia_3d_pipeline_flow.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/nvidia_3d_pipeline_flow.md) |
| **B5** | **Hiện thực hóa (Implementation)** | Lập trình Backend (`notebook/backend/`) và Frontend Web UI (`notebook/frontend/`) | ✅ Hoàn thành | Mã nguồn Python FastAPI + Three.js Viewer |
| **B6** | **Kiểm thử và Đánh giá (Testing & Eval)** | Kiểm định hình học Mesh Health (Watertight, 0 non-manifold edges, Latency) | ✅ Hoàn thành | Slicer in 3D kiểm tra đạt chuẩn xanh 100% |
| **B7** | **Kết luận & Bàn giao (Conclusion)** | Bàn giao tài liệu, Runbook Colab 1-click, đồng bộ Git | ✅ Hoàn thành | Branch `P6-FullStack-Cloud` |

---

## 📌 BẢNG ĐẶC TẢ TÍNH NĂNG CHUẨN NVIDIA DUAL-ENGINE (F1 → F6)

### 1. Phân hệ 1: Tiền Xử Lý Dữ Liệu & Bảo Toàn Quang Học (Data & Preprocessing - F1)
- `F1.1 - Validating Loader`: Kiểm tra định dạng (JPG/PNG), lọc file hỏng, giữ nguyên kênh Native Alpha nếu có (PNG RGBA).
- `F1.2 - CIE Lab Luminance Synchronization`:
  - Chọn ảnh chính diện (#0) làm **Anchor View**.
  - Thực hiện **Histogram Matching duy nhất trên kênh độ sáng L (CIE Lab)** cho $N-1$ ảnh còn lại theo Anchor View, giữ nguyên hai kênh sắc độ $a, b$ để bảo toàn màu sắc nguyên bản của vật thể.
- `F1.3 - Dual-Tier Background Segmentation`:
  - **Tầng 1 (Studio Solid Backdrop):** Khoảng cách sắc độ Lab $\|[a, b] - \text{corner}_{ab}\|$, phân ngưỡng tự động Otsu, lọc hình thái học Closing (vá lỗ phản quang) và Opening (xóa nhiễu viền nền).
  - **Tầng 2 (Natural Complex Background):** AI Rembg / BiRefNet vá kín lỗ hổng phản quang.
- `F1.4 - Global Uniform Scale & Camera Intrinsics Compensation`:
  - Áp dụng DUY NHẤT một tỉ lệ phóng $s = \frac{S_{\text{target}} \cdot \eta}{\max_i(\max(H_i, W_i))}$ cho toàn bộ chuỗi ảnh.
  - Khi có độ dời tâm $(\Delta x_i, \Delta y_i)$, **cập nhật bù trừ ma trận nội thông số camera $K_i \to K_i'$**:
    $$K_i' = \begin{bmatrix} s \cdot f_x & 0 & s \cdot c_x + \Delta x_i \\ 0 & s \cdot f_y & s \cdot c_y + \Delta y_i \\ 0 & 0 & 1 \end{bmatrix}$$
- `F1.5 - Hungarian Viewpoint Assignment`: Nhận diện các mặt tự động và gán tối ưu 1-1 các góc vật lý $[0^\circ, 90^\circ, 180^\circ, 270^\circ, +85^\circ, -85^\circ]$.

### 2. Phân hệ 2: Trích Xuất Chiều Sâu & Hình Học (Depth & Geometry AI - F2)
- `F2.1 - Single-View Generative Mesh Engine`: `TripoSR` (~1.7GB, ViT + Triplane NeRF + Scikit-Image Marching Cubes) sinh mesh kín nước 100% (watertight) trong **1.5 giây**.
- `F2.2 - Multi-View SOTA Engine`: **Tencent Hunyuan3D-2mv** (DiT Flow Matching Pipeline - `tencent/Hunyuan3D-2mv`). Nhận trực tiếp đa ảnh (Front, Right, Back, Left) thông qua bộ Hungarian Viewpoint Assignment, sinh khối 3D đặc kín nước 100% chuẩn CAD/Game, tối ưu hóa bộ nhớ cho Colab T4 GPU (~6-8GB VRAM). Chi tiết xem tại [docs/ke_hoach_nang_cap_model_multiview.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/ke_hoach_nang_cap_model_multiview.md).
- `F2.3 - Dual-Pose Extrinsics Engine`:
  - GPU Mode (Colab): Tự động giải ma trận quay $R_i$, tịnh tiến $T_i$ và tiêu cự $f_i$ tự do.
  - CPU Mode (Local): Turntable Rig kết hợp khóa mặt phẳng đáy (Ground Plane Anchor).

### 3. Phân hệ 3: Cổng Kiểm Định Chất Lượng & Cứu Hộ (Quality Gate & Fail-safe - F3)
- `F3.1 - Cosine Angle Verification`: $\Delta\theta \ge 5^\circ$ giữa hai camera kề nhau để chống suy biến ma trận hình học.
- `F3.2 - Co-visibility Graph Check`: Đồ thị quan sát liên thông 1 thành phần với góc quét $\ge 45^\circ$.
- `F3.3 - Fail-Safe Fallback`: Khi đa ảnh không đạt chuẩn $\to$ Tự động chuyển về Anchor View #0 ở chế độ Đơn Ảnh (`TripoSR`), sinh mô hình Watertight sắc nét trong **1.5s**, bảo đảm hệ thống không bao giờ bị gián đoạn.

### 4. Phân hệ 4: Dựng Khối Thể Tích & Triệt Tiêu Phần Dư (3D Volumetric Mesh - F4)
- `F4.1 - True Multi-View Silhouette Space Carving (Visual Hull)`: Chiếu chùm tia voxel qua ma trận $K_i', R_i, T_i$; gọt sạch voxel nằm ngoài Alpha Mask $\to$ Triệt tiêu $100\%$ vây/phần dư thừa thãi.
- `F4.2 - Ray TSDF Truncation`: Tích lũy độ sâu quan sát $d_i(\mathbf{P})$ khôi phục chi tiết phần lõm bề mặt.
- `F4.3 - Marching Cubes Watertight Extraction`: Trích xuất Iso-surface tại $SDF = 0.0$ với lớp đệm không khí bảo vệ ở 6 mặt ngoài $\implies$ **Kín nước 100% (Watertight Manifold), 0 cạnh hở, 1 khối duy nhất**.
- `F4.4 - Quadric Mesh Decimation`: Tối ưu hóa số lượng đa giác tam giác về khoảng $15,000 \sim 25,000$ mặt, tăng tốc UV unwrapping.

### 5. Phân hệ 5: Trải Phẳng UV & Nướng Màu Chân Thực (Texture & UV Shading - F5)
- `F5.1 - XAtlas UV Parameterization`: Trải phẳng các mảng tam giác vào không gian UV $[0, 1] \times [0, 1]$ không bị chồng lấn.
- `F5.2 - Angle-Weighted Color Blending (NVIDIA Fresnel Principle)`: Trọng số hòa trộn màu lũy thừa $\text{Weight} = \max(0, \vec{n} \cdot \vec{v}_{\text{cam}})^3$, loại bỏ ánh sáng chói bóng và xử lý che khuất (Occlusion) qua Z-buffer.
- `F5.3 - PBR GLB Packaging`: Xuất mô hình chuẩn nhị phân `.glb` nhúng kèm Albedo Texture Map (+Y Up, chân đặt tại $Y_{\min} = 0$).

### 6. Phân hệ 6: Điện Toán Đám Mây & Giao Diện Web 3D (Full-Stack & Cloud - F6)
- `F6.1 - Asynchronous Job Polling Engine`: `POST /generate-3d/job/` phản hồi tức thì ($<100$ms), frontend polling `GET /generate-3d/job/{id}` mỗi 1.5s, miễn nhiễm hoàn toàn lỗi HTTP 524 / 100s timeout của Cloudflare Tunnel trên Google Colab.
- `F6.2 - Three.js Interactive Viewer`: Web UI nhẹ, hiển thị mô hình 3D xoay 360°, chế độ khung dây (Wireframe), tự động xoay và tải `.glb`.
- `F6.3 - Runbook Colab 1-click`: Chạy trơn tru trên Colab T4 GPU, tự động dọn dẹp tiến trình cũ, khởi động máy chủ dưới 5 giây.

---

## 🏗️ CẤU TRÚC THƯ MỤC CHUẨN HOÁ

```text
Img2d-to-3d/
├── input/                   # Thư mục nhận ảnh đầu vào từ người dùng
├── output/                  # Thư mục xuất file mô hình .glb
├── docs/                    # Tài liệu kiến trúc và hướng dẫn
│   ├── require.md           # Đặc tả yêu cầu & phạm vi
│   ├── plan.md              # Kế hoạch tổng thể (Master Plan v2)
│   ├── planforAI.md         # Phân chia chi tiết cho các thành viên & AI
│   ├── status.md            # Bảng theo dõi tiến độ thực tế
│   └── nvidia_3d_pipeline_flow.md # Luồng kỹ thuật chi tiết
└── notebook/
    ├── frontend/            # Giao diện Web 3D (HTML5 + Three.js)
    │   └── index.html
    ├── demo_colab.ipynb     # Notebook Colab 1-click chạy tự động
    └── backend/             # Máy chủ FastAPI và các module lõi P1 → P5
        ├── app.py           # API Controller & Asynchronous Job Polling
        ├── preprocess.py    # P1: Lab Optical Norm + Intrinsics K Compensation + Dual-tier Mask
        ├── engine_depth.py  # P2: Depth-Anything-V2 + DA3 Edge Filter
        ├── quality_gate.py  # P3: Cổng kiểm tra góc chụp 3 lớp & Fail-safe
        ├── engine_tsdf_mesh.py # P4: Space Carving + TSDF + Marching Cubes
        ├── texture_blender.py  # P5: XAtlas UV + Fresnel Angle-Weighted Blending
        └── utils_3d.py      # Đóng gói xuất file .glb
```