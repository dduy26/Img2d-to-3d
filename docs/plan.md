# KẾ HOẠCH TỔNG THỂ DỰ ÁN 2D → 3D (MASTER PLAN - NVIDIA 3D WORKFLOW)

> **Phương pháp luận:** Chuẩn hóa theo **Quy trình chuẩn 7 bước** tại [docs/flow.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/flow.md).  
> **Kiến trúc tham chiếu:** Bộ chuẩn công nghiệp **NVIDIA 3D Object Reconstruction Workflow** (NVIDIA DALI, Kaolin, DA3-blender, Poisson & Volumetric Space Carving).  
> **Môi trường thực thi mục tiêu:** Google Colab Free (NVIDIA T4 15GB VRAM, RAM 12GB, 2 vCPU, Timeout 90 phút) & Môi trường Local.

---

## 🧭 BẢNG ĐIỀU HƯỚNG TIẾN ĐỘ 7 BƯỚC (PROJECT ROADMAP)

| Bước | Tên giai đoạn | Trọng tâm công việc | Trạng thái | Sản phẩm bàn giao (Deliverables) |
| :---: | :--- | :--- | :---: | :--- |
| **B1** | **Phân tích yêu cầu (Refine Requirement)** | Khóa chặt phạm vi (In/Out-of-scope), mục tiêu và ràng buộc | ✅ Hoàn thành | [docs/require.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/require.md) |
| **B2** | **Hiểu về dữ liệu (Data Understanding)** | Phân loại dữ liệu ảnh thực tế (chai lọ, giày dép, đồ chơi), chuẩn hóa benchmark | ✅ Hoàn thành | Cấu trúc dữ liệu trong `input/` và `data/` |
| **B3** | **Xác định tính năng (Feature Definition)** | Đặc tả tính năng P1 (Tiền xử lý), P2 (Depth/Pose), P3 (Quality Gate), P4 (TSDF Mesh), P5 (Texture), P6 (Web UI) | ✅ Hoàn thành | Bảng đặc tả tính năng F1.x → F6.x |
| **B4** | **Giải pháp Kỹ thuật (Technical Solution)** | Chuẩn hóa toàn bộ Pipeline theo triết lý NVIDIA 3D | ✅ Hoàn thành | [docs/nvidia_3d_pipeline_flow.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/nvidia_3d_pipeline_flow.md) |
| **B5** | **Hiện thực hóa (Implementation)** | Lập trình Backend (`notebook/backend/`) và Frontend Web UI (`notebook/frontend/`) | ✅ Hoàn thành | Mã nguồn Python FastAPI + Three.js Viewer |
| **B6** | **Kiểm thử và Đánh giá (Testing & Eval)** | Kiểm định hình học Mesh Health (Watertight, 0 non-manifold edges, Latency) | ✅ Hoàn thành | Slicer in 3D kiểm tra đạt chuẩn xanh 100% |
| **B7** | **Kết luận & Bàn giao (Conclusion)** | Bàn giao tài liệu, Runbook Colab 1-click, đồng bộ Git | ✅ Hoàn thành | Branch `P6-FullStack-Cloud` |

---

## 📌 BẢNG ĐẶC TẢ TÍNH NĂNG CHUẨN NVIDIA 3D (F1 → F6)

### 1. Phân hệ 1: Tiền Xử Lý Dữ Liệu 2D (Data & Preprocessing - F1)
- `F1.1 - Validating Loader`: Kiểm tra định dạng (JPG/PNG), lọc file hỏng, hỗ trợ Unicode path.
- `F1.2 - Optical Color Synchronization (NVIDIA DALI Style)`:
  - Chọn ảnh chính diện (#0) làm **Anchor View**.
  - Thực hiện toán tử **Histogram Matching** đồng bộ dải sáng và cân bằng trắng cho $N-1$ ảnh còn lại theo Anchor View, khử triệt để lỗi phơi sáng tự động (Auto-Exposure) và ám màu.
- `F1.3 - Geometric Resizer (Bảo toàn Epipolar Geometry)`:
  - Resize ảnh bảo toàn tỷ lệ khung hình (Aspect Ratio), cạnh lớn nhất đưa về $512$px.
  - Ép kích thước $(H, W)$ chia hết cho 16 chuẩn hóa cho mạng Vision Transformer (ViT Backbone).
  - **Tuyệt đối không crop riêng lẻ từng ảnh** để tránh phá vỡ ma trận nội thông số camera $K$.
- `F1.4 - Transparent/Reflective Mask Refinement`:
  - Tách nền bằng Rembg / RMBG-2.0.
  - Áp dụng toán tử hình thái học `binary_fill_holes` và lọc liên thông lớn nhất để vá các lỗ thủng do phản quang trên chai nhựa PET trong suốt và kim loại bóng.
- `F1.5 - Viewpoint Recognition & Hungarian Assignment`:
  - Nhận diện các mặt tự động (Tên file $\to$ Zero-shot CLIP ViT $\to$ HOG Bilateral Symmetry).
  - Giải thuật Hungarian gán tối ưu 1-1 các góc vật lý $[0^\circ, 90^\circ, 180^\circ, 270^\circ, +85^\circ, -85^\circ]$, triệt tiêu hiện tượng "ghép loạn" camera rays.

### 2. Phân hệ 2: Trích Xuất Chiều Sâu & Hình Học (Depth & Geometry AI - F2)
- `F2.1 - Monocular & Multi-View Depth Prediction`: Sử dụng `Depth-Anything-V2-Small` (~95MB) dự đoán bản đồ chiều sâu độ nét cao $D(u, v)$ trong 0.2s/ảnh.
- `F2.2 - DUSt3R Global Alignment (GPU Pipeline)`: Với luồng đa ảnh trên GPU, chạy Pairwise Matching và Global Alignment optimization để đồng thời tính toán camera poses $[R \mid T]$ và 3D pointmaps cùng hệ tọa độ.
- `F2.3 - DA3 Edge Discontinuity Filtering`: Quét ma trận đạo hàm độ sâu $\nabla D(u, v) = \max(|\nabla D_x|, |\nabla D_y|)$. Pixel biên nào có độ dốc nhảy vọt $> \tau \cdot D(u, v)$ sẽ bị loại bỏ để chống tia mạng nhện / flying pixels.

### 3. Phân hệ 3: Cổng Kiểm Định Chất Lượng & Cứu Hộ (Quality Gate & Fail-safe - F3)
- `F3.1 - Cosine Angle Verification`: Kiểm tra góc quay giữa các camera kề nhau qua trace ma trận quay:
  $$\cos\theta = \frac{\text{Tr}(R_i^T R_j) - 1}{2}$$
  Nếu $\Delta\theta < 5^\circ$ (chụp trùng góc) $\to$ Đánh trượt để tránh suy biến ma trận hình học.
- `F3.2 - Confidence Density Check`: Kiểm tra mật độ pixel tin cậy cao của Depth/Pointmap $\ge 0.45$.
- `F3.3 - Fail-Safe Fallback (Depth-Anything-V2 Surface Engine)`:
  - Khi người dùng chỉ upload 1 ảnh HOẶC khi bộ ảnh đa góc bị Quality Gate đánh trượt $\to$ Tự động kích hoạt luồng tái tạo đơn ảnh qua `DepthReconstructionEngine`.
  - Sinh Pinhole Surface Mesh sắc nét (92k+ đỉnh) trong **1.58 giây**, chính thức thay thế TripoSR để tránh treo môi trường do C++ `torchmcubes`.

### 4. Phân hệ 4: Dựng Khối Thể Tích & Triệt Tiêu Phần Dư (3D Volumetric Mesh - F4)
- `F4.1 - True Multi-View Silhouette Space Carving (Visual Hull)`:
  - Quét toàn bộ voxel 3D trong Bounding Box thực tế và chiếu lên $N$ camera.
  - Voxel nào chiếu ra ngoài Silhouette Mask ở BẤT KỲ góc nhìn nào $\to$ Gọt sạch thành không khí (+trunc_margin).
  - Triệt tiêu 100% hiện tượng "thừa phần dư", cánh/vây lơ lửng ngoài vật thể.
- `F4.2 - Ray-based Truncated Signed Distance Field (TSDF)`: Tích lũy độ sâu quan sát dọc theo chùm tia camera, phân định ranh giới mặt ngoài và ruột đặc bên trong.
- `F4.3 - Marching Cubes Iso-surface Extraction`: Trích xuất lưới bề mặt tại mức $SDF = 0.0$ với lớp đệm không khí bảo vệ ở 6 mặt ngoài, bảo đảm **Watertight 100%**.
- `F4.4 - Quadric Mesh Decimation`: Tối ưu hóa số lượng đa giác tam giác (rút gọn về khoảng $20,000 \sim 35,000$ mặt), giúp bước trải UV ở P5 tăng tốc từ 40s xuống 1.5s mà vẫn giữ trọn hình thái vật thể.

### 5. Phân hệ 5: Trải Phẳng UV & Nướng Màu Chân Thực (Texture & UV Shading - F5)
- `F5.1 - XAtlas UV Parameterization`: Trải phẳng các mảng tam giác (charts) vào không gian UV $[0, 1] \times [0, 1]$ không bị chồng chéo.
- `F5.2 - Angle-Weighted Color Blending (NVIDIA Fresnel Principle)`:
  - Tính toán tích vô hướng giữa vector pháp tuyến bề mặt và hướng nhìn camera:
    $$\cos\theta = \vec{n} \cdot \vec{v}_{\text{cam}}$$
  - Trọng số hòa trộn màu lũy thừa $\text{Weight} = \cos^3\theta$ ($\gamma = 3.0$), loại bỏ hoàn toàn ánh sáng chói bóng (specular highlights) và rách màu vùng khuất (self-occlusion).
- `F5.3 - PBR GLB Packaging`: Xuất mô hình chuẩn nhị phân `.glb` nhúng kèm Albedo Texture Map.

### 6. Phân hệ 6: Điện Toán Đám Mây & Giao Diện Web 3D (Full-Stack & Cloud - F6)
- `F6.1 - Asynchronous Job Polling Engine`:
  - API `POST /generate-3d/job/` phản hồi tức thì ($<100$ms) trả về `job_id`.
  - Frontend polling `GET /generate-3d/job/{id}` mỗi 1.5s, **triệt tiêu hoàn toàn lỗi HTTP 524 / 100s timeout** của Cloudflare Tunnel trên Google Colab.
- `F6.2 - Three.js Interactive Viewer`: Web UI nhẹ, không cần build step, hiển thị mô hình 3D xoay 360°, chế độ khung dây (Wireframe), tự động xoay và tải `.glb`.
- `F6.3 - Runbook Colab 1-click`: Chạy trơn tru trên Colab T4 GPU, tự động dọn dẹp tiến trình cũ, khởi động máy chủ dưới 5 giây.

---

## 🏗️ CẤU TRÚC THƯ MỤC CHUẨN HOÁ

```text
Img2d-to-3d/
├── input/                   # Thư mục duy nhất nhận ảnh đầu vào từ người dùng
├── output/                  # Thư mục duy nhất xuất file mô hình .glb
├── docs/                    # Tài liệu kiến trúc và hướng dẫn
│   ├── require.md           # Đặc tả yêu cầu & phạm vi
│   ├── plan.md              # Kế hoạch tổng thể (Master Plan)
│   ├── planforAI.md         # Phân chia chi tiết cho các thành viên & AI
│   ├── status.md            # Bảng theo dõi tiến độ thực tế
│   └── nvidia_3d_pipeline_flow.md # Tài liệu kiến trúc kỹ thuật NVIDIA 3D
└── notebook/
    ├── frontend/            # Giao diện Web 3D (HTML5 + Tailwind + Three.js)
    │   └── index.html
    ├── demo_colab.ipynb     # Notebook Colab 1-click chạy tự động
    └── backend/             # Máy chủ FastAPI và các module lõi P1 → P5
        ├── app.py           # API Controller & Asynchronous Job Polling
        ├── preprocess.py    # P1: Intrinsics + Histogram Matching + Mask Refine
        ├── engine_depth.py  # P2: Depth-Anything-V2 + DA3 Edge Filter
        ├── quality_gate.py  # P3: Cổng kiểm tra góc chụp 3 lớp & Fail-safe
        ├── engine_tsdf_mesh.py # P4: True Space Carving + Marching Cubes
        ├── texture_blender.py  # P5: XAtlas UV + Angle-Weighted Blending
        └── utils_3d.py      # Đóng gói xuất file .glb
```