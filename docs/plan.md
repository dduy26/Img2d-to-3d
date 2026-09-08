# KẾ HOẠCH TỔNG THỂ DỰ ÁN 2D → 3D (MASTER IMPLEMENTATION PLAN)

> **Căn cứ phương pháp luận:** Chuẩn hóa nghiêm ngặt theo **Quy trình chuẩn 7 bước** tại [docs/flow.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/flow.md).  
> **Căn cứ kỹ thuật:** Dựa trên kết quả đánh giá phản biện chuyên sâu tại [docs/danh_gia_va_luong_hoat_dong_2d_to_3d.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/danh_gia_va_luong_hoat_dong_2d_to_3d.md), phân tích 3 repo tham khảo tại [docs/phan_tich_chuyen_sau_reference_repos.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/phan_tich_chuyen_sau_reference_repos.md) và đặc tả yêu cầu tại [docs/require.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/require.md).  
> **Môi trường thực thi mục tiêu:** Google Colab Free (NVIDIA T4 15GB VRAM, RAM 12GB, 2 vCPU, Timeout 90 phút).

---

## 🧭 BẢNG ĐIỀU HƯỚNG TIẾN ĐỘ 7 BƯỚC (PROJECT ROADMAP)

| Bước | Tên giai đoạn | Trọng tâm công việc | Trạng thái | Sản phẩm bàn giao (Deliverables) |
| :---: | :--- | :--- | :---: | :--- |
| **B1** | **Phân tích yêu cầu (Refine Requirement)** | Khóa chặt phạm vi (In/Out-of-scope), mục tiêu và ràng buộc | ✅ Hoàn thành | [docs/require.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/require.md) |
| **B2** | **Hiểu về dữ liệu (Data Understanding)** | Phân loại dữ liệu ảnh (đơn/đa ảnh), chuẩn hóa bộ benchmark test | 🟡 Đang thực hiện | Cấu trúc thư mục dữ liệu & Dataset mẫu trong `data/` |
| **B3** | **Xác định tính năng (Feature Definition)** | Đặc tả tính năng: Tiền xử lý, Tái tạo 3D (2 Option) & Web UI | ✅ Hoàn thành | Bảng đặc tả tính năng F1.x → F3.x |
| **B4** | **Giải pháp Kỹ thuật (Technical Solution)** | Kiến trúc 2 Option: Phần Logic (Backend/Geometry) & Phần AI | ✅ Hoàn thành | Sơ đồ kiến trúc & Bóc tách giải thuật chi tiết |
| **B5** | **Hiện thực hóa (Implementation)** | Lập trình Backend (`notebook/backend/`) và Frontend (`notebook/frontend/`) | ⏳ Tiếp theo | Mã nguồn Python FastAPI + Giao diện React Three.js |
| **B6** | **Kiểm thử và Đánh giá (Testing & Eval)** | Đánh giá 2 tầng: Tầng mô hình (Model) & Tầng toàn luồng (Full Flow) | ⏳ Tiếp theo | Báo cáo benchmark (FPS, VRAM, Watertightness, Latency) |
| **B7** | **Kết luận & Bàn giao (Conclusion)** | Đúc kết kinh nghiệm, so sánh 2 Option, viết Runbook Colab 1-click | ⏳ Tiếp theo | Báo cáo tổng kết & Notebook demo |

---

## 📌 CHI TIẾT KẾ HOẠCH TRIỂN KHAI TỪNG BƯỚC

---

### BƯỚC 1: PHÂN TÍCH YÊU CẦU (REFINE REQUIREMENT)
- **Tình trạng:** Đã hoàn thiện và chuẩn hóa trong [docs/require.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/require.md).
- **Các nguyên tắc cốt lõi đã chốt:**
  1. *Khóa phạm vi (Scope):* Tập trung vào vật thể tĩnh đơn lập (Rigid static object); không train mô hình từ đầu; không làm cảnh lớn (scene) hoặc vật thể động/rigging hoạt hình.
  2. *Đầu vào:* 1 ảnh đơn (Single-view) hoặc $N = 2 \sim 8$ ảnh (Multi-view).
  3. *Đầu ra:* 1 file `.glb` nhúng Mesh + UV Map + Base-Color Texture (không nhầm lẫn với PBR).
  4. *Ràng buộc tài nguyên:* VRAM $\le 10$ GB, RAM $\le 8$ GB, thời gian chạy $\le 25$ giây trên Colab T4.

---

### BƯỚC 2: HIỂU VỀ DỮ LIỆU (DATA UNDERSTANDING)
- **Mục tiêu:** Nắm vững đặc tính quang học, cấu trúc hình học của dữ liệu đầu vào và thiết lập bộ dữ liệu kiểm chuẩn (Benchmark Dataset).
- **Phân tích đặc tính dữ liệu ảnh:**
  - *Độ phân giải & Tỉ lệ (Resolution & Aspect Ratio):* Ảnh chụp từ điện thoại thường có tỉ lệ $4:3, 16:9$, dọc hoặc ngang. Cần chuẩn hóa qua loader về kích thước chuẩn (cạnh lớn nhất 512px) bảo toàn aspect ratio.
  - *Độ chồng lấn (Visual Overlap trong Đa ảnh):* Góc xoay giữa 2 ảnh kề nhau lý tưởng là $\approx 45^\circ \sim 60^\circ$ (tương ứng $6 \sim 8$ ảnh quanh vật thể) để đảm bảo độ chồng lấn $\ge 40\%$. Nếu chụp 4 ảnh ($90^\circ$), overlap chỉ đạt $\sim 15\%$, nguy cơ trôi pose cao.
  - *Đặc điểm nền (Background):* Nền chứa texture phong phú hỗ trợ DUSt3R tìm điểm đặc trưng neo camera. Nền đơn sắc đòi hỏi bản thân vật thể phải có nhiều vân chi tiết.
- **Kế hoạch chuẩn bị dữ liệu kiểm thử (Benchmark Testcases):**
  ```
  data/
  ├── input/
  │   ├── single_view/                 # Testcase cho luồng Đơn ảnh
  │   │   ├── 01_rigid_cube_chair.jpg  # Khối đặc, góc cạnh rõ ràng
  │   │   ├── 02_organic_statue.jpg    # Bề mặt hữu cơ gồ ghề, chi tiết phức tạp
  │   │   └── 03_smooth_ceramic_mug.jpg# Bề mặt trơn nhẵn, ít texture
  │   └── multi_view/                  # Testcase cho luồng Đa ảnh
  │       ├── dataset_shoe_6views/     # Giày thể thao (6 góc quanh thân, nghiêng 30°)
  │       ├── dataset_toy_figure_6views/# Tượng đồ chơi (6 góc có nền tự nhiên)
  │       └── dataset_4views_orthogonal/# 4 góc trực giao chuẩn (Front, Right, Back, Left)
  └── output/
      └── models_glb/                  # Lưu trữ file .glb xuất ra sau khi test
  ```

---

### BƯỚC 3: XÁC ĐỊNH TÍNH NĂNG (FEATURE DEFINITION)

Hệ thống được phân rã thành 3 nhóm tính năng chính:

#### 1. Nhóm Tính năng Tiền xử lý (Preprocessing Features - F1)
- `F1.1 - Validating Loader`: Kiểm tra số lượng ảnh ($N = 1$ cho đơn ảnh, hoặc $N = 4 \sim 8$ cho đa ảnh), định dạng hợp lệ (JPG/PNG), lọc ảnh mờ nhòe.
- `F1.2A - Single-view Preprocessor (Cho N = 1)`:
  - Tách nền RMBG-2.0 lấy Alpha Mask.
  - **Canh tâm & Scale Normalization:** Crop Bounding Box $\to$ Canh giữa tâm $\to$ Scale bảo toàn Aspect Ratio để vật thể chiếm $80\% \sim 85\%$ khung hình $512 \times 512$ $\to$ Square Letterbox Padding.
  - Cập nhật ma trận Camera Intrinsics $K \to K'$ theo hệ số scale $s$ và offset dời tâm $(\Delta x, \Delta y)$ để bảo toàn tính hội tụ quang học 3D.
- `F1.2B - Multi-view Geometric Resizer (Cho N = 4 ~ 8 ảnh, tối ưu 6 ảnh)`:
  - Giới hạn tải lên tối ưu từ $4$ đến $8$ ảnh (nếu người dùng tải thừa $>8$ ảnh, hệ thống tự động lọc giữ 6~8 ảnh có góc phân bổ đều nhất).
  - Sử dụng DUSt3R standard image loader (Resize giữ aspect ratio về max 512px).
  - **TUYỆT ĐỐI KHÔNG crop riêng lẻ từng ảnh** nhằm bảo toàn quan hệ hình học quang học và vị trí tâm camera giữa các góc nhìn.
  - Chạy RMBG-2.0 song song trích xuất Alpha Mask $M_i$ (lưu tạm để loại bỏ điểm nền ở khâu Fusion).
  - Cân bằng màu sắc & ánh sáng giữa các ảnh (Multi-view Histogram Matching).

#### 2. Nhóm Tính năng Tái tạo 3D Đa Chế Độ (3D Reconstruction Engines - F2)

Hệ thống cung cấp **2 LỰA CHỌN (OPTIONS)** linh hoạt:

* **OPTION 1: HƯỚNG TÁI TẠO HÌNH HỌC CHUYÊN SÂU (GEOMETRIC PIPELINE)**  
  *(Trọng tâm học thuật, thể hiện rõ bản chất xử lý ảnh số & thị giác máy tính)*
  - `F2.1A - Single-view Depth Engine`: Ảnh 2D $\to$ Depth-Anything-V2-Metric $\to$ Back-projection $\to$ Poisson Surface Reconstruction $\to$ Camera Texture Projection.
  - `F2.1B - Multi-view DUSt3R Engine (Pipeline v1)`: $N$ ảnh $\to$ DUSt3R Pairwise Matching & Global Alignment $\to$ Xuất đồng thời Camera Poses $[R \mid T]$, tiêu cự $K$, 3D Point-maps và Confidence map trong cùng hệ tọa độ. *(Không trộn Depth-Anything vào để tránh lệch scale).*
  - `F2.1C - Quality Gate Validation`: Kiểm soát chất lượng sau Global Alignment (kiểm tra đồ thị Co-visibility liên thông, mật độ pixel confidence cao, và alignment loss).
  - `F2.1D - Background Point Pruning`: Áp mặt nạ Alpha Mask RMBG-2.0 để loại bỏ 100% điểm thuộc về hậu cảnh trên Point-map.
  - `F2.1E - TSDF Volumetric Fusion & Marching Cubes`: Tích lũy các điểm 3D vào lưới thể tích Voxel TSDF qua Open3D $\to$ Marching Cubes trích xuất bề mặt lưới tam giác kín nước 360°.
  - `F2.1F - Base-Color Texture Blending`: XAtlas mở phẳng UV $\to$ Angle-weighted Color Blending nướng màu khuếch tán từ $N$ ảnh gốc (Base-Color Texture, không gọi PBR).

* **OPTION 2: HƯỚNG MÔ HÌNH SINH TRỰC TIẾP (FAST FEED-FORWARD LRM PIPELINE)**  
  *(Trọng tâm tốc độ, demo tức thì & đóng vai trò túi khí cứu hộ Fail-safe)*
  - `F2.2A - TripoSR Single-view Engine`: Đưa 1 ảnh (đã tách nền) qua Transformer TripoSR $\to$ Sinh thẳng Textured Mesh hoàn chỉnh trong 1-2 giây.
  - `F2.2B - LGM 4-View Engine`: Nhận 4 góc trực giao chuẩn $\to$ Sinh trực tiếp 3D Gaussians / Mesh trong ~5 giây.
  - `F2.2C - Fail-safe Auto Fallback`: Khi Option 1B (DUSt3R đa ảnh) bị Quality Gate đánh trượt (do góc chụp không đủ overlap) $\to$ Tự động chọn ảnh nét nhất chuyển sang TripoSR để xuất ngay file `.glb`, kèm thông báo cảnh báo trên giao diện.

#### 3. Nhóm Tính năng Giao diện & Trực quan hóa (Web UI & Viewer - F3)
- `F3.1 - Upload Zone & Mode Selector`: Kéo thả ảnh; cho phép chọn chế độ (Đơn ảnh / Đa ảnh) và chọn giải thuật (Option 1 Hình học / Option 2 Tốc độ).
- `F3.2 - 2D Inspection Panel`: Xem ảnh gốc và mặt nạ tách nền.
- `F3.3 - 3D Interactive Canvas`: Khung nhìn WebGL (Three.js / `<model-viewer>`) xoay 360°, zoom, pan, bật/tắt lưới tam giác (Wireframe mode).
- `F3.4 - GLB Export & Download`: Tải file `.glb` về máy tính.

---

### BƯỚC 4: GIẢI PHÁP KỸ THUẬT (TECHNICAL SOLUTION ARCHITECTURE)

Hệ thống được chia tách mạch lạc thành 2 phần: **Phần Logic** và **Phần AI**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PHẦN LOGIC (BACKEND & I/O)                           │
│ • Web Server: FastAPI xử lý REST API bất đồng bộ (async).                              │
│ • Geometry & Math: Open3D Scalable TSDF, Marching Cubes CPU, Back-projection.          │
│ • UV & Mesh Processing: xatlas-python, PyMCubes, Trimesh export GLB.                   │
│ • Validation: Quality Gate (Phân tích đồ thị liên thông NetworkX, tính reprojection). │
│ • Cache Manager: Quản lý cache weights model trên Colab (tránh tải lại sau 90p).       │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PHẦN AI (NEURAL MODELS)                              │
│ • Phân đoạn ảnh: briaai/RMBG-2.0 (Torch, <0.5GB VRAM).                                 │
│ • Hình học đa ảnh: DUSt3R (ViT-Large backbone, tự sinh Pose + Focal + Pts3D, ~5GB VRAM)│
│ • Độ sâu đơn ảnh: Depth-Anything-V2-Small (DINOv2 backbone, ~0.3GB VRAM).              │
│ • Sinh khối nhanh: TripoSR (Transformer LRM, ~6GB VRAM, inference 1.5s).               │
│ • Multi-view nhanh: LGM (4-view Gaussian Model, ~7.5GB VRAM).                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### BƯỚC 5: HIỆN THỰC HÓA (IMPLEMENTATION ROADMAP)

Kế hoạch xây dựng mã nguồn theo cấu trúc thư mục mô-đun hóa:

```
ImgToModel/
├── notebook/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py                  # Entrypoint FastAPI Server
│   │   ├── preprocess.py            # Module loader, resize và tách nền RMBG-2.0
│   │   ├── quality_gate.py          # Module 3 lớp kiểm tra lỗi DUSt3R
│   │   ├── pipeline_geometric.py    # Option 1: DUSt3R + TSDF + Marching Cubes + XAtlas
│   │   ├── pipeline_feedforward.py  # Option 2: TripoSR & LGM Generator
│   │   └── utils_3d.py              # Export Trimesh GLB, tính normals, làm mịn Laplace
│   └── frontend/
│       ├── package.json
│       ├── index.html
│       ├── src/
│       │   ├── main.jsx             # Entrypoint React
│       │   ├── App.jsx              # Giao diện chính điều khiển pipeline
│       │   ├── components/
│       │   │   ├── UploadZone.jsx   # Kéo thả ảnh đơn/đa ảnh
│       │   │   ├── ModeToggle.jsx   # Nút chuyển Option 1 / Option 2
│       │   │   ├── Viewer3D.jsx     # Canvas Three.js xoay 360 độ
│       │   │   └── ProgressBar.jsx  # Tiến trình thực thi từng chặng
│       │   └── index.css            # Styling Vanilla CSS hiện đại (Dark Glassmorphism)
```

#### Lộ trình thực hiện chi tiết (Phân Chia Công Việc Cho Nhóm 6 Người - Kịch Bản 2 Đa Ảnh Làm Trước):

Để 6 thành viên có thể làm việc song song (Parallel Development) mà không bị nghẽn (bottleneck), nhóm áp dụng nguyên tắc **Hợp đồng giao diện (Interface Contract)**: mỗi người phụ trách 1 module độc lập có quy định rõ ràng về Input và Output.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        SƠ ĐỒ PHÂN CHIA 6 THÀNH VIÊN (KỊCH BẢN 2)                       │
│                                                                                        │
│ [Thành viên 1] ──► preprocess.py (Loader, RMBG-2.0 Alpha Mask, Histogram Matching)     │
│       │                                                                                │
│       ▼                                                                                │
│ [Thành viên 2] ──► engine_dust3r.py (Pairwise Matching, Global Alignment, Point-maps)   │
│       │                                                                                │
│       ▼                                                                                │
│ [Thành viên 3] ──► quality_gate.py + engine_triposr.py (Kiểm tra 3 lớp & Cứu hộ)      │
│       │                                                                                │
│  (Pass ✅)                                                                             │
│       ▼                                                                                │
│ [Thành viên 4] ──► engine_tsdf_mesh.py [Phần Hình Học] (Voxel TSDF & Marching Cubes)   │
│       │                                                                                │
│       ▼                                                                                │
│ [Thành viên 5] ──► utils_3d.py + texture_blender.py (XAtlas UV, Color Blending, GLB)   │
│       │                                                                                │
│       ▼                                                                                │
│ [Thành viên 6] ──► main.py (FastAPI), React Three.js Frontend & Colab Runbook Tunnel   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

| STT | Vị trí đảm nhiệm | File mã nguồn chịu trách nhiệm | Input nhận vào | Output bàn giao | Tiêu chí nghiệm thu (DoD) |
|:---:|:---|:---|:---|:---|:---|
| **P1** | **Data & Preprocessing Engineer** | `notebook/backend/preprocess.py` & `data/input/multi_view/` | $N=4\sim 8$ ảnh chụp từ người dùng | `images_tensor`, `alpha_masks`, `focals_init` | Tách sạch nền không lẹm vật thể, giữ nguyên epipolar geometry, xử lý $\le 1.5$s. |
| **P2** | **Pose & 3D Geometry Engineer** | `notebook/backend/engine_dust3r.py` | `images_tensor` từ P1 | Ma trận $R, T$, tiêu cự $K$, `pts3d`, `confidence` | Global Alignment hội tụ, Poses và Point-maps cùng 1 hệ tọa độ, VRAM $\le 5$GB. |
| **P3** | **Quality Gate & Fail-safe Engineer** | `notebook/backend/quality_gate.py` & `engine_triposr.py` | Pose, confidence, loss từ P2 | Cờ `is_valid` (True/False); nếu False $\to$ Model `.glb` cứu hộ từ TripoSR | Bắt đúng 100% ảnh thiếu overlap/lệch góc; fallback trả ra model 3D hợp lệ $\le 2$s. |
| **P4** | **3D Volumetric Mesh Engineer** | `notebook/backend/engine_tsdf_mesh.py` (Phần hình học) | `pts3d`, `poses`, `alpha_masks` | Lưới tam giác thô `raw_mesh (V, F, N)` | Marching Cubes sinh lưới khép kín 360°, dọn sạch cụm rác, chạy CPU $\le 2$s. |
| **P5** | **Texture & UV Shading Engineer** | `notebook/backend/utils_3d.py` & `texture_blender.py` | `raw_mesh` từ P4, ảnh gốc, `poses` | File `.glb` hoàn chỉnh có Albedo Texture | Trải UV XAtlas không chồng lấn (zero overlap), hòa trộn màu mượt không vệt cắt. |
| **P6** | **Full-Stack & Cloud Deployment Lead** | `notebook/backend/main.py`, `frontend/`, `demo_colab.ipynb` | Toàn bộ module của P1 $\to$ P5 | API REST, Web UI Three.js, Notebook Colab 1-click | Chạy mượt trên Colab T4, tunnel Cloudflare mở web từ máy tính, tải được file `.glb`. |

---

### BƯỚC 6: KIỂM THỬ VÀ ĐÁNH GIÁ (TESTING & EVALUATION PLAN)

Thực hiện đánh giá nghiêm ngặt ở 2 tầng theo đúng phương pháp luận của `flow.md`:

#### 1. Tầng 1: Đánh giá ở mức mô hình (Model-Level Evaluation)
- **Tính trọn vẹn hình học (Geometric Integrity):**
  - Kiểm tra tính kín nước: Hàm `mesh.is_watertight` của Trimesh (Bắt buộc `True` với Option 1B đa ảnh).
  - Không có mặt lộn ngược: Kiểm tra pháp tuyến bề mặt (Normal consistency check).
  - Triệt tiêu mạng nhện/điểm rác: Kiểm tra số lượng thành phần cô lập (Connected components $\le 2$).
- **Độ phân giải lưới đa giác:** Đảm bảo số mặt tam giác nằm trong khoảng $50.000 \le \text{Faces} \le 150.000$ (vừa đủ nét mà không làm đơ trình duyệt).
- **Độ trung thực màu sắc (Texture Fidelity):** So sánh trực quan màu sắc khuếch tán nướng trên UV với ảnh chụp gốc.

#### 2. Tầng 2: Đánh giá ở mức toàn bộ luồng (Full-Flow Evaluation)
- **Thời gian đáp ứng (End-to-End Latency):**
  - Luồng Đơn ảnh (Option 1A / Option 2A): $\le 3$ giây.
  - Luồng Đa ảnh 6 ảnh (Option 1B): $\le 10$ giây.
- **Mức tiêu thụ tài nguyên GPU/RAM (Resource Benchmark trên Colab T4):**
  - Mức đỉnh VRAM $\le 6.0$ GB (giới hạn an toàn của T4 là 15GB).
  - Mức tiêu thụ RAM hệ thống $\le 7.0$ GB (giới hạn an toàn của Colab là 12GB).
- **Tỷ lệ thành công (Pipeline Robustness):**
  - Chạy thử nghiệm trên 10 bộ dữ liệu benchmark khác nhau.
  - Tỷ lệ ra được file `.glb` hợp lệ: Mục tiêu **100%** (nhờ cơ chế Quality Gate + TripoSR Fallback).

---

### BƯỚC 7: KẾT LUẬN & BÀN GIAO (CONCLUSION & FINAL DELIVERABLES)

- **So sánh & Tổng kết:**
  - Đối chiếu ưu/nhược điểm thực nghiệm giữa Option 1 (Hình học TSDF) và Option 2 (Feed-forward LRM).
  - Rút ra bài học chuyên môn về Epipolar Geometry, Point Cloud Pruning và Color Blending.
- **Sản phẩm đóng gói nghiệm thu:**
  1. Toàn bộ mã nguồn sạch, có docstring giải thích chi tiết.
  2. File Jupyter Notebook chạy thử nghiệm 1-click (`demo_colab.ipynb`) tích hợp ngrok mở Web UI.
  3. Báo cáo nghiệm thu đầy đủ số liệu đo đạc (latency, VRAM, hình ảnh so sánh trước/sau khi chuyển đổi).