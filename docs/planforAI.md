# KẾ HOẠCH TRIỂN KHAI CHI TIẾT THEO CHUẨN NVIDIA 3D WORKFLOW (PLAN FOR AI & TEAM)

> **Mục tiêu:** Bản hướng dẫn chi tiết về mã nguồn, thuật toán, interface contract và kiểm thử cho từng phân hệ (P1 → P6) nhằm đảm bảo sự đồng bộ tuyệt đối giữa các thành viên và các phiên làm việc của AI Agent.
> **Nguyên tắc kỹ thuật:** Bám sát chuẩn công nghiệp NVIDIA 3D (DALI, Kaolin, DA3, True Space Carving & Marching Cubes).

---

## 1️⃣ PHÂN HỆ 1: DATA & PREPROCESSING ENGINEER (P1 - DUY)
* **File chịu trách nhiệm:** `notebook/backend/preprocess.py`
* **Vấn đề cốt lõi:** Lệch phơi sáng (Auto-Exposure), ám màu (White-Balance) giữa các góc chụp; thủng lỗ trên chai nhựa trong suốt/phản quang; sai thứ tự góc ảnh gây "ghép loạn" camera.
* **Giải pháp chuẩn NVIDIA 3D:**
  1. **Histogram Matching (Anchor View):** Ép biểu đồ màu của $N-1$ ảnh theo ảnh chính diện (#0).
  2. **ViT Resize:** Resize bảo toàn Aspect Ratio, cạnh lớn nhất $512$px, $(H, W)$ chia hết cho 16. Tuyệt đối không crop đơn lập.
  3. **Vá lỗ PET / Thuỷ tinh (`refine_alpha_mask`):** Dùng `scipy.ndimage.binary_fill_holes` và lọc Connected Components để bịt kín vùng phản xạ.
  4. **Nhận diện góc nhìn (`classify_viewpoints`):** Kết hợp Tên file $\to$ CLIP ViT $\to$ HOG Bilateral Symmetry và giải thuật Hungarian gán cặp 1-1 góc chuẩn.

### Hợp đồng giao diện (Interface Contract):
```python
def preprocess_multiview(
    image_paths: List[str], 
    target_size: int = 512, 
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Returns:
        {
            'images_rgb': List[np.ndarray],              # Ảnh RGB chuẩn hóa cho P5 nướng màu
            'images_normalized': Tensor (N, 3, H, W),    # Chuẩn hóa ImageNet cho ViT
            'alpha_masks': List[np.ndarray],             # Mask nhị phân {0, 1} đã vá lỗ cho P4 Space Carving
            'viewpoint_assignments': List[Dict],         # Danh sách góc [0, 90, 180, 270, 85, -85]
            'original_sizes': List[Tuple[int, int]],
            'scale_factors': List[float],
            'focal_lengths': List[Tuple[float, float]],
            'num_images': int
        }
    """
```

---

## 2️⃣ PHÂN HỆ 2: DEPTH & 3D GEOMETRY AI ENGINEER (P2 - HUY)
* **File chịu trách nhiệm:** `notebook/backend/engine_hunyuan3d.py` & `notebook/backend/engine_depth.py`
* **Vấn đề cốt lõi:** Sinh hình khối 3D đa ảnh hoàn chỉnh không bị rỗng đáy (thin hollow shell), ước tính chiều sâu chính xác và loại bỏ các điểm rách biên (Edge-Bleeding) gây mạng nhện.
* **Giải pháp chuẩn SOTA & NVIDIA 3D:**
  1. **Tencent Hunyuan3D-2mv DiT Engine (`engine_hunyuan3d.py`):**
     - Sử dụng mô hình tạo sinh Diffusion Transformer (DiT) kết hợp Flow Matching (`tencent/Hunyuan3D-2mv`, `hunyuan3d-dit-v2-mv`).
     - Nhận diện và sắp xếp 4 góc nhìn chuẩn (`front`, `right`, `back`, `left`) từ thuật toán Hungarian của P1.
     - Sinh khối 3D đặc kín nước 100% (Watertight Solid Mesh), chuẩn CAD/Game asset, có đế và lòng vật thể hoàn chỉnh, tối ưu VRAM ~6-8GB trên Colab GPU T4.
     - Tích hợp trực tiếp Multi-View Texture Blender phủ màu chân thực từ ảnh chụp thực tế.
  2. **Monocular Depth & DA3 Edge Discontinuity Filter (`engine_depth.py`):**
     - `Depth-Anything-V2-Small` dự đoán $D(u, v)$ trong 0.2s/ảnh phục vụ preview hoặc TSDF mesh cục bộ.
     - Lọc viền bất liên tục: $\nabla D(u, v) = \max\big(|\nabla D_x|, |\nabla D_y|\big)$, triệt tiêu các pixel viền có $\nabla D > \tau \cdot D(u, v)$.

---

## 3️⃣ PHÂN HỆ 3: QUALITY GATE & DUAL-STREAM ROUTING (P3 - ĐỨC)
* **File chịu trách nhiệm:** `notebook/backend/quality_gate.py`
* **Vấn đề cốt lõi:** Người dùng chụp thiếu góc (overlap $< 40\%$) hoặc chụp 2 ảnh trùng góc làm suy biến ma trận hình học; điều phối luồng 1 ảnh vs đa ảnh.
* **Giải pháp chuẩn NVIDIA 3D:**
  1. **Lớp 1 - Cosine Angle Verification:** Tính $\cos\theta = \frac{\text{Tr}(R_i^T R_j) - 1}{2}$. Nếu góc quay $\Delta\theta < 5^\circ \to$ Cảnh báo ảnh trùng góc.
  2. **Lớp 2 - Confidence Density Check:** Đảm bảo mật độ điểm quan sát tin cậy $\ge 0.45$.
  3. **Lớp 3 - Dual-Stream Routing & Fail-Safe Fallback:**
     - **Luồng 1 (1 ảnh):** Kích hoạt **TripoSR** (~1.5s, ViT + Triplane NeRF + Marching Cubes, Watertight Mesh) trên Colab GPU; hoặc **Depth-Anything-V2 Surface Engine** (~1.58s) khi chạy offline trên Local CPU.
     - **Luồng 2 ($N \ge 2$ ảnh):** Tự động điều hướng sang **Tencent Hunyuan3D-2mv DiT Flow Matching Pipeline** kèm Multi-View Texture Blender.
     - **Fail-safe Fallback:** Khi bộ ảnh đa góc không đạt chuẩn (trùng góc hoặc lỗi hình học) $\to$ Tự động chuyển về Anchor View #0 ở chế độ Đơn Ảnh (`TripoSR`), bảo đảm hệ thống luôn xuất ra file 3D chất lượng mà không bao giờ bị crash.

---

## 4️⃣ PHÂN HỆ 4: 3D VOLUMETRIC MESH ENGINEER (P4 - THỌ)
* **File chịu trách nhiệm:** `notebook/backend/engine_tsdf_mesh.py`
* **Vấn đề cốt lõi:** Mô hình bị nhô các cánh/vây thừa ra ngoài ("phần dư") và phần mềm in 3D Slicer báo lỗi cạnh phi đa tạp (non-manifold edges: 1 cạnh nối $>2$ mặt).
* **Giải pháp chuẩn NVIDIA 3D:**
  1. **True Multi-View Silhouette Space Carving (Visual Hull):**
     - Voxel 3D chỉ tồn tại nếu nằm trong vùng vật thể ở **TẤT CẢ các ảnh**:
       $$\text{Solid}(x, y, z) = \prod_{i=1}^{N} \text{Mask}_i\big(\pi_i(x, y, z)\big)$$
     - Gọt sạch 100% mọi vây thừa, cánh lơ lửng nằm ngoài silhouette.
  2. **Ray-based Truncated Signed Distance Field (TSDF):** Gọt không gian tự do phía trước bề mặt quan sát ($Z_{\text{cam}} < D_{\text{surf}} - \mu$).
  3. **Marching Cubes Iso-surface Extraction:** Trích xuất bề mặt kín nước (Watertight) 100%, $0$ cạnh biên, $0$ cạnh phi đa tạp.
  4. **Quadric Mesh Decimation:** Rút gọn lưới về $20,000 \sim 35,000$ tam giác, giúp khâu trải UV ở P5 tăng tốc từ 40s xuống 1.5s mà không giảm độ nét.

---

## 5️⃣ PHÂN HỆ 5: TEXTURE & UV SHADING ENGINEER (P5 - THÔNG)
* **File chịu trách nhiệm:** `notebook/backend/texture_blender.py` & `utils_3d.py`
* **Vấn đề cốt lõi:** Hiện tượng tự che khuất (Self-occlusion) khiến mặt sau bị lem màu mặt trước; ánh sáng chói lóa (Specular reflections) làm loang lổ bề mặt.
* **Giải pháp chuẩn NVIDIA 3D:**
  1. **XAtlas UV Parameterization:** Trải phẳng UV tối ưu vào không gian $[0, 1] \times [0, 1]$.
  2. **Angle-Weighted Blending (NVIDIA Fresnel Principle):**
     - Tính tích vô hướng giữa pháp tuyến mặt và tia nhìn camera $\cos\theta = \vec{n} \cdot \vec{v}_{\text{cam}}$.
     - Trọng số lũy thừa bậc 3: $\text{Weight} = \cos^3\theta$.
     - Dập tắt bóng gương và loại bỏ hoàn toàn các tam giác quay lưng về phía camera.
  3. **Xuất GLB PBR:** Đóng gói file `.glb` chuẩn nhị phân chứa mesh, UV và texture Albedo.

---

## 6️⃣ PHÂN HỆ 6: FULL-STACK & CLOUD DEPLOYMENT (P6 - LEAD)
* **File chịu trách nhiệm:** `notebook/backend/app.py`, `notebook/frontend/index.html`, `notebook/demo_colab.ipynb`
* **Vấn đề cốt lõi:** Cloudflare Tunnel trên Google Colab tự động ngắt kết nối (HTTP 524) khi request kéo dài quá 100 giây.
* **Giải pháp chuẩn:**
  1. **Asynchronous Job Polling:**
     - Endpoint `POST /generate-3d/job/` tạo `job_id` và trả kết quả ngay lập tức ($<100$ms).
     - Frontend polling `GET /generate-3d/job/{id}` mỗi 1.5s, kết nối luôn thông suốt không bao giờ timeout.
  2. **Three.js Web UI Viewer:** Tải và tương tác mô hình 3D xoay 360°, bật/tắt wireframe, tải file `.glb`.
  3. **Runbook Colab 1-click:** Loại bỏ phụ thuộc C++, khởi động dưới 5 giây trên Colab T4 GPU.