<!-- đây là nơi để mọi người viết plan của mình vào khi dùng ai agent 
Lưu ý:
viết kế tiếp theo sau k cần tạo thêm file mới , chỉ tạo thêm ở folder notebook với cây đã được để sẵn trong plan.md
Vui lòng k sửa các file khác , chỉ sửa file này và cập nhật status.md sau mỗi step -->
Tạo thêm branch mới để test ,nếu check ok thì tính sau:))

---

# 🚀 KẾ HOẠCH TRIỂN KHAI CHI TIẾT: THÀNH VIÊN 1 (DUY)
## VỊ TRÍ: DATA & PREPROCESSING ENGINEER (KỊCH BẢN 2 - Multi-view)

> **Người thực hiện:** Duy (Thành viên 1 - P1)  
> **Lý thuyết nền tảng:** [docs/lythuyet.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/lythuyet.md)  
> **Căn cứ plan tổng thể:** [docs/plan.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/plan.md)  
> **Phạm vi mã nguồn chịu trách nhiệm:**
> - `notebook/backend/preprocess.py` (Chế độ tiền xử lý Đa ảnh `preprocess_multiview`).
> - `notebook/backend/test_preprocess.py` (Test suite 500 test cases).
> - `data/input/multi_view/` (Bộ dữ liệu ảnh kiểm thử đa góc nhìn).

---

## 📦 PHẦN 1: CẤU TRÚC FILE & HỢP ĐỒNG GIAO DIỆN (INTERFACE CONTRACT)

### 1.1 Cây thư mục liên quan trực tiếp đến Duy:
```
ImgToModel/
├── data/
│   └── input/
│       ├── multi_view/              # Ảnh đa góc nhìn (giữ nguyên cấu trúc gốc)
│       └── single_view/             # Ảnh đơn (giữ nguyên cấu trúc gốc)
├── notebook/
│   └── backend/
│       ├── preprocess.py            # FILE CHÍNH CỦA DUY
│       └── test_preprocess.py       # TEST SUITE CHỈ CHO PHẦN PREPROCESSING CỦA DUY
└── docs/
    ├── lythuyet.md                  # Lý thuyết nền tảng (đọc trước khi code)
    ├── planforAI.md                 # Kế hoạch hành động này
    └── status.md                    # Cập nhật tiến độ sau mỗi step (bổ sung, không ghi đè)
```

### 1.2 Hợp đồng giao diện (Hàm bàn giao cho P2 và P4):

```python
def preprocess_multiview(
    image_paths: List[str], 
    target_size: int = 512, 
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Tiền xử lý đa ảnh cho Pipeline v1:
    - Kiểm tra & lọc ảnh hợp lệ
    - Resize chuẩn DUSt3R (bảo toàn epipolar geometry)
    - Tách Alpha Mask bằng RMBG-2.0
    - Cân bằng sáng Histogram Matching
    
    Returns:
        dict chứa:
            'images_dust3r': Tensor (N, 3, H, W) chuẩn hóa ImageNet cho DUSt3R
            'alpha_masks': Tensor (N, 1, H, W) nhị phân {0, 1} cho P4 (Point Pruning)
            'images_rgb_clean': List[np.ndarray] ảnh RGB gốc đã resize cho P5 (Texturing)
            'original_sizes': List[Tuple[int, int]] kích thước gốc (H, W)
            'scale_factors': List[float] tỉ lệ co dãn mỗi ảnh
            'filenames': List[str] tên file gốc
    """
```

---

## 🛠️ PHẦN 2: LỘ TRÌNH THỰC HIỆN (TUẦN TỰ NGHIÊM NGẶT)

**Nguyên tắc:** Mỗi step phải hoàn thành + update status.md trước khi chuyển sang step kế tiếp.  
**Thứ tự bắt buộc:** Chuẩn bị data → Viết code → Audit code → Viết 500 test cases → Pass toàn bộ test → Mới tạo file backend chính thức theo cây.

---

### Step 1: Chuẩn bị dữ liệu benchmark đa góc
- [x] Sắp xếp 6 ảnh chụp xung quanh vật thể tĩnh vào `data/input/multi_view/` (góc chụp cách nhau $\approx 60°$, cùng độ cao).
- [x] Chuẩn bị thêm các bộ ảnh edge-case: ảnh quá sáng, quá tối, ảnh bị blur, ảnh sai định dạng, ảnh trùng lặp (`generate_test_images.py`).
- [x] Giữ nguyên `data/input/single_view/` cho Kịch bản 1 sau này.

---

### Step 2: Viết code module `preprocess.py` (Draft đầu tiên)
- [x] **Hàm `validate_and_load_images()`:** Kiểm tra file hợp lệ (định dạng, đọc được, không bị hỏng).
- [x] **Hàm `subsample_images()`:** Nếu $N > 8$ tự động lấy mẫu đều Uniform Subsampling về $6 \sim 8$ ảnh.
- [x] **Hàm `dust3r_resize()`:** Resize ảnh bảo toàn aspect ratio, $\max(H,W) = 512$, $(H,W)$ chia hết cho 16.
- [x] **Hàm `extract_alpha_masks()`:** Tích hợp RMBG-2.0, chạy tuần tự, xuất Mask nhị phân.
- [x] **Hàm `histogram_match()`:** Cân bằng sáng theo ảnh tham chiếu $I_0$.
- [x] **Hàm `preprocess_multiview()`:** Đóng gói tổng thể, gọi tuần tự các hàm trên.

---

### Step 3: Audit Code (Rà soát chất lượng mã nguồn)
- [x] **Kiểm tra logic:** Mỗi hàm xử lý đúng 1 nhiệm vụ duy nhất (Single Responsibility).
- [x] **Kiểm tra an toàn bộ nhớ:** Không giữ tensor/array lớn thừa trong RAM; giải phóng đúng cách sau khi dùng xong (`try...finally`, `del`, `empty_cache`).
- [x] **Kiểm tra edge cases trong code:**
  - Xử lý khi `image_paths` rỗng hoặc chứa đường dẫn không tồn tại.
  - Xử lý khi ảnh có kích thước quá nhỏ ($< 32 \times 32$) hoặc quá lớn ($> 10000 \times 10000$).
  - Xử lý khi ảnh là grayscale (1 kênh) hoặc RGBA (hòa nền trắng thay vì viền đen).
  - Xử lý file rỗng (0 bytes) và file corrupt.
- [x] **Kiểm tra docstring & type hints:** Tất cả các hàm public có docstring chi tiết Input/Output và type hints.
- [x] **Kiểm tra naming convention:** Tuân thủ `snake_case`, rõ ràng, không viết tắt gây nhầm lẫn.
- [x] **Kiểm tra import:** Import tường minh, không dùng wildcard import (`*`).

---

### Step 4: Viết Test Suite 500 Cases (`test_preprocess.py`)
> **Lưu ý:** Test cases chỉ kiểm tra các hàm trong phần việc của Duy (`preprocess.py`). Không test code của P2~P6.

- [x] Tổ chức test cases theo 5 nhóm chức năng (mỗi nhóm $\ge 100$ cases, tổng 500 cases trong `notebook/backend/test_preprocess.py`):
  - Nhóm A: Test Validate & Load Images (100 cases)
  - Nhóm B: Test Subsampling Logic (100 cases)
  - Nhóm C: Test DUSt3R Resize (100 cases)
  - Nhóm D: Test RMBG-2.0 Alpha Mask (100 cases)
  - Nhóm E: Test Histogram Matching & Tích Hợp End-to-End (100 cases)

#### Nhóm A: Test Validate & Load Images (~100 cases)
- Đầu vào hợp lệ: JPG, PNG, WEBP, BMP (nhiều kích thước khác nhau).
- Đầu vào không hợp lệ: file hỏng, file rỗng, file không phải ảnh (PDF, TXT, ZIP), đường dẫn không tồn tại.
- Ảnh đặc biệt: grayscale, RGBA (4 kênh), ảnh 1x1 pixel, ảnh cực lớn (8000x6000).
- Trùng lặp: 2 ảnh giống hệt nhau trong danh sách đầu vào.
- Unicode path: đường dẫn chứa ký tự tiếng Việt, dấu cách, ký tự đặc biệt.

#### Nhóm B: Test Subsampling Logic (~100 cases)
- $N = 1, 2, 3$: Kiểm tra báo lỗi / cảnh báo đúng.
- $N = 4, 5, 6, 7, 8$: Kiểm tra giữ nguyên toàn bộ, không lọc.
- $N = 9, 10, 12, 15, 20, 50, 100$: Kiểm tra thuật toán Uniform Subsampling trả về đúng $6 \sim 8$ ảnh.
- Kiểm tra ảnh đầu tiên và ảnh cuối cùng luôn được giữ lại.
- Kiểm tra tính ổn định: gọi hàm 2 lần cùng input phải cho cùng output (deterministic).

#### Nhóm C: Test DUSt3R Resize (~100 cases)
- Ảnh ngang (landscape): 1920x1080, 4000x3000, 800x600, ...
- Ảnh dọc (portrait): 1080x1920, 3000x4000, 600x800, ...
- Ảnh vuông (square): 1024x1024, 512x512, 256x256, ...
- Ảnh kích thước lẻ: 1001x777, 513x513, 17x17, ...
- Kiểm tra: $\max(H_{\text{out}}, W_{\text{out}}) \le 512$, $(H_{\text{out}} \mod 16) = 0$, $(W_{\text{out}} \mod 16) = 0$.
- Kiểm tra: aspect ratio bảo toàn (sai lệch $\le 2\%$).
- Kiểm tra: pixel values nằm trong $[0, 255]$ (uint8) hoặc $[0.0, 1.0]$ (float32) tùy giai đoạn.

#### Nhóm D: Test RMBG-2.0 Alpha Mask (~100 cases)
- Ảnh có nền đơn sắc (trắng, đen, xanh lá).
- Ảnh có nền phức tạp (vật thể trên bàn gỗ, sàn nhà, ngoài trời).
- Ảnh vật thể chiếm toàn bộ khung hình (không có nền).
- Kiểm tra output shape: $(1, H, W)$ hoặc $(H, W)$.
- Kiểm tra giá trị: $M \in \{0, 1\}$ (nhị phân, không có giá trị lẻ).
- Kiểm tra: Mask không toàn 0 (mất vật thể) và không toàn 1 (không tách được nền).
- Kiểm tra: Mask khớp kích thước với ảnh đã resize.

#### Nhóm E: Test Histogram Matching & Tích Hợp End-to-End (~100 cases)
- 6 ảnh cùng điều kiện sáng: Output phải gần giống input (không thay đổi quá nhiều).
- 6 ảnh khác điều kiện sáng (sáng / tối / ám vàng): Histogram sau matching phải hội tụ gần nhau.
- Kiểm tra: pixel values vẫn nằm trong $[0, 255]$ (không clipping ra ngoài).
- **Test End-to-End `preprocess_multiview()`:** 6 ảnh chuẩn → dict output chứa đủ 6 keys, đúng shape, đúng dtype.
- Kiểm tra: thời gian xử lý $\le 5.0$s (CPU) hoặc $\le 2.0$s (GPU T4).
- Kiểm tra: không memory leak (RAM usage trước và sau gọi hàm chênh lệch $\le 100$MB).

---

### Step 5: Chạy toàn bộ 500 test cases & Fix bugs
- [x] Chạy `python notebook/backend/test_preprocess.py` hoặc `pytest notebook/backend/test_preprocess.py -v`.
- [x] **Mục tiêu: 500/500 PASSED, 0 FAILED, 0 ERROR.**
- [x] Đã hoàn thiện toàn bộ edge-cases, audit logic và dọn dẹp file test sau nghiệm thu.

---

### Step 6: Tạo file backend chính thức theo cây & Bàn giao
- [x] Đảm bảo `notebook/backend/preprocess.py` nằm đúng vị trí theo cây thư mục đã chốt trong [docs/plan.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/plan.md).
- [x] Đảm bảo `notebook/backend/__init__.py` tồn tại (Python package).
- [x] Bàn giao output cho Thành viên 2 (P2: DUSt3R) và Thành viên 4 (P4: TSDF Mesh).
- [x] Cập nhật `docs/status.md` đánh dấu P1 hoàn thành.

---

## 🎯 PHẦN 3: TIÊU CHÍ NGHIỆM THU (DEFINITION OF DONE)

| # | Tiêu chí | Ngưỡng đạt |
|:---:|:---|:---|
| 1 | Tính trọn vẹn quang học | $100\%$ ảnh đầu ra không bị crop méo tâm; aspect ratio bảo toàn |
| 2 | Chất lượng Alpha Mask | Mask bao khít vật thể, giá trị chuẩn nhị phân $\{0, 1\}$ |
| 3 | Test cases coverage | $\ge 500$ test cases, tỷ lệ pass $100\%$ (500/500) |
| 4 | Audit code | Qua rà soát: logic, bộ nhớ, edge cases, docstring, naming |
| 5 | Thời gian xử lý | $\le 2.0$s (GPU T4) hoặc $\le 5.0$s (CPU) cho 6 ảnh |
| 6 | Tính độc lập | Chạy `test_preprocess.py` thành công mà không cần code của P2~P6 |


---

# 🚀 KẾ HOẠCH TRIỂN KHAI CHI TIẾT: THÀNH VIÊN 2 (HUY)
## VỊ TRÍ: ƯỚC LƯỢNG 3D & CAMERA POSE (DUSt3R)

> **Người thực hiện:** HUY (Thành viên 2 - P2)  
> **Lý thuyết nền tảng:** [docs/lythuyet.md](file:///C:/Users/ADMIN/Downloads/Img2d-to-3d-main/Img2d-to-3d-main/docs/lythuyet.md)  
> **Căn cứ plan tổng thể:** [docs/plan.md](file:///C:/Users/ADMIN/Downloads/Img2d-to-3d-main/Img2d-to-3d-main/docs/plan.md)  
> **Phạm vi mã nguồn chịu trách nhiệm:**
> - `notebook/backend/engine_dust3r.py` (Lõi DUSt3REngine).
> - `notebook/backend/test_dust3r.py` (Test suite cho DUSt3R).

---

## 📦 PHẦN 1: CẤU TRÚC & GIAO DIỆN

### 1.1 Hợp đồng dữ liệu (Interface Contract):
- **Input:** dict từ `preprocess_multiview` (của P1), bao gồm `images_dust3r` (Tensor N,3,H,W chuẩn hóa ImageNet), `original_sizes`.
- **Output:**
  - `pointmaps_3d`: Điểm 3D tại mỗi pixel (N, H, W, 3) trong cùng hệ tọa độ toàn cục.
  - `confidence_masks`: Độ tin cậy tại mỗi điểm (N, H, W).
  - `camera_poses`: Danh sách ma trận 4x4 (R, T) của các góc nhìn.
  - `focal_lengths`: Tiêu cự ước lượng (f_x, f_y).

## 🛠️ PHẦN 2: LỘ TRÌNH THỰC HIỆN

### Step 1: Viết module `engine_dust3r.py`
- Tạo class `DUSt3REngine`.
- Hàm `load_model()`: Khởi tạo mô hình ViT từ pretrained weights.
- Hàm `run_pairwise_matching()`: Tính toán pose và point-map giữa các cặp ảnh.
- Hàm `run_global_alignment()`: Tối ưu Global alignment.
- Hàm `process()`: pipeline chính gọi các hàm trên.

### Step 2: Tự Audit Code
- Tối ưu GPU VRAM (dọn dẹp cache, dùng context `torch.no_grad()`).
- Bắt lỗi ngoại lệ (OOM, input không hợp lệ).
- Nhận xét tự Audit Code ghi tại cuối file dưới dạng comment.

### Step 3: Viết Test Case `test_dust3r.py`
- Viết unit tests kiểm thử độc lập các hàm của `DUSt3REngine`.
- Sử dụng mock data (tensor) để test logic.

---
## PHẦN 3: CẤU TRÚC FILE & HỢP ĐỒNG GIAO DIỆN (INTERFACE CONTRACT)

### 3.1 Cây thư mục liên quan trực tiếp đến Quality Gate & Fail-safe Engine:
```text
Img2d-to-3d/
├── notebook/
│   └── backend/
│       ├── tsr/                     # Thư mục mã nguồn lõi của TripoSR (cần clone từ repo gốc)
│       ├── app.py                   # FILE CHÍNH - Máy chủ FastAPI điều phối luồng
│       ├── quality_gate.py          # Module cổng lọc chất lượng (Cosine Similarity)
│       ├── engine_triposr.py        # Module động cơ cứu hộ dự phòng
│       ├── temp_uploads/            # Thư mục tự động tạo lưu ảnh input
│       └── outputs/                 # Thư mục tự động tạo lưu kết quả .glb
└── docs/
    └── status.md                    # Cập nhật tiến độ sau mỗi step (bổ sung, không ghi đè)
```

### 3.2 Hợp đồng giao diện (Các API & Hàm cốt lõi bàn giao cho toàn hệ thống):

```Python
# ==========================================
# 1. API ĐIỀU PHỐI (app.py)
# ==========================================
@app.post("/generate-3d/")
async def generate_3d(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Endpoint chính nhận yêu cầu tạo 3D. 
    Điều phối dữ liệu qua Quality Gate và quyết định gọi DUST3R (Main) hay TripoSR (Fail-safe).

    Returns:
        dict chứa:
            'status': str ("success" hoặc "failed")
            'quality_passed': bool (Kết quả đánh giá từ cổng kiểm định)
            'gate_reason': str (Lý do chi tiết nếu bị đánh rớt)
            'execution_time_seconds': float (KPI đo lường thời gian <= 2s)
            'output_file': str (Đường dẫn tới file .glb thành phẩm)
    """

# ==========================================
# 2. CỔNG KIỂM ĐỊNH (quality_gate.py)
# ==========================================
class QualityGate:
    def evaluate(
        self, 
        poses: List[np.ndarray], 
        confidence_map: np.ndarray, 
        ba_loss: float
    ) -> Tuple[bool, str]:
        """
        Đánh giá chất lượng dữ liệu 3D không gian dựa trên ngưỡng toán học.
        - Kiểm tra độ lệch góc Camera bằng Cosine Similarity
        - Kiểm tra Confidence & Bundle Adjustment Loss

        Returns:
            Tuple[bool, str] chứa (is_valid, reason)
        """

# ==========================================
# 3. ĐỘNG CƠ CỨU HỘ (engine_triposr.py)
# ==========================================
class TripoSREngine:
    def run_fallback(
        self, 
        image_path: str, 
        output_glb_path: str
    ) -> Tuple[bool, str, float]:
        """
        Kích hoạt quy trình tạo 3D khẩn cấp khi Quality Gate báo lỗi.
        - Tiền xử lý: Tách nền rembg, ép định dạng RGB phông trắng.
        - Xử lý: Nặn mesh 3D qua TripoSR (resolution=128 tối ưu RAM).
        - Hậu xử lý: Trực tiếp xuất file mesh.export().

        Returns:
            Tuple[bool, str, float] chứa (success, model_path, execution_time)
        """
```

---

# 🚀 KẾ HOẠCH TRIỂN KHAI CHI TIẾT: THÀNH VIÊN 4 (NVIDIA 3D RECONSTRUCTION)
## VỊ TRÍ: 3D VOLUMETRIC MESH ENGINEER

> **Người thực hiện:** P4 (3D Mesh) & P5 (Texture Blending)  
> **Lý thuyết nền tảng:** [docs/lythuyet.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/lythuyet.md), [docs/phan_tich_chuyen_sau_reference_repos.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/phan_tich_chuyen_sau_reference_repos.md) (Phần 4: Chi tiết thuật toán & bản chất toán học)  
> **Căn cứ plan tổng thể:** [docs/plan.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/plan.md)  
> **Phạm vi mã nguồn chịu trách nhiệm:**
> - `notebook/backend/engine_tsdf_mesh.py` (Lõi P4: Lọc biên độ sâu, Pruning điểm nền, Voxel TSDF Grid, Marching Cubes).
> - `notebook/backend/texture_blender.py` (Lõi P5: XAtlas UV Unwrapping, Angle-Weighted Color Blending, Xuất file .glb).
> - `notebook/backend/test_tsdf_pipeline.py` (Test suite kiểm thử độc lập P4 & P5).
> - `notebook/backend/app.py` (Nối P4 & P5 vào nhánh Quality PASS thay cho mock TripoSR).

---

## 📦 PHẦN 1: CẤU TRÚC FILE & HỢP ĐỒNG GIAO DIỆN (INTERFACE CONTRACT)

### 1.1 Hợp đồng dữ liệu đầu vào (Nhận từ P1 Preprocess & P2 DUSt3R):
- `pointmaps_3d`: np.ndarray hoặc Tensor $(N, H, W, 3)$ — Tọa độ điểm 3D trong hệ quy chiếu thế giới chung.
- `confidence_masks`: np.ndarray hoặc Tensor $(N, H, W)$ — Bản đồ độ tin cậy từ DUSt3R.
- `alpha_masks`: List[np.ndarray] $(N, H, W)$ giá trị $\{0, 1\}$ — Mặt nạ phân đoạn từ RMBG-2.0.
- `images_rgb`: List[np.ndarray] $(N, H, W, 3)$ uint8 — Ảnh RGB gốc đã cân bằng sáng (dành cho texturing).
- `camera_poses`: List[np.ndarray] ma trận $4 \times 4$ $[R_i \mid T_i]$ của từng góc nhìn.
- `focal_lengths`: List[Tuple[float, float]] $(f_x, f_y)$ tiêu cự camera.

### 1.2 Hợp đồng dữ liệu đầu ra:
- `TSDFMeshEngine.reconstruct(...)`: Trả về `Trimesh` object chứa vertices $(V, 3)$, faces $(F, 3)$, vertex normals $(V, 3)$.
- `TextureBlender.process_and_export(...)`: Trả về `(success: bool, glb_path: str)` — File `.glb` hoàn chỉnh có Base-Color UV Texture map $1024 \times 1024$.

---

## 🛠️ PHẦN 2: LỘ TRÌNH THỰC HIỆN TUẦN TỰ (4 THUẬT TOÁN)

- [x] **Thuật toán 1 (Lọc viền độ sâu & Pruning điểm nền):**
  - Cắt tỉa 100% pixel nền qua Alpha Mask $\mathbf{M}_i(u, v) = 0$.
  - Lọc điểm nhiễu $\mathbf{C}_i(u, v) < \tau_{\text{conf}}$.
  - Tính gradient độ sâu $G_x = |D(u+1, v) - D(u-1, v)|, G_y = |D(u, v+1) - D(u, v-1)|$, loại bỏ điểm mép rách $\max(G_x, G_y) > \tau \cdot D(u, v)$.
- [x] **Thuật toán 2 (Lưới thể tích TSDF Volumetric Fusion):**
  - Thiết lập Voxel grid 3D theo Bounding Box của điểm hợp lệ.
  - Chiếu voxel $\mathbf{p}$ về ảnh camera $i$, tính khoảng cách có dấu $d_i(\mathbf{p}) = D_i(\mathbf{x}_i) - z$.
  - Cắt ngắn $[-\mu, +\mu]$ thành $\text{tsdf}_i(\mathbf{p})$.
  - Tích lũy liên tục có trọng số theo confidence: $D_{\text{new}}(\mathbf{p}), W_{\text{new}}(\mathbf{p})$.
- [x] **Thuật toán 3 (Trích xuất Iso-surface Marching Cubes):**
  - Quét 8 đỉnh của từng voxel qua Look-up Table 256 cấu hình tam giác tại $D = 0$.
  - Nội suy vị trí đỉnh, tính toán vertex normals và dọn dẹp các mảnh tam giác rời rạc.
- [x] **Thuật toán 4 (Trải UV XAtlas & Nướng màu Angle-Weighted Blending):**
  - Tham số hóa UV vào khung $[0, 1] \times [0, 1]$.
  - Tính vector nhìn $\vec{v}_i$, pháp tuyến $\vec{n}$, góc $\cos \theta_i = \max(0, \vec{n} \cdot \vec{v}_i)$.
  - Trọng số $W_i = (\cos \theta_i)^\gamma$ ($\gamma \approx 2 \sim 4$), ray-cast kiểm tra che khuất.
  - Hòa trộn màu albedo khuếch tán, xuất file `.glb` chuẩn qua Trimesh.
- [x] **Kiểm thử & Nối luồng:**
  - Viết `test_tsdf_pipeline.py` kiểm thử độc lập 100% pass trên dữ liệu mô phỏng.
  - Nối vào `app.py`, hoàn thiện luồng End-to-End.

---

# 🚀 KẾ HOẠCH TRIỂN KHAI CHI TIẾT: THÀNH VIÊN 5
## VỊ TRÍ: TEXTURE & UV SHADING ENGINEER

> **Người thực hiện:** Phước (Thành viên 5 - P5)
> **Lý thuyết nền tảng:** [docs/lythuyet.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/lythuyet.md)
> **Căn cứ plan tổng thể:** [docs/plan.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/plan.md)
> **Phạm vi mã nguồn chịu trách nhiệm:** (tạo 3 file mới trong `notebook/backend/`)
> - `utils_3d.py`: helper cho validation, camera projection, visibility, sampling và GLB export.
> - `texture_blender.py`: XAtlas UV, angle-weighted Base-Color blending, texture baking và GLB assembly.
> - `test_texture_blender.py`: test độc lập cho interface P5.

---

## 📦 PHẦN 1: HỢP ĐỒNG BÀN GIAO

### Input từ các phần trước
- P1 `preprocess_multiview()`: `images_rgb`, danh sách ảnh `(H, W, 3)` `uint8` đã resize và histogram-match.
- P2 `DUSt3REngine.process()`: `camera_poses` camera-to-world `3x4`/`4x4` và `focal_lengths` `(fx, fy)`.
- P4 `TSDFMeshEngine.reconstruct()`: `trimesh.Trimesh` có vertices, faces và normals.

### Output
- `TextureBlender.process_and_export(...) -> (success: bool, glb_path: str)`.
- Một file `.glb` chứa mesh, UV và Base-Color/Albedo texture.
- Không gọi output là PBR texture vì pipeline không tạo normal, roughness, metallic hoặc AO map.

## 🛠️ PHẦN 2: LỘ TRÌNH P5

- [x] Tạo helper dùng trực tiếp contract P1/P2/P4 trong `utils_3d.py`.
- [x] Tạo `TextureBlender` trong `texture_blender.py`.
- [x] Dùng XAtlas khi có thư viện; có fallback UV deterministic để test không phụ thuộc bắt buộc vào XAtlas.
- [x] Chiếu vertices về từng ảnh và blend màu theo `max(0, n dot v)^gamma`.
- [x] Lọc điểm không hợp lệ/không nhìn thấy bằng triangle-rasterized depth-buffer trước khi đóng góp màu.
- [x] Bake màu trực tiếp theo texel UV, nội suy barycentric world position và blend từ các view nhìn thấy.
- [x] Xuất GLB và kiểm tra lại UV/material image trong test.
- [x] Viết test cho projection, visibility, UV, blending, baking, GLB và input mismatch.
- [x] Hoàn tất code và bộ test P5; các file đã được duyệt: `utils_3d.py`, `texture_blender.py`, `test_texture_blender.py`.
- [x] Chạy `pytest notebook/backend/test_texture_blender.py -q`: 4 passed với XAtlas backend thật.
- [x] Bàn giao interface cho P6; P5 không tự sửa `app.py`.

## ✅ DEFINITION OF DONE

| Tiêu chí | Điều kiện đạt |
|:---|:---|
| UV | UV có shape `(N, 2)`, nằm trong `[0, 1]`; ưu tiên XAtlas |
| Color blending | Dùng ảnh P1, pose/focal P2 và normal từ mesh P4 |
| Visibility | Điểm bị che khuất không đóng góp màu trong depth-buffer |
| Texture | Có RGB Base-Color texture được bake trên UV atlas |
| GLB | Load lại được bằng Trimesh, có UV và material image |
| Isolation | Test P5 không gọi `app.py` và không sửa module P1-P4 |