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
