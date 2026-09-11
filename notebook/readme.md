# Cổng Kiểm Định Chất Lượng & Động Cơ Cứu Hộ 3D (Quality Gate & Fail-safe Engine)

Phân hệ này là một phần của hệ thống tạo mô hình 3D từ ảnh 2D. Nhiệm vụ chính của phân hệ là kiểm tra chất lượng dữ liệu đầu vào (Quality Gate) và cung cấp một cơ chế tạo mô hình 3D dự phòng siêu tốc bằng **TripoSR** trong trường hợp dữ liệu không đạt chuẩn (Fail-safe), đảm bảo KPI thời gian phản hồi $\le 2$s.

---

## Mục lục
1. [Cấu trúc thư mục](#1-cấu-trúc-thư-mục)
2. [Yêu cầu hệ thống & Cài đặt](#2-yêu-cầu-hệ-thống--cài-đặt)
3. [Luồng hoạt động (Workflow)](#3-luồng-hoạt-động-workflow)
4. [Hướng dẫn chạy & Nghiệm thu](#4-hướng-dẫn-chạy--nghiệm-thu)
---

## 1. Cấu trúc thư mục

Để hệ thống hoạt động đúng, kiến trúc thư mục cần được tổ chức như sau:

```text
Img2d-to-3d/
└── notebook/
    └── backend/
        ├── tsr/                  # Thư mục mã nguồn lõi của TripoSR
        ├── app.py                # Máy chủ FastAPI (API Điều phối)
        ├── quality_gate.py       # Cổng lọc chất lượng (Thuật toán Cosine Similarity)
        ├── engine_triposr.py     # Động cơ Cứu hộ (TripoSR + Rembg)
        ├── temp_uploads/         # (Tự động tạo khi run app.py) Chứa ảnh user upload
        └── outputs/              # (Tự động tạo khi run app.py) Chứa model .glb xuất ra
```
## 2. Yêu cầu hệ thống & Cài đặt

- **Ngôn ngữ:** Python 3.10
- **Môi trường:** Trình biên dịch C++ (Microsoft C++ Build Tools) để build thư viện `torchmcubes`.

**Cài đặt các thư viện cần thiết:**
Mở Terminal, di chuyển vào thư mục dự án và chạy các lệnh sau:

```bash
pip install torch torchvision
pip install fastapi uvicorn trimesh rembg onnxruntime
```
**Cài đặt mã nguồn TripoSR (Bắt buộc):**
Hệ thống yêu cầu mã nguồn lõi của TripoSR để khởi tạo mô hình Cứu hộ. Thực hiện các bước sau:

1. Mở Terminal ở một thư mục bất kỳ và tải kho lưu trữ chính thức của TripoSR:

   ```bash
   git clone https://github.com/VAST-AI-Research/TripoSR.git
   ```
2. Mở thư mục `TripoSR` vừa tải về, copy toàn bộ thư mục `tsr/`.
3. Dán thư mục `tsr/` vào bên trong thư mục `notebook/backend/` của dự án để cấu trúc khớp với sơ đồ trên.

*(Lưu ý: Trọng số mô hình `model.ckpt` nặng ~2GB sẽ được hệ thống tự động kết nối và tải về từ Hugging Face trong lần khởi chạy server đầu tiên).*
## 3. Luồng hoạt động (Workflow)

Hệ thống được cấu thành từ 3 file chính, hoạt động theo dây chuyền:

1. **`app.py` (Nhạc trưởng):** Khởi tạo máy chủ ở cổng `8000`. Nhận ảnh từ người dùng, lưu vào `temp_uploads` và kích hoạt luồng kiểm định.
2. **`quality_gate.py` (Bộ lọc):**
   - Đánh giá Pose (góc camera) bằng Cosine Similarity.
   - Bắt chính xác 100% các góc chụp bị trùng lặp (ví dụ: góc lệch = 0 độ) hoặc điểm Confidence thấp.
   - Nếu lỗi -> Kích hoạt cứu hộ.
3. **`engine_triposr.py` (Cứu hộ):**
   - Được nạp vào RAM ngay khi boot server để tối ưu tốc độ.
   - Xóa nền ảnh gốc, đắp nền trắng và đưa ảnh về chuẩn RGB.
   - Feed-forward qua TripoSR, nặn lưới 3D (Mesh) và xuất thẳng ra định dạng chuẩn `.glb` vào thư mục `outputs`.
## 4. Hướng dẫn chạy & Nghiệm thu

### Bước 1: Khởi động máy chủ
Di chuyển vào thư mục `backend` và chạy lệnh Uvicorn:

```bash
cd notebook/backend
python -m uvicorn app:app --reload
```
*Đợi đến khi Terminal báo:* `Application startup complete.`

### Bước 2: Upload ảnh test
1. Mở trình duyệt web truy cập vào giao diện Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. Mở rộng mục `POST /generate-3d/` → Chọn **Try it out**.
3. Upload 1 bức ảnh (Nên chọn ảnh có vật thể rõ ràng, độ tương phản cao với nền).
4. Bấm **Execute**.

### Bước 3: Nghiệm thu kết quả
- **Thời gian (KPI):** Kiểm tra log trả về trên web, `execution_time_seconds` phải ≤ 2.0 giây.
- **Model 3D:** Mở thư mục `outputs`, tìm file `.glb` vừa được tạo.
- **Cách xem:** Sử dụng phần mềm **3D Viewer** (có sẵn trên Windows) hoặc kéo thả file vào web [glTF Viewer](https://gltf-viewer.donmccurdy.com/) để chiêm ngưỡng mô hình 360 độ.
