# 📚 LÝ THUYẾT NỀN TẢNG: THÀNH VIÊN 1 (DUY - PREPROCESSING ENGINEER)

> **Mục đích:** Tài liệu cơ sở lý thuyết & bản chất toán học của từng thành phần kỹ thuật mà Duy (P1) chịu trách nhiệm.  
> **Tham chiếu:** [docs/phan_tich_chuyen_sau_reference_repos.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/phan_tich_chuyen_sau_reference_repos.md), [docs/plan.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/plan.md).

---

## 1. Bản Chất Quang Học & Hình Học Epipolar

### 1.1 Tại sao Đa Ảnh TUYỆT ĐỐI KHÔNG ĐƯỢC Crop Riêng Lẻ?

**Nguyên lý:** Trong thị giác máy tính đa góc nhìn, mối quan hệ không gian giữa 2 ảnh được ràng buộc bởi **Hình học Epipolar (Epipolar Geometry)** thông qua ma trận cơ bản $F$ (Fundamental Matrix):

$$\mathbf{x}_2^T F \mathbf{x}_1 = 0$$

Trong đó $\mathbf{x}_1, \mathbf{x}_2$ là tọa độ đồng nhất của cùng một điểm vật thể trên ảnh 1 và ảnh 2. Ma trận $F$ phụ thuộc trực tiếp vào ma trận nội thông camera (Intrinsics $K$) và tâm quang học $(c_x, c_y)$ — vị trí tia sáng đi vuông góc qua tâm cảm biến.

**Ma trận Camera Intrinsics:**

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Trong đó:
- $f_x, f_y$: Tiêu cự theo pixel trên trục $x$ và $y$.
- $(c_x, c_y)$: Tâm quang học (Principal Point) — thường nằm ở trung tâm ảnh.

### 1.2 Cạm Bẫy Khi Crop Bounding Box Riêng Lẻ

Nếu ảnh 1 crop theo khung $[x_1, y_1, w_1, h_1]$, ảnh 2 crop theo khung $[x_2, y_2, w_2, h_2]$:
- Tâm quang học $(c_x, c_y)$ của từng ảnh sẽ bị dịch chuyển một khoảng ngẫu nhiên: $(\Delta x_1, \Delta y_1) \neq (\Delta x_2, \Delta y_2)$.
- **Hậu quả:** Ràng buộc Epipolar bị bẻ gãy hoàn toàn. DUSt3R (sử dụng Vision Transformer) dựa vào positional embedding để ước lượng tia chiếu chéo giữa các ảnh. Nếu ảnh bị crop xô lệch tâm, DUSt3R sẽ hiểu nhầm góc chụp, dẫn đến việc ước lượng ma trận xoay $[R|T]$ sai lệch hoàn toàn hoặc thuật toán Global Alignment không thể hội tụ.

### 1.3 Giải Pháp Chuẩn: DUSt3R Standard Loader

- Giữ nguyên toàn bộ khung hình quang học gốc.
- Chỉ thực hiện phép co dãn đồng tỉ lệ (Uniform Scaling) sao cho cạnh lớn nhất bằng $512\text{px}$.
- Kích thước $(H, W)$ phải chia hết cho $16$ (để vừa khớp với Patch Size $16 \times 16$ của mô hình ViT).
- Tuyệt đối không cắt xén bất đối xứng.

**Công thức tính tỉ lệ co dãn:**

$$s = \frac{T}{\max(H_{\text{orig}}, W_{\text{orig}})} \quad (T = 512)$$

$$H_{\text{new}} = \text{round\_to\_16}(H_{\text{orig}} \times s)$$
$$W_{\text{new}} = \text{round\_to\_16}(W_{\text{orig}} \times s)$$

Trong đó:
$$\text{round\_to\_16}(x) = \left\lfloor \frac{x + 8}{16} \right\rfloor \times 16$$

---

## 2. Bản Chất Mô Hình RMBG-2.0 & Alpha Mask

### 2.1 Kiến Trúc RMBG-2.0 (Bilateral Reference Network — BiRefNet)

- Mạng nơ-ron phân đoạn ảnh độ phân giải cao, chuyên trích xuất mặt nạ mờ (Alpha Matte):
  $$\mathbf{M} \in [0, 1]^{H \times W}$$
- Sử dụng kiến trúc Encoder-Decoder với cơ chế tham chiếu song phương (Bilateral Reference) giúp giữ sắc viền cực kỳ chi tiết ngay cả khi vật thể có tóc, lông, hay viền trong suốt.

### 2.2 Tại Sao KHÔNG Xóa Nền Trước Khi Đưa Vào DUSt3R?

- DUSt3R hoạt động dựa trên cơ chế tìm kiếm điểm tương đồng (Dense Feature Matching) trên **toàn bộ khung hình**.
- Những vật thể chụp thực tế (tượng gốm trắng, đồ chơi trơn nhẵn, khối hình học) thường rất **nghèo nàn về vân bề mặt (textureless)**. Để tính toán chính xác camera đang đứng ở đâu, DUSt3R **rất cần bám vào các điểm đặc trưng xung quanh**: vân mặt bàn, góc phòng, sàn nhà, bóng đổ.
- Nếu xóa nền thành màu đen tuyền $[0, 0, 0]$ ngay từ đầu, hơn $60\% \sim 70\%$ điểm bám quang học bị biến mất $\to$ DUSt3R thất bại hoàn toàn trong việc tìm góc camera.

### 2.3 Quy Trình Chuẩn Xác: Chạy Song Song, Dùng Sau

1. **RMBG-2.0 chạy ngầm độc lập** (tuần tự từng ảnh để không tốn VRAM).
2. Trích xuất ra mảng nhị phân Alpha Mask $\mathbf{M}_i$.
3. Đóng gói bàn giao cho **Thành viên 4 (P4)**.
4. P4 sẽ dùng Mask này để **Cắt bỏ các điểm 3D thuộc hậu cảnh (Background Point Pruning)** *sau khi* DUSt3R đã hoàn thành định vị không gian 3D.

**Nhị phân hóa Alpha Mask (Thresholding):**

$$\mathbf{M}_i^{\text{binary}}(u, v) = \begin{cases} 1 & \text{nếu } \mathbf{M}_i(u, v) > \tau \\ 0 & \text{ngược lại} \end{cases} \quad (\tau = 0.5)$$

---

## 3. Bản Chất Cân Bằng Sáng (Histogram Matching)

### 3.1 Vấn Đề Thực Tế

Khi người dùng cầm điện thoại đi vòng quanh vật thể chụp $N$ ảnh:
- Tính năng Auto-Exposure (AE) và Auto-White-Balance (AWB) liên tục thay đổi ISO, tốc độ màn trập và nhiệt độ màu.
- Kết quả: Ảnh phía trước sáng trắng, ảnh phía sau tối om hoặc ám vàng.

### 3.2 Nguyên Lý Toán Học

1. Chọn ảnh tham chiếu $I_{\text{ref}}$ (ảnh có độ phơi sáng chuẩn nhất).
2. Tính hàm phân phối tích lũy (CDF) cho từng ảnh:
   $$T_i(r) = \sum_{k=0}^{r} p_i(k)$$
   Trong đó $p_i(k)$ là xác suất pixel có giá trị cường độ $k$ (histogram chuẩn hóa).
3. Chuyển đổi không gian màu sang **LAB** (Luminance, A: Green-Red, B: Blue-Yellow) để chỉ cân bằng kênh độ sáng L, hoặc cân bằng độc lập trên từng kênh RGB:
   $$s = T_{\text{ref}}^{-1}(T_i(r))$$
4. Kết quả: Đưa dải tương phản của toàn bộ $N$ ảnh về cùng một hệ quy chiếu ánh sáng.

### 3.3 Tác Dụng Trong Pipeline

- Giúp DUSt3R so khớp điểm ảnh nhạy hơn (ít bị nhiễu do chênh lệch sáng).
- Giúp **Thành viên 5 (P5)** khi hòa trộn màu texture không bị hiện tượng **sọc loang lổ giữa các mặt**.

---

## 4. Bản Chất Chọn Lọc Số Lượng Ảnh ($N = 4 \sim 8$)

### 4.1 Toán Học Tổ Hợp — Ngăn Ngừa Tràn VRAM GPU T4

DUSt3R so khớp từng cặp ảnh. Số lượng cặp cần xử lý:

$$K = C_N^2 = \frac{N(N-1)}{2}$$

| Số ảnh $N$ | Số cặp $K$ | VRAM ước tính | Trạng thái T4 (15GB) |
|:---:|:---:|:---:|:---|
| 4 | 6 | ~2.0 GB | ✅ An toàn |
| 6 | 15 | ~5.0 GB | ✅ An toàn (Tối ưu) |
| 8 | 28 | ~9.3 GB | ⚠️ Gần giới hạn |
| 10 | 45 | ~15.0 GB | ❌ Sát giới hạn, rủi ro OOM |
| 12 | 66 | ~22.0 GB | ❌ Tràn VRAM, crash ngay lập tức |

### 4.2 Thuật Toán Uniform Subsampling

Nếu người dùng tải lên $N > 8$ ảnh:
1. Sắp xếp ảnh theo tên file (giả định thứ tự chụp tuần tự quanh vật thể).
2. Tính bước nhảy (stride): $\text{stride} = \lceil N / N_{\text{target}} \rceil$ với $N_{\text{target}} = 6$.
3. Lấy mẫu đều: $\text{selected} = [0, \text{stride}, 2 \times \text{stride}, \ldots]$ cho đến khi đủ $N_{\text{target}}$ ảnh.
4. Đảm bảo ảnh đầu tiên và ảnh cuối cùng luôn được giữ lại (để bao phủ $360^\circ$).

### 4.3 Quy Tắc Kiểm Tra Đầu Vào

- $N < 2$: Báo lỗi `ValueError("Cần ít nhất 2 ảnh cho multi-view reconstruction")`.
- $2 \le N < 4$: Cảnh báo `Warning("Số ảnh ít, chất lượng 3D có thể giảm")`, vẫn chạy.
- $4 \le N \le 8$: Giữ nguyên toàn bộ (Trạng thái lý tưởng).
- $N > 8$: Tự động Uniform Subsampling về $6 \sim 8$ ảnh, ghi log thông báo.

---

## 5. Bản Chất Định Dạng Tensor Đầu Ra Cho DUSt3R ViT

### 5.1 Chuẩn Hóa Pixel (Normalization)

DUSt3R sử dụng Vision Transformer (ViT-Large) với ImageNet normalization:

$$\hat{I}(c, h, w) = \frac{I(c, h, w) / 255.0 - \mu_c}{\sigma_c}$$

Trong đó:
- $\mu = [0.485, 0.456, 0.406]$ (mean ImageNet RGB)
- $\sigma = [0.229, 0.224, 0.225]$ (std ImageNet RGB)

### 5.2 Thứ Tự Kênh & Kiểu Dữ Liệu

| Thuộc tính | Giá trị chuẩn |
|:---|:---|
| Layout | `(N, C, H, W)` — Batch, Channel, Height, Width |
| Kênh màu | RGB (không phải BGR) |
| Kiểu dữ liệu | `torch.float32` |
| Phạm vi giá trị | Sau chuẩn hóa ImageNet: xấp xỉ $[-2.1, 2.6]$ |
| Kích thước $H, W$ | Chia hết cho 16 |
