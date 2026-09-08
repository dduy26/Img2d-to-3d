# TÀI LIỆU ĐẶC TẢ YÊU CẦU & PHẠM VI DỰ ÁN (PROJECT REQUIREMENTS & SCOPE)

> **Căn cứ tài liệu:** Tuân thủ theo **Bước 1: Phân tích yêu cầu (Refine Requirement)** trong [flow.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/flow.md).  
> **Dự án:** Chuyển đổi ảnh 2D thành mô hình 3D (2D Images to 3D Textured Mesh).  
> **Môi trường mục tiêu:** Google Colab Free (NVIDIA T4 15GB VRAM, RAM 12GB, Timeout 90 phút).

---

## 📌 1. MỤC TIÊU DỰ ÁN (PROJECT OBJECTIVES)
1. Xây dựng pipeline tự động hóa hoàn chỉnh chuyển đổi từ ảnh 2D sang mô hình 3D có đầy đủ bề mặt lưới tam giác (Mesh) và kết cấu vân màu sắc (Texture map).
2. Hỗ trợ 2 chế độ linh hoạt:
   - **Chế độ 1: Đơn ảnh (Single-view 2D → 3D):** Tập trung vào việc hiểu sâu kỹ thuật xử lý ảnh (Depth map → Point Cloud → Poisson/Mesh Reconstruction) và feed-forward nhanh.
   - **Chế độ 2: Đa ảnh (Multi-view $N$ ảnh 2D → 3D, $N = 4 \sim 8$):** Tận dụng thị sai hình học để tạo mô hình 360 độ hoàn chỉnh, khép kín (Watertight) qua TSDF Volumetric Fusion và Marching Cubes.
3. Cung cấp giao diện trực quan cho phép người dùng upload ảnh, kiểm tra ảnh tiền xử lý, kích hoạt suy luận và tương tác trực tiếp (xoay, zoom, panning) với mô hình 3D trên trình duyệt.

---

## 📌 2. XÁC ĐỊNH PHẠM VI (SCOPE BOUNDARIES)

### 2.1. Trong phạm vi (IN-SCOPE) ✅
- **Loại đối tượng:** Vật thể tĩnh đơn lập (Rigid static object) có ranh giới rõ ràng với hậu cảnh (ví dụ: đồ chơi, nhân vật mô hình, giày dép, đồ gia dụng, đồ gỗ nội thất, tượng, linh kiện...).
- **Đầu vào:**
  - Định dạng: `.jpg`, `.jpeg`, `.png`.
  - Độ phân giải: Tối thiểu $512 \times 512$ pixel.
  - Số lượng: $1$ ảnh (đơn ảnh) hoặc $4 \sim 8$ ảnh chụp quanh vật thể (đa góc nhìn).
  - Tách nền & Chuẩn hóa: Dùng DUSt3R Image Loader (resize giữ aspect ratio, không crop riêng lẻ làm méo camera parameters); RMBG-2.0 chạy song song lấy Alpha mask lọc điểm sau.
- **Mô hình & Thuật toán (Pipeline v1):**
  - Đa ảnh (Multi-view): DUSt3R (Pairwise matching + Global Alignment tự sinh Poses, Focals và 3D Point-maps đồng nhất; không trộn Depth-Anything vào).
  - Kiểm soát chất lượng (Quality Gate): Kiểm tra đồ thị Co-visibility liên thông, mật độ pixel confidence cao, và loss alignment; tự động **Fallback về TripoSR Single-view** trên ảnh nét nhất nếu fail.
  - Đơn ảnh (Single-view): TripoSR (trực tiếp) hoặc Depth-Anything-V2 + Poisson (hình học).
  - Dựng hình khối đa ảnh: Background Point Pruning (áp mask RMBG-2.0) $\to$ Open3D Scalable TSDF Integration $\to$ Marching Cubes (kín nước 360°).
  - Vân bề mặt: XAtlas UV Unwrapping + Angle-weighted RGB color blending (**Base-Color Texture**, không gọi là PBR Texture).
- **Đầu ra:**
  - Định dạng: Duy nhất **1 file nhị phân `.glb` (GLTF 2.0 Binary)** chứa toàn bộ Mesh + UV + Base-Color Texture.
- **Giao diện:** Web UI tinh gọn (Vite/React hoặc Gradio/Streamlit chạy trên Colab qua ngrok/localtunnel).

### 2.2. Ngoài phạm vi (OUT-OF-SCOPE) ❌
- **Huấn luyện mô hình từ đầu (Training from scratch):** Không train mô hình, chỉ sử dụng mô hình pre-trained SOTA phục vụ suy luận (Inference only).
- **Cảnh không gian mở (Scenes):** Không xử lý cảnh phòng ốc lớn (indoor scenes), phong cảnh ngoài trời (outdoor landscape) hoặc bài toán xe tự hành (autonomous driving scenes).
- **Vật thể chuyển động (Dynamic Objects):** Không xử lý video hoặc chuỗi ảnh có vật thể biến dạng, cử động trong quá trình chụp.
- **Hoạt hình & Khung xương (Rigging / Skinning / Animation):** Không gắn xương hay sinh animation cho mô hình.
- **Hạ tầng phức tạp:** Không xây dựng hệ thống phân tán microservices, database người dùng, cổng thanh toán.

---

## 📌 3. RÀNG BUỘC KỸ THUẬT & TÀI NGUYÊN (CONSTRAINTS)

| Ràng buộc | Ngưỡng cho phép | Giải pháp kiểm soát |
| :--- | :--- | :--- |
| **GPU VRAM** | $\le 12$ GB (trên 15 GB của T4) | Xử lý batch tuần tự cho RMBG, dùng FP16 cho Depth Anything và DUSt3R. |
| **RAM Hệ thống** | $\le 10$ GB (trên 12 GB Colab) | Giải phóng biến tạm (`gc.collect()`, `torch.cuda.empty_cache()`), không tích lũy tensor lớn trên RAM. |
| **Thời gian xử lý (Latency)** | $\le 30$ giây / mô hình | Tối ưu TSDF voxel size ($v_{\text{size}} \approx 3\text{mm} \sim 5\text{mm}$), chạy Marching Cubes đa luồng Open3D. |
| **Bản quyền (License)** | Phù hợp nghiên cứu / học thuật | Các mô hình được chọn đều có license Apache 2.0, MIT hoặc CC BY-NC 4.0. |

---

## 📌 4. TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA)

### Tầng 1: Đánh giá ở mức mô hình (Model-Level)
1. **Chất lượng hình học (Geometry):** Mô hình 3D tái tạo đúng hình dạng vật thể gốc, không bị biến dạng méo tỉ lệ; với luồng đa ảnh, lưới đa giác phải kín nước (watertight), không bị rách cạnh hay mạng nhện (triệt tiêu flying pixels nhờ Edge Filtering).
2. **Chất lượng Texture:** Màu sắc bề mặt phản ánh trung thực ảnh chụp gốc, không bị mờ nhạt hoặc nứt đường nối (seam artifacts).

### Tầng 2: Đánh giá ở mức toàn bộ luồng (Full-Flow Level)
1. **Tính trơn tru (End-to-End Reliability):** Pipeline chạy liên tục từ lúc người dùng tải ảnh lên đến khi xuất ra file `.glb` mà không phát sinh lỗi tràn bộ nhớ (Out-Of-Memory - OOM).
2. **Trải nghiệm tương tác:** File `.glb` có thể load và xoay mượt mà với 60 FPS trên khung nhìn 3D Viewer của Frontend.