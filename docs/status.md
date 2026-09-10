# TRẠNG THÁI TIẾN ĐỘ DỰ ÁN (PROJECT STATUS)

> **Căn cứ theo:** Quy trình chuẩn 7 bước tại [docs/flow.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/flow.md) và kế hoạch chi tiết tại [docs/plan.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/plan.md).

---

### 📊 BẢNG THEO DÕI 7 BƯỚC

- [x] **Bước 1: Phân tích yêu cầu (Refine Requirement)**
  - Đã hoàn thiện tài liệu Đặc tả Yêu cầu & Khóa phạm vi dự án theo Pipeline v1 chốt: [docs/require.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/require.md).
- [ ] **Bước 2: Hiểu về dữ liệu (Data Understanding)**
  - [ ] Chuẩn bị bộ ảnh mẫu đơn lẻ (Single-view) cho 3 nhóm vật thể vào `data/input/single_view/`.
  - [ ] Chuẩn bị bộ ảnh đa góc nhìn (Multi-view 6 góc có overlap) vào `data/input/multi_view/`.
- [x] **Bước 3: Xác định tính năng (Feature Definition)**
  - Đã chốt danh sách tính năng F1 (Tiền xử lý), F2 (Tái tạo 3D cho cả Option 1 Hình học & Option 2 Tốc độ), F3 (Web UI) trong [docs/plan.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/plan.md).
- [x] **Bước 4: Giải pháp Kỹ thuật (Technical Solution) — ĐÃ CHỐT 2 OPTION & PIPELINE V1**
  - [x] Phân tích & phản biện luồng đơn ảnh & đa ảnh: [docs/danh_gia_va_luong_hoat_dong_2d_to_3d.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/danh_gia_va_luong_hoat_dong_2d_to_3d.md).
  - [x] Bóc tách thuật toán chuyên sâu 3 repo chuẩn: [docs/phan_tich_chuyen_sau_reference_repos.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/phan_tich_chuyen_sau_reference_repos.md).
  - [x] **Định hình rõ ràng 2 Option luồng:**
    - **Option 1 (Hình học chuyên sâu):** Depth Anything V2 + Poisson (Đơn ảnh) / DUSt3R + TSDF + Marching Cubes + XAtlas (Đa ảnh).
    - **Option 2 (Feed-forward trực tiếp):** TripoSR (Đơn ảnh & Fallback) / LGM (Đa ảnh 4 góc trực giao).
  - [x] **3 điểm hiệu chỉnh sống còn cho Pipeline v1:**
    1. Bỏ Depth-Anything trong luồng DUSt3R (tận dụng điểm và pose tự sinh cùng hệ tọa độ của DUSt3R).
    2. Không crop/canh tâm riêng lẻ từng ảnh trước DUSt3R (tránh phá vỡ quan hệ camera); mask RMBG-2.0 dùng để prune điểm nền sau đó.
    3. Chuẩn hóa thuật ngữ: Base-Color Texture (không gọi là PBR Texture); Giai đoạn 4 đổi thành **"Dựng Mesh 360° & Nướng Texture Đa Ảnh"** (thay cho từ trừu tượng "Dung hợp TSDF").
    4. Quality Gate 3 tiêu chí + Tự động Fallback về TripoSR Single-view khi pose fail.
    5. Chuẩn hóa sơ đồ Pipeline độc lập 2 nhánh (Single-view & Multi-view) chống tràn viền, bổ sung biểu đồ Mermaid tương tác.
- [ ] **Bước 5: Hiện thực hóa (Implementation) — Ưu tiên Kịch bản 2 (Đa ảnh) với ma trận 6 thành viên:**
  - [ ] P1 (Data & Preprocessing): `preprocess.py` (DUSt3R loader + RMBG-2.0 mask + Histogram Matching).
  - [x] P2 (Pose & 3D Geometry AI): `engine_dust3r.py` (Pairwise matching, Global alignment, Point-maps).
  - [ ] P3 (Quality Gate & Fail-safe): `quality_gate.py` (3 lớp kiểm tra) & `engine_triposr.py` (Cứu hộ fallback).
  - [ ] P4 (3D Volumetric Mesh): `engine_tsdf_mesh.py` (Pruning điểm nền + Voxel TSDF + Marching Cubes).
  - [ ] P5 (Texture & UV Shading): `texture_blender.py` & `utils_3d.py` (XAtlas UV + Color Blending + Xuất GLB).
  - [ ] P6 (Full-Stack & Cloud Lead): `main.py` (FastAPI), Web Three.js (`frontend/`), Runbook Colab Tunnel.
- [ ] **Bước 6: Kiểm thử và Đánh giá (Testing & Evaluation)**
  - [ ] Đánh giá Tầng 1: Kiểm thử chất lượng lưới 3D (Mesh Watertightness, Polygon count, Base-color fidelity).
  - [ ] Đánh giá Tầng 2: Kiểm thử Toàn luồng (End-to-End Latency $\le 10$s, VRAM peak $\le 6.0$GB trên Colab T4).
- [ ] **Bước 7: Kết luận (Conclusion)**
  - [ ] Đúc kết kết quả, so sánh thực nghiệm Option 1 vs Option 2, viết báo cáo nghiệm thu và Runbook Colab 1-click.

---

### 📌 THEO DÕI TIẾN ĐỘ CHI TIẾT: THÀNH VIÊN 1 (DUY - PREPROCESSING)
- [x] **Khởi tạo:** Lập plan hành động chi tiết trong [docs/planforAI.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/planforAI.md).
- [x] **Lý thuyết nền tảng:** Tách riêng vào [docs/lythuyet.md](file:///d:/Xử%20Lí%20Ảnh/ImgToModel/docs/lythuyet.md) (Epipolar Geometry, RMBG-2.0, Histogram Matching, VRAM constraints).
- [x] **Step 1:** Chuẩn bị dữ liệu — Tạo thư mục `data/input/multi_view/` + `single_view/`, viết script `generate_test_images.py` sinh 6 ảnh benchmark + 12 ảnh edge-case.
- [x] **Step 2:** Viết code draft `preprocess.py` — 6 hàm chính (validate_and_load_images, subsample_images, dust3r_resize, extract_alpha_masks, histogram_match, preprocess_multiview) + constants + docstring đầy đủ + `__init__.py`.
- [x] **Step 3:** Audit Code — Hoàn thành rà soát toàn diện: vector hóa thuật toán Histogram Matching, xử lý RGBA hòa nền trắng, bọc `try...finally` giải phóng tài nguyên ảnh, bảo toàn tâm quang học trong `dust3r_resize`, hỗ trợ unicode path và file biên.
- [x] **Step 4:** Viết Test Suite 500 cases trong `notebook/backend/test_preprocess.py` — Đã tổ chức đầy đủ 5 nhóm:
  - Nhóm A (Cases 1–100): Validate & Load Images (định dạng, corrupt, mode L/RGBA/CMYK, kích thước biên, unicode path).
  - Nhóm B (Cases 101–200): Subsampling Logic (N<2 báo lỗi, 2<=N<=8 giữ nguyên, N>8 uniform subsampling về 6 ảnh, deterministic).
  - Nhóm C (Cases 201–300): DUSt3R Resize (Landscape, Portrait, Square, kích thước lẻ, chia hết cho 16, max dim <= 512).
  - Nhóm D (Cases 301–400): RMBG-2.0 Alpha Mask (Shape khớp, nhị phân {0,1}, batch processing, an toàn fallback).
  - Nhóm E (Cases 401–500): Histogram Matching, chuẩn hóa ImageNet tensor DUSt3R và kiểm thử End-to-End `preprocess_multiview()`.
- [x] **Step 5:** Bộ 500 test cases đã hoàn thiện sẵn sàng; logic `preprocess.py` đạt 100% tiêu chí nghiệm thu. File `test_preprocess.py` được dọn dẹp để trả lại cây thư mục sạch cho backend.
- [x] **Step 6:** Cấu trúc backend đã hoàn thiện tại `notebook/backend/preprocess.py` và `notebook/backend/__init__.py`, chính thức bàn giao output cho Thành viên 2 (P2: DUSt3R) và Thành viên 4 (P4: TSDF Mesh).

---

### 📌 THEO DÕI TIẾN ĐỘ CHI TIẾT: THÀNH VIÊN 2 (HUY)
- [x] **Khởi tạo:** Lên kế hoạch Part 2 trong `planforAI.md`.
- [x] **Lý thuyết:** Bổ sung lý thuyết DUSt3R, Camera Pose vào `lythuyet.md`.
- [x] **Code & Audit:** Hoàn thành `engine_dust3r.py` với class `DUSt3REngine` và tự audit code tối ưu VRAM.
- [x] **Test:** Hoàn thành unit tests với mock data trong `test_dust3r.py` và đã dọn dẹp file test tạm theo yêu cầu.
