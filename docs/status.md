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
  - [ ] P2 (Pose & 3D Geometry AI): `engine_dust3r.py` (Pairwise matching, Global alignment, Point-maps).
  - [ ] P3 (Quality Gate & Fail-safe): `quality_gate.py` (3 lớp kiểm tra) & `engine_triposr.py` (Cứu hộ fallback).
  - [ ] P4 (3D Volumetric Mesh): `engine_tsdf_mesh.py` (Pruning điểm nền + Voxel TSDF + Marching Cubes).
  - [ ] P5 (Texture & UV Shading): `texture_blender.py` & `utils_3d.py` (XAtlas UV + Color Blending + Xuất GLB).
  - [ ] P6 (Full-Stack & Cloud Lead): `main.py` (FastAPI), Web Three.js (`frontend/`), Runbook Colab Tunnel.
- [ ] **Bước 6: Kiểm thử và Đánh giá (Testing & Evaluation)**
  - [ ] Đánh giá Tầng 1: Kiểm thử chất lượng lưới 3D (Mesh Watertightness, Polygon count, Base-color fidelity).
  - [ ] Đánh giá Tầng 2: Kiểm thử Toàn luồng (End-to-End Latency $\le 10$s, VRAM peak $\le 6.0$GB trên Colab T4).
- [ ] **Bước 7: Kết luận (Conclusion)**
  - [ ] Đúc kết kết quả, so sánh thực nghiệm Option 1 vs Option 2, viết báo cáo nghiệm thu và Runbook Colab 1-click.