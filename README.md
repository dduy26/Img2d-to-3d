# 🌟 2D → 3D AI Reconstruction Studio

Hệ thống tái tạo mô hình 3D nguyên khối từ ảnh 2D trên Google Colab (GPU Tesla T4) và Local Client.

---

## 🚀 Kiến Trúc Hai Luồng AI SOTA (Dual-AI Engine)

1. **⚡ Luồng 1: Đơn Ảnh (Single-View - 1 ảnh)**
   - **Mô hình:** **TripoSR** (Stability AI / VAST-AI Research).
   - **Kiến trúc:** Vision Transformer (ViT) + Triplane NeRF + Marching Cubes thuần túy.
   - **Đặc tính:** Sinh mô hình 3D hoàn chỉnh chỉ trong **~1.5 giây**, 100% Watertight kín nước.

2. **🌐 Luồng 2: Đa Ảnh (Multi-View - N ảnh, N ≥ 2)**
   - **Mô hình:** **Tencent Hunyuan3D-2mv** (Tencent AI Lab).
   - **Kiến trúc:** DiT Flow Matching Pipeline (`tencent/Hunyuan3D-2mv`, `hunyuan3d-dit-v2-mv`).
   - **Đặc tính:** Hỗ trợ trực tiếp chuỗi góc chụp đa hướng (`front`, `right`, `back`, `left`), tích hợp thuật toán gán góc Hungarian Bipartite Matching (P1), sinh lưới 3D đặc kín nước 100% (Watertight Solid Mesh), chuẩn CAD/Game asset, xuất duy nhất **1 file nhị phân `.glb`**.

---

## 📁 Cấu Trúc Dự Án

```text
notebook/
├── backend/
│   ├── app.py                 # FastAPI Web & API Server
│   ├── engine_hunyuan3d.py    # Tencent Hunyuan3D-2mv DiT Multi-View Engine
│   ├── engine_depth.py        # Depth-Anything-V2 Engine
│   ├── preprocess.py          # P1: Chuẩn hóa quang học, Alpha Mask, Hungarian Matching
│   ├── quality_gate.py        # P3: Kiểm định chất lượng đa góc nhìn
│   ├── texture_blender.py     # P5: Hòa trộn màu Fresnel & UV Baking
│   └── utils_3d.py            # Tiện ích xử lý ma trận và hình học 3D
├── frontend/
│   └── index.html             # Web UI 3D tương tác (Three.js CDN)
├── demo_colab.ipynb           # Sổ tay chạy trọn gói trên Google Colab (T4 GPU)
└── local_app.py               # Trình khách Gradio UI chạy cục bộ

tests/
├── run_all_tests.py           # Bộ chạy kiểm thử toàn diện (24/24 tests PASS)
├── test_p1_optical_normalization.py
├── test_p1_background_segmentation.py
├── test_p1_aspect_ratio_consistency.py
├── test_p1_camera_intrinsics_math.py
├── test_p4_space_carving_tsdf.py
├── test_p5_texture_blending.py
├── test_e2e_objaverse_train.py
└── test_p6_hunyuan3d_engine.py
```

---

## 🧪 Chạy Kiểm Thử (Tự Động & Độc Lập)

Toàn bộ 24 bài kiểm thử đơn vị và tích hợp E2E chạy độc lập trên CPU không yêu cầu GPU:

```bash
python tests/run_all_tests.py
```

---

## 💻 Chạy Cục Bộ (Local Backend)

```bash
uvicorn notebook.backend.app:app --host 0.0.0.0 --port 8000
```
- Mở trình duyệt tại: `http://localhost:8000/` để tương tác trực tiếp với giao diện Web UI 3D.
- Hoặc khởi chạy ứng dụng Gradio Client:
  ```bash
  python notebook/local_app.py
  ```

---

## ☁️ Chạy Trên Google Colab (GPU T4)

1. Mở file `notebook/demo_colab.ipynb` trên Google Colab.
2. Chọn Runtime: **Python 3 ▸ T4 GPU**.
3. Bấm **Run all**: Hệ thống tự động cài đặt môi trường, nạp mô hình và tạo đường dẫn công khai qua **Cloudflare Tunnel**.
4. Dán URL Cloudflare vào Local Gradio App hoặc dùng trực tiếp viewer 3D trong Colab.
