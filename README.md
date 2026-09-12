# Multi-View 2D → 3D Pipeline

Chuyển đổi N ảnh 2D thành mô hình 3D trên Google Colab (GPU T4).

## Cấu trúc

```
src/
├── preprocess.py        # Validate ảnh, letterbox, background mask
├── dust3r_pipeline.py   # DUSt3R inference + quality gate
├── fusion.py            # TSDF Fusion → mesh → export GLB
└── fallback.py          # TripoSR single-view khi DUSt3R fail

tests/
├── test_preprocess.py
├── test_dust3r_pipeline.py
├── test_fusion.py
└── test_fallback.py

notebook/
└── multi_view_colab.ipynb   # Notebook chạy trên Colab
```

## Chạy tests (local, không cần GPU)

```bash
cd Img2d-to-3d
python -m pytest tests/ -v
```

notebook/frontend/
└── index.html               # Web UI (Three.js CDN) — 1 file, không build
```

## Chạy nhanh (local, CPU)

```bash
cd notebook/backend && python -m uvicorn app:app
```
Mở http://127.0.0.1:8000/ → upload ảnh → tải `.glb`. Chạy `python notebook/backend/test_pipeline.py` để kiểm thử 7/7.

## Chạy trên Colab GPU

Mở `notebook/demo_colab.ipynb` → Runtime ▸ T4 GPU → Run all → mở URL Cloudflare in ra ở Cell 4.

## Quality Gate

DUSt3R output bị reject nếu:
- Ít hơn 30% pixel có confidence > 0.5 trên bất kỳ ảnh nào
- Pairwise overlap trung bình < 2%
- Đồ thị ảnh–ảnh không liên thông (có ảnh cô lập)

Khi fail → fallback sang TripoSR single-view trên ảnh tốt nhất.
