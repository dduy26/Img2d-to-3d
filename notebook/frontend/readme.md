# Frontend — Web UI (P6)

Web UI tĩnh (1 file `index.html`, không cần build/node_modules) do FastAPI phục vụ cùng origin.

## Chạy local

```bash
cd notebook/backend
python -m uvicorn app:app --reload
```

Mở http://127.0.0.1:8000/ → giao diện upload + viewer 3D.

## Chạy trên Colab (1-click, mở từ máy tính)

Mở `notebook/demo_colab.ipynb` → **Runtime ▸ Change runtime type ▸ T4 GPU** → Run all.
Cell 4 in ra URL Cloudflare `https://xxxx.trycloudflare.com` → mở URL đó để dùng Web UI (Swagger ở `/docs`).

## Luồng dữ liệu

```
index.html ──POST /generate-3d/ (multipart, N ảnh)──► app.py
                                                        │
                            ┌───────────────────────────┴───────────────────────────┐
                            │ N=1 → TripoSR            N≥2 → P1→P2→P3 Quality Gate   │
                            │                              PASS → P4 TSDF → P5 GLB  │
                            │                              FAIL → TripoSR fallback  │
                            └───────────────────────────┬───────────────────────────┘
                    {"output_file": "outputs/result_x.glb"}
                                                        │
index.html ◄──GET /outputs/result_x.glb (Three.js GLTFLoader)── StaticFiles
```

## Vì sao chỉ có index.html, không có React/Vite

`plan.md` ghi React + Vite. Nhưng UI chỉ cần: kéo thả ảnh → gọi 1 API → xem 1 file GLB.
Ladder: three.js đã đủ, thêm Vite/React = thêm `npm install`, `npm run build`, một
`node_modules/` 200MB trong repo mà không đổi tính năng nào. Khi nào cần routing/state
phức tạp mới tách build step (lúc đó thay `index.html` bằng output của Vite là xong,
API contract ở trên không đổi).

## API contract (đã có sẵn từ P1–P5, P6 không sửa)

| Endpoint | Input | Output |
| :-- | :-- | :-- |
| `POST /generate-3d/` | `files`: 1 hoặc 4–8 ảnh | `status, mode, pipeline_type, quality_passed, gate_reason, execution_time_seconds, output_file` |
| `POST /generate-3d/single/` | `file`: 1 ảnh | `status, execution_time_seconds, output_file` |
| `GET /api/health` | — | `{status, device, frontend}` |
| `GET /outputs/<name>.glb` | — | file nhị phân GLB |
