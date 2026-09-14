# Dữ liệu & độ phủ góc chụp — vì sao mesh thiếu mặt

> Tài liệu P6 (Thông). Trả lời câu hỏi: **"tại sao output chỉ ra 1 mặt, mất đế và phần vòm?"**

## 1. Kết luận ngắn

Thiếu mặt **không phải lỗi code P4/P5**. TSDF là fusion thuần hình học: *mặt không camera nào thấy thì
không dựng được*. Bộ ảnh GSO cũ (5 ảnh) chỉ đi **một vòng ngang** ⇒ mặt đáy không ai thấy ⇒ mesh hở.

Hai lỗi code thật đi kèm đã sửa (xem §4):

| Lỗi | Triệu chứng bạn thấy | Trạng thái |
| :-- | :-- | :-- |
| P5 lấy màu không truyền mask nền | Texture **bạc màu** (nền trắng bake lên vật) | ✅ đã sửa |
| P3 không kiểm camera có bao quanh | `quality_passed: true` cho bộ ảnh không thấy đáy | ✅ đã bổ sung (tầng keo P6) |

## 2. Số đo bộ ảnh GSO cũ (`data/gso/`, 5 ảnh)

| Ảnh | Vật chiếm khung | Nền |
| :-- | --: | --: |
| `view_01` | 12.8 % | 87.2 % |
| `view_02` | 14.1 % | 85.9 % |
| `view_03` | 5.8 % | 94.2 % |
| `view_04` | 14.5 % | 85.5 % |
| `view_05` | 6.2 % | 93.8 % |

Góc chụp: 1 ảnh 3/4, 2 ảnh bên **cùng độ cao**, 1 mũi, 1 gót. **Không ảnh nào thấy toàn bộ mặt dưới.**

## 3. Bộ dữ liệu thay thế — DX.GL Objaverse-1K

Nguồn: <https://huggingface.co/datasets/dxgl/objaverse-1k> — CC-BY 4.0, tải trực tiếp, không cần đăng ký.

| Thuộc tính | Giá trị | Vì sao quan trọng với ta |
| :-- | :-- | :-- |
| Số vật | 1026 | chọn được giày / ghế / cốc… |
| **Góc/vật** | **196** | phủ **cả mặt cầu (±89°)** → có ảnh từ trên xuống *và* từ dưới lên |
| Độ phân giải | 1024×1024 | gấp đôi GSO (640×460) |
| Kèm theo | **mask PNG** + poses (nerfstudio) + depth 8/16-bit + normals | mask ⇒ **không cần HF_TOKEN** |
| Dung lượng | ~130 MB / vật | tải 1 vật trong ~1 phút |
| Layout mỗi zip | `images/frame_NNNNN.png`, `masks/…`, `transforms.json`, `points3D.ply` | |

So với các nguồn khác:

| Nguồn | Góc/vật | Vấn đề |
| :-- | --: | :-- |
| Google Scanned Objects (bộ cũ) | **5**, một vòng ngang | hở đáy — chính là lỗi đang gặp |
| **DX.GL Objaverse-1K** | **196**, cả mặt cầu | ✅ chọn dùng |
| OmniObject3D | 100 | tốt, nhưng phải đăng ký OpenDataLab, ~1.2 TB nếu tải cả bộ |
| CO3D (Meta) | nhiều, ảnh chụp thật | bản nhỏ đã 8.9 GB, license CC BY-NC |
| Objaverse_2d_renders (ShapeSplats) | 72, nửa mặt cầu trên | vẫn **không có đáy** |

**Tốt nhất vẫn là tự chụp**: nền trơn → đi vòng quanh **8 tấm cách đều ~45°**, và
**thêm 2–4 tấm chúc từ trên xuống + 2–4 tấm từ dưới lên**, giữ nguyên khoảng cách máy, **không xoay vật**.

## 4. Đã sửa gì

1. **`utils_3d.sample_rgb_nearest` nhận thêm `mask`** — pixel rơi vào nền trả `NaN` để nơi gọi BỎ mẫu.
2. **`texture_blender`** — `blend_colors_for_vertices`, `bake_texture_from_views`, `process_and_export`
   truyền `alpha_masks` xuống. Không truyền (mặc định `None`) thì giữ nguyên hành vi cũ ⇒ test cũ không đổi.
3. **`app.py`** — P5 gọi `process_and_export(..., alpha_masks=preprocess_result["alpha_masks"])`.
4. **`preprocess.py`** — giữ kênh alpha của ảnh gốc (`validate_and_load_images` trả thêm khoá `alpha`);
   `masks_from_alpha()` dựng mask từ chính kênh đó, dùng **thay** RMBG-2.0 khi mọi ảnh đều có alpha.
   ⇒ ảnh render nền trong suốt **không cần HF_TOKEN**; RMBG chỉ còn cho ảnh chụp thật.
5. **`app.audit_view_coverage()`** — đo camera có **bao quanh** vật không, bằng `|cos|` lớn nhất giữa
   hướng camera và 3 trục chính của vật (SVD). In ra log + trả trong `view_coverage` của API.
   Đặt ở tầng keo P6 nên **không sửa `quality_gate.py` của TV3**.
6. **`notebook/demo_colab.ipynb`** — Cell 1 trỏ đúng nhánh `P6-tsdf-fix`; Cell 2 tải 8 ảnh phủ mặt cầu
   từ Objaverse-1K (chọn góc bằng farthest-point sampling, không cần biết trục nào là "trên");
   Cell 4 ưu tiên `data/objaverse/`.

## 5. Đọc log thế nào

```
[P3→P6] Độ phủ gốc chụp: góc nhìn tốt nhất theo 3 trục vật: [19, 68, 34]°
[P3→P6] Bộ ảnh KHÔNG bao quanh vật đủ -> phần bị thiếu là do GÓC CHỤP, không phải lỗi P4/P5.
```
Con số là **góc lệch** giữa hướng camera tốt nhất và từng trục vật. Trục nào lệch > ~53° là
hướng đó **không ai thấy**. Với vật đặt đứng, trục bị lệch lớn chính là **mặt đáy**.

## 6. Chạy lại kiểm thử

```bash
python notebook/backend/test_p6_texture_mask.py   # 11 phép kiểm: mask nền + độ phủ gốc chụp
python notebook/backend/utils_3d.py               # tự kiểm sample_rgb_nearest
python notebook/backend/test_tsdf_synthetic.py    # P4 đối chiếu ground truth tổng hợp
```
