# Hệ Thống Kiểm Thử Toàn Diện (ImgToModel Test Suite)

Thư mục này chứa toàn bộ các bài kiểm thử chuẩn hóa (Unit Tests & End-to-End Integration Tests) bao quát tất cả các vấn đề kỹ thuật và phân hệ trong pipeline tái tạo 3D từ ảnh 2D theo chuẩn NVIDIA:

---

## 1. Cấu Trúc Thư Mục `tests/`

```text
tests/
├── data/
│   ├── objaverse_apple/                  # Dữ liệu 4 góc chụp thực tế từ dxgl/objaverse-1k (apple.zip)
│   │   ├── front.png, right.png, back.png, left.png
│   └── objaverse_train/                  # Dữ liệu vật thể bất đối xứng từ dxgl/objaverse-1k (train.zip)
│       ├── images/ (front, right, back, left)
│       ├── masks/  (ground truth masks)
│       └── transforms.json (ground truth camera parameters)
│
├── test_p1_optical_normalization.py     # Test cân bằng quang học CIE Lab (Luminance-only, bảo toàn Hue/Chroma)
├── test_p1_background_segmentation.py   # Test tách nền đa tầng: Native Alpha & Studio Lab Chroma + Otsu
├── test_p1_aspect_ratio_consistency.py  # Test bảo toàn tỉ lệ hình học Front vs Side (triệt tiêu lỗi squish/crop)
├── test_p1_camera_intrinsics_math.py    # Test chứng minh toán học bù trừ ma trận Camera Intrinsics K -> K'
├── test_p4_space_carving_tsdf.py        # Test lưới 3D thể tích kín nước (Watertight: True, 0 boundary edges, 1 khối)
├── test_p5_texture_blending.py          # Test trải UV & nướng màu góc nhìn Fresnel với K' bù trừ
├── test_e2e_objaverse_train.py          # Test tích hợp đầu-cuối toàn pipeline trên ảnh Objaverse thực tế
└── run_all_tests.py                     # Bộ chạy kiểm nghiệm tự động một lệnh duy nhất (All-in-one Runner)
```

---

## 2. Các Vấn Đề Kỹ Thuật Được Kiểm Tra

| File Kiểm Thử | Vấn Đề Giải Quyết & Kiểm Tra | Tiêu Chí Đo Lường Thành Công |
| :--- | :--- | :--- |
| `test_p1_optical_normalization.py` | Cân bằng độ sáng không làm lệch tông màu vật thể | Sai lệch góc sắc tướng $\Delta\text{Hue} \le 2.0^\circ$, độ sáng bám sát anchor view. |
| `test_p1_background_segmentation.py` | Tách nền studio trắng/đen có bóng mờ không dùng ngưỡng cứng | Giữ nguyên Native Alpha PNG RGBA; tách phông studio đạt phân ngưỡng sạch, không viền hộp. |
| `test_p1_aspect_ratio_consistency.py` | Khắc phục lỗi mặt bên (Left/Right) bị co dãn tỉ lệ theo mặt trước | Global Uniform Scale đồng nhất $100\%$, tỉ lệ $w/h$ vật thể dài ngang được bảo toàn 1:1. |
| `test_p1_camera_intrinsics_math.py` | Bù trừ ma trận $K_i'$ khi căn giữa vật thể lệch tâm | Sai số tọa độ điểm ảnh giữa canvas và tia chiếu $\pi(K_i', R, t, X) < 10^{-5}$ pixel. |
| `test_p4_space_carving_tsdf.py` | Triệt tiêu vĩnh viễn "phần dư" (fins/wings chìa ra ngoài) | `watertight == True`, `boundary_edges == 0`, `components == 1`, thể tích dương. |
| `test_p5_texture_blending.py` | Gán màu đỉnh với tâm quang học bù trừ $c_x', c_y'$ | Xuất file `.glb` hợp lệ, đỉnh nhận đúng màu theo góc chiếu camera, không bị mặc định màu xám. |
| `test_e2e_objaverse_train.py` | Vận hành toàn bộ chuỗi P1 $\to$ P5 trên dataset Objaverse | Trả về file `.glb`, mesh đạt chuẩn Slicer in 3D (xanh lá cây), hoàn tất trong $< 5$ giây. |

---

## 3. Cách Chạy Kiểm Thử

### Chạy toàn bộ test suite (Khuyến nghị):
```bash
python tests/run_all_tests.py
```

### Chạy từng bài test riêng lẻ:
```bash
python tests/test_p1_aspect_ratio_consistency.py
python tests/test_p1_camera_intrinsics_math.py
python tests/test_e2e_objaverse_train.py
```
