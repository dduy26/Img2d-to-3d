# TÀI LIỆU TRAO ĐỔI & THIẾT KẾ KIẾN TRÚC DỰ ÁN 2D → 3D

> **Mục đích tài liệu:** Ghi lại chi tiết quá trình phân tích, đánh giá phản biện kế hoạch ban đầu và xây dựng luồng kiến trúc chuẩn (Pipeline), lựa chọn mô hình, thuật toán cùng tài nguyên thực thi cho dự án chuyển đổi ảnh 2D sang mô hình 3D trên Google Colab Free (NVIDIA T4).

---

## 📌 PHẦN 1: ĐÁNH GIÁ & PHẢN BIỆN KẾ HOẠCH BAN ĐẦU

### 🤖 Trợ lý AI (AI Assistant)

> **Nội dung trao đổi / Đặt vấn đề:**
>
> "OK, mình đã đọc kỹ kế hoạch rồi. Để mình phân tích thẳng thắn luôn — vì đây là yêu cầu phân tích và đánh giá, mình sẽ nói đầy đủ.
>
> **Đánh giá kế hoạch 2D → 3D**
>
> **Tổng quan:**
> Kế hoạch về mặt tổng thể là đúng hướng, các thuật toán được chọn đều là state-of-the-art và phù hợp với Colab free (T4, 15GB VRAM). Tuy nhiên có một số vấn đề cần sửa mà leadership của bạn đang hiểu sai hoặc diễn đạt chưa chính xác.
>
> **Phân tích từng bước:**
>
> 1. **Tách nền (Background Removal) — ✅ Đúng, nhưng cần bổ sung chi tiết:**
>    - Chọn đúng: RMBG-2.0 (BriaAI) hoặc RMBG-1.4 đều chạy tốt trên T4, <0.5GB VRAM, <1 giây.
>    - _Nhưng cần lưu ý:_
>      - RMBG-2.0 không phải "thay thế cho Rembg cũ" — Rembg là toolkit wraps nhiều model, RMBG là model riêng. Hai thứ khác nhau hoàn toàn.
>      - Không bắt buộc phải tách nền. TripoSR thực tế chạy tốt hơn VỚI nền phức tạp hơn là nền trong suốt — vì model được train trên ảnh có nền. Tách nền chỉ cần thiết nếu vật thể có nền rất ồn (nhiều vật thể xen kẽ). Leadership đang khẳng định "bắt buộc" là quá cứng.
>      - Output: Ảnh PNG với alpha channel là đúng, nhưng bạn cần đảm bảo ảnh RGB bên trong vẫn giữ nguyên màu — vì sau này dùng cho texture projection.
>    - _Đề xuất sửa:_ Bước này nên là tùy chọn, không bắt buộc. Thêm logic: nếu ảnh đầu vào đã rõ vật thể → bỏ qua bước này, tiết kiệm thời gian.
> 2. **Suy luận 3D — ⚠️ Có chỗ hiểu sai nghiêm trọng:**
>    - **Chế độ 1: Depth Anything V2 — ⚠️ NHẦM LẪN QUAN TRỌNG:**
>      - Depth Anything V2 KHÔNG tạo mesh. Nó chỉ tạo depth map — một ảnh grayscale mỗi pixel là khoảng cách tới camera.
>      - Depth map ≠ Point Cloud ≠ Mesh. Depth map chỉ là bước 1 trong luồng reconstruction, KHÔNG PHẢI "luồng reconstruction hoàn chỉnh."
>      - Sau khi có depth map, bạn vẫn cần thêm:
>        - Back-projection depth map + camera intrinsics → Point Cloud (Open3D hoặc Trimesh).
>        - Point Cloud → Mesh (bằng Poisson Surface Reconstruction hoặc Ball Pivoting Algorithm).
>        - Texture projection từ ảnh gốc lên mesh.
>      - Leadership viết như thể Depth Anything tự chạy xong ra mesh — sai. Phần mesh generation step 3 trong kế hoạch chỉ đúng khi dùng depth map làm input, nhưng leadership đang tách rời 2 thứ mà thực ra chúng phụ thuộc nhau.
>      - VRAM đúng: Depth Anything V2 Small ~95MB, ~0.2s trên T4. ✅
>    - **Chế độ 2: TripoSR — ✅ Đúng hướng, nhưng cần hiểu đúng:**
>      - Đúng: TripoSR là feed-forward, không cần optimization lặp, VRAM ~4-6GB trên T4, chạy được.
>      - _Nhưng cần hiểu:_
>        - TripoSR đã bao gồm luôn bước mesh extraction — output ra mesh luôn (không cần bạn tự làm Marching Cubes hay Poisson riêng). Plan đang viết như thể TripoSR chỉ ra "density field" rồi bạn phải tự convert → sai.
>        - TripoSR output ra textured mesh luôn (có UV map + texture), không phải NeRF thuần. Plan đang nhầm khái niệm.
>        - SV3D (Stable Video 3D) cần multi-view input — không chạy từ 1 ảnh duy nhất được. Nếu bạn chỉ có 1 ảnh 2D → bỏ SV3D, chỉ dùng TripoSR.
>        - TripoSR không free license hoàn toàn — model weights dùng license riêng, cần check trước khi commercialize (nhưng với demo sinh viên thì OK).
> 3. **Mesh Generation — ⚠️ Sai lệch lớn giữa plan và thực tế:**
>    - Vấn đề chính: Leadership viết như thể mesh generation là bước riêng biệt, độc lập. Thực tế:
>      - _Nếu dùng Depth Anything V2:_ CÓ THỰC SỰ cần Point Cloud → Mesh (Poisson/BPA). NHƯNG phải viết code thủ công: `depth_to_pointcloud` → `trimesh`.
>      - _Nếu dùng TripoSR:_ KHÔNG CẦN bước này — TripoSR đã return mesh trực tiếp. Bạn chỉ cần load mesh từ output.
>    - Marching Cubes không phù hợp cho case depth map → mesh. Poisson Surface Reconstruction hoặc Ball Pivoting Algorithm mới đúng. Marching Cubes dùng cho volume data (CT scan, MRI) — không phải depth map.
>    - _Đề xuất sửa:_ Bỏ Marching Cubes. Dùng:
>      - Depth Anything path: Open3D `create_from_point_cloud_poisson()` hoặc Trimesh `ball_pivoting`.
>      - TripoSR path: Bỏ bước này hoàn toàn, dùng output mesh trực tiếp.
> 4. **Render & Export — ✅ Đúng, nhưng phức tạp hóa không cần thiết:**
>    - Đúng: `.glb` là format tốt nhất, nhúng mesh + texture + UV map. ✅
>    - _Nhưng:_
>      - XAtlas / BFF UV Unwrapping — không cần làm thủ công. TripoSR đã unwrap sẵn UV. Depth Anything path thì UV unwrapping rất đơn giản (chỉ có 1 ảnh 2D, chỉ cần project texture lên mesh bằng camera projection — không cần atlas phức tạp).
>      - Texturing bằng projection từ ảnh gốc — đúng hướng, nhưng cần hiểu: nếu depth map path, texture quality phụ thuộc vào camera intrinsics bạn ước tính đúng không.
>
> **Các vấn đề khác cần lưu ý:**
>
> - **Colab Free — thực tế giới hạn:**
>   - GPU: T4 15GB VRAM ✅ (TripoSR ~5GB, Depth Anything ~0.5GB — OK).
>   - RAM: ~12GB → TripoSR cần ~8GB RAM khi load, gần chạm giới hạn.
>   - Session timeout: ~90 phút không tương tác → cần checkpoint model sau khi download.
>   - Disk: ~78GB → model weights + output OK.
>   - Kết nối internet: HuggingFace download chậm, cần `hf` CLI hoặc `gdown` để cache.
> - **Không train model — Plan đã đúng:** Cả Depth Anything V2 và TripoSR đều là pre-trained models, inference-only. Bạn KHÔNG cần train. Plan đúng ở chỗ này. ✅
> - **"Hiểu được code source" — đây là khó khăn lớn nhất:**
>   - TripoSR repo: code tương đối clean, dùng PyTorch, dễ đọc. ✅
>   - Depth Anything V2: code đơn giản, inference vài dòng code. ✅
>   - Nhưng phần chuyển depth map → mesh cần viết thêm code thủ công — plan không mention phần này. ⚠️

### 🔄 Kế hoạch sửa đề xuất

#### • Luồng 1: Depth Reconstruction (Hướng tiếp cận hình học, hiểu sâu bản chất)

```
[Ảnh 2D Input]
      ↓
[RMBG-2.0] (Tách nền - Tùy chọn)
      ↓
[Depth Anything V2 Small] (Dự đoán Depth Map)
      ↓
[Open3D: Back-projection] (Depth Map → Point Cloud 3D)
      ↓
[Open3D: Poisson Reconstruction] (Point Cloud → Triangular Mesh 3D)
      ↓
[Camera Projection] (Chiếu ảnh 2D gốc tạo Texture + UV)
      ↓
[Xuất file .glb hoàn chỉnh]
```

#### • Luồng 2: Feed-forward Direct (Hướng tiếp cận Deep Learning trực tiếp, nhanh gọn)

```
[Ảnh 2D Input]
      ↓
[RMBG-2.0] (Tách nền - Tùy chọn)
      ↓
[TripoSR Transformer] (Dự đoán trực tiếp ra Textured Mesh + UV)
      ↓
[Xuất file .glb hoàn chỉnh]
```

> **Điểm khác biệt cốt lõi:** TripoSR đã tích hợp sẵn toàn bộ bước tái tạo lưới và nướng texture, trong khi Depth Anything V2 yêu cầu nhóm phải tự lập trình pipeline chuyển đổi từ depth map sang mesh.

---

### 📊 Bảng kết luận đánh giá ban đầu

| Hạng mục phân tích            |    Đánh giá    | Chi tiết nhận xét & Hướng xử lý                                                                               |
| :---------------------------- | :------------: | :------------------------------------------------------------------------------------------------------------ |
| **Thuật toán được chọn**      |   ✅ Phù hợp   | Rất phù hợp với tài nguyên Colab free (GPU NVIDIA T4 15GB VRAM).                                              |
| **Hiểu về VRAM requirements** |  ✅ Chuẩn xác  | TripoSR chiếm khoảng 4-6GB VRAM (mức trung bình khá, không phải "siêu nhẹ" như plan gốc mô tả).               |
| **Hiểu về luồng xử lý**       |   ⚠️ Cần sửa   | **Depth Anything V2 không tạo mesh** mà chỉ tạo depth map. TripoSR đã bao gồm sẵn khâu tạo mesh.              |
| **Mesh generation step**      |  ⚠️ Sai lệch   | **Marching Cubes không phù hợp** cho depth map/point cloud. Cần thay bằng **Poisson Surface Reconstruction**. |
| **Bắt buộc xóa nền**          |  ⚠️ Quá cứng   | Nên chuyển thành **Tùy chọn (Optional)**; TripoSR chạy tốt ngay cả trên ảnh có nền phức tạp.                  |
| **Format xuất file**          |  ✅ Chính xác  | Chọn định dạng **.glb** là tối ưu nhất vì nhúng toàn bộ mesh, UV map và texture vào 1 file.                   |
| **Khả thi trên Colab free**   | ✅ Rất khả thi | Cả 2 luồng đều kiểm chứng thực thi tốt trên môi trường Colab miễn phí.                                        |

---

## 📌 PHẦN 2: TỔNG KẾT KẾ HOẠCH & CÁC ĐIỂM CẦN SỬA

### 👤 Người dùng (User) - Câu hỏi 1

> **"Tổng kết lại giúp tôi , và luồng hoạt động của nó + những điểm cần sửa và sửa giúp tôi"**

---

### 🤖 Trợ lý AI (AI Assistant) - Trả lời 1

Dưới đây là phần tổng kết lại toàn bộ kế hoạch, luồng hoạt động đã sửa và các điểm sai lệch cần hiệu chỉnh:

### 1. Luồng hoạt động (Đã sửa)

- **Luồng 1: Reconstruction (Hiểu kỹ thuật hơn)**

  ```
  Ảnh 2D → [Tùy chọn: RMBG-2.0 xóa nền]
         → [Depth Anything V2 Small: depth map]
         → [Open3D: depth map → Point Cloud → Poisson mesh]
         → [Texture projection: chiếu màu từ ảnh gốc]
         → Xuất .glb
  ```

  - **Thời gian thực thi:** ~2-3 giây total.
  - **VRAM sử dụng:** ~1GB.

- **Luồng 2: Feed-forward (Nhanh, ít code hơn)**

  ```
  Ảnh 2D → [Tùy chọn: RMBG-2.0 xóa nền]
         → [TripoSR: ảnh vào → mesh trực tiếp (đã có UV + texture)]
         → Xuất .glb
  ```

  - **Thời gian thực thi:** ~1 giây total.
  - **VRAM sử dụng:** ~6GB.

---

### 2. Bảng tổng hợp những điểm cần sửa trong plan gốc

|   #   | Lỗi trong plan gốc                                          | Hiệu chỉnh chuẩn xác                                                                                                    |
| :---: | :---------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------- |
| **1** | _"Bắt buộc phải tách nền"_                                  | **Tùy chọn (Optional).** TripoSR chạy tốt hơn với nền phức tạp. Chỉ cần dùng khi ảnh có nhiều vật thể xen kẽ gây nhiễu. |
| **2** | _Depth Anything V2 tạo mesh_                                | **Sai.** Model chỉ tạo depth map (ảnh grayscale). Cần bổ sung code: `depth map` → `point cloud` → `mesh` bằng Open3D.   |
| **3** | _TripoSR ra "NeRF/Plücker coordinates" rồi phải tự convert_ | **Sai.** TripoSR tạo thẳng mesh hoàn chỉnh + UV + texture. Bước 3 & 4 đã được tích hợp sẵn bên trong.                   |
| **4** | _SV3D dùng được với 1 ảnh_                                  | **Sai.** SV3D cần multi-view input (nhiều góc nhìn). Loại bỏ nếu chỉ xử lý đơn ảnh 2D.                                  |
| **5** | _Marching Cubes dùng cho depth map → mesh_                  | **Sai thuật toán.** Dùng **Poisson Surface Reconstruction** hoặc **Ball Pivoting Algorithm (BPA)**.                     |
| **6** | _TripoSR nặng 4-6GB VRAM là "siêu nhẹ"_                     | 4-6GB trên GPU T4 15GB là khả thi nhưng không phải siêu nhẹ; đặc biệt tải nặng RAM hệ thống (~8GB / 12GB RAM Colab).    |

---

### 3. Khuyến nghị cho nhóm sinh viên

- **Nếu muốn hiểu kỹ thuật + học được nhiều:** Dùng **Luồng 1 (Depth Anything + Open3D)**. Nhóm sẽ phải tự tay lập trình khâu mesh extraction thủ công — đây chính là phần giá trị nhất để hiểu bản chất chuyển đổi depth map thành 3D.
- **Nếu muốn demo nhanh, kết quả đẹp:** Dùng **Luồng 2 (TripoSR)**. Ít code hơn, output đã có texture sẵn, nhưng mang tính chất black-box nhiều hơn.
- **Tính khả thi:** Cả 2 luồng đều kiểm chứng chạy tốt trên Colab free T4. ✅

---

## 📌 PHẦN 3: LUỒNG HOẠT ĐỘNG CHUẨN CỦA DỰ ÁN (CHI TIẾT MÔ HÌNH & THUẬT TOÁN)

### 👤 Người dùng (User) - Câu hỏi 2

> **"cho tôi luồng hoạt động chuẩn của dự án + thuật toán + mô hình + những điểm cần thiết"**

---

### 🤖 Trợ lý AI (AI Assistant) - Trả lời 2

Dưới đây là thiết kế kiến trúc chuẩn hóa toàn diện cho dự án **2D → 3D** với đầy đủ tham số kỹ thuật, thuật toán, mô hình và lưu ý môi trường:

### 1. Tổng quan kiến trúc Pipeline

```
Ảnh 2D (JPG/PNG)
    ↓
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 1: TIỀN XỬ LÝ ẢNH                                │
│  - Thuật toán: Image Segmentation                       │
│  - Mô hình: RMBG-2.0 (BriaAI)                           │
│  - Thư viện: transformers + PIL                         │
│  - VRAM: ~0.5GB | Thời gian: < 1s                       │
└─────────────────────────────────────────────────────────┘
    ↓
Ảnh PNG đã xử lý (RGB + Alpha tùy chọn)
    ↓
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 2: SUY LUẬN 3D (LỰA CHỌN 1 TRONG 2 LUỒNG)         │
│                                                         │
│  [LUỒNG A] Depth Reconstruction (Kỹ thuật sâu)         │
│  [LUỒNG B] Feed-forward Direct (Nhanh, nguyên khối)    │
└─────────────────────────────────────────────────────────┘
    ↓
Mesh 3D (Vertices + Faces + UV Map + Texture)
    ↓
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 3: XUẤT FILE HOÀN CHỈNH                           │
│  - Định dạng: .glb (GLTF Binary)                        │
│  - Thư viện: trimesh                                    │
│  - VRAM: 0 | Thời gian: < 10ms                          │
└─────────────────────────────────────────────────────────┘
    ↓
File .glb (Mở trực tiếp trên Blender, Web 3D Viewer, Game Engine)
```

---

### 2. Chi tiết LUỒNG A: Depth Reconstruction (Hiểu sâu kỹ thuật)

#### Bước 2A.1: Tạo Depth Map

- **Mục tiêu:** Dự đoán bản đồ độ sâu từ đơn ảnh.

| Thông số                  | Giá trị                                                                  |
| :------------------------ | :----------------------------------------------------------------------- |
| **Mô hình**               | **Depth-Anything-V2-Small**                                              |
| **Tác giả**               | Lihe Yang et al. (HKU + TikTok)                                          |
| **Kiến trúc**             | DPT head + ViT-Small (DINOv2 backbone)                                   |
| **Số tham số**            | 24.8M                                                                    |
| **Kích thước checkpoint** | ~95MB                                                                    |
| **Input**                 | Ảnh RGB đã tiền xử lý, resize 518×518                                    |
| **Output**                | **Depth map** (ảnh grayscale biểu diễn khoảng cách tương đối tới camera) |
| **VRAM tiêu thụ**         | ~0.3GB                                                                   |
| **Thời gian chạy**        | ~0.2s trên GPU T4                                                        |
| **Bản quyền (License)**   | Apache 2.0 ✅                                                            |
| **Thư viện chính**        | PyTorch hoặc HuggingFace Transformers                                    |

> **Output:** Depth map — **KHÔNG PHẢI mesh**. Đây là tập hợp các giá trị Z (chiều sâu) cho từng pixel.

---

#### Bước 2A.2: Chuyển Depth Map → Point Cloud

- **Mục tiêu:** Ánh xạ từ không gian 2D pixel sang không gian 3D thế giới thực.

| Thông số            | Giá trị                                                      |
| :------------------ | :----------------------------------------------------------- |
| **Thuật toán**      | **Back-projection** (Pinhole Camera Model)                   |
| **Thư viện**        | Open3D (`open3d.geometry.PointCloud`)                        |
| **Yêu cầu đầu vào** | Depth map + Ma trận Camera Intrinsics ($f_x, f_y, c_x, c_y$) |
| **VRAM**            | 0 (thực thi trên CPU)                                        |
| **Thời gian chạy**  | ~50ms                                                        |

- **Công thức toán học Back-projection:**
  Với mỗi pixel tại tọa độ $(u, v)$ có giá trị độ sâu $z$:
  $$X = \frac{(u - c_x) \times z}{f_x}$$
  $$Y = \frac{(v - c_y) \times z}{f_y}$$
  $$Z = z$$
  _Kết quả:_ Tập hợp điểm Point Cloud gồm các bộ ba tọa độ $(X, Y, Z)$ trong không gian 3D.

---

#### Bước 2A.3: Chuyển Point Cloud → 3D Mesh

- **Mục tiêu:** Nối các điểm rời rạc thành các mặt tam giác (triangular surface mesh).

| Thông số                | Giá trị                                                                           |
| :---------------------- | :-------------------------------------------------------------------------------- |
| **Thuật toán**          | **Poisson Surface Reconstruction**                                                |
| **Thư viện**            | Open3D (`open3d.geometry.TriangleMesh.create_from_point_cloud_poisson`)           |
| **VRAM**                | 0 (thực thi trên CPU)                                                             |
| **Thời gian chạy**      | ~200ms – 1s (tùy mật độ điểm)                                                     |
| **Thuật toán thay thế** | Ball Pivoting Algorithm (BPA) — xử lý lâu hơn, thích hợp hơn khi point cloud thưa |

---

#### Bước 2A.4: Texture Projection

- **Mục tiêu:** Chiếu màu sắc gốc của ảnh 2D lên bề mặt lưới 3D vừa tái tạo.

| Thông số            | Giá trị                                                       |
| :------------------ | :------------------------------------------------------------ |
| **Thuật toán**      | **Camera Projection** (chiếu ngược tọa độ ảnh 2D lên 3D mesh) |
| **Thư viện**        | Open3D hoặc Trimesh                                           |
| **Yêu cầu đầu vào** | Ma trận Camera Intrinsics + Ảnh RGB gốc + Depth map bước 2A.2 |
| **Output**          | Tọa độ UV Map + Ảnh Texture                                   |

---

### 3. Chi tiết LUỒNG B: Feed-forward Direct (Nhanh, ít code)

#### Bước 2B: Tạo Mesh trực tiếp từ ảnh

- **Mục tiêu:** Đi tắt từ ảnh đơn thẳng ra mô hình 3D nguyên khối có sẵn bề mặt và màu sắc.

| Thông số                | Giá trị                                                               |
| :---------------------- | :-------------------------------------------------------------------- |
| **Mô hình**             | **TripoSR**                                                           |
| **Tác giả**             | Dmitry Tochilkin et al. (Tripo AI + Stability AI)                     |
| **Kiến trúc**           | LRM (Large Reconstruction Model) dựa trên Transformer                 |
| **Số tham số**          | Hàng trăm triệu parameters (~hundreds of millions)                    |
| **Input**               | Ảnh RGB 512×512 (đã xóa nền)                                          |
| **Output**              | **Textured mesh** (.obj hoặc .ply) — **ĐÃ CÓ UV map + vertex colors** |
| **VRAM tiêu thụ**       | ~6GB trên GPU T4                                                      |
| **Thời gian chạy**      | ~0.5s trên A100, ~1-2s trên GPU T4                                    |
| **Bản quyền (License)** | MIT ✅                                                                |
| **Thư viện chính**      | PyTorch + Trimesh                                                     |

- **Lệnh chạy chuẩn CLI:**
  ```bash
  python run.py input_image.png --output-dir output/ --bake-texture
  ```

> **Lưu ý quan trọng về cờ `--bake-texture`:**
> Không có flag này chỉ ra vertex colors (màu ở đỉnh), có flag này mới bake texture image cho mesh — **PHẢI dùng flag này nếu muốn output đẹp**.
>
> TripoSR đã bao gồm: Mesh extraction + UV unwrapping + texture baking → **BỎ BƯỚC 3 trong plan gốc**.

---

### 4. So sánh Mesh Generation & Extraction giữa 2 Luồng

| Tiêu chí                | Luồng A (Depth Reconstruction) | Luồng B (Feed-forward Direct) |
| :---------------------- | :----------------------------- | :---------------------------- |
| **Cần bước này không?** | ✅ **CÓ**                      | ❌ **KHÔNG**                  |
| **Thuật toán áp dụng**  | Poisson Surface Reconstruction | Đã có sẵn trong TripoSR       |
| **Thư viện thực hiện**  | Open3D                         | —                             |
| **Mức chiếm dụng VRAM** | 0 GB                           | —                             |

> **Marching Cubes KHÔNG PHÙ HỢP cho cả 2 luồng:**
>
> - Dùng cho volume data (CT scan, MRI), không phải depth map hay point cloud.
> - Poisson Reconstruction đúng thuật toán hơn cho bài toán này.

---

### 5. Định dạng Xuất file & Render (Render & Export)

| Thông số kỹ thuật        | Cấu hình chuẩn hóa                                                                   | Chi tiết & Ghi chú triển khai                                                                                                                                                                  |
| :----------------------- | :----------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Định dạng file xuất**  | **.glb** (GLTF 2.0 Binary)                                                           | **Không dùng .obj rời rạc.** File `.glb` nhúng toàn bộ Geometry (Mesh), Tọa độ UV Map và Texture Image vào một file nhị phân duy nhất, mở xem được ngay trên Web 3D, Blender hoặc Game Engine. |
| **Thư viện xử lý xuất**  | **Trimesh**                                                                          | Gọi lệnh đơn giản: `mesh.export('output.glb')`.                                                                                                                                                |
| **Cơ chế UV Unwrapping** | • **Luồng A:** Camera Projection thủ công<br>• **Luồng B:** TripoSR tự động mở phẳng | Không cần dùng thuật toán trải phẳng phức tạp như XAtlas hay BFF (vốn chỉ cần cho game engine cao cấp).                                                                                        |
| **Mức tiêu thụ VRAM**    | **0 GB**                                                                             | Toàn bộ khâu xuất file và render nhẹ được xử lý hoàn toàn trên CPU.                                                                                                                            |
| **Thời gian thực thi**   | **< 10 ms**                                                                          | Tốc độ xuất tức thì, không gây nghẽn pipeline.                                                                                                                                                 |

> **Về thuật toán XAtlas / BFF:** KHÔNG CẦN cho project này. Hai thuật toán UV unwrapping phức tạp này dùng cho game engine cần atlas optimization — project sinh viên không cần.

---

### 6. Bảng tổng hợp tài nguyên trên Colab Free (T4 15GB VRAM, 12GB RAM)

| Mô hình                     |    VRAM    |    RAM    | Disk  |  Thời gian load lần đầu  |
| :-------------------------- | :--------: | :-------: | :---: | :----------------------: |
| **RMBG-2.0**                |   0.5GB    |    1GB    | 884MB |  ~30s (từ HuggingFace)   |
| **Depth Anything V2 Small** |   0.3GB    |   0.5GB   | 95MB  |           ~10s           |
| **TripoSR**                 |    6GB     |    8GB    | 1.5GB |           ~60s           |
| **Open3D (CPU)**            |     0      |   0.2GB   |   —   |      Instant (pip)       |
| **Tổng (Luồng A)**          |  **~1GB**  | **~2GB**  |   —   |    An toàn tuyệt đối     |
| **Tổng (Luồng B)**          | **~6.5GB** | **~10GB** |   —   | ⚠️ Gần chạm giới hạn RAM |

---

### 7. Các điểm kỹ thuật cần thiết khi triển khai

#### 1. Cache model weights (Chạy 1 lần đầu, sau đó dùng lại)

```python
!pip install -q huggingface_hub
from huggingface_hub import snapshot_download

snapshot_download("stabilityai/TripoSR", local_dir="./triposr_weights")
```

_Colab free timeout 90 phút → lần 2 load từ local, không cần download lại._

#### 2. Camera Intrinsics (chỉ cần cho Luồng A - Ước tính nếu không biết thông số camera)

```python
fx = fy = max(width, height)  # focal length tương đối
cx, cy = width / 2, height / 2  # principal point
```

#### 3. GPU verification

```python
import torch

assert torch.cuda.is_available(), "Cần GPU T4 để chạy"
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")
```

#### 4. Bảng đối chiếu bản quyền (License check)

| Mô hình               | License      | OK cho demo sinh viên? |
| :-------------------- | :----------- | :--------------------: |
| **RMBG-2.0**          | CC BY-NC 4.0 |  ✅ (non-commercial)   |
| **Depth Anything V2** | Apache 2.0   |           ✅           |
| **TripoSR**           | MIT          |           ✅           |

---

### 8. Khuyến nghị

- **Muốn hiểu kỹ thuật + học được nhiều:** Bắt đầu bằng **Luồng A (Depth Anything + Open3D)**. Viết được phần `depth` → `point cloud` → `mesh` là hiểu cách 3D reconstruction thực sự hoạt động.
- **Muốn demo nhanh, kết quả đẹp:** Dùng **Luồng B (TripoSR)**. Ít code hơn, output có texture sẵn, nhưng TripoSR là "black box lớn" — khó hiểu bên trong.
- **Cả 2 đều chạy được trên Colab free.** Luồng A an toàn hơn về VRAM. Luồng B gần chạm giới hạn RAM (~10GB/12GB) — cần đảm bảo không chạy thêm process nào khác.
