# 📚 CƠ SỞ LÝ THUYẾT & NỀN TẢNG TOÁN HỌC HỆ THỐNG TÁI TẠO 3D (DUAL-STREAM ARCHITECTURE)

> **Mục đích tài liệu:** Cung cấp cơ sở lý thuyết toán học, thị giác máy tính và kiến trúc học sâu (Deep Learning) của hệ thống chuyển đổi ảnh 2D sang mô hình 3D nguyên khối, phục vụ việc đối soát, nghiệm thu và nghiên cứu mở rộng.  
> **Kiến trúc trọng tâm:** **Hai luồng AI SOTA (Dual-AI Engine)**:
> 1. **Luồng 1 (Single-View - 1 ảnh):** TripoSR (ViT + Triplane NeRF + Marching Cubes, ~1.5s, 100% Watertight Mesh).
> 2. **Luồng 2 (Multi-View - $N \ge 2$ ảnh):** Tencent Hunyuan3D-2mv DiT Flow Matching Pipeline + P1 Hungarian Viewpoint Assignment + P5 Multi-View Texture Blender (Fresnel $\cos^3\theta$ + Z-buffer Occlusion culling).

---

## 1. HÌNH HỌC QUANG HỌC & BÙ TRỪ MA TRẬN CAMERA INTRINSICS

### 1.1 Nguyên lý Hình học Chiếu (Projective Geometry)
Trong không gian 3 chiều, một điểm thế giới $\mathbf{P} = [X, Y, Z]^T$ được ánh xạ lên mặt phẳng cảm biến ảnh 2D tại điểm $\mathbf{p} = [u, v, 1]^T$ thông qua ma trận nội thông số camera (Camera Intrinsics $K$) và ma trận ngoại thông số (Extrinsics $[R \mid T]$):

$$\mathbf{p} \sim K \cdot [R \mid T] \cdot \mathbf{P}$$

Trong đó ma trận nội thông $K$ biểu diễn các đặc tính quang học của cảm biến:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

- $f_x, f_y$: Tiêu cự theo trục pixel $x$ và $y$.
- $(c_x, c_y)$: Tâm quang học (Principal Point) — điểm mà trục quang học xuyên vuông góc qua tâm cảm biến.

### 1.2 Tại sao Tuyệt Đối Không Được Cắt (Crop) Riêng Rẽ Từng Ảnh?
Khi người dùng chụp $N$ bức ảnh đa góc quanh vật thể:
- Nếu ảnh $i$ bị crop bounding box độc lập: tâm quang học thực tế của ảnh bị dịch chuyển một đại lượng $(\Delta x_i, \Delta y_i) \neq (\Delta x_j, \Delta y_j)$.
- **Hậu quả nghiêm trọng:** Tâm quang học bị xô lệch phá vỡ hoàn toàn mối liên hệ không gian giữa các góc nhìn. Khi nướng màu (Texture Projection) hoặc nội suy hình học, các tia chiếu từ camera bị lệch hướng, tạo ra hiện tượng mô hình méo mó hoặc texture bị chồng bóng ma (ghosting artifacts).

### 1.3 Thuật toán Bảo Toàn Tỉ Lệ & Bù Trừ $K \to K'$ (Uniform Scale & Centering)
Để đưa mọi kích thước ảnh về chuẩn $512 \times 512$ cho mạng học sâu mà **không làm biến dạng hình học**:
1. Tính tỉ lệ phóng đồng nhất duy nhất $s$:
   $$s = \frac{S_{\text{canvas}} \cdot \eta}{\max(H_{\text{orig}}, W_{\text{orig}})} \quad (\eta \approx 0.85)$$
2. Tính độ dời tâm $(\Delta x, \Delta y)$ khi đặt ảnh vào giữa canvas vuông:
   $$\Delta x = \frac{S_{\text{canvas}} - s \cdot W_{\text{orig}}}{2}, \quad \Delta y = \frac{S_{\text{canvas}} - s \cdot H_{\text{orig}}}{2}$$
3. **Bù trừ toán học vào ma trận Camera Intrinsics $K \to K'$:**
   $$K' = \begin{bmatrix} s \cdot f_x & 0 & s \cdot c_x + \Delta x \\ 0 & s \cdot f_y & s \cdot c_y + \Delta y \\ 0 & 0 & 1 \end{bmatrix}$$
Công thức này chứng minh rằng: Tọa độ chiếu 3D qua $K'$ trên canvas $512\times 512$ trùng khớp $100\%$ với vị trí chiếu vật lý trên ảnh gốc ban đầu.

---

## 2. PHÂN ĐOẠN NỀN TỰ ĐỘNG & XỬ LÝ LỖ THỦNG PHẢN XẠ (ALPHA MATTING)

### 2.1 Kiến trúc BiRefNet (Bilateral Reference Network / Rembg)
Các mô hình tạo sinh 3D hiện đại (TripoSR, Hunyuan3D-2mv) được huấn luyện trên tập dữ liệu vật thể 3D độc lập (Objaverse). Nếu giữ nguyên nền nhà, mặt bàn, hay góc phòng, mạng nơ-ron sẽ cố gắng nặn luôn cả mặt bàn thành khối 3D dính liền vào đáy vật thể.
- **RMBG / BiRefNet:** Sử dụng kiến trúc Encoder-Decoder với liên kết song phương (Bilateral Connections), trích xuất mặt nạ $\mathbf{M} \in [0, 1]^{H \times W}$ tách biệt sắc nét từng sợi viền mảnh.

### 2.2 Thuật toán Vá Lỗ Vật Thể Trong Suốt / Phản Xạ (`refine_alpha_mask`)
- **Vấn đề thực tế:** Vật thể trơn bóng (kim loại sáng, thủy tinh, chai nhựa PET, da bóng của giày) thường tạo ra các đốm phản quang (specular highlights). Thuật toán tách nền dựa trên độ tương phản thường hiểu nhầm các đốm sáng trắng này là hậu cảnh, làm thủng lỗ to giữa thân vật thể.
- **Giải pháp Toán - Hình thái học:**
  1. Sử dụng thuật toán `scipy.ndimage.binary_fill_holes` bịt kín các vùng rỗng nằm lọt hoàn toàn bên trong thân vật thể.
  2. Phân tích thành phần liên thông (Connected Components Labeling): Chỉ giữ lại khối vật thể lớn nhất và các đảo pixel hợp lệ, triệt tiêu toàn bộ nhiễu lơ lửng xung quanh.

---

## 3. ĐỒNG BỘ ÁNH SÁNG TRONG KHÔNG GIAN CIE LAB (HISTOGRAM MATCHING)

### 3.1 Vấn đề Lệch Phơi Sáng (Auto-Exposure Artifacts)
Khi người dùng cầm điện thoại di chuyển quanh vật thể, cơ chế Auto-Exposure (AE) và Auto-White-Balance (AWB) tự động điều chỉnh tốc độ màn trập và ISO:
- Ảnh chụp từ góc thuận sáng thì sáng chói; góc ngược sáng thì tối sầm hoặc đổi nhiệt độ màu (ám vàng/ám xanh).
- Nếu nướng trực tiếp lên mô hình 3D, bề mặt sẽ xuất hiện các vệt loang lổ, phân ranh giới sáng tối tại các đường nối mí (seam lines).

### 3.2 Thuật toán Cân Bằng Duy Nhất Kênh Độ Sáng $L$
Hệ thống chuyển đổi ảnh sang không gian màu **CIE $L^*a^*b^*$**:
- Kênh $L^*$ đại diện cho cường độ sáng nhận thức (Perceptual Luminance).
- Hai kênh $a^*$ (Trục Đỏ - Xanh lá) và $b^*$ (Trục Vàng - Xanh dương) đại diện cho sắc độ màu (Chrominance).

**Quy trình chuẩn hóa:**
1. Chọn ảnh chính diện (#0) làm **Anchor View**.
2. Tính hàm phân phối tích lũy (Cumulative Distribution Function - CDF) của kênh $L^*$ trên ảnh Anchor:
   $$T_{\text{anchor}}(r) = \sum_{k=0}^{r} p_{\text{anchor}}(k)$$
3. Ánh xạ biểu đồ histogram kênh $L^*$ của $N-1$ ảnh còn lại theo hàm $T_{\text{anchor}}^{-1}$:
   $$L'_i = T_{\text{anchor}}^{-1}\big(T_i(L_i)\big)$$
4. **Giữ nguyên $100\%$ hai kênh sắc thái $a^*, b^*$:** Giúp các ảnh đồng bộ hoàn hảo về độ sáng nhưng **tuyệt đối không bị biến đổi màu sắc gốc** của vật thể.

---

## 4. GIẢI THUẬT GÁN GÓC NHÌN TỐI ƯU HUNGARIAN (BIPARTITE MATCHING)

### 4.1 Đặt Vấn Đề
Mô hình Multi-View DiT (`Hunyuan3D-2mv`) yêu cầu đầu vào là 4 góc nhìn trực giao chuẩn:
$$\mathcal{V}_{\text{canonical}} = \{\text{front } (0^\circ), \text{right } (90^\circ), \text{back } (180^\circ), \text{left } (270^\circ)\}$$
Tuy nhiên, người dùng tải ảnh lên theo thứ tự ngẫu nhiên hoặc tên file không có quy luật.

### 4.2 Ma Trận Chi Phí & Tối Ưu Tuyến Tính
1. **Phân loại đa tầng (3-Tier Classification):**
   - Tầng 1: Heuristic tên file (`front`, `back`, `left`, `right`).
   - Tầng 2: Zero-shot CLIP ViT Embedding tính độ tương đồng ngữ nghĩa.
   - Tầng 3: Tính đối xứng trục HOG Bilateral Symmetry (ảnh mặt trước và mặt sau có tính đối xứng trục đứng cao hơn ảnh mặt bên).
2. **Gán cặp 1-1 tối ưu (Hungarian Algorithm):**
   Xây dựng ma trận chi phí $C \in \mathbb{R}^{N \times 4}$ với $C_{ij} = 1 - \text{Confidence}(i \to \text{view}_j)$.
   Giải bài toán gán cặp cực tiểu tổng chi phí bằng `scipy.optimize.linear_sum_assignment`:
   $$\min_{\pi} \sum_{i} C_{i, \pi(i)}$$
Giải thuật đảm bảo mỗi góc nhìn chuẩn có đúng 1 bức ảnh đại diện tốt nhất, triệt tiêu $100\%$ lỗi gán trùng góc.

---

## 5. BẢN CHẤT KIẾN TRÚC AI 2 LUỒNG: SO SÁNH & NGUYÊN LÝ HOẠT ĐỘNG

### 5.1 Bảng So Sánh Hai Luồng

| Tiêu chí | ⚡ Luồng 1: Single-View (TripoSR) | 🌐 Luồng 2: Multi-View (Tencent Hunyuan3D-2mv) |
| :--- | :--- | :--- |
| **Đầu vào** | 1 ảnh 2D duy nhất | $N \ge 2$ ảnh chụp quanh vật thể (tối ưu 4-8 ảnh) |
| **Mô hình lõi** | **TripoSR** (Stability AI / VAST) | **Tencent Hunyuan3D-2mv** (Tencent AI Lab) |
| **Kiến trúc mạng** | ViT Encoder + Triplane NeRF Generator | Diffusion Transformer (DiT) + Flow Matching |
| **Thời gian suy luận** | **~1.5 giây** | **~20 - 30 giây** |
| **Yêu cầu VRAM** | ~4 - 6 GB VRAM | ~6 - 8 GB VRAM (chế độ FP16) |
| **Độ kín nước** | **100% Watertight Solid Mesh** | **100% Watertight Solid Mesh (Chuẩn CAD)** |
| **Tô màu bề mặt** | Vertex Colors trực tiếp từ NeRF field | **Multi-View Texture Blender (Fresnel $\cos^3\theta$ + Z-buffer)** |
| **Định dạng xuất** | 1 file `.glb` duy nhất | 1 file `.glb` duy nhất |

### 5.2 Tại sao Hunyuan3D-2mv DiT Giải Quyết Triệt Để Lỗi "Vỏ Sò Rỗng" của DUSt3R?
- **Hạn chế của phương pháp Stereo Photogrammetry truyền thống (như DUSt3R):**
  DUSt3R chỉ hồi quy tọa độ 3D $X, Y, Z$ cho các pixel nhìn thấy trực tiếp (Line-of-Sight). Khi chụp vật thể đặt trên bàn (như đôi giày), camera không thể nhìn thấy mặt đế áp vào bàn hay lòng bên trong giày. DUSt3R không có tri thức tạo sinh (generative prior), dẫn đến mô hình bị rỗng ruột, mỏng dính như vỏ sò và không thể in 3D.
- **Đột phá của DiT Flow Matching trong Hunyuan3D-2mv:**
  Hunyuan3D-2mv kết hợp cơ chế chú ý chéo đa hướng (Multi-view Cross Attention) với trường vận tốc Flow Matching. Mô hình đồng thời:
  1. Ràng buộc hình học chính xác theo các chi tiết có trong ảnh chụp thực tế.
  2. Tự động suy luận trường thể tích đặc (SDF/Occupancy) tại các vùng khuất (mặt đế, lòng giày), tạo ra một khối 3D đặc kín nước $100\%$, đạt chuẩn công nghiệp in 3D và đồ họa game.

---

## 6. NGUYÊN LÝ NƯỚNG MÀU FRESNEL VÀ KHỬ CHE KHUẤT (TEXTURE BLENDER)

### 6.1 Trọng Số Hòa Trộn Fresnel Bậc 3 ($\cos^3\theta$)
Khi chiếu màu từ nhiều góc chụp lên các đỉnh của mô hình 3D:
- Đối với mỗi đỉnh $\mathbf{v}$ có vector pháp tuyến đơn vị $\vec{n}$:
- Vector hướng nhìn từ tâm camera $i$ đến đỉnh $\mathbf{v}$ là $\vec{d}_i = \frac{\mathbf{C}_i - \mathbf{v}}{\|\mathbf{C}_i - \mathbf{v}\|}$.
- Tích vô hướng $\cos\theta_i = \vec{n} \cdot \vec{d}_i$ xác định góc hợp bởi hướng bề mặt và hướng nhìn của camera.

**Hàm trọng số góc nhìn NVIDIA Fresnel:**
$$w_i(\mathbf{v}) = \begin{cases} (\vec{n} \cdot \vec{d}_i)^3 & \text{nếu } \vec{n} \cdot \vec{d}_i > 0 \\ 0 & \text{nếu } \vec{n} \cdot \vec{d}_i \le 0 \end{cases}$$

- **Lũy thừa bậc 3 ($\cos^3\theta$):** Giảm nhanh trọng số khi góc nhìn nghiêng đi, loại bỏ triệt để hiện tượng phản chiếu ánh sáng chói (specular highlights) và nhòe màu biên.
- Các bề mặt quay lưng về phía camera ($\vec{n} \cdot \vec{d}_i \le 0$) tự động nhận trọng số bằng 0.

### 6.2 Khử Che Khuất Bằng Bộ Đệm Chiều Sâu (Z-Buffer Occlusion Culling)
- Hiện tượng tự che khuất (Self-occlusion) xảy ra khi mặt sau của vật thể vô tình bị gán màu từ camera mặt trước do tia chiếu xuyên thấu qua thân vật thể.
- **Thuật toán giải quyết:**
  Đối với mỗi tia chiếu, tính khoảng cách từ camera đến đỉnh $D_{\text{point}} = \|\mathbf{C}_i - \mathbf{v}\|$.
  So sánh với độ sâu bề mặt gần nhất $D_{\text{min}}$ theo hướng tia:
  $$\text{Nếu } D_{\text{point}} > D_{\text{min}} + \epsilon \implies \text{Đỉnh bị che khuất (Occluded)} \to w_i(\mathbf{v}) = 0$$

### 6.3 Hòa Trộn Màu Sắc Cuối Cùng
Màu sắc RGB tại mỗi đỉnh được tổng hợp bằng trung bình trọng số chuẩn hóa:

$$\mathbf{C}(\mathbf{v}) = \frac{\sum_{i=1}^N w_i(\mathbf{v}) \cdot \mathbf{I}_i(\pi_i(\mathbf{v}))}{\sum_{i=1}^N w_i(\mathbf{v})}$$

Màu sắc này được gán trực tiếp vào thuộc tính đỉnh `COLOR_0` và đóng gói vào chuẩn nhị phân GLTF 2.0 (`.glb`), cho phép hiển thị màu sắc chân thực, sắc nét ngay lập tức trên Three.js, Windows 3D Viewer hay Blender mà không bị suy hao chất lượng.
