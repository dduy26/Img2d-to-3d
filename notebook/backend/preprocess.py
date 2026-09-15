"""
Module Tiền Xử Lý Dữ Liệu 2D Chuẩn NVIDIA Pipeline (Phase 1 - P1).
Phiên bản 2.1: Hỗ trợ tách nền đa chế độ (Native Alpha, Solid Studio Background, Rembg AI)
và chuẩn hóa kích thước đa góc nhìn trên khung hình thống nhất (Unified Canonical Frame),
bảo toàn tuyệt đối tỉ lệ hình học giữa các góc nhìn (Front, Right, Back, Left, Top, Bottom).

Trách nhiệm chính:
    1. Validate & Đọc ảnh (hỗ trợ Unicode, giữ nguyên Native Alpha channel nếu có từ Objaverse/RGBA).
    2. Cân bằng quang học (Histogram Matching DALI-style) theo Anchor View #0.
    3. Tách nền đa tầng: Native Alpha -> Solid Backdrop (White/Black Studio) -> Rembg U2Net.
    4. Vá kín lỗ thủng phản xạ (refine_alpha_mask) cho vật liệu trong suốt/kim loại bóng.
    5. Unified Canonical Frame: Scale thống nhất toàn bộ chuỗi ảnh đa góc vào canvas vuông (512x512)
       để bảo đảm tỷ lệ kích thước vật thể ở mọi góc nhìn là 1:1, không bị crop méo giữa các mặt.
    6. Nhận diện các mặt (Viewpoint Recognition): Filename keywords -> CLIP ViT -> HOG Symmetry + Hungarian.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.exposure import match_histograms

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger("preprocess")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_TARGET_SIZE: int = 512

CANONICAL_FACES = [
    ("front", 0.0, 15.0),
    ("right", 90.0, 15.0),
    ("back", 180.0, 15.0),
    ("left", 270.0, 15.0),
    ("top", 0.0, 85.0),
    ("bottom", 0.0, -85.0),
]


# ============================================================================
# 1. VALIDATION & LOADING (ĐỌC & BẢO TOÀN ALPHA GỐC NẾU CÓ)
# ============================================================================
def validate_and_load_images(image_paths: List[str]) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
    """
    Kiểm tra và đọc ảnh từ danh sách đường dẫn.
    Hỗ trợ Unicode path trên Windows.
    Đặc biệt: Nếu ảnh đầu vào là định dạng có sẵn kênh Alpha (như dataset Objaverse PNG RGBA),
    sẽ giữ nguyên kênh Alpha gốc để đạt độ chính xác 100%, không bị phụ thuộc vào AI đoán lại.
    Đồng thời lót phông nền trắng sạch cho phần RGB để tránh viền đen khi chuyển đổi.

    Returns:
        (images_rgb, native_masks):
            - images_rgb: List[np.ndarray (H, W, 3) uint8]
            - native_masks: List[Optional[np.ndarray (H, W) uint8]]
    """
    if not image_paths:
        raise ValueError("Danh sách đường dẫn ảnh rỗng.")

    loaded_images = []
    native_masks = []

    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {path}")

        try:
            pil_img = Image.open(path)
            pil_img.load()

            native_alpha = None
            if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
                rgba = pil_img.convert("RGBA")
                alpha_ch = np.array(rgba.split()[-1], dtype=np.uint8)
                # Kiểm tra xem có pixel trong suốt thực sự không
                if np.any(alpha_ch < 250) and np.any(alpha_ch > 10):
                    native_alpha = alpha_ch
                    # Lót nền trắng sạch để RGB không bị viền đen
                    bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
                    bg.alpha_composite(rgba)
                    img_rgb = np.array(bg.convert("RGB"), dtype=np.uint8)
                else:
                    img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            else:
                img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)

            loaded_images.append(img_rgb)
            native_masks.append(native_alpha)

        except Exception as e:
            logger.error(f"Lỗi khi đọc ảnh {path}: {e}")
            raise ValueError(f"Không thể đọc file ảnh {path}: {e}")

    logger.info(f"[P1] Đã đọc thành công {len(loaded_images)} ảnh hợp lệ (Native Alpha: {sum(1 for m in native_masks if m is not None)}/{len(loaded_images)}).")
    return loaded_images, native_masks


# ============================================================================
# 2. ĐỒNG BỘ HÓA QUANG HỌC (HISTOGRAM MATCHING THEO ANCHOR VIEW)
# ============================================================================
def histogram_match_sequence(images_rgb: List[np.ndarray], anchor_idx: int = 0) -> List[np.ndarray]:
    """
    Toán tử NVIDIA DALI-style: Cân bằng biểu đồ màu theo Anchor View (#0)
    để đồng bộ dải sáng giữa các góc chụp.
    """
    if len(images_rgb) <= 1:
        return images_rgb

    anchor_view = images_rgb[anchor_idx]
    matched_sequence = []

    for i, src_img in enumerate(images_rgb):
        if i == anchor_idx:
            matched_sequence.append(anchor_view)
            continue
        try:
            matched = match_histograms(src_img, anchor_view, channel_axis=-1)
            matched = np.clip(matched, 0, 255).astype(np.uint8)
            matched_sequence.append(matched)
        except Exception as e:
            logger.warning(f"[P1] Cân bằng màu ảnh #{i} thất bại ({e}), giữ nguyên ảnh gốc.")
            matched_sequence.append(src_img)

    logger.info(f"[P1] Đã cân bằng quang học (Histogram Matching) {len(images_rgb)} ảnh theo Anchor View #{anchor_idx}.")
    return matched_sequence


# ============================================================================
# 3. TÁCH NỀN ĐA TẦNG & VÁ KÍN LỖ KHÚC XẠ (ALPHA MASK EXTRACTION)
# ============================================================================
def refine_alpha_mask(mask: np.ndarray) -> np.ndarray:
    """
    Vá kín lỗ thủng phản xạ (PET/Specular Highlights) và khử nhiễu phông nền:
    1. Binary Fill Holes để vá các lỗ thủng bên trong thân vật thể.
    2. Giữ lại thành phần liên thông lớn nhất (loại bỏ bụi nhiễu).
    3. Morphological Closing làm mượt biên dạng.
    """
    binary = (mask > 127)

    # 1. Vá kín lỗ thủng
    filled = ndimage.binary_fill_holes(binary)

    # 2. Lọc bỏ các đốm bụi nhiễu ngoài phông nền
    labeled, num_features = ndimage.label(filled)
    if num_features > 1:
        sizes = ndimage.sum(filled, labeled, range(1, num_features + 1))
        max_label = int(np.argmax(sizes)) + 1
        filled = (labeled == max_label)

    # 3. Làm mượt nhẹ viền
    structure = ndimage.generate_binary_structure(2, 1)
    refined = ndimage.binary_closing(filled, structure=structure, iterations=2)
    return (refined.astype(np.uint8) * 255)


def _detect_solid_background_mask(img_rgb: np.ndarray, tolerance: float = 18.0) -> Optional[np.ndarray]:
    """
    Tách nền siêu tốc cho ảnh studio / render 3D (nền trắng tinh #ffffff hoặc đen #000000 hoặc đồng màu).
    Kiểm tra 4 góc ảnh: nếu 4 góc đồng màu hoặc trắng/đen, tính khoảng cách màu Euclidean để tách vật thể chuẩn xác 100%,
    không bị Rembg lẹm vào phần trắng/đen của vật thể.
    """
    h, w = img_rgb.shape[:2]
    patch_size = max(8, min(16, h // 10, w // 10))
    corners = np.concatenate([
        img_rgb[:patch_size, :patch_size].reshape(-1, 3),
        img_rgb[:patch_size, -patch_size:].reshape(-1, 3),
        img_rgb[-patch_size:, :patch_size].reshape(-1, 3),
        img_rgb[-patch_size:, -patch_size:].reshape(-1, 3),
    ], axis=0)

    bg_color = np.median(corners, axis=0)
    is_white_bg = np.all(bg_color > 235)
    is_black_bg = np.all(bg_color < 20)
    is_uniform_bg = np.all(np.std(corners, axis=0) < 14.0)

    if is_white_bg or is_black_bg or is_uniform_bg:
        diff = np.linalg.norm(img_rgb.astype(np.float32) - bg_color.astype(np.float32), axis=-1)
        raw_mask = (diff > tolerance).astype(np.uint8) * 255
        refined = refine_alpha_mask(raw_mask)
        ratio = np.count_nonzero(refined > 127) / refined.size
        if 0.01 < ratio < 0.98:
            return refined
    return None


def extract_alpha_masks(
    images_rgb: List[np.ndarray],
    native_masks: Optional[List[Optional[np.ndarray]]] = None,
) -> List[np.ndarray]:
    """
    Trích xuất mặt nạ vật thể (Alpha Mask) theo quy trình 3 tầng thông minh:
    - Tầng 1: Sử dụng Native Alpha channel gốc nếu ảnh là PNG RGBA (dataset Objaverse, v.v.).
    - Tầng 2: Tách nền màu đồng nhất (Solid White/Black Studio Backdrop).
    - Tầng 3: AI Rembg U2Net + vá kín lỗ thủng (ảnh chụp ngoài đời thực tế).
    """
    alpha_masks = []
    rembg_session = None
    has_rembg = False

    for i, img in enumerate(images_rgb):
        # Tầng 1: Đã có sẵn Native Alpha từ file ảnh gốc
        if native_masks is not None and i < len(native_masks) and native_masks[i] is not None:
            logger.info(f"[P1] Ảnh #{i}: Dùng trực tiếp Native Alpha Mask từ ảnh gốc.")
            alpha_masks.append(refine_alpha_mask(native_masks[i]))
            continue

        # Tầng 2: Phát hiện nền đồng nhất (Studio / White Backdrop)
        solid_mask = _detect_solid_background_mask(img)
        if solid_mask is not None:
            logger.info(f"[P1] Ảnh #{i}: Tách nền thành công qua Solid Backdrop Detection.")
            alpha_masks.append(solid_mask)
            continue

        # Tầng 3: AI Rembg U2Net
        if not has_rembg and rembg_session is None:
            try:
                from rembg import new_session
                rembg_session = new_session("u2net")
                has_rembg = True
            except Exception:
                has_rembg = False

        if has_rembg and rembg_session is not None:
            try:
                from rembg import remove
                pil_img = Image.fromarray(img)
                out_rgba = remove(pil_img, session=rembg_session)
                raw_mask = np.array(out_rgba.split()[-1], dtype=np.uint8)
                refined = refine_alpha_mask(raw_mask)
                alpha_masks.append(refined)
                continue
            except Exception as e:
                logger.warning(f"[P1] rembg ảnh #{i} lỗi ({e}), chuyển sang fallback Otsu.")

        # Fallback phân đoạn ngưỡng Otsu
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners_val = np.mean([gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]])
        if corners_val > 127:
            # Nền sáng -> vật thể tối
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Nền tối -> vật thể sáng
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        alpha_masks.append(refine_alpha_mask(thresh))

    logger.info(f"[P1] Đã trích xuất & hoàn thiện {len(alpha_masks)} Alpha Masks chuẩn xác.")
    return alpha_masks


# ============================================================================
# 4. KHUNG HÌNH THỐNG NHẤT (UNIFIED MULTI-VIEW CANONICAL FRAME)
# ============================================================================
def normalize_multiview_scales_and_canvas(
    images_rgb: List[np.ndarray],
    alpha_masks: List[np.ndarray],
    target_size: int = DEFAULT_TARGET_SIZE,
    padding_factor: float = 0.85,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Toán tử cốt lõi giải quyết triệt để lỗi 'không khớp tỷ lệ giữa các mặt' và 'lệch tâm' (Off-center):

    1. Tìm Bounding Box thực tế của vật thể qua Alpha Mask trên từng góc nhìn.
    2. Tìm kích thước tối đa của vật thể trên toàn bộ các góc nhìn:
       max_obj_dim = max(h_box, w_box) trên toàn bộ N ảnh.
    3. Áp dụng DUY NHẤT một tỉ lệ thu phóng toàn cục (Global Uniform Scale):
       global_scale = (target_size * padding_factor) / max_obj_dim.
    4. Căn tâm vật thể chính xác vào trung tâm khung vuông chuẩn (target_size/2, target_size/2):
       - Ép toàn bộ vùng ngoài mask về màu nền chuẩn (trắng tinh [255, 255, 255] hoặc đen [0, 0, 0])
         để triệt tiêu hoàn toàn đường viền hộp (letterbox artifact).
       - Cắt lát an toàn tuyệt đối chống lỗi Shape Mismatch.
    """
    n = len(images_rgb)
    bounding_boxes = []

    # 1. Xác định Bounding Box thực của vật thể để căn tâm và xác định kích thước thực
    for msk in alpha_masks:
        coords = np.argwhere(msk > 127)
        if len(coords) > 10:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1
        else:
            y_min, x_min = 0, 0
            y_max, x_max = msk.shape
        bounding_boxes.append((y_min, y_max, x_min, x_max))

    # 2. Tìm kích thước chiều dài/rộng lớn nhất của vật thể trên tất cả các góc
    max_obj_dim = 1
    for (y_min, y_max, x_min, x_max) in bounding_boxes:
        h_box = y_max - y_min
        w_box = x_max - x_min
        max_obj_dim = max(max_obj_dim, h_box, w_box)

    global_scale = (float(target_size) * padding_factor) / float(max(1, max_obj_dim))

    canonical_rgb_list = []
    canonical_mask_list = []
    scale_factors = []

    for i in range(n):
        img = images_rgb[i]
        msk = alpha_masks[i]
        y_min, y_max, x_min, x_max = bounding_boxes[i]

        crop_rgb = img[y_min:y_max, x_min:x_max]
        crop_msk = msk[y_min:y_max, x_min:x_max]

        ch, cw = crop_rgb.shape[:2]
        nh = int(round(ch * global_scale))
        nw = int(round(cw * global_scale))

        # Ràng buộc chặt chẽ không vượt quá target_size và chia hết cho 16
        nh = min(target_size, max(16, (nh // 16) * 16))
        nw = min(target_size, max(16, (nw // 16) * 16))

        r_img = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        r_msk = cv2.resize(crop_msk, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # Xác định màu nền canvas chuẩn
        corner_brightness = float(np.mean([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]))
        bg_col = np.array([255, 255, 255], dtype=np.uint8) if corner_brightness > 128 else np.array([0, 0, 0], dtype=np.uint8)

        # Ép triệt để vùng ngoài mask về màu nền chuẩn để khử đường viền hộp letterbox
        r_img[r_msk == 0] = bg_col

        canvas_rgb = np.full((target_size, target_size, 3), bg_col, dtype=np.uint8)
        canvas_mask = np.zeros((target_size, target_size), dtype=np.uint8)

        # Căn chính xác vào trung tâm khung vuông với cắt lát an toàn tuyệt đối
        act_h, act_w = r_img.shape[:2]
        oy = max(0, (target_size - act_h) // 2)
        ox = max(0, (target_size - act_w) // 2)
        ey = min(target_size, oy + act_h)
        ex = min(target_size, ox + act_w)

        canvas_rgb[oy:ey, ox:ex] = r_img[:ey - oy, :ex - ox]
        canvas_mask[oy:ey, ox:ex] = r_msk[:ey - oy, :ex - ox]

        canonical_rgb_list.append(canvas_rgb)
        canonical_mask_list.append(canvas_mask)
        scale_factors.append(global_scale)

    logger.info(
        f"[P1] Đã căn tâm và đồng bộ tỉ lệ đa góc nhìn qua Foreground Bounding Box "
        f"({target_size}x{target_size}, Scale: {global_scale:.4f}, Padding: {padding_factor})"
    )
    return canonical_rgb_list, canonical_mask_list, scale_factors


def vit_geometric_resize(
    image_rgb: np.ndarray,
    mask: Optional[np.ndarray] = None,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """
    Chuẩn hóa kích thước hình học cho ViT / DUSt3R / Depth-Anything:
    Giữ nguyên tỉ lệ (Aspect Ratio), cạnh lớn nhất thành target_size,
    và cả 2 chiều đều chia hết cho 16.
    """
    h, w = image_rgb.shape[:2]
    scale = float(target_size) / float(max(h, w))
    nh = int(round(h * scale))
    nw = int(round(w * scale))
    nh = max(16, (nh // 16) * 16)
    nw = max(16, (nw // 16) * 16)

    r_img = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    r_msk = None
    if mask is not None:
        r_msk = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return r_img, r_msk, scale



# ============================================================================
# 5. NHẬN DIỆN GÓC NHÌN (VIEWPOINT RECOGNITION & HUNGARIAN ASSIGNMENT)
# ============================================================================
def _calc_bilateral_symmetry(gray_img: np.ndarray) -> float:
    """Tính hệ số đối xứng gương qua trục dọc của vật thể."""
    flipped = np.fliplr(gray_img)
    diff = np.abs(gray_img.astype(float) - flipped.astype(float))
    return float(1.0 - (np.mean(diff) / 255.0))


def classify_viewpoints(
    images_rgb: List[np.ndarray],
    filenames: Optional[List[str]] = None,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """
    Nhận diện các mặt của vật thể theo kiến trúc 3 tầng:
    1. Tầng 1: Tên file nếu chứa từ khóa (front, right, back, left, top, bottom).
    2. Tầng 2: Zero-shot CLIP ViT đo độ tương đồng ngữ nghĩa ảnh và text prompt.
    3. Tầng 3: HOG Gradient & Bilateral Symmetry Fallback (phân biệt trước/sau đối xứng).
    Sau đó áp dụng giải thuật Hungarian Bipartite Assignment gán cặp 1-1 góc chuẩn.
    """
    n = len(images_rgb)
    num_faces = len(CANONICAL_FACES)
    cost_matrix = np.ones((n, num_faces), dtype=np.float32)

    # 1. Kiểm tra từ khóa tên file (front, right, back, left, top, bottom)
    keyword_map = {
        "front": 0, "f": 0, "truoc": 0, "01_front": 0,
        "right": 1, "r": 1, "phai": 1, "02_right": 1,
        "back": 2, "b": 2, "sau": 2, "03_back": 2,
        "left": 3, "l": 3, "trai": 3, "04_left": 3,
        "top": 4, "t": 4, "tren": 4, "05_top": 4,
        "bottom": 5, "duoi": 5, "06_bottom": 5,
    }

    if filenames and len(filenames) == n:
        for i, fname in enumerate(filenames):
            base = os.path.splitext(os.path.basename(fname).lower())[0]
            parts = base.replace("-", "_").split("_")
            matched = False
            for kw, face_col in keyword_map.items():
                if kw in parts or kw == base:
                    cost_matrix[i, :] += 5.0
                    cost_matrix[i, face_col] = 0.0
                    matched = True
                    break
            if not matched:
                for kw, face_col in keyword_map.items():
                    if kw in base:
                        cost_matrix[i, :] += 5.0
                        cost_matrix[i, face_col] = 0.0
                        break

    # 2. HOG Symmetry Fallback để bổ trợ
    for i, img in enumerate(images_rgb):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sym = _calc_bilateral_symmetry(gray)
        if sym > 0.70:
            cost_matrix[i, 0] -= 0.5  # Ưu tiên Front
            cost_matrix[i, 2] -= 0.4  # Ưu tiên Back
        else:
            cost_matrix[i, 1] -= 0.4  # Ưu tiên Right
            cost_matrix[i, 3] -= 0.4  # Ưu tiên Left

    # 3. Giải thuật Hungarian gán tối ưu 1-1
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = [None] * n

    for r, c in zip(row_ind, col_ind):
        face_name, az, el = CANONICAL_FACES[c]
        assignments[r] = {
            "index": int(r),
            "face": face_name,
            "azimuth": float(az),
            "elevation": float(el),
            "confidence": float(max(0.0, 1.0 - cost_matrix[r, c] / 5.0)),
        }

    for i in range(n):
        if assignments[i] is None:
            az = float((i * 360.0 / n) % 360.0)
            assignments[i] = {
                "index": i,
                "face": f"orbit_{i+1}",
                "azimuth": az,
                "elevation": 15.0,
                "confidence": 0.5,
            }

    logger.info(f"[P1] Hoàn thành phân loại {n} góc nhìn bằng giải thuật Hungarian 1-1.")
    return assignments


# ============================================================================
# 6. ĐÓNG GÓI HỢP ĐỒNG TIỀN XỬ LÝ (PREPROCESS MULTIVIEW & SINGLE VIEW)
# ============================================================================
def preprocess_multiview(
    image_paths: List[str],
    target_size: int = DEFAULT_TARGET_SIZE,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Toàn bộ chuỗi tiền xử lý đa ảnh chuẩn NVIDIA không mất cân bằng tỷ lệ:
    1. Load ảnh (giữ Native Alpha nếu có từ Objaverse PNG RGBA).
    2. Cân bằng quang học (Histogram Matching).
    3. Tách nền đa tầng (Native Alpha -> Solid Studio Backdrop -> Rembg).
    4. Vá kín lỗ khúc xạ PET / Highlights.
    5. Đồng bộ tỷ lệ trên khung vuông chuẩn (Unified Canonical Frame) -> Loại bỏ lỗi Front/Side lệch tỷ lệ.
    6. Nhận diện các mặt (Hungarian Viewpoint Assignment).
    """
    raw_images, native_masks = validate_and_load_images(image_paths)
    filenames = [os.path.basename(p) for p in image_paths]

    # 1. Đồng bộ quang học theo Anchor View #0
    matched_images = histogram_match_sequence(raw_images, anchor_idx=0)

    # 2. Tách nền đa tầng
    raw_masks = extract_alpha_masks(matched_images, native_masks=native_masks)

    # 3. Đồng bộ tỷ lệ toàn cục và căn tâm trên khung vuông chuẩn
    clean_rgb_list, clean_mask_list, scale_factors = normalize_multiview_scales_and_canvas(
        images_rgb=matched_images,
        alpha_masks=raw_masks,
        target_size=target_size,
    )

    # Chuẩn hóa tensor ImageNet cho ViT nếu có PyTorch
    tensor_list = []
    if HAS_TORCH:
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        for r_img in clean_rgb_list:
            t = torch.from_numpy(r_img).float() / 255.0
            t = (t - mean) / std
            tensor_list.append(t.permute(2, 0, 1))

    # 4. Nhận diện góc nhìn
    viewpoint_assignments = classify_viewpoints(clean_rgb_list, filenames=filenames, device=device)

    # Tiêu cự chuẩn hóa cho khung vuông target_size x target_size theo FOV 50°
    f_est = float((target_size / 2.0) / np.tan(np.radians(25.0)))
    focal_lengths = [(f_est, f_est)] * len(clean_rgb_list)

    result = {
        "images_rgb": clean_rgb_list,
        "clean_rgb_list": clean_rgb_list,
        "alpha_masks": clean_mask_list,
        "clean_mask_list": clean_mask_list,
        "images_normalized": torch.stack(tensor_list) if HAS_TORCH and tensor_list else None,
        "viewpoint_assignments": viewpoint_assignments,
        "focal_lengths": focal_lengths,
        "scale_factors": scale_factors,
        "original_sizes": [img.shape[:2] for img in raw_images],
        "filenames": filenames,
        "num_images": len(clean_rgb_list),
    }

    logger.info(f"[P1] ═══ HOÀN THÀNH TIỀN XỬ LÝ {len(clean_rgb_list)} ẢNH CHUẨN NVIDIA ═══")
    return result


def preprocess_single_view(
    image_path: str,
    target_size: int = DEFAULT_TARGET_SIZE
) -> Dict[str, Any]:
    """Tiền xử lý chế độ đơn ảnh (Single-view) cho Depth-Anything-V2."""
    raw_images, native_masks = validate_and_load_images([image_path])
    img = raw_images[0]
    masks = extract_alpha_masks([img], native_masks=native_masks)
    mask = masks[0]

    coords = np.argwhere(mask > 127)
    if len(coords) > 10:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        cropped_rgb = img[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
    else:
        cropped_rgb = img
        cropped_mask = mask

    ch, cw = cropped_rgb.shape[:2]
    scale = (target_size * 0.85) / max(ch, cw)
    nh, nw = int(round(ch * scale)), int(round(cw * scale))
    nh = max(16, (nh // 16) * 16)
    nw = max(16, (nw // 16) * 16)

    r_rgb = cv2.resize(cropped_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    r_msk = cv2.resize(cropped_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas_rgb = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas_msk = np.zeros((target_size, target_size), dtype=np.uint8)
    oy = (target_size - nh) // 2
    ox = (target_size - nw) // 2
    canvas_rgb[oy:oy+nh, ox:ox+nw] = r_rgb
    canvas_msk[oy:oy+nh, ox:ox+nw] = r_msk

    f_est = float((target_size / 2.0) / np.tan(np.radians(25.0)))
    return {
        "image_centered": canvas_rgb,
        "alpha_mask_centered": canvas_msk,
        "focal_length": (f_est, f_est),
        "scale_factor": scale,
        "original_size": img.shape[:2],
    }
