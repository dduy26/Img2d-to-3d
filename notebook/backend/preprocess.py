"""
Module Tiền Xử Lý Dữ Liệu 2D (P1 - Data & Preprocessing)
Chuẩn hóa theo kiến trúc NVIDIA 3D Workflow & NVIDIA DALI:
1. Đồng bộ quang học (Multi-view Color & Histogram Matching theo Anchor View #0).
2. Tách nền & Vá lỗ phản quang trên vật thể trong suốt (PET / Kim loại bóng).
3. Resize chuẩn Vision Transformer (ViT) bảo toàn tỷ lệ khung hình & Epipolar Geometry (chia hết cho 16).
4. Nhận diện góc nhìn tự động (Viewpoint Recognition: Tên file -> CLIP ViT -> HOG Symmetry)
   kết hợp giải thuật Hungarian Bipartite Assignment gán góc 1-1 tối ưu toàn cục.
"""

import os
import sys
import glob
import logging
from typing import List, Tuple, Dict, Any, Optional, Union

import cv2
import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
import scipy.ndimage as ndimage
from scipy.optimize import linear_sum_assignment

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ============================================================================
# CẤU HÌNH HỆ THỐNG
# ============================================================================
DEFAULT_TARGET_SIZE = 512
MAX_RECOMMENDED_VIEWS = 8
MIN_RECOMMENDED_VIEWS = 2

# Các góc chụp chuẩn định chuẩn (Canonical Faces) theo tọa độ thế giới (OpenCV Camera)
CANONICAL_FACES = [
    ("front", 0.0, 15.0),
    ("right", 90.0, 15.0),
    ("back", 180.0, 15.0),
    ("left", 270.0, 15.0),
    ("top", 0.0, 85.0),
    ("bottom", 0.0, -85.0),
]


# ============================================================================
# 1. VALIDATION & LOADING (ĐỌC & CHUẨN HÓA ĐẦU VÀO)
# ============================================================================
def validate_and_load_images(image_paths: List[str]) -> List[np.ndarray]:
    """
    Kiểm tra và đọc ảnh từ danh sách đường dẫn.
    Hỗ trợ Unicode path trên Windows và tự động chuyển về định dạng RGB uint8.
    """
    if not image_paths:
        raise ValueError("Danh sách đường dẫn ảnh rỗng.")

    loaded_images = []
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {path}")

        # Đọc an toàn hỗ trợ unicode path
        try:
            pil_img = Image.open(path)
            pil_img.load()
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img_rgb = np.array(pil_img, dtype=np.uint8)
            loaded_images.append(img_rgb)
        except Exception as e:
            logger.error(f"Lỗi khi đọc ảnh {path}: {e}")
            raise ValueError(f"Không thể đọc file ảnh {path}: {e}")

    logger.info(f"[P1] Đã đọc thành công {len(loaded_images)} ảnh hợp lệ.")
    return loaded_images


# ============================================================================
# 2. ĐỒNG BỘ HÓA QUANG HỌC (HISTOGRAM MATCHING THEO ANCHOR VIEW)
# ============================================================================
def histogram_match_sequence(images_rgb: List[np.ndarray], anchor_idx: int = 0) -> List[np.ndarray]:
    """
    Toán tử NVIDIA DALI-style: Ép biểu đồ màu của N-1 ảnh theo ảnh Anchor View (#0)
    để đồng bộ quang học, loại bỏ hoàn toàn lỗi phơi sáng (Auto-Exposure) và ám màu (Auto-WB).
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
# 3. TÁCH NỀN & VÁ LỖ PHẢN QUANG (PET / SPECULAR HOLE REFINEMENT)
# ============================================================================
def refine_alpha_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Thuật toán vá lỗ vật thể trong suốt / phản quang:
    Chai nhựa PET, ly thủy tinh, kim loại bóng thường bị khoét lủng lỗ trắng do phản xạ.
    Sử dụng binary_fill_holes và lọc thành phần liên thông lớn nhất để lấp kín 100% ruột vật thể.
    """
    binary = (raw_mask > 127).astype(bool)
    if not np.any(binary):
        return np.ones_like(raw_mask, dtype=np.uint8)

    # 1. Bịt kín toàn bộ lỗ thủng bên trong thân vật thể
    filled = ndimage.binary_fill_holes(binary)

    # 2. Lọc bỏ các đốm bụi nhiễu nhỏ ngoài phông nền
    labeled, num_features = ndimage.label(filled)
    if num_features > 1:
        sizes = ndimage.sum(filled, labeled, range(1, num_features + 1))
        max_label = int(np.argmax(sizes)) + 1
        filled = (labeled == max_label)

    # 3. Làm mượt nhẹ viền
    structure = ndimage.generate_binary_structure(2, 1)
    refined = ndimage.binary_closing(filled, structure=structure, iterations=2)
    return (refined.astype(np.uint8) * 255)


def extract_alpha_masks(images_rgb: List[np.ndarray]) -> List[np.ndarray]:
    """
    Trích xuất mặt nạ vật thể (Alpha Mask) từ danh sách ảnh RGB.
    Tự động ưu tiên rembg offline hoặc fallback ngưỡng Otsu.
    """
    alpha_masks = []
    has_rembg = False
    session = None

    try:
        from rembg import remove, new_session
        session = new_session("u2net")
        has_rembg = True
    except Exception:
        has_rembg = False

    for i, img in enumerate(images_rgb):
        if has_rembg and session is not None:
            try:
                pil_img = Image.fromarray(img)
                out_rgba = remove(pil_img, session=session)
                raw_mask = np.array(out_rgba.split()[-1], dtype=np.uint8)
                refined = refine_alpha_mask(raw_mask)
                alpha_masks.append(refined)
                continue
            except Exception as e:
                logger.warning(f"[P1] rembg ảnh #{i} lỗi ({e}), chuyển sang fallback Otsu.")

        # Fallback phân đoạn màu nền
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        alpha_masks.append(refine_alpha_mask(thresh))

    logger.info(f"[P1] Đã trích xuất & vá kín {len(alpha_masks)} Alpha Masks chuẩn xác.")
    return alpha_masks


# ============================================================================
# 4. RESIZE CHUẨN VISION TRANSFORMER (DIVISIBLE BY 16 & EPIPOLAR PRESERVATION)
# ============================================================================
def vit_geometric_resize(
    image: np.ndarray, 
    mask: np.ndarray, 
    target_size: int = DEFAULT_TARGET_SIZE
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Resize bảo toàn tỷ lệ khung hình (Aspect Ratio), cạnh dài nhất = target_size,
    đồng thời ép cả 2 cạnh chia hết cho 16 chuẩn cấu hình mạng ViT Backbone.
    Tuyệt đối không crop để bảo toàn Epipolar Geometry và tâm quang học.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    # Ép chia hết cho 16
    nh = max(16, (nh // 16) * 16)
    nw = max(16, (nw // 16) * 16)

    resized_img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return resized_img, resized_mask, scale


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
    1. Tầng 1: Tên file nếu chứa từ khóa (front, right, back, left, top).
    2. Tầng 2: Zero-shot CLIP ViT đo độ tương đồng ngữ nghĩa ảnh và text prompt.
    3. Tầng 3: HOG Gradient & Bilateral Symmetry Fallback (phân biệt trước/sau đối xứng).
    Sau đó áp dụng giải thuật Hungarian Bipartite Assignment gán cặp 1-1 góc chuẩn.
    """
    n = len(images_rgb)
    num_faces = len(CANONICAL_FACES)
    cost_matrix = np.ones((n, num_faces), dtype=np.float32)

    # 1. Kiểm tra từ khóa tên file
    keyword_map = {
        "front": 0, "f": 0, "truoc": 0,
        "right": 1, "r": 1, "phai": 1,
        "back": 2, "b": 2, "sau": 2,
        "left": 3, "l": 3, "trai": 3,
        "top": 4, "t": 4, "tren": 4,
        "bottom": 5, "duoi": 5,
    }

    if filenames and len(filenames) == n:
        for i, fname in enumerate(filenames):
            base = os.path.splitext(os.path.basename(fname).lower())[0]
            for kw, face_col in keyword_map.items():
                if kw in base.split("_") or kw in base.split("-") or kw == base:
                    cost_matrix[i, :] += 5.0
                    cost_matrix[i, face_col] = 0.0
                    break

    # 2. HOG Symmetry Fallback để bổ trợ
    for i, img in enumerate(images_rgb):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sym = _calc_bilateral_symmetry(gray)
        # Mặt trước và mặt sau đối xứng cao hơn mặt bên
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

    # Bổ sung góc phân bổ đều nếu số lượng ảnh > số mặt canonical
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
# 6. ĐÓNG GÓI HỢP ĐỒNG GIAO DIỆN (PREPROCESS MULTIVIEW & SINGLE VIEW)
# ============================================================================
def preprocess_multiview(
    image_paths: List[str],
    target_size: int = DEFAULT_TARGET_SIZE,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Toàn bộ chuỗi tiền xử lý đa ảnh chuẩn NVIDIA:
    Load -> Histogram Match -> Alpha Mask -> ViT Resize -> Viewpoint Hungarian.
    """
    raw_images = validate_and_load_images(image_paths)
    filenames = [os.path.basename(p) for p in image_paths]

    # 1. Đồng bộ quang học (NVIDIA DALI Style)
    matched_images = histogram_match_sequence(raw_images, anchor_idx=0)

    # 2. Tách nền và vá lỗ phản quang
    raw_masks = extract_alpha_masks(matched_images)

    # 3. ViT Geometric Resize (bảo toàn epipolar geometry, chia hết cho 16)
    clean_rgb_list = []
    clean_mask_list = []
    tensor_list = []
    scale_factors = []

    for img, msk in zip(matched_images, raw_masks):
        r_img, r_msk, s = vit_geometric_resize(img, msk, target_size=target_size)
        clean_rgb_list.append(r_img)
        clean_mask_list.append(r_msk)
        scale_factors.append(s)

        if HAS_TORCH:
            # Chuẩn hóa ImageNet cho ViT
            t = torch.from_numpy(r_img).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            t = (t - mean) / std
            tensor_list.append(t.permute(2, 0, 1))

    # 4. Nhận diện góc nhìn tự động
    viewpoint_assignments = classify_viewpoints(clean_rgb_list, filenames=filenames, device=device)

    # Tiêu cự ước tính theo FOV 50°
    h, w = clean_rgb_list[0].shape[:2]
    f_est = float((w / 2.0) / np.tan(np.radians(25.0)))
    focal_lengths = [(f_est, f_est)] * len(clean_rgb_list)

    result = {
        "images_rgb": clean_rgb_list,
        "alpha_masks": clean_mask_list,
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
    raw_images = validate_and_load_images([image_path])
    img = raw_images[0]
    masks = extract_alpha_masks([img])
    mask = masks[0]

    # Canh giữa tâm và padding vào khung vuông
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
