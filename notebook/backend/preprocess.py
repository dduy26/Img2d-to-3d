"""
Module tiền xử lý ảnh đa góc nhìn (Multi-view Preprocessing) cho Pipeline v1.
Thành viên 1 (Duy) - Data & Preprocessing Engineer.

Chức năng chính:
    - Kiểm tra & đọc ảnh hợp lệ (validate_and_load_images)
    - Lọc số lượng ảnh tối ưu (subsample_images)
    - Resize chuẩn DUSt3R (dust3r_resize)
    - Trích xuất Alpha Mask bằng RMBG-2.0 (extract_alpha_masks)
    - Cân bằng sáng giữa các góc nhìn (histogram_match)
    - Đóng gói tổng thể (preprocess_multiview)

Lý thuyết nền tảng: docs/lythuyet.md
Kế hoạch hành động: docs/planforAI.md
"""

import os
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np

try:
    from PIL import Image, ImageOps
except ImportError:
    raise ImportError("Cần cài Pillow: pip install Pillow")

# Cấu hình logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ============================================================================
# CONSTANTS
# ============================================================================

# Kích thước mục tiêu tối đa cho DUSt3R ViT backbone
DEFAULT_TARGET_SIZE: int = 512

# Patch size của Vision Transformer (ViT) - kích thước ảnh phải chia hết cho giá trị này
VIT_PATCH_SIZE: int = 16

# Giới hạn số lượng ảnh đa góc nhìn
MIN_MULTIVIEW_IMAGES: int = 2
OPTIMAL_MIN_IMAGES: int = 4
MAX_MULTIVIEW_IMAGES: int = 8
TARGET_SUBSAMPLE_COUNT: int = 6

# Ngưỡng nhị phân hóa Alpha Mask
ALPHA_THRESHOLD: float = 0.5

# Kích thước ảnh tối thiểu và tối đa cho phép
MIN_IMAGE_DIM: int = 32
MAX_IMAGE_DIM: int = 10000

# Định dạng ảnh hỗ trợ
SUPPORTED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# ImageNet normalization (cho DUSt3R ViT backbone)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ============================================================================
# HÀM 1: VALIDATE & LOAD IMAGES
# ============================================================================

def validate_and_load_images(image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
    """
    Kiểm tra tính hợp lệ và đọc danh sách ảnh đầu vào.

    Kiểm tra:
        - Đường dẫn file tồn tại
        - File có phần mở rộng hợp lệ (JPG, PNG, WEBP, BMP, TIFF)
        - File đọc được bằng Pillow (không bị hỏng, không phải file rỗng/giả)
        - Ảnh có kích thước hợp lý (>= MIN_IMAGE_DIM, <= MAX_IMAGE_DIM)
        - Tự động chuyển đổi sang RGB (xử lý grayscale, RGBA nền trắng, CMYK)

    Args:
        image_paths: Danh sách đường dẫn tuyệt đối hoặc tương đối tới các file ảnh.

    Returns:
        Danh sách dict, mỗi phần tử chứa:
            - 'image': PIL.Image.Image (chế độ RGB)
            - 'path': str (đường dẫn tuyệt đối chuẩn hóa)
            - 'filename': str (tên file)
            - 'original_size': Tuple[int, int] (W, H gốc)

    Raises:
        ValueError: Nếu không có ảnh hợp lệ nào hoặc image_paths rỗng.
    """
    if not image_paths:
        raise ValueError("Danh sách đường dẫn ảnh rỗng.")

    valid_images: List[Dict[str, Any]] = []
    skipped: List[str] = []
    seen_paths: set = set()

    for path_input in image_paths:
        path_str = str(path_input)
        path = Path(path_input)

        # Kiểm tra file tồn tại
        if not path.is_file():
            logger.warning(f"Bỏ qua: File không tồn tại — {path_str}")
            skipped.append(path_str)
            continue

        # Kiểm tra file rỗng (0 bytes)
        try:
            if path.stat().st_size == 0:
                logger.warning(f"Bỏ qua: File rỗng (0 bytes) — {path.name}")
                skipped.append(path_str)
                continue
        except OSError:
            skipped.append(path_str)
            continue

        # Kiểm tra phần mở rộng
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Bỏ qua: Định dạng không hỗ trợ ({ext}) — {path.name}")
            skipped.append(path_str)
            continue

        # Cảnh báo đường dẫn trùng lặp
        resolved_str = str(path.resolve())
        if resolved_str in seen_paths:
            logger.warning(f"Lưu ý: Phát hiện file trùng lặp đường dẫn — {path.name}")
        seen_paths.add(resolved_str)

        # Đọc file bằng Pillow và kiểm tra toàn vẹn
        try:
            with Image.open(str(path)) as raw_img:
                raw_img.verify()

            # Mở lại để load dữ liệu thực sự (vì verify() làm vô hiệu handle)
            img = Image.open(str(path))
            img.load()  # Ép giải mã dữ liệu pixel để phát hiện ảnh hỏng ngầm
        except Exception as e:
            logger.warning(f"Bỏ qua: Không đọc được file ảnh — {path.name} ({e})")
            skipped.append(path_str)
            continue

        # Xử lý xoay theo EXIF orientation nếu có
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        w_orig, h_orig = img.size

        # Kiểm tra kích thước biên
        if w_orig < MIN_IMAGE_DIM or h_orig < MIN_IMAGE_DIM:
            logger.warning(
                f"Bỏ qua: Ảnh quá nhỏ ({w_orig}x{h_orig}, tối thiểu {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}) — {path.name}"
            )
            skipped.append(path_str)
            img.close()
            continue

        if w_orig > MAX_IMAGE_DIM or h_orig > MAX_IMAGE_DIM:
            logger.warning(
                f"Bỏ qua: Ảnh quá lớn ({w_orig}x{h_orig}, tối đa {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}) — {path.name}"
            )
            skipped.append(path_str)
            img.close()
            continue

        # Chuyển đổi an toàn sang RGB:
        # Nếu là RGBA hoặc LA: hòa trộn nền trắng để tránh viền đen khi khử alpha
        if img.mode in ("RGBA", "LA"):
            try:
                bg = Image.new("RGB", img.size, (255, 255, 255))
                alpha_channel = img.split()[-1]
                bg.paste(img.convert("RGB"), mask=alpha_channel)
                img.close()
                img = bg
            except Exception:
                img = img.convert("RGB")
        elif img.mode != "RGB":
            try:
                converted = img.convert("RGB")
                img.close()
                img = converted
            except Exception as e:
                logger.warning(f"Bỏ qua: Không thể chuyển sang RGB — {path.name} ({e})")
                skipped.append(path_str)
                img.close()
                continue

        valid_images.append({
            "image": img,
            "path": resolved_str,
            "filename": path.name,
            "original_size": (w_orig, h_orig),
        })

    if not valid_images:
        raise ValueError(
            f"Không có ảnh hợp lệ nào trong {len(image_paths)} file đầu vào. "
            f"Các file bị bỏ qua: {skipped}"
        )

    logger.info(
        f"Đã đọc {len(valid_images)}/{len(image_paths)} ảnh hợp lệ. "
        f"Bỏ qua {len(skipped)} file."
    )
    return valid_images


# ============================================================================
# HÀM 2: SUBSAMPLE IMAGES
# ============================================================================

def subsample_images(
    images: List[Dict[str, Any]],
    max_count: int = MAX_MULTIVIEW_IMAGES,
    target_count: int = TARGET_SUBSAMPLE_COUNT,
) -> List[Dict[str, Any]]:
    """
    Lọc số lượng ảnh tối ưu cho DUSt3R multi-view.

    Quy tắc:
        - N < MIN_MULTIVIEW_IMAGES: Raise ValueError.
        - MIN_MULTIVIEW_IMAGES <= N < OPTIMAL_MIN_IMAGES: Cảnh báo, vẫn giữ nguyên.
        - OPTIMAL_MIN_IMAGES <= N <= max_count: Giữ nguyên toàn bộ.
        - N > max_count: Uniform Subsampling về target_count ảnh,
          luôn giữ ảnh đầu tiên và ảnh cuối cùng. Thuật toán deterministic.

    Args:
        images: Danh sách ảnh đã validate (output từ validate_and_load_images).
        max_count: Số lượng ảnh tối đa cho phép trước khi subsample (mặc định 8).
        target_count: Số lượng ảnh mục tiêu khi cần subsample (mặc định 6).

    Returns:
        Danh sách ảnh đã lọc (cùng format dict như input).

    Raises:
        ValueError: Nếu số ảnh < MIN_MULTIVIEW_IMAGES.
    """
    n = len(images)

    if n < MIN_MULTIVIEW_IMAGES:
        raise ValueError(
            f"Cần ít nhất {MIN_MULTIVIEW_IMAGES} ảnh cho multi-view reconstruction. "
            f"Chỉ nhận được {n} ảnh."
        )

    if n < OPTIMAL_MIN_IMAGES:
        logger.warning(
            f"Số ảnh ({n}) ít hơn mức tối ưu ({OPTIMAL_MIN_IMAGES}). "
            f"Chất lượng tái tạo 3D có thể giảm."
        )

    if n <= max_count:
        logger.info(f"Giữ nguyên toàn bộ {n} ảnh (trong ngưỡng cho phép <= {max_count}).")
        return images

    # Đảm bảo target_count hợp lý
    effective_target = min(target_count, max_count)
    effective_target = max(MIN_MULTIVIEW_IMAGES, effective_target)

    logger.info(f"Subsample: {n} ảnh → {effective_target} ảnh (Uniform Subsampling).")

    # Thuật toán Uniform Subsampling:
    # Phân bố đều các mốc chỉ số từ 0 đến n - 1, làm tròn số nguyên
    raw_indices = np.round(np.linspace(0, n - 1, effective_target)).astype(int).tolist()

    # Bảo đảm duy nhất và giữ thứ tự
    seen = set()
    indices = []
    for idx in raw_indices:
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)

    # Nếu trùng lặp do làm tròn, bổ sung các index kế cận còn trống
    if len(indices) < effective_target:
        for candidate in range(n):
            if candidate not in seen:
                indices.append(candidate)
                seen.add(candidate)
                if len(indices) == effective_target:
                    break
        indices.sort()

    # Luôn đảm bảo giữ chỉ số đầu tiên (0) và chỉ số cuối cùng (n - 1)
    indices[0] = 0
    indices[-1] = n - 1
    indices = sorted(list(set(indices)))

    selected = [images[i] for i in indices]
    logger.info(f"Đã chọn {len(selected)} ảnh tại các vị trí index: {indices}")
    return selected


# ============================================================================
# HÀM 3: DUST3R STANDARD IMAGE RESIZE
# ============================================================================

def _round_to_patch_size(value: int, patch_size: int = VIT_PATCH_SIZE) -> int:
    """Làm tròn giá trị về bội số gần nhất của patch_size (luôn >= patch_size)."""
    rounded = int(round(value / patch_size)) * patch_size
    return max(patch_size, rounded)


def dust3r_resize(
    img: Union[Image.Image, np.ndarray],
    target_size: int = DEFAULT_TARGET_SIZE,
    patch_size: int = VIT_PATCH_SIZE,
) -> Tuple[Image.Image, float]:
    """
    Resize ảnh theo chuẩn DUSt3R Standard Loader.

    Bảo toàn tuyệt đối:
        - Aspect ratio gốc (co dãn đồng tỉ lệ)
        - Tâm quang học (không crop bất đối xứng)
        - Cả chiều dài và chiều rộng chia hết cho patch_size (16)
        - max(H_new, W_new) <= target_size (nếu target_size chia hết cho patch_size)

    Xem chi tiết lý thuyết: docs/lythuyet.md (Mục 1.3)

    Args:
        img: Ảnh PIL.Image.Image hoặc np.ndarray (RGB).
        target_size: Kích thước cạnh lớn nhất mục tiêu (mặc định 512).
        patch_size: Patch size của ViT backbone (mặc định 16).

    Returns:
        Tuple gồm:
            - Ảnh đã resize (PIL.Image.Image, RGB)
            - Tỉ lệ co dãn (float): max(w_new / w_orig, h_new / h_orig)
    """
    # Chuyển đổi np.ndarray sang PIL nếu cần
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pil_img = img
    else:
        raise TypeError(f"Loại ảnh không hỗ trợ: {type(img)}. Cần PIL.Image hoặc np.ndarray.")

    w_orig, h_orig = pil_img.size
    if w_orig <= 0 or h_orig <= 0:
        raise ValueError(f"Kích thước ảnh không hợp lệ: {w_orig}x{h_orig}")

    # Đảm bảo target_size là bội số của patch_size
    target_size_aligned = _round_to_patch_size(target_size, patch_size)

    # Tính tỉ lệ co dãn theo cạnh lớn nhất
    max_dim = max(w_orig, h_orig)
    scale = target_size_aligned / max_dim

    # Tính kích thước mới và căn chỉnh về bội số patch_size
    w_calc = int(round(w_orig * scale))
    h_calc = int(round(h_orig * scale))

    w_new = _round_to_patch_size(w_calc, patch_size)
    h_new = _round_to_patch_size(h_calc, patch_size)

    # Giới hạn không để kích thước vượt quá target_size_aligned
    if w_orig >= h_orig:
        w_new = min(w_new, target_size_aligned)
        h_new = _round_to_patch_size(int(round(w_new * (h_orig / w_orig))), patch_size)
    else:
        h_new = min(h_new, target_size_aligned)
        w_new = _round_to_patch_size(int(round(h_new * (w_orig / h_orig))), patch_size)

    w_new = max(patch_size, w_new)
    h_new = max(patch_size, h_new)

    resized = pil_img.resize((w_new, h_new), Image.LANCZOS)
    actual_scale = max(w_new / w_orig, h_new / h_orig)

    logger.debug(
        f"Resize: ({w_orig}x{h_orig}) → ({w_new}x{h_new}), "
        f"scale={actual_scale:.4f}, "
        f"divisible_by_{patch_size}: W={w_new % patch_size == 0}, H={h_new % patch_size == 0}"
    )

    return resized, actual_scale


# ============================================================================
# HÀM 4: EXTRACT ALPHA MASKS (RMBG-2.0)
# ============================================================================

def extract_alpha_masks(
    images: List[Union[Image.Image, np.ndarray]],
    device: str = "cpu",
    threshold: float = ALPHA_THRESHOLD,
) -> List[np.ndarray]:
    """
    Trích xuất Alpha Mask nhị phân cho từng ảnh bằng RMBG-2.0.

    Chạy tuần tự từng ảnh để tránh tràn VRAM.
    KHÔNG xóa nền ảnh — chỉ xuất mask để P4 (TSDF Mesh) dùng cho Point Pruning.
    Xem lý do chi tiết: docs/lythuyet.md (Mục 2.2)

    Args:
        images: Danh sách ảnh PIL.Image.Image hoặc np.ndarray (RGB, đã resize).
        device: Thiết bị chạy model ('cpu' hoặc 'cuda').
        threshold: Ngưỡng nhị phân hóa alpha matte (mặc định 0.5).

    Returns:
        Danh sách np.ndarray, mỗi mask có shape (H, W), dtype uint8, giá trị {0, 1}.
    """
    masks: List[np.ndarray] = []
    if not images:
        return masks

    # Chuẩn hóa đầu vào về PIL Image
    pil_images: List[Image.Image] = []
    for item in images:
        if isinstance(item, np.ndarray):
            pil_images.append(Image.fromarray(item))
        elif isinstance(item, Image.Image):
            pil_images.append(item)
        else:
            raise TypeError(f"Ảnh không đúng định dạng: {type(item)}")

    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
        has_torch = True
    except ImportError:
        has_torch = False

    if not has_torch:
        logger.warning(
            "Không tìm thấy torch/transformers. "
            "Tạo mask giả lập (toàn 1) để tiếp tục pipeline mà không cần GPU/model."
        )
        for img in pil_images:
            w, h = img.size
            masks.append(np.ones((h, w), dtype=np.uint8))
        return masks

    # Nếu không có HF token trong môi trường, ưu tiên dùng rembg (u2net) offline trực tiếp
    # để tránh gọi briaai/RMBG-2.0 bị lỗi 401 Unauthorized (gated repo)
    has_hf_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not has_hf_token:
        try:
            import rembg
            logger.info("Chạy tách nền bằng rembg (u2net) offline...")
            for img in pil_images:
                rgba = rembg.remove(img)
                alpha = np.array(rgba.split()[3])
                mask = (alpha > int(threshold * 255)).astype(np.uint8)
                masks.append(mask)
            logger.info(f"  ✓ Đã trích xuất {len(masks)} Alpha Masks bằng rembg!")
            return masks
        except Exception as rembg_err:
            logger.warning(f"rembg offline không khả dụng: {rembg_err}. Thử tải RMBG-2.0...")

    # Tải model RMBG-2.0 an toàn (nếu có token hoặc rembg không có)
    logger.info(f"Đang tải model briaai/RMBG-2.0 trên {device}...")
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", trust_remote_code=True
        )
        model = model.to(device)
        model.eval()
    except Exception as e:
        logger.warning(f"Không thể tải RMBG-2.0: {e}. Kích hoạt fallback sang rembg (u2net)...")
        try:
            import rembg
            for img in pil_images:
                rgba = rembg.remove(img)
                alpha = np.array(rgba.split()[3])
                mask = (alpha > int(threshold * 255)).astype(np.uint8)
                masks.append(mask)
            logger.info(f"  ✓ Đã trích xuất {len(masks)} Alpha Masks thành công bằng rembg!")
            return masks
        except Exception as rembg_err:
            logger.warning(f"rembg fallback cũng thất bại: {rembg_err}. Tạo mask toàn 1.")
            for img in pil_images:
                w, h = img.size
                masks.append(np.ones((h, w), dtype=np.uint8))
            return masks

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])

    try:
        for i, img in enumerate(pil_images):
            logger.info(f"  RMBG-2.0: Xử lý ảnh {i + 1}/{len(pil_images)}...")
            orig_w, orig_h = img.size

            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)

            if isinstance(output, (list, tuple)):
                pred = output[0]
            else:
                pred = output

            # Xử lý dimensions an toàn
            while pred.dim() > 2:
                pred = pred[0]

            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)

            pred_resized = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            mask_np = (pred_resized.cpu().numpy() > threshold).astype(np.uint8)
            masks.append(mask_np)

            del input_tensor, output, pred, pred_resized
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Đã trích xuất {len(masks)} Alpha Masks.")
    return masks


# ============================================================================
# HÀM 5: HISTOGRAM MATCHING (CÂN BẰNG SÁNG)
# ============================================================================

def histogram_match(
    images: List[np.ndarray],
    reference_index: int = 0,
) -> List[np.ndarray]:
    """
    Cân bằng sáng giữa các ảnh đa góc nhìn bằng Histogram Matching.

    Dùng ảnh tại reference_index làm ảnh tham chiếu.
    Cân bằng độc lập trên từng kênh RGB bằng thuật toán CDF mapping vector hóa.
    Xem chi tiết toán học: docs/lythuyet.md (Mục 3)

    Args:
        images: Danh sách np.ndarray, mỗi ảnh shape (H, W, 3), dtype uint8.
        reference_index: Chỉ số ảnh tham chiếu (mặc định 0 = ảnh đầu tiên).

    Returns:
        Danh sách np.ndarray đã cân bằng sáng, cùng shape và dtype như input.
    """
    if not images:
        return []

    if len(images) == 1:
        return [images[0].copy()]

    if reference_index < 0 or reference_index >= len(images):
        reference_index = 0
        logger.warning("reference_index không hợp lệ, tự động dùng ảnh đầu tiên làm tham chiếu.")

    ref_img = images[reference_index]
    matched_images: List[np.ndarray] = []

    for i, img in enumerate(images):
        if i == reference_index:
            matched_images.append(img.copy())
            continue

        matched = _match_histogram_single(img, ref_img)
        matched_images.append(matched)

    logger.info(
        f"Histogram Matching: Đã cân bằng {len(images)} ảnh "
        f"theo ảnh tham chiếu #{reference_index}."
    )
    return matched_images


def _match_histogram_single(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Cân bằng histogram của ảnh source theo ảnh reference một cách vector hóa.
    Xử lý độc lập trên từng kênh màu.

    Args:
        source: Ảnh nguồn, shape (H, W, C) hoặc (H, W), dtype uint8.
        reference: Ảnh tham chiếu, shape (H, W, C) hoặc (H, W), dtype uint8.

    Returns:
        Ảnh đã cân bằng, cùng shape và dtype uint8.
    """
    if source.ndim == 2:
        src_expanded = source[:, :, np.newaxis]
        ref_expanded = reference[:, :, np.newaxis] if reference.ndim == 2 else reference[:, :, :1]
        is_2d = True
    else:
        src_expanded = source
        ref_expanded = reference
        is_2d = False

    num_channels = min(src_expanded.shape[2], ref_expanded.shape[2])
    result = src_expanded.copy()

    for ch in range(num_channels):
        src_ch = src_expanded[:, :, ch].ravel()
        ref_ch = ref_expanded[:, :, ch].ravel()

        # Tính histogram phân bố tần suất 256 mức xám
        src_hist, _ = np.histogram(src_ch, bins=256, range=(0, 256))
        ref_hist, _ = np.histogram(ref_ch, bins=256, range=(0, 256))

        # Tính hàm phân phối tích lũy CDF
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)

        # Chuẩn hóa CDF về đoạn [0, 1]
        src_max = src_cdf[-1]
        ref_max = ref_cdf[-1]
        src_cdf_norm = src_cdf / src_max if src_max > 0 else src_cdf
        ref_cdf_norm = ref_cdf / ref_max if ref_max > 0 else ref_cdf

        # Xây dựng lookup table vector hóa:
        # Với mỗi mức xám s in [0, 255], tìm r sao cho |ref_cdf(r) - src_cdf(s)| nhỏ nhất
        diff_matrix = np.abs(ref_cdf_norm[:, np.newaxis] - src_cdf_norm[np.newaxis, :])
        lookup = np.argmin(diff_matrix, axis=0).astype(np.uint8)

        # Áp dụng bảng tra
        result[:, :, ch] = lookup[src_expanded[:, :, ch]]

    if is_2d:
        return result[:, :, 0]
    return result


# ============================================================================
# HÀM 6: PREPROCESS MULTIVIEW (ĐÓNG GÓI TỔNG THỂ)
# ============================================================================

def preprocess_multiview(
    image_paths: List[Union[str, Path]],
    target_size: int = DEFAULT_TARGET_SIZE,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Hàm đóng gói tổng thể: Tiền xử lý ảnh đa góc nhìn cho Pipeline v1.

    Quy trình tuần tự:
        1. Validate & Load ảnh hợp lệ (RGB, kích thước chuẩn)
        2. Subsample nếu N > 8 về 6 ảnh (Uniform Subsampling)
        3. Resize chuẩn DUSt3R (bảo toàn epipolar geometry, chia hết cho 16)
        4. Trích xuất Alpha Mask bằng RMBG-2.0 (Mask nhị phân {0, 1})
        5. Cân bằng sáng Histogram Matching theo ảnh đầu tiên
        6. Chuẩn hóa Tensor ImageNet cho DUSt3R

    Args:
        image_paths: Danh sách đường dẫn file ảnh đầu vào.
        target_size: Kích thước cạnh lớn nhất mục tiêu (mặc định 512).
        device: Thiết bị chạy RMBG-2.0 ('cpu' hoặc 'cuda').

    Returns:
        Dict chứa 7 trường dữ liệu chuẩn:
            'images_rgb': List[np.ndarray] ảnh RGB đã resize (H, W, 3) uint8 — cho P5 Texturing
            'images_normalized': np.ndarray (N, 3, H, W) float32 chuẩn hóa ImageNet — cho P2 DUSt3R
            'alpha_masks': List[np.ndarray] mỗi mask shape (H, W) uint8 {0,1} — cho P4 Point Pruning
            'original_sizes': List[Tuple[int, int]] kích thước gốc (W, H)
            'scale_factors': List[float] tỉ lệ co dãn mỗi ảnh
            'filenames': List[str] tên file gốc
            'num_images': int số lượng ảnh hợp lệ sau xử lý
    """
    logger.info(f"═══ BẮT ĐẦU PREPROCESSING {len(image_paths)} ẢNH ═══")

    # --- Bước 1: Validate & Load ---
    logger.info("[1/5] Kiểm tra & đọc ảnh...")
    loaded = validate_and_load_images(image_paths)

    try:
        # --- Bước 2: Subsample ---
        logger.info("[2/5] Kiểm tra số lượng ảnh...")
        filtered = subsample_images(loaded)

        # --- Bước 3: DUSt3R Resize ---
        logger.info("[3/5] Resize chuẩn DUSt3R (bảo toàn epipolar geometry)...")
        resized_images: List[Image.Image] = []
        scale_factors: List[float] = []
        original_sizes: List[Tuple[int, int]] = []
        filenames: List[str] = []

        for item in filtered:
            resized_img, scale = dust3r_resize(item["image"], target_size=target_size)
            resized_images.append(resized_img)
            scale_factors.append(scale)
            original_sizes.append(item["original_size"])
            filenames.append(item["filename"])

        # Chuyển sang numpy array RGB
        images_rgb: List[np.ndarray] = [np.array(img) for img in resized_images]

        # --- Bước 4: Trích xuất Alpha Mask ---
        logger.info("[4/5] Trích xuất Alpha Mask (RMBG-2.0)...")
        alpha_masks = extract_alpha_masks(resized_images, device=device)

        # --- Bước 5: Cân bằng sáng ---
        logger.info("[5/5] Cân bằng sáng (Histogram Matching)...")
        images_rgb_matched = histogram_match(images_rgb, reference_index=0)

        # --- Chuẩn hóa Tensor cho DUSt3R ViT ---
        images_normalized = _normalize_for_dust3r(images_rgb_matched)

        result = {
            "images_rgb": images_rgb_matched,
            "images_normalized": images_normalized,
            "alpha_masks": alpha_masks,
            "original_sizes": original_sizes,
            "scale_factors": scale_factors,
            "filenames": filenames,
            "num_images": len(images_rgb_matched),
        }

        logger.info(
            f"═══ HOÀN THÀNH PREPROCESSING: {result['num_images']} ảnh, "
            f"output tensor: ({result['num_images']}, 3, "
            f"{images_normalized.shape[2]}, {images_normalized.shape[3]}) ═══"
        )
        return result

    finally:
        # Luôn giải phóng tài nguyên ảnh PIL
        for item in loaded:
            try:
                item["image"].close()
            except Exception:
                pass


# ============================================================================
# HÀM 7: PREPROCESS SINGLE VIEW (ĐÓNG GÓI CHO KỊCH BẢN 1 ẢNH — F1.2A)
# ============================================================================

def _center_crop_and_pad(
    image_rgb: np.ndarray,
    alpha_mask: np.ndarray,
    target_size: int = DEFAULT_TARGET_SIZE,
    fill_ratio: float = 0.82,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[float, float]]:
    """
    Canh tâm vật thể dựa trên Bounding Box của Alpha Mask, co dãn bảo toàn
    Aspect Ratio để vật thể chiếm ~80-85% khung hình, rồi đặt vào
    Square Letterbox Padding target_size × target_size.

    Đặc tả F1.2A trong docs/plan.md:
        Crop Bounding Box → Canh giữa tâm → Scale bảo toàn Aspect Ratio
        → Square Letterbox Padding 512×512

    Args:
        image_rgb: Ảnh RGB (H, W, 3) uint8.
        alpha_mask: Mask nhị phân (H, W) uint8 {0, 1}.
        target_size: Kích thước khung vuông đầu ra (mặc định 512).
        fill_ratio: Tỉ lệ vật thể chiếm trong khung (mặc định 0.82 ≈ 82%).

    Returns:
        Tuple gồm:
            - image_padded: np.ndarray (target_size, target_size, 3) uint8
            - mask_padded: np.ndarray (target_size, target_size) uint8
            - crop_scale: float — hệ số co dãn khi scale vào khung
            - offset: Tuple[float, float] — (delta_x, delta_y) dời tâm
    """
    h, w = image_rgb.shape[:2]

    # Tìm Bounding Box của vật thể từ alpha mask
    ys, xs = np.where(alpha_mask > 0)
    if len(ys) == 0:
        # Mask rỗng (không phát hiện được vật thể) → dùng toàn bộ ảnh
        logger.warning("Alpha mask rỗng, sử dụng toàn bộ ảnh làm vùng quan tâm.")
        x_min, y_min, x_max, y_max = 0, 0, w, h
    else:
        x_min, y_min = int(xs.min()), int(ys.min())
        x_max, y_max = int(xs.max()) + 1, int(ys.max()) + 1

    # Mở rộng Bounding Box thêm 5% margin mỗi chiều để tránh cắt sát mép
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    margin_x = int(bbox_w * 0.05)
    margin_y = int(bbox_h * 0.05)
    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(w, x_max + margin_x)
    y_max = min(h, y_max + margin_y)

    # Crop vùng chứa vật thể
    cropped_rgb = image_rgb[y_min:y_max, x_min:x_max]
    cropped_mask = alpha_mask[y_min:y_max, x_min:x_max]

    crop_h, crop_w = cropped_rgb.shape[:2]

    # Tính scale để vật thể chiếm fill_ratio khung target_size
    scale_x = (target_size * fill_ratio) / crop_w
    scale_y = (target_size * fill_ratio) / crop_h
    crop_scale = min(scale_x, scale_y)  # Bảo toàn Aspect Ratio

    new_w = max(1, int(round(crop_w * crop_scale)))
    new_h = max(1, int(round(crop_h * crop_scale)))

    # Resize ảnh crop
    from PIL import Image as _PILImage
    pil_cropped = _PILImage.fromarray(cropped_rgb)
    pil_resized = pil_cropped.resize((new_w, new_h), _PILImage.LANCZOS)
    resized_rgb = np.array(pil_resized)

    # Resize mask crop (nearest neighbor để giữ nhị phân)
    pil_mask_cropped = _PILImage.fromarray(cropped_mask * 255)
    pil_mask_resized = pil_mask_cropped.resize((new_w, new_h), _PILImage.NEAREST)
    resized_mask = (np.array(pil_mask_resized) > 127).astype(np.uint8)

    # Letterbox Padding: đặt vật thể vào giữa khung vuông
    image_padded = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    mask_padded = np.zeros((target_size, target_size), dtype=np.uint8)

    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2

    image_padded[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_rgb
    mask_padded[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_mask

    delta_x = float(offset_x - x_min * crop_scale)
    delta_y = float(offset_y - y_min * crop_scale)

    logger.debug(
        f"Center crop: bbox=({x_min},{y_min},{x_max},{y_max}), "
        f"crop=({crop_w}x{crop_h}), scale={crop_scale:.4f}, "
        f"placed=({new_w}x{new_h}) at offset=({offset_x},{offset_y})"
    )

    return image_padded, mask_padded, crop_scale, (delta_x, delta_y)


def preprocess_single_view(
    image_path: Union[str, Path],
    target_size: int = DEFAULT_TARGET_SIZE,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Hàm tiền xử lý ảnh đơn (Single-view) cho Kịch bản 1 — Đặc tả F1.2A.

    Quy trình tuần tự:
        1. Validate & Load ảnh (tái sử dụng validate_and_load_images)
        2. Resize giữ aspect ratio (tái sử dụng dust3r_resize)
        3. Trích xuất Alpha Mask bằng RMBG-2.0 (tái sử dụng extract_alpha_masks)
        4. Canh tâm & Scale Normalization: Crop BBox → Center → Letterbox Padding
        5. Ước tính Camera Intrinsics (focal length) từ kích thước ảnh

    Args:
        image_path: Đường dẫn file ảnh đầu vào (1 file duy nhất).
        target_size: Kích thước cạnh khung vuông đầu ra (mặc định 512).
        device: Thiết bị chạy RMBG-2.0 ('cpu' hoặc 'cuda').

    Returns:
        Dict chứa các trường dữ liệu chuẩn:
            'image_rgb': np.ndarray (H, W, 3) uint8 — ảnh gốc đã resize (chưa canh tâm)
            'image_centered': np.ndarray (target_size, target_size, 3) uint8 — ảnh đã canh tâm + padding
            'alpha_mask': np.ndarray (H, W) uint8 {0,1} — mask trên ảnh đã resize
            'alpha_mask_centered': np.ndarray (target_size, target_size) uint8 {0,1} — mask đã canh tâm
            'focal_length': Tuple[float, float] — Camera Intrinsics ước tính (fx, fy)
            'camera_intrinsics': Dict — Ma trận K đầy đủ {'fx', 'fy', 'cx', 'cy'}
            'original_size': Tuple[int, int] — (W, H) gốc
            'scale_factor': float — tỉ lệ co dãn resize
            'crop_scale': float — tỉ lệ co dãn khi canh tâm
            'filename': str — tên file gốc
    """
    logger.info(f"═══ BẮT ĐẦU PREPROCESSING ĐƠN ẢNH: {image_path} ═══")

    # --- Bước 1: Validate & Load ---
    logger.info("[1/5] Kiểm tra & đọc ảnh...")
    loaded = validate_and_load_images([str(image_path)])
    item = loaded[0]

    try:
        # --- Bước 2: Resize giữ aspect ratio ---
        logger.info("[2/5] Resize ảnh...")
        resized_img, scale_factor = dust3r_resize(item["image"], target_size=target_size)
        image_rgb = np.array(resized_img)

        # --- Bước 3: Trích xuất Alpha Mask ---
        logger.info("[3/5] Trích xuất Alpha Mask (RMBG-2.0)...")
        masks = extract_alpha_masks([resized_img], device=device)
        alpha_mask = masks[0]

        # --- Bước 4: Canh tâm & Scale Normalization ---
        logger.info("[4/5] Canh tâm & Scale Normalization (F1.2A)...")
        image_centered, mask_centered, crop_scale, offset = _center_crop_and_pad(
            image_rgb, alpha_mask, target_size=target_size
        )

        # --- Bước 5: Ước tính Camera Intrinsics ---
        logger.info("[5/5] Ước tính Camera Intrinsics...")
        # Theo tài liệu danh_gia_va_luong_hoat_dong_2d_to_3d.md:
        # fx = fy = max(width, height) — focal length tương đối
        # cx, cy = width/2, height/2 — principal point
        h_img, w_img = image_rgb.shape[:2]
        fx = fy = float(max(w_img, h_img))
        cx, cy = w_img / 2.0, h_img / 2.0

        # Cập nhật Camera Intrinsics K → K' theo crop_scale và offset (F1.2A)
        fx_prime = fx * crop_scale
        fy_prime = fy * crop_scale
        cx_prime = cx * crop_scale + offset[0]
        cy_prime = cy * crop_scale + offset[1]

        result = {
            "image_rgb": image_rgb,
            "image_centered": image_centered,
            "alpha_mask": alpha_mask,
            "alpha_mask_centered": mask_centered,
            "focal_length": (fx_prime, fy_prime),
            "camera_intrinsics": {
                "fx": fx_prime, "fy": fy_prime,
                "cx": cx_prime, "cy": cy_prime,
                "fx_original": fx, "fy_original": fy,
                "cx_original": cx, "cy_original": cy,
            },
            "original_size": item["original_size"],
            "scale_factor": scale_factor,
            "crop_scale": crop_scale,
            "filename": item["filename"],
        }

        logger.info(
            f"═══ HOÀN THÀNH PREPROCESSING ĐƠN ẢNH: "
            f"rgb={image_rgb.shape}, centered={image_centered.shape}, "
            f"focal=({fx_prime:.1f}, {fy_prime:.1f}) ═══"
        )
        return result

    finally:
        try:
            item["image"].close()
        except Exception:
            pass


def _normalize_for_dust3r(images: List[np.ndarray]) -> np.ndarray:
    """
    Chuẩn hóa danh sách ảnh RGB thành tensor (N, 3, H, W) float32
    theo chuẩn ImageNet normalization cho DUSt3R ViT backbone.

    Xem chi tiết: docs/lythuyet.md (Mục 5)

    Args:
        images: Danh sách np.ndarray, mỗi ảnh shape (H, W, 3), dtype uint8.

    Returns:
        np.ndarray shape (N, 3, H, W), dtype float32, chuẩn hóa ImageNet.
    """
    if not images:
        return np.zeros((0, 3, 0, 0), dtype=np.float32)

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    batch = np.zeros((len(images), 3, max_h, max_w), dtype=np.float32)

    for i, img in enumerate(images):
        h, w = img.shape[:2]
        transposed = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        batch[i, :, :h, :w] = transposed

    batch = (batch - mean) / std
    return batch
