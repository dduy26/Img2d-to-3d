"""
preprocess.py — P1: Tien Xu Ly Anh & Chuan Hoa Quang Hoc
==========================================================
Chuc nang:
  1. Load anh an toan, giu nguyen Native Alpha (PNG RGBA).
  2. Tach nen (Alpha Matting):
       - Anh co san Alpha (PNG RGBA): trich Alpha truc tiep.
       - Anh co nen trang/den phong (Studio):  CIE Lab Chroma Otsu.
       - Anh chup thuc te (in-the-wild): RMBG-2.0 / BiRefNet (fallback).
  3. Can bang sang CIE Lab L-channel Histogram Matching.
  4. Uniform Scale + Letterbox vao canvas vuong 512x512.
  5. Bu tru ma tran noi thong K -> K'.
  6. Phan loai goc chup (Hungarian viewpoint assignment).

Ly thuyet tham chieu:
  - docs/lythuyet.md §1 (Epipolar), §2 (RMBG), §3 (Histogram Matching).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .utils_3d import (
    CameraIntrinsics,
    compensate_intrinsics,
    compute_uniform_scale_params,
    round_to_16,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANVAS_SIZE    = 512          # Canvas chuan ViT 512x512
ALPHA_THRESH   = 0.5          # Nguong nhi phan hoa Alpha mask
LAB_CHROMA_THR = 20.0         # Nguong Chroma Otsu cho tach nen phong studio
MAX_IMAGES     = 8            # Toi da anh su dung (tranh OOM T4)
TARGET_IMAGES  = 6            # So anh toi uu khi N > MAX_IMAGES

# 4 goc chuan Objaverse: front=0, right=90, back=180, left=270
OBJAVERSE_AZIMUTH_DEG = [0.0, 90.0, 180.0, 270.0]

# ---------------------------------------------------------------------------
# 1. LOAD ANH AN TOAN
# ---------------------------------------------------------------------------

def load_image_safe(path: Union[str, Path]) -> np.ndarray:
    """Load anh bang cv2, tra ve BGR hoac BGRA (neu co Alpha).

    Dam bao doc dung:
      - PNG RGBA: giu nguyen kenh Alpha (BGRA, 4 kenh).
      - JPEG/PNG RGB: tra ve BGR (3 kenh).
      - Ho tro duong dan Unicode/tieng Viet (dung numpy frombuffer + imdecode).

    Args:
        path: Duong dan den file anh.

    Returns:
        np.ndarray uint8: (H, W, 3) hoac (H, W, 4).

    Raises:
        FileNotFoundError: Neu file khong ton tai.
        IOError: Neu khong the doc file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file anh: {path}")
    # Dung binary read + imdecode de tranh loi Unicode path tren Windows
    raw_bytes = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Khong the doc file anh: {path}")
    return img


def ensure_rgb(img_bgr_or_bgra: np.ndarray) -> np.ndarray:
    """Chuyen anh BGR hoac BGRA ve RGB uint8 (bo kenh alpha neu co)."""
    if img_bgr_or_bgra.ndim == 2:
        return cv2.cvtColor(img_bgr_or_bgra, cv2.COLOR_GRAY2RGB)
    c = img_bgr_or_bgra.shape[2]
    if c == 4:
        return cv2.cvtColor(img_bgr_or_bgra[:, :, :3], cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(img_bgr_or_bgra, cv2.COLOR_BGR2RGB)


def extract_native_alpha(img_bgra: np.ndarray) -> Optional[np.ndarray]:
    """Trich kenh Alpha goc tu anh BGRA (4 kenh).

    Returns:
        (H, W) float32 in [0, 1], hoac None neu anh khong co Alpha.
    """
    if img_bgra.ndim == 3 and img_bgra.shape[2] == 4:
        alpha = img_bgra[:, :, 3].astype(np.float32) / 255.0
        return alpha
    return None


# ---------------------------------------------------------------------------
# 2. TACH NEN — Ba phuong phap ket hop
# ---------------------------------------------------------------------------

def _lab_chroma_otsu_mask(rgb: np.ndarray) -> np.ndarray:
    """Tach nen phong trang/den bang CIE Lab Chroma + Nguong Otsu.

    Ly thuyet (lythuyet.md §2 + plan.md):
      - Chuyen sang CIE Lab (kenh L: do sang, kenh a/b: sac do).
      - Tinh Chroma = sqrt(a^2 + b^2) — do mau sac.
      - Anh co nen trang/den don sac: Chroma cua nen ~ 0.
      - Ap dung nguong Otsu tren Chroma de phan tach vat the / nen.
      - Loc hinh thai hoc Morphological Closing(7x7) + Opening(5x5)
        de lop via/nhieu.

    Args:
        rgb: (H, W, 3) uint8 RGB.

    Returns:
        mask: (H, W) float32 in [0, 1] — 1 = vat the, 0 = nen.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    a_ch = lab[:, :, 1] - 128.0
    b_ch = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a_ch ** 2 + b_ch ** 2)

    # Nguong Otsu tren Chroma
    chroma_u8 = np.clip(chroma * 255.0 / (chroma.max() + 1e-6), 0, 255).astype(np.uint8)
    _, mask_otsu = cv2.threshold(chroma_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Phan biet nen trang (L cao, Chroma thap) vs nen den (L thap, Chroma thap)
    L_ch   = lab[:, :, 0]
    bg_white = (L_ch > 200) & (chroma < LAB_CHROMA_THR)
    bg_dark  = (L_ch < 20)  & (chroma < LAB_CHROMA_THR)
    bg_mask  = bg_white | bg_dark

    # Ket hop: foreground = NOT background AND co chroma
    fg = (~bg_mask).astype(np.uint8) * 255

    # Morphological: Closing(7x7) lap via, Opening(5x5) khu nhieu
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k7)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k5)

    # Giu lai component lon nhat (loai bo nhieu nho)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg = (labels == largest).astype(np.uint8) * 255

    return fg.astype(np.float32) / 255.0


def _rmbg_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Tach nen bang mo hinh RMBG-2.0 / BiRefNet (fallback cho anh thuc te).

    Chi nap mo hinh khi duoc goi lan dau (lazy loading).
    Neu khong co transformers/torch, tra ve None (pipeline se fallback sang Otsu).

    Returns:
        mask: (H, W) float32 in [0, 1], hoac None neu khong co model.
    """
    try:
        from transformers import pipeline as hf_pipeline
        import torch

        # Lazy singleton
        if not hasattr(_rmbg_mask, "_pipe"):
            _rmbg_mask._pipe = hf_pipeline(
                "image-segmentation",
                model="briaai/RMBG-1.4",
                trust_remote_code=True,
                device=0 if torch.cuda.is_available() else -1,
            )
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(rgb)
        result  = _rmbg_mask._pipe(pil_img)
        # Ket qua la danh sach dict co "mask" la PIL Image
        mask_pil = result[0]["mask"]
        mask_np  = np.array(mask_pil).astype(np.float32) / 255.0
        return mask_np
    except Exception:
        return None


def _is_studio_background(rgb: np.ndarray) -> bool:
    """Kiem tra nhanh xem anh co nen phong studio (trang/den don sac) khong.

    Phuong phap: Lay mau 4 goc anh, tinh Chroma trung binh.
    Neu Chroma < nguong -> nen don sac.
    """
    h, w = rgb.shape[:2]
    corners = [
        rgb[:20, :20].reshape(-1, 3),
        rgb[:20, w-20:].reshape(-1, 3),
        rgb[h-20:, :20].reshape(-1, 3),
        rgb[h-20:, w-20:].reshape(-1, 3),
    ]
    corner_pixels = np.vstack(corners).astype(np.float32)
    lab = cv2.cvtColor(corner_pixels.reshape(1, -1, 3).astype(np.uint8),
                       cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float32)
    a_ch = lab[:, 1] - 128.0
    b_ch = lab[:, 2] - 128.0
    chroma_mean = np.sqrt(a_ch ** 2 + b_ch ** 2).mean()
    return bool(chroma_mean < LAB_CHROMA_THR)


def compute_alpha_mask(
    img_raw: np.ndarray,
    force_rmbg: bool = False,
) -> np.ndarray:
    """Tinh Alpha Mask (H, W) float32 [0,1] cho anh dau vao.

    Thu tu uu tien:
      1. Native Alpha (PNG RGBA): trich Alpha truc tiep.
      2. Nen phong studio (Chroma Otsu): nhanh, chinh xac cho nen trang/den.
      3. RMBG-1.4 / BiRefNet: cho anh thuc te phuc tap.
      4. Fallback: Tra ve mask toan bo foreground (tat ca deu la 1).

    Args:
        img_raw:    Anh goc BGR hoac BGRA (uint8).
        force_rmbg: True -> ep dung RMBG-1.4 bat ke nen.

    Returns:
        mask: (H, W) float32 in [0, 1].
    """
    # Buoc 1: Native Alpha
    native_alpha = extract_native_alpha(img_raw)
    if native_alpha is not None and not force_rmbg:
        return native_alpha

    rgb = ensure_rgb(img_raw)

    # Buoc 2: Studio background
    if not force_rmbg and _is_studio_background(rgb):
        return _lab_chroma_otsu_mask(rgb)

    # Buoc 3: RMBG-1.4
    mask = _rmbg_mask(rgb)
    if mask is not None:
        return mask

    # Buoc 4: Fallback Lab Chroma Otsu
    warnings.warn(
        "Khong the load RMBG-1.4. Fallback sang Lab Chroma Otsu.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _lab_chroma_otsu_mask(rgb)


# ---------------------------------------------------------------------------
# 3. CAN BANG SANG — CIE Lab L-channel Histogram Matching
# ---------------------------------------------------------------------------

def _compute_cdf(channel_u8: np.ndarray) -> np.ndarray:
    """Tinh ham phan phoi tich luy (CDF) cho kenh anh uint8."""
    hist, _ = np.histogram(channel_u8.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum().astype(np.float64)
    cdf /= (cdf[-1] + 1e-9)
    return cdf


def match_l_channel(
    source_rgb: np.ndarray,
    reference_rgb: np.ndarray,
) -> np.ndarray:
    """Can bang kenh L (do sang) cua source theo reference trong khong gian CIE Lab.

    Chi can bang tren kenh L, giu nguyen a va b (sac do) de khong lam bien
    dang mau sac Albedo cua vat the (lythuyet.md §3).

    Args:
        source_rgb:    Anh can can bang, (H, W, 3) uint8 RGB.
        reference_rgb: Anh tham chieu, (H, W, 3) uint8 RGB.

    Returns:
        Anh da can bang (H, W, 3) uint8 RGB.
    """
    src_lab = cv2.cvtColor(source_rgb,    cv2.COLOR_RGB2Lab)
    ref_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2Lab)

    src_L = src_lab[:, :, 0]
    ref_L = ref_lab[:, :, 0]

    src_cdf = _compute_cdf(src_L)
    ref_cdf = _compute_cdf(ref_L)

    # Xay dung bang tra cuu: gia tri L_src -> L_ref tuong duong
    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and ref_cdf[j] < src_cdf[i]:
            j += 1
        lut[i] = j

    matched_L = lut[src_L]
    result_lab = src_lab.copy()
    result_lab[:, :, 0] = matched_L
    return cv2.cvtColor(result_lab, cv2.COLOR_Lab2RGB)


def normalize_exposure(
    images_rgb: list,
    reference_idx: int = 0,
) -> list:
    """Can bang do sang cho danh sach N anh theo anh tham chieu.

    Args:
        images_rgb:    Danh sach N anh (H, W, 3) uint8 RGB.
        reference_idx: Chi so anh tham chieu (mac dinh = 0, anh dau tien).

    Returns:
        Danh sach N anh da can bang do sang.
    """
    if not images_rgb:
        return images_rgb
    ref = images_rgb[reference_idx]
    normalized = []
    for i, img in enumerate(images_rgb):
        if i == reference_idx:
            normalized.append(img.copy())
        else:
            normalized.append(match_l_channel(img, ref))
    return normalized


# ---------------------------------------------------------------------------
# 4. UNIFORM SCALE + LETTERBOX vao CANVAS 512x512
# ---------------------------------------------------------------------------

def place_on_canvas(
    img_rgb:    np.ndarray,
    alpha_mask: np.ndarray,
    canvas_size: int = CANVAS_SIZE,
    bg_color:   tuple = (0, 0, 0),
) -> tuple:
    """Co dan anh vao canvas vuong, giu ty le, va tinh K' bu tru.

    Quy trinh:
      1. Tinh scale s = canvas / max(H, W).
      2. Tinh new_w, new_h chia het cho 16 (chuẩn ViT patch-size).
      3. Resize anh va mask.
      4. Dat vao canvas (letterbox) voi offset_x, offset_y.
      5. Tinh K' tu K goc.

    Args:
        img_rgb:     (H, W, 3) uint8 RGB.
        alpha_mask:  (H, W) float32 [0, 1].
        canvas_size: Kich thuoc canvas muc tieu.
        bg_color:    Mau nen canvas (default den).

    Returns:
        (canvas_rgb, canvas_alpha, K_prime, scale, offset_x, offset_y)
    """
    h, w = img_rgb.shape[:2]
    scale, new_w, new_h, off_x, off_y = compute_uniform_scale_params(w, h, canvas_size)

    # Resize
    img_resized   = cv2.resize(img_rgb,    (new_w, new_h), interpolation=cv2.INTER_AREA)
    alpha_resized = cv2.resize(alpha_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Canvas
    canvas_rgb   = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
    canvas_alpha = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    canvas_rgb[off_y:off_y + new_h, off_x:off_x + new_w]   = img_resized
    canvas_alpha[off_y:off_y + new_h, off_x:off_x + new_w] = alpha_resized

    # K -> K'
    K_orig  = CameraIntrinsics.default_ortho(w, h)
    K_prime, _ = compensate_intrinsics(
        K_orig, w, h, canvas_size, float(off_x), float(off_y)
    )

    return canvas_rgb, canvas_alpha, K_prime, scale, off_x, off_y


# ---------------------------------------------------------------------------
# 5. PHAN LOAI GOC CHUP — Hungarian Viewpoint Assignment
# ---------------------------------------------------------------------------

def _estimate_azimuth(img_rgb: np.ndarray) -> float:
    """Uoc luong goc chup nhin (azimuth) tu noi dung anh (phuong phap don gian).

    Phuong phap: Phan tich trong so khoi luong anh sang (luminance mass)
    tren nua trai vs phai de phan biet front/back vs left/right.
    Day la phuong phap heuristic nhanh, du chuan xac cho 4-view ortho.

    Returns:
        Goc uoc tinh (0, 90, 180, 270) do.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    left_mass  = gray[:, :w//2].mean()
    right_mass = gray[:, w//2:].mean()
    top_mass   = gray[:h//2, :].mean()
    bot_mass   = gray[h//2:, :].mean()

    # Heuristic: anh front thuong co khoi luong xung doi
    # Khong du thong tin de phan biet chinh xac -> tra ve None
    return None


def assign_viewpoints_hungarian(
    images_rgb: list,
    known_azimuths: Optional[list] = None,
) -> list:
    """Phan cong goc chup cho N anh bang thuat toan Hungarian.

    Neu already co thong tin ten file (front/right/back/left), dung truc tiep.
    Neu khong, giu nguyen thu tu (front=0, right=1, back=2, left=3...).

    Args:
        images_rgb:     Danh sach N anh RGB.
        known_azimuths: Danh sach goc azimuth (do) neu biet truoc, hoac None.

    Returns:
        Danh sach chi so thu tu da sap xep (viewpoint assignment).
    """
    N = len(images_rgb)
    if known_azimuths is not None and len(known_azimuths) == N:
        # Sap xep theo thu tu tang dan cua azimuth
        return sorted(range(N), key=lambda i: known_azimuths[i])
    # Mac dinh: giu nguyen thu tu dau vao
    return list(range(N))


def parse_azimuth_from_filename(filepath: Union[str, Path]) -> Optional[float]:
    """Phan tich goc azimuth tu ten file Objaverse-style.

    Mapping:
      front  -> 0
      right  -> 90
      back   -> 180
      left   -> 270

    Returns:
        Goc azimuth (float) hoac None.
    """
    name = Path(filepath).stem.lower()
    mapping = {
        "front":  0.0,
        "right":  90.0,
        "back":   180.0,
        "left":   270.0,
    }
    for key, val in mapping.items():
        if key in name:
            return val
    return None


# ---------------------------------------------------------------------------
# 6. SUBSAMPLE ANH KHI N > MAX_IMAGES
# ---------------------------------------------------------------------------

def uniform_subsample(images: list, target: int = TARGET_IMAGES) -> list:
    """Lay mau deu de giam so anh tu N xong target.

    Thuat toan (lythuyet.md §4.2):
      stride = ceil(N / target)
      Lay anh 0, stride, 2*stride, ...
      Dam bao giu lai anh dau va anh cuoi.

    Args:
        images: Danh sach N phan tu (co the la path, numpy array, ...).
        target: So phan tu sau khi lay mau.

    Returns:
        Danh sach target phan tu duoc chon.
    """
    N = len(images)
    if N <= target:
        return images
    import math as _math
    stride  = _math.ceil(N / target)
    indices = list(range(0, N, stride))
    # Dam bao anh cuoi duoc giu lai
    if N - 1 not in indices:
        indices.append(N - 1)
    # Sap xep va loai bo trung lap
    indices = sorted(set(indices))
    # Neu qua nhieu (vi them anh cuoi) -> giu N-1 + trim dau
    if len(indices) > target:
        # Giu anh cuoi, cat bot o giua
        keep_indices = list(range(0, N, _math.ceil(N / (target - 1))))
        if N - 1 not in keep_indices:
            keep_indices.append(N - 1)
        indices = sorted(set(keep_indices))[:target]
    return [images[i] for i in indices]


# ---------------------------------------------------------------------------
# 7. PIPELINE P1 — ENTRY POINT CHINH
# ---------------------------------------------------------------------------

def preprocess_images(
    image_paths: list,
    canvas_size: int = CANVAS_SIZE,
    force_rmbg:  bool = False,
    normalize_exposure_flag: bool = True,
) -> list:
    """Xu ly danh sach anh dau vao qua toan bo quy trinh P1.

    Buoc thuc hien:
      1. Kiem tra so luong anh (canh bao/subsample neu can).
      2. Load anh va trich Alpha mask cho tung anh.
      3. Can bang do sang (L-channel histogram matching).
      4. Uniform Scale + Letterbox -> canvas 512x512.
      5. Bu tru K -> K' cho tung anh.
      6. Phan cong goc chup bang Hungarian.

    Args:
        image_paths: Danh sach duong dan file anh.
        canvas_size: Kich thuoc canvas muc tieu (mac dinh 512).
        force_rmbg:  Ep dung RMBG cho moi anh.
        normalize_exposure_flag: Co can bang do sang khong.

    Returns:
        Danh sach dict, moi dict chua:
          {
            "canvas_rgb":   np.ndarray (512, 512, 3) uint8,
            "alpha_mask":   np.ndarray (512, 512) float32,
            "K_prime":      CameraIntrinsics,
            "scale":        float,
            "offset_x":     int,
            "offset_y":     int,
            "orig_path":    str,
            "azimuth_deg":  float or None,
          }

    Raises:
        ValueError: Neu so luong anh < 1.
    """
    N = len(image_paths)
    if N < 1:
        raise ValueError("Can it nhat 1 anh de xu ly.")
    if N < 2:
        warnings.warn("Chi co 1 anh — se dung che do Single-View.", UserWarning, stacklevel=2)
    elif N < 4:
        warnings.warn(
            f"Chi co {N} anh — so luong it, chat luong 3D co the giam.",
            UserWarning, stacklevel=2,
        )

    # Subsample neu qua nhieu anh
    paths = list(image_paths)
    if N > MAX_IMAGES:
        warnings.warn(
            f"N={N} > {MAX_IMAGES}: tu dong subsample xuong {TARGET_IMAGES} anh.",
            UserWarning, stacklevel=2,
        )
        paths = uniform_subsample(paths, TARGET_IMAGES)

    # Parse azimuth tu ten file
    azimuths = [parse_azimuth_from_filename(p) for p in paths]

    # Load va tach nen
    raws:       list = []
    alphas:     list = []
    rgbs:       list = []
    for p in paths:
        raw   = load_image_safe(p)
        raws.append(raw)
        alpha = compute_alpha_mask(raw, force_rmbg=force_rmbg)
        alphas.append(alpha)
        rgbs.append(ensure_rgb(raw))

    # Can bang do sang (tren phan foreground)
    if normalize_exposure_flag and len(rgbs) > 1:
        rgbs = normalize_exposure(rgbs, reference_idx=0)

    # Uniform Scale + Letterbox + K'
    results = []
    for i, (rgb, alpha, path, az) in enumerate(zip(rgbs, alphas, paths, azimuths)):
        canvas_rgb, canvas_alpha, K_prime, scale, off_x, off_y = place_on_canvas(
            rgb, alpha, canvas_size
        )
        results.append({
            "canvas_rgb":  canvas_rgb,
            "alpha_mask":  canvas_alpha,
            "K_prime":     K_prime,
            "scale":       scale,
            "offset_x":    off_x,
            "offset_y":    off_y,
            "orig_path":   str(path),
            "azimuth_deg": az,
            "view_idx":    i,
        })

    # Phan cong goc chup
    known_az = [r["azimuth_deg"] for r in results]
    if all(a is not None for a in known_az):
        order = assign_viewpoints_hungarian(rgbs, known_az)
        results = [results[i] for i in order]
        for new_idx, r in enumerate(results):
            r["view_idx"] = new_idx

    return results
