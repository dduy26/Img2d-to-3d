"""
engine_depth.py — P2: Uoc Luong Do Sau & Loc Vien
==================================================
Chuc nang:
  1. Tai mo hinh Depth-Anything-V2-Small tu cache cuc bo (local_files_only=True).
  2. Du doan ban do do sau (Depth Map) cho moi anh canvas 512x512.
  3. Loc vien "flying pixels" bang bo loc gradient do sau (DA3 gradient edge filter).
  4. Tra ve ban do do sau chuan hoa [0, 1] de dung cho TSDF fusion.

Rang buoc phan cung:
  - Mo hinh Depth-Anything-V2-Small: ~95MB VRAM, chay duoc tren RTX 3050 6GB.
  - Doi voi CPU: toc do ~1.5 giay/anh tren i5-12450HX.

Ly thuyet:
  - DA3 gradient edge filter: loc vien dot gay (|gradient D| > nguong) de
    loai bo flying pixels xuat hien o ranh gioi foreground/background.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME        = "depth-anything/Depth-Anything-V2-Small-hf"
GRADIENT_THR_REL  = 0.10   # Nguong gradient tuong doi (10% dai dong)
CANVAS_SIZE       = 512

# ---------------------------------------------------------------------------
# 1. MODEL LOADING — Lazy singleton, local_files_only
# ---------------------------------------------------------------------------

_depth_model   = None
_depth_pipe    = None
_device_str    = None


def _load_depth_pipeline(device: Optional[str] = None) -> object:
    """Nap pipeline Depth-Anything-V2-Small lan dau, luu singleton.

    Uu tien:
      1. local_files_only=True (chay hoan toan offline neu da cache).
      2. Neu cache khong co -> download tu HuggingFace (log canh bao).

    Args:
        device: "cuda", "cpu", hoac None (tu dong phat hien).

    Returns:
        transformers.Pipeline object.
    """
    global _depth_pipe, _device_str

    if _depth_pipe is not None:
        return _depth_pipe

    try:
        import torch
        from transformers import pipeline as hf_pipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _device_str = device

        # Thu tai tu cache cuc bo truoc
        try:
            _depth_pipe = hf_pipeline(
                "depth-estimation",
                model=MODEL_NAME,
                device=0 if device == "cuda" else -1,
                local_files_only=True,
            )
        except OSError:
            warnings.warn(
                f"Model '{MODEL_NAME}' chua co trong cache cuc bo. "
                "Se download tu HuggingFace (can ket noi internet).",
                RuntimeWarning,
                stacklevel=3,
            )
            _depth_pipe = hf_pipeline(
                "depth-estimation",
                model=MODEL_NAME,
                device=0 if device == "cuda" else -1,
                local_files_only=False,
            )

        return _depth_pipe

    except ImportError as e:
        raise ImportError(
            "Can cai dat torch va transformers: pip install torch transformers"
        ) from e


# ---------------------------------------------------------------------------
# 2. INFER DEPTH — Du doan ban do do sau
# ---------------------------------------------------------------------------

def infer_depth_map(
    canvas_rgb: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """Du doan ban do do sau cho anh canvas RGB.

    Args:
        canvas_rgb: (H, W, 3) uint8 RGB — anh da duoc co dan vao canvas.
        device:     "cuda" / "cpu" / None (tu dong).

    Returns:
        depth_norm: (H, W) float32 in [0, 1], 1 = gan camera, 0 = xa.
    """
    pipe = _load_depth_pipeline(device)

    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(canvas_rgb)
    result  = pipe(pil_img)

    # Ket qua la dict co "predicted_depth" (torch.Tensor hoac np.ndarray)
    depth_raw = result["predicted_depth"]

    # Chuyen sang numpy neu can
    try:
        import torch
        if isinstance(depth_raw, torch.Tensor):
            depth_raw = depth_raw.squeeze().cpu().numpy()
    except ImportError:
        pass

    depth_raw = np.asarray(depth_raw, dtype=np.float32)

    # Resize ve kich thuoc canvas neu mo hinh tra ve kich thuoc khac
    h, w = canvas_rgb.shape[:2]
    if depth_raw.shape != (h, w):
        depth_raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_LINEAR)

    # Chuan hoa ve [0, 1] — Depth-Anything-V2 tra ve do sau tuong doi
    d_min, d_max = depth_raw.min(), depth_raw.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth_raw - d_min) / (d_max - d_min)
    else:
        depth_norm = np.ones_like(depth_raw, dtype=np.float32) * 0.5

    return depth_norm.astype(np.float32)


def infer_depth_batch(
    canvases_rgb: list,
    device: Optional[str] = None,
) -> list:
    """Du doan do sau cho danh sach N anh, tra ve danh sach N depth map.

    Args:
        canvases_rgb: Danh sach N anh (H, W, 3) uint8 RGB.
        device:       "cuda" / "cpu" / None.

    Returns:
        Danh sach N ban do do sau (H, W) float32.
    """
    return [infer_depth_map(rgb, device=device) for rgb in canvases_rgb]


# ---------------------------------------------------------------------------
# 3. DA3 GRADIENT EDGE FILTER — Loc flying pixels
# ---------------------------------------------------------------------------

def compute_depth_gradient(depth_map: np.ndarray) -> np.ndarray:
    """Tinh do lon gradient cua ban do do sau bang Sobel operator.

    |gradient D| = sqrt((dD/dx)^2 + (dD/dy)^2)

    Args:
        depth_map: (H, W) float32 do sau.

    Returns:
        gradient_mag: (H, W) float32 — do lon gradient.
    """
    gx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def filter_flying_pixels(
    depth_map:  np.ndarray,
    alpha_mask: np.ndarray,
    gradient_thr_rel: float = GRADIENT_THR_REL,
) -> tuple:
    """Loc flying pixels tai ranh gioi foreground/background.

    Ly thuyet:
      - Flying pixels: pixel do sau bi nhiem gia tri sai tai vung chuyển tiep.
      - Phuong phap: Tinh gradient do sau, danh dau vung co gradient cao
        (|grad D| > thr) la vung loai bo.
      - Ket hop voi alpha mask de chi giu pixels foreground co gradient thap.

    Args:
        depth_map:        (H, W) float32 [0, 1].
        alpha_mask:       (H, W) float32 [0, 1].
        gradient_thr_rel: Nguong gradient tuong doi (0.10 = 10% dai dong).

    Returns:
        (filtered_depth, valid_mask)
        - filtered_depth: (H, W) float32 — do sau da loc.
        - valid_mask:     (H, W) float32 — 1 = hieu le, 0 = da loc.
    """
    grad_mag = compute_depth_gradient(depth_map)
    d_range  = depth_map.max() - depth_map.min()
    if d_range < 1e-6:
        d_range = 1.0
    thr = gradient_thr_rel * d_range

    # Vung co gradient cao -> ranh gioi, can loc bo
    sharp_edge = grad_mag > thr

    # Foreground mask (alpha > 0.5)
    fg = alpha_mask > 0.5

    # Valid pixels: foreground va KHONG phai vung ranh gioi sac net
    valid = fg & ~sharp_edge

    # Do gian no nhe (dilate) vung loai bo de khong de lai via
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    invalid_region = (~valid).astype(np.uint8)
    invalid_region_dilated = cv2.dilate(invalid_region, kernel, iterations=1)
    valid_final = (invalid_region_dilated == 0) & fg

    filtered_depth = depth_map.copy()
    filtered_depth[~valid_final] = 0.0

    return filtered_depth, valid_final.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. DEPTH MAP -> POINT CLOUD (utility)
# ---------------------------------------------------------------------------

def depth_to_pointcloud(
    depth_map:    np.ndarray,
    alpha_mask:   np.ndarray,
    K_prime:      object,
    z_scale:      float = 1.0,
) -> tuple:
    """Chuyen ban do do sau thanh dam may diem 3D trong he toa do camera.

    Cong thuc unproject (nguoc phoi canh):
        X = (u - cx') * Z / fx'
        Y = (v - cy') * Z / fy'
        Z = depth * z_scale

    Args:
        depth_map:  (H, W) float32 do sau [0, 1].
        alpha_mask: (H, W) float32 [0, 1].
        K_prime:    CameraIntrinsics da bu tru.
        z_scale:    He so nhan chieu sau tuyet doi (m).

    Returns:
        (points_3d, colors_rgb_placeholder)
        - points_3d: (N, 3) float32 — toa do XYZ.
    """
    h, w = depth_map.shape
    fx, fy = K_prime.fx, K_prime.fy
    cx, cy = K_prime.cx, K_prime.cy

    u_grid = np.arange(w, dtype=np.float32)
    v_grid = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u_grid, v_grid)

    fg_mask = alpha_mask > 0.5
    Z = depth_map[fg_mask] * z_scale
    u_fg = uu[fg_mask]
    v_fg = vv[fg_mask]

    X = (u_fg - cx) * Z / (fx + 1e-9)
    Y = (v_fg - cy) * Z / (fy + 1e-9)

    points_3d = np.stack([X, Y, Z], axis=1).astype(np.float32)
    return points_3d


# ---------------------------------------------------------------------------
# 5. PIPELINE P2 — ENTRY POINT
# ---------------------------------------------------------------------------

def estimate_depth_pipeline(
    preprocessed_views: list,
    device: Optional[str] = None,
    filter_edges: bool = True,
) -> list:
    """Thuc hien uoc luong do sau cho toan bo danh sach views tu P1.

    Args:
        preprocessed_views: Ket qua tu preprocess.preprocess_images().
        device:             "cuda" / "cpu" / None.
        filter_edges:       Co loc flying pixels khong.

    Returns:
        Danh sach dict (mo rong tu preprocessed_views), them:
          {
            ...keys tu P1...,
            "depth_map":    (H, W) float32 [0, 1],
            "valid_mask":   (H, W) float32 [0, 1],
            "depth_points": (N, 3) float32 toa do XYZ,
          }
    """
    results = []
    for view in preprocessed_views:
        canvas_rgb = view["canvas_rgb"]
        alpha_mask = view["alpha_mask"]
        K_prime    = view["K_prime"]

        # Infer depth
        depth_raw = infer_depth_map(canvas_rgb, device=device)

        # Loc flying pixels
        if filter_edges:
            depth_filtered, valid_mask = filter_flying_pixels(depth_raw, alpha_mask)
        else:
            depth_filtered = depth_raw
            valid_mask     = (alpha_mask > 0.5).astype(np.float32)

        # Tao dam may diem 3D so bo (camera-space)
        pts_3d = depth_to_pointcloud(depth_filtered, valid_mask, K_prime)

        result = dict(view)
        result.update({
            "depth_map":    depth_filtered,
            "valid_mask":   valid_mask,
            "depth_points": pts_3d,
        })
        results.append(result)

    return results
