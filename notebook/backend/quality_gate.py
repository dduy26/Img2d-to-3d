"""
quality_gate.py — P3: Cong Kiem Soat Chat Luong & Fail-safe Fallback
=====================================================================
Chuc nang:
  1. Kiem dinh danh sach views sau P2 theo nhieu tieu chi:
       - Dien tich tien canh (foreground area) > 1% canvas.
       - Do bao phu goc (angle coverage) > 45 do.
       - So luong views hop le >= 2 (cho multi-view).
  2. Neu vuot nguong -> cho phep chay engine multi-view (Branch B).
  3. Neu khong dat -> tu dong chuyen sang che do Single-View Fallback (Branch A)
     tu anh neo (anchor image) net nhat.

Tieu chi lua chon anchor:
  - Chon anh co dien tich foreground lon nhat va gradient do sau thap nhat
    (nen nhat, tin cay nhat).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FOREGROUND_RATIO  = 0.0005   # Toi thieu 0.05% dien tich foreground (ho tro vat the dai/hep)
MIN_ANGLE_COVERAGE    = 45.0     # Toi thieu 45 do bao phu goc
MIN_VIEWS_MULTIVIEW   = 2        # Toi thieu 2 views de chay multi-view

# ---------------------------------------------------------------------------
# 1. TIEU CHI KIEM DINH
# ---------------------------------------------------------------------------

@dataclass
class ViewQuality:
    """Chat luong cua mot view rieng le."""
    view_idx:          int
    orig_path:         str
    foreground_ratio:  float   # Ty le pixel foreground / tong pixel
    depth_grad_mean:   float   # Gradient do sau trung binh (thap = tot)
    is_valid:          bool    # Dat tieu chi hay khong
    reason:            str = ""


@dataclass
class QualityGateResult:
    """Ket qua tong the cua cong kiem soat."""
    mode:              str        # "multi_view" hoac "single_view_fallback"
    valid_views:       list       # Danh sach view dict (da loc) du tieu chuan
    anchor_view:       dict       # Anh neo tot nhat (cho fallback)
    view_qualities:    list       # Danh sach ViewQuality cho tung view
    angle_coverage:    float      # Do bao phu goc do
    reason:            str        # Giai thich quyet dinh
    warnings:          list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. TINH CHAT LUONG TUNG VIEW
# ---------------------------------------------------------------------------

def _foreground_ratio(alpha_mask: np.ndarray) -> float:
    """Tinh ty le pixel foreground (alpha > 0.5) tren tong pixel."""
    total = alpha_mask.size
    fg    = np.sum(alpha_mask > 0.5)
    return float(fg) / (total + 1e-9)


def _depth_grad_mean(depth_map: np.ndarray) -> float:
    """Tinh do lon gradient do sau trung binh."""
    import cv2
    gx = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(grad_mag.mean())


def evaluate_view_quality(view: dict) -> ViewQuality:
    """Danh gia chat luong cua mot view tu P2.

    Args:
        view: Dict view tu estimate_depth_pipeline().

    Returns:
        ViewQuality.
    """
    alpha   = view.get("alpha_mask",  np.zeros((512, 512), dtype=np.float32))
    depth   = view.get("depth_map",   np.zeros((512, 512), dtype=np.float32))
    fg_ratio   = _foreground_ratio(alpha)
    grad_mean  = _depth_grad_mean(depth)

    is_valid = fg_ratio >= MIN_FOREGROUND_RATIO
    reason   = ""
    if not is_valid:
        reason = f"Foreground ratio {fg_ratio:.4f} < {MIN_FOREGROUND_RATIO}"

    return ViewQuality(
        view_idx         = view.get("view_idx", 0),
        orig_path        = view.get("orig_path", ""),
        foreground_ratio = fg_ratio,
        depth_grad_mean  = grad_mean,
        is_valid         = is_valid,
        reason           = reason,
    )


# ---------------------------------------------------------------------------
# 3. DO BAO PHU GOC
# ---------------------------------------------------------------------------

def compute_angle_coverage(views: list) -> float:
    """Tinh do bao phu goc (do) tu danh sach views co azimuth_deg.

    Neu khong co thong tin azimuth, gia dinh 90*N do cho N views.

    Returns:
        Do bao phu goc (do), toi da 360.
    """
    azimuths = [v.get("azimuth_deg") for v in views]
    known    = [a for a in azimuths if a is not None]
    if len(known) < 2:
        # Uoc tinh tu so luong views
        n = len(views)
        return min(360.0, max(0.0, n * 90.0 - 90.0))
    # Tinh "angular span" theo danh sach azimuth da biet
    azs = sorted(known)
    # Tinh khoang trong lon nhat giua cac goc lien tiep (vong)
    gaps = []
    for i in range(len(azs) - 1):
        gaps.append(azs[i + 1] - azs[i])
    # Khoang cach tu goc cuoi den goc dau (vong)
    gaps.append((azs[0] + 360.0) - azs[-1])
    # Do bao phu = 360 - khoang trong lon nhat
    coverage = 360.0 - max(gaps)
    return max(0.0, coverage)


# ---------------------------------------------------------------------------
# 4. CHON ANH NEO (ANCHOR)
# ---------------------------------------------------------------------------

def select_anchor_view(views: list, view_qualities: list) -> dict:
    """Chon anh neo tot nhat: foreground lon nhat, gradient thap nhat.

    Score = foreground_ratio / (depth_grad_mean + 1e-6)
    -> Score cao = foreground lon, do sau muot.

    Args:
        views:          Danh sach view dict.
        view_qualities: Danh sach ViewQuality tuong ung.

    Returns:
        View dict cua anh neo.
    """
    if not views:
        raise ValueError("Khong co view nao de chon anchor.")
    scores = []
    for vq in view_qualities:
        score = vq.foreground_ratio / (vq.depth_grad_mean + 1e-6)
        scores.append(score)
    best_idx = int(np.argmax(scores))
    return views[best_idx]


# ---------------------------------------------------------------------------
# 5. QUALITY GATE — ENTRY POINT
# ---------------------------------------------------------------------------

def run_quality_gate(
    depth_views: list,
    min_fg_ratio:       float = MIN_FOREGROUND_RATIO,
    min_angle_coverage: float = MIN_ANGLE_COVERAGE,
    min_views:          int   = MIN_VIEWS_MULTIVIEW,
) -> QualityGateResult:
    """Chay cong kiem soat chat luong cho toan bo danh sach views tu P2.

    Logic quyet dinh:
      1. Danh gia tung view -> LocFilter valid views (fg_ratio >= min).
      2. Tinh do bao phu goc cua tap valid views.
      3. Neu so valid views >= min_views VA angle_coverage >= min_angle:
           -> mode = "multi_view" (Branch B).
      4. Nguoc lai:
           -> mode = "single_view_fallback" (Branch A).
      5. Luon chon anchor (anh neo) tot nhat du o mode nao.

    Args:
        depth_views:        Ket qua tu estimate_depth_pipeline().
        min_fg_ratio:       Nguong foreground ratio (mac dinh 1%).
        min_angle_coverage: Nguong bao phu goc (mac dinh 45 do).
        min_views:          So views toi thieu cho multi-view (mac dinh 2).

    Returns:
        QualityGateResult.
    """
    warn_list = []

    # Buoc 1: Danh gia tung view
    qualities = [evaluate_view_quality(v) for v in depth_views]
    valid_views = [
        v for v, q in zip(depth_views, qualities) if q.is_valid
    ]
    invalid_count = len(depth_views) - len(valid_views)
    if invalid_count > 0:
        warn_list.append(
            f"{invalid_count} view bi loai do foreground ratio qua nho."
        )

    # Buoc 2: Do bao phu goc
    coverage = compute_angle_coverage(valid_views) if valid_views else 0.0

    # Buoc 3: Chon anchor (tu valid views neu co, neu khong thi tu tat ca)
    try_views = valid_views if valid_views else depth_views
    # Re-compute qualities cho try_views (tranh so sanh dict co numpy array)
    try_quals = [evaluate_view_quality(v) for v in try_views]
    anchor = select_anchor_view(try_views, try_quals)


    # Buoc 4: Quyet dinh mode
    enough_views = len(valid_views) >= min_views
    enough_angle = coverage >= min_angle_coverage

    if enough_views and enough_angle:
        mode   = "multi_view"
        reason = (
            f"Du dieu kien multi-view: {len(valid_views)} views hop le, "
            f"coverage={coverage:.1f} do."
        )
    else:
        mode   = "single_view_fallback"
        reason_parts = []
        if not enough_views:
            reason_parts.append(
                f"Chi co {len(valid_views)} views hop le (can >= {min_views})."
            )
        if not enough_angle:
            reason_parts.append(
                f"Do bao phu goc {coverage:.1f} do (can >= {min_angle_coverage} do)."
            )
        reason = "Fallback sang Single-View: " + " | ".join(reason_parts)
        warnings.warn(reason, UserWarning, stacklevel=2)

    return QualityGateResult(
        mode           = mode,
        valid_views    = valid_views,
        anchor_view    = anchor,
        view_qualities = qualities,
        angle_coverage = coverage,
        reason         = reason,
        warnings       = warn_list,
    )
