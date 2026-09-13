"""
P2 — DUSt3R Multi-view Reconstruction Engine.

Chế độ thực thi:
  100% THẬT — nạp naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt, chạy pairwise inference +
  global alignment, trả pointmap 3D / confidence / camera pose / focal
  trong CÙNG một hệ toạ độ world (P4 TSDF tiêu thụ trực tiếp).
  Đã xóa bỏ hoàn toàn chế độ mock data ngẫu nhiên.
        

Bật chế độ thật trên Colab:
    git clone --recursive https://github.com/naver/dust3r.git /content/dust3r
    pip install -q roma tqdm matplotlib einops      # KHÔNG cài torch/torchvision lại
    export PYTHONPATH=/content/dust3r               # (notebook P6 set sẵn khi spawn uvicorn)
"""

import sys
import os
from pathlib import Path

# Tự động nạp đường dẫn dust3r và croco (hỗ trợ cả chạy local trong notebook/backend/dust3r và /content/dust3r trên Colab)
_backend_dir = Path(__file__).resolve().parent
_dust3r_dir = _backend_dir / "dust3r"
_croco_dir = _dust3r_dir / "croco"
for _p in [_dust3r_dir, _croco_dir, Path("/content/dust3r"), Path("/content/dust3r/croco")]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

DUST3R_IMPORT_ERROR = None
try:
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    HAS_DUST3R = True
except Exception as e:
    HAS_DUST3R = False
    DUST3R_IMPORT_ERROR = e

import numpy as np
import logging

# Weights DUSt3R gốc trên HuggingFace (~2.3GB, tải ở lần chạy đầu)
DEFAULT_WEIGHTS = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"


def _to_map(x, rank):
    """Chuẩn hoá output của scene.get_pts3d()/get_conf() về np.ndarray đúng `rank` chiều.

    DUSt3R tuỳ cấu hình có thể trả tensor trực tiếp, list tensor, hoặc tensor đã
    stack theo số cặp ảnh (symmetrize). Cắt dần chiều ngoài cùng cho tới đúng rank
    thay vì đoán shape — sai shape sẽ lộ ra ngay ở dòng log thay vì tạo dữ liệu rác.
    """
    if isinstance(x, (list, tuple)):
        x = x[0]
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    while x.ndim > rank:
        x = x[0]
    return x


def _conf_to_unit(raw):
    """Đưa confidence thô của DUSt3R về [0, 1] để P3 (Quality Gate) & P4 dùng được.

    P3 có ngưỡng mean_confidence >= 0.45 và P4 lọc conf >= 0.35 — cả hai đều giả định
    thang [0,1]. Confidence gốc của DUSt3R dương nhưng không chặn trên, nên chuẩn hoá
    theo percentile 95 thay vì max: một điểm nhiễu cực đại không làm sập cả thang.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.size == 0:
        return raw
    # DUSt3R để NaN ở pixel không quan sát được -> NaN sẽ lan vào TSDF nếu không chặn
    raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
    if float(raw.max()) <= 1.0001:               # đã ở thang [0,1]
        return np.clip(raw, 0.0, 1.0)
    scale = float(np.percentile(raw, 95))
    return np.clip(raw / max(scale, 1e-6), 0.0, 1.0)


class DUSt3REngine:
    def __init__(self, model_name="dust3r", device=None, weights=DEFAULT_WEIGHTS,
                 image_size=512, niter=300, batch_size=1):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
        self.weights = weights
        self.image_size = int(image_size)
        self.niter = int(niter)
        self.batch_size = int(batch_size)
        self.model = None
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Nạp model DUSt3R thật từ HuggingFace weights. Đã xóa bỏ hoàn toàn mock data."""
        if not HAS_DUST3R:
            err_detail = f" (Chi tiết lỗi: {DUST3R_IMPORT_ERROR})" if DUST3R_IMPORT_ERROR else ""
            raise RuntimeError(
                f"Không tìm thấy hoặc không thể import package 'dust3r'{err_detail}. "
                "Đã gỡ bỏ hoàn toàn chế độ mock data; vui lòng kiểm tra cài đặt dust3r hoặc chạy trên môi trường có GPU/dust3r."
            )
        self.logger.info(f"Đang nạp DUSt3R weights {self.weights} trên {self.device}...")
        self.model = AsymmetricCroCo3DStereo.from_pretrained(self.weights).to(self.device)
        self.model.eval()
        self.logger.info("DUSt3R loaded successfully.")

    def run_real(self, image_paths):
        """
        Chạy DUSt3R thật: load ảnh -> pairwise inference -> global alignment.

        Trả về pointmap/pose/focal trong cùng hệ world, kèm `geometry` để tầng gọi
        (app.py) căn lại alpha mask & ảnh RGB cho khớp từng pixel.
        """
        import time
        t0 = time.time()
        if self.model is None:
            self.load_model()

        # load_images tự resize (cạnh dài = 512) + chuẩn hoá [-1,1] đúng như mạng được train.
        # square_ok=True: ảnh vuông giữ vuông (mặc định DUSt3R crop mất 1/4 chiều cao).
        views = load_images(list(image_paths), size=self.image_size, square_ok=True, verbose=False)
        self.logger.info(f"[P2] Đã nạp {len(views)} view: {views[0]['true_shape'].tolist()}")

        pairs = make_pairs(views, scene_graph="complete", prefilter=None, symmetrize=True)
        self.logger.info(f"[P2] Pairwise inference trên {len(pairs)} cặp ảnh...")
        with torch.no_grad():
            output = inference(pairs, self.model, self.device, batch_size=self.batch_size)

        self.logger.info(f"[P2] Global alignment (PointCloudOptimizer, niter={self.niter})...")
        scene = global_aligner(output, device=self.device,
                               mode=GlobalAlignerMode.PointCloudOptimizer)
        ba_loss = scene.compute_global_alignment(init="mst", niter=self.niter,
                                                 schedule="cosine", lr=0.01)

        pts = [_to_map(p, 3) for p in scene.get_pts3d()]              # mỗi view (H_i, W_i, 3)
        conf = [_conf_to_unit(_to_map(c, 2)) for c in scene.get_conf(mode="id")]
        # Các view có thể lệch kích thước -> pad về (H, W) lớn nhất.
        n = len(pts)
        h = max(p.shape[0] for p in pts)
        w = max(p.shape[1] for p in pts)
        pointmaps_3d = np.zeros((n, h, w, 3), dtype=np.float32)
        confidence_masks = np.zeros((n, h, w), dtype=np.float32)
        for i, (p, c) in enumerate(zip(pts, conf)):
            hi, wi = p.shape[:2]
            pointmaps_3d[i, :hi, :wi] = p
            confidence_masks[i, :hi, :wi] = c

        poses = scene.get_im_poses().detach().cpu().numpy()
        raw_focals = scene.get_focals().detach().cpu().numpy()
        raw_focals = np.squeeze(raw_focals)
        focal_lengths = []
        for i in range(n):
            if raw_focals.ndim == 0:
                f_val = float(raw_focals)
                focal_lengths.append((f_val, f_val))
            elif raw_focals.ndim == 1:
                f_val = float(raw_focals[i])
                focal_lengths.append((f_val, f_val))
            else:
                focal_lengths.append((float(raw_focals[i, 0]), float(raw_focals[i, -1])))

        camera_poses = [np.asarray(poses[i], dtype=np.float32) for i in range(n)]

        self.logger.info(
            f"[P2] XONG trong {time.time() - t0:.1f}s — {n} view, pointmap {pointmaps_3d.shape}, "
            f"ba_loss={float(ba_loss):.4f}, focal~{focal_lengths[0][0]:.0f}px, "
            f"conf_mean={float(confidence_masks.mean()):.3f}"
        )
        return {
            "pointmaps_3d": pointmaps_3d,
            "confidence_masks": confidence_masks,
            "camera_poses": camera_poses,
            "focal_lengths": focal_lengths,
            "geometry": (pointmaps_3d.shape[1], pointmaps_3d.shape[2]),
            "ba_loss": float(ba_loss),
            "backend": "dust3r-real",
        }

    def process(self, input_dict):
        """
        Pipeline xử lý 3D từ ảnh thật bằng DUSt3R.
        Đã xóa bỏ hoàn toàn mock data giả lập.
        """
        image_paths = input_dict.get('image_paths')
        if not image_paths:
            raise ValueError("DUSt3REngine.process() yêu cầu 'image_paths' chứa danh sách ảnh thật.")

        if not HAS_DUST3R:
            raise RuntimeError(
                "Chưa cài đặt package 'dust3r'. Đã xóa bỏ toàn bộ mock data; "
                "bắt buộc phải có DUSt3R thật để tái tạo 3D từ ảnh."
            )

        try:
            result = self.run_real(image_paths)
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result
        except Exception as e:
            self.logger.error(f"Lỗi trong quá trình chạy DUSt3R thật: {e}", exc_info=True)
            raise


# ============================================================================
# SELF-CHECK — chạy: python notebook/backend/engine_dust3r.py
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Confidence thô của DUSt3R phải quy về [0,1] để P3/P4 dùng được
    raw = np.array([[0.5, 3.0, 12.0]], dtype=np.float32)
    got = _conf_to_unit(raw)
    assert got.shape == raw.shape and got.min() >= 0.0 and got.max() <= 1.0, got
    assert np.all(np.diff(got[0]) > 0), "conf phải giữ thứ tự đơn điệu sau chuẩn hoá"

    # _to_map phải cắt được tensor đã stack theo số cặp ảnh về đúng rank
    assert _to_map(np.zeros((3, 8, 6, 3), np.float32), 3).shape == (8, 6, 3)
    assert _to_map([np.zeros((8, 6, 3), np.float32)], 3).shape == (8, 6, 3)
    assert _to_map(np.zeros((3, 8, 6), np.float32), 2).shape == (8, 6)

    print("SELF-CHECK PASS: DUSt3R engine sạch sẽ, 100% không còn mock data.")
