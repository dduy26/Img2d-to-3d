"""
P2 — DUSt3R Multi-view Reconstruction Engine.

Hai chế độ, tự chọn theo môi trường:

  THẬT  — có package `dust3r` trong PYTHONPATH:
          nạp naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt, chạy pairwise inference +
          global alignment, trả pointmap 3D / confidence / camera pose / focal
          trong CÙNG một hệ toạ độ world (P4 TSDF tiêu thụ trực tiếp).
  MOCK  — không có `dust3r`:
          giữ nguyên hành vi cũ (dữ liệu NGẪU NHIÊN) để test offline P4/P5/app
          chạy được trên máy không GPU. Hình dạng 3D sẽ KHÔNG lấy từ ảnh.

Bật chế độ thật trên Colab:
    git clone --recursive https://github.com/naver/dust3r.git /content/dust3r
    pip install -q roma tqdm matplotlib einops      # KHÔNG cài torch/torchvision lại
    export PYTHONPATH=/content/dust3r               # (notebook P6 set sẵn khi spawn uvicorn)
"""

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    HAS_DUST3R = True
except ImportError:
    HAS_DUST3R = False

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
        """
        Nạp DUSt3R thật nếu có package; ngược lại đánh dấu mock để pipeline vẫn chạy.
        """
        if not HAS_DUST3R:
            self.model = "Mock DUSt3R Model"
            self.logger.warning(
                "Chưa cài package `dust3r` -> chạy MOCK: hình dạng 3D là NGẪU NHIÊN, "
                "không lấy từ ảnh. Xem docstring đầu file để bật chế độ thật."
            )
            return
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
        # get_conf() mặc định trả conf_trf = log(x) (âm ở vùng ít tin cậy) -> dùng 'id'
        # để lấy confidence thô dương, rồi tự quy về [0,1] cho P3/P4.
        conf = [_conf_to_unit(_to_map(c, 2)) for c in scene.get_conf(mode="id")]
        poses = scene.get_im_poses().detach().cpu().numpy()
        focals = np.atleast_2d(scene.get_focals().detach().cpu().numpy())

        # Các view có thể lệch kích thước -> pad về (H, W) lớn nhất.
        # Vùng pad có confidence = 0 nên P4 tự loại, không phải sửa P4.
        n = len(pts)
        h = max(p.shape[0] for p in pts)
        w = max(p.shape[1] for p in pts)
        pointmaps_3d = np.zeros((n, h, w, 3), dtype=np.float32)
        confidence_masks = np.zeros((n, h, w), dtype=np.float32)
        for i, (p, c) in enumerate(zip(pts, conf)):
            hi, wi = p.shape[:2]
            pointmaps_3d[i, :hi, :wi] = p
            confidence_masks[i, :hi, :wi] = c

        camera_poses = [np.asarray(poses[i], dtype=np.float32) for i in range(n)]
        focal_lengths = [(float(focals[i][0]), float(focals[i][-1])) for i in range(n)]

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

    # ── ĐƯỜNG MOCK (giữ nguyên hành vi cũ cho test offline) ──────────────────

    def run_pairwise_matching(self, images_dust3r):
        """
        Tính toán matching giữa các cặp ảnh.
        Input:
            images_dust3r: Tensor (N, 3, H, W)
        Output:
            pairwise_pointmaps: List các tensor point-maps
            pairwise_confidence: List các tensor confidence
        """
        self.logger.info("Running pairwise matching...")
        N, _, H, W = images_dust3r.shape
        # Mock output
        if HAS_TORCH:
            pairwise_pointmaps = [torch.randn(H, W, 3) for _ in range(N - 1)]
            pairwise_confidence = [torch.rand(H, W) for _ in range(N - 1)]
        else:
            pairwise_pointmaps = [np.random.randn(H, W, 3).astype(np.float32) for _ in range(N - 1)]
            pairwise_confidence = [np.random.rand(H, W).astype(np.float32) for _ in range(N - 1)]
        return pairwise_pointmaps, pairwise_confidence

    def run_global_alignment(self, pairwise_pointmaps, pairwise_confidence, N, H, W):
        """
        Tối ưu Global alignment để đưa tất cả point-maps về một hệ tọa độ.
        """
        self.logger.info("Running global alignment...")
        if HAS_TORCH:
            pointmaps_3d = torch.from_numpy(np.random.uniform(-0.3, 0.3, (N, H, W, 3)).astype(np.float32)).to(self.device)
            confidence_masks = torch.from_numpy((0.6 + 0.3 * np.random.rand(N, H, W)).astype(np.float32)).to(self.device)
        else:
            pointmaps_3d = np.random.uniform(-0.3, 0.3, (N, H, W, 3)).astype(np.float32)
            confidence_masks = (0.6 + 0.3 * np.random.rand(N, H, W)).astype(np.float32)

        # Mock camera poses và focal lengths (phân bổ camera quanh trục Y nhìn về tâm)
        camera_poses = []
        for i in range(N):
            angle = i * (2.0 * np.pi / max(N, 1))
            cam_pos = np.array([2.0 * np.sin(angle), 0.0, 2.0 * np.cos(angle)], dtype=np.float32)
            forward = -cam_pos / np.linalg.norm(cam_pos)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(up, forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)

            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = forward
            c2w[:3, 3] = cam_pos
            camera_poses.append(c2w)
        focal_lengths = [(500.0, 500.0) for _ in range(N)]

        return pointmaps_3d, confidence_masks, camera_poses, focal_lengths

    def process(self, input_dict):
        """
        Pipeline chính.

        input_dict:
            image_paths    : List[str] đường dẫn ảnh (dùng cho DUSt3R thật)
            images_dust3r  : Tensor/Ndarray (N, 3, H, W) (dùng cho chế độ mock)
        """
        try:
            image_paths = input_dict.get('image_paths')
            images_dust3r = input_dict.get('images_dust3r')
            if image_paths is None and images_dust3r is None:
                raise ValueError("Input dictionary must contain 'image_paths' hoặc 'images_dust3r'")

            # ── Đường THẬT ──
            if HAS_DUST3R and image_paths:
                try:
                    result = self.run_real(image_paths)
                except Exception:
                    # KHÔNG rơi về mock: mock tạo hình NGẪU NHIÊN, im lặng đổi sang mock
                    # chỉ tạo ra file .glb trông như thành công. Lỗi phải nổi lên.
                    self.logger.error("DUSt3R thật chạy lỗi (không fallback mock):", exc_info=True)
                    raise
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return result

            if image_paths and not HAS_DUST3R:
                self.logger.warning(
                    "Chưa cài `dust3r` -> chạy MOCK: hình dạng 3D NGẪU NHIÊN, không lấy từ ảnh."
                )

            # ── Đường MOCK ──
            if images_dust3r is None:
                raise ValueError("Chế độ mock cần 'images_dust3r' tensor")

            N, _, H, W = images_dust3r.shape

            if HAS_TORCH and hasattr(images_dust3r, 'shape') and not isinstance(images_dust3r, np.ndarray):
                with torch.no_grad():
                    pairwise_pointmaps, pairwise_confidence = self.run_pairwise_matching(images_dust3r)
                    pointmaps_3d, confidence_masks, camera_poses, focal_lengths = self.run_global_alignment(
                        pairwise_pointmaps, pairwise_confidence, N, H, W
                    )
            else:
                pairwise_pointmaps, pairwise_confidence = self.run_pairwise_matching(images_dust3r)
                pointmaps_3d, confidence_masks, camera_poses, focal_lengths = self.run_global_alignment(
                    pairwise_pointmaps, pairwise_confidence, N, H, W
                )

            # Dọn dẹp cache GPU để tránh OOM
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'pointmaps_3d': pointmaps_3d,
                'confidence_masks': confidence_masks,
                'camera_poses': camera_poses,
                'focal_lengths': focal_lengths,
                'geometry': None,        # mock dùng đúng geometry của ảnh đầu vào
                'ba_loss': 1.0,          # placeholder như trước
                'backend': 'mock',
            }
        except Exception as e:
            self.logger.error(f"Error during DUSt3R processing: {e}")
            raise


# ============================================================================
# SELF-CHECK — chạy: python notebook/backend/engine_dust3r.py
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eng = DUSt3REngine(device="cpu")

    # Confidence thô của DUSt3R phải quy về [0,1] để P3/P4 dùng được
    raw = np.array([[0.5, 3.0, 12.0]], dtype=np.float32)
    got = _conf_to_unit(raw)
    assert got.shape == raw.shape and got.min() >= 0.0 and got.max() <= 1.0, got
    assert np.all(np.diff(got[0]) > 0), "conf phải giữ thứ tự đơn điệu sau chuẩn hoá"

    # _to_map phải cắt được tensor đã stack theo số cặp ảnh về đúng rank
    assert _to_map(np.zeros((3, 8, 6, 3), np.float32), 3).shape == (8, 6, 3)
    assert _to_map([np.zeros((8, 6, 3), np.float32)], 3).shape == (8, 6, 3)
    assert _to_map(np.zeros((3, 8, 6), np.float32), 2).shape == (8, 6)

    # Chế độ mock trả đủ khoá hợp đồng cho app.py
    out = eng.process({"images_dust3r": np.zeros((2, 3, 64, 64), np.float32)})
    for key in ("pointmaps_3d", "confidence_masks", "camera_poses", "focal_lengths",
                "geometry", "ba_loss", "backend"):
        assert key in out, f"thiếu khoá {key}"
    assert out["pointmaps_3d"].shape == (2, 64, 64, 3)
    assert out["confidence_masks"].shape == (2, 64, 64)
    assert len(out["camera_poses"]) == 2 and len(out["focal_lengths"]) == 2
    assert out["backend"] == "mock" and out["geometry"] is None

    print("SELF-CHECK PASS: hợp đồng dữ liệu P2 (real + mock) đúng.")
