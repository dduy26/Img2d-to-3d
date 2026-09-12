try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np
import logging

class DUSt3REngine:
    def __init__(self, model_name="dust3r", device=None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
        self.model = None
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """
        Khởi tạo mô hình ViT từ pretrained weights (giả lập mock hoặc load thực tế).
        """
        self.logger.info(f"Loading {self.model_name} model on {self.device}...")
        # Mock load model
        self.model = "Mock DUSt3R Model"
        self.logger.info("Model loaded successfully.")

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
        Pipeline chính gọi các hàm trên.
        """
        try:
            images_dust3r = input_dict.get('images_dust3r')
            if images_dust3r is None:
                raise ValueError("Input dictionary must contain 'images_dust3r'")
            
            if self.model is None:
                self.load_model()
            
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
                'focal_lengths': focal_lengths
            }
        except Exception as e:
            self.logger.error(f"Error during DUSt3R processing: {e}")
            raise

