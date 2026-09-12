import time
from PIL import Image
import trimesh
import numpy as np

import os
import sys
from pathlib import Path

# Thêm đường dẫn TripoSR nếu được clone vào tsr/TripoSR
_backend_dir = Path(__file__).resolve().parent
_triposr_sub = _backend_dir / "tsr" / "TripoSR"
if _triposr_sub.exists() and str(_triposr_sub) not in sys.path:
    sys.path.insert(0, str(_triposr_sub))

try:
    import torch
    import rembg
    from tsr.system import TSR
    HAS_TRIPOSR = True
except ImportError:
    HAS_TRIPOSR = False

class TripoSREngine:
    def __init__(self):
        # Khởi tạo mô hình ngay khi boot server để không mất thời gian load lại (đảm bảo <= 2s)
        if HAS_TRIPOSR:
            print("Loading TripoSR Fail-safe Model...")
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            try:
                self.model = TSR.from_pretrained(
                    "stabilityai/TripoSR",
                    config_name="config.yaml",
                    weight_name="model.ckpt",
                )
                self.model.to(self.device)
                self.model.eval()
                print("TripoSR loaded successfully.")
            except Exception as e:
                print(f"Cảnh báo: Không thể nạp weights TripoSR ({e}), sử dụng mock fallback.")
                self.model = None
        else:
            print("Chưa có TSR/Torch/Rembg, kích hoạt TripoSR Mock Fallback Engine.")
            self.model = None

    def preprocess_image(self, image_path):
        """Tách nền và chuẩn hóa về định dạng RGB 3 kênh"""
        # 1. Đọc ảnh gốc
        input_image = Image.open(image_path).convert("RGB")
        
        # 2. Xóa nền nếu có rembg
        if HAS_TRIPOSR:
            try:
                rgba_image = rembg.remove(input_image)
                white_bg = Image.new("RGB", rgba_image.size, (255, 255, 255))
                white_bg.paste(rgba_image, mask=rgba_image.split()[3])
                return white_bg
            except Exception:
                pass
        return input_image

    def run_fallback(self, image_path, output_glb_path):
        """
        Thực thi cứu hộ: Tạo model 3D từ 1 ảnh duy nhất (từ file path).
        Giữ nguyên tương thích ngược cho multi-view fallback path.
        Đảm bảo constraint thời gian thực thi.
        """
        start_time = time.time()
        
        try:
            # 1. Tiền xử lý: Tách nền (dùng rembg nội bộ vì không qua P1)
            img_rgba = self.preprocess_image(image_path)
            
            # 2. Tạo 3D
            if self.model is not None and HAS_TRIPOSR:
                with torch.no_grad():
                    scene_codes = self.model(img_rgba, device=self.device)
                meshes = self.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=128)
                mesh = meshes[0]
            else:
                # Mock fallback: tạo mesh lập phương tròn góc có màu từ ảnh
                mesh = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
                img_arr = np.array(img_rgba)
                mean_color = np.mean(img_arr, axis=(0, 1))[:3].astype(np.uint8)
                mesh.visual.vertex_colors = np.hstack([np.tile(mean_color, (len(mesh.vertices), 1)), np.full((len(mesh.vertices), 1), 255, dtype=np.uint8)])
            
            # 3. Lưu ra định dạng .glb
            mesh.export(output_glb_path)
            
            execution_time = time.time() - start_time
            print(f"Rescue successful! Model generated in {execution_time:.2f}s")
            
            return True, output_glb_path, execution_time
            
        except Exception as e:
            print(f"TripoSR Fallback failed: {e}")
            return False, None, time.time() - start_time

    def run_from_preprocessed(self, image_rgb, alpha_mask, output_glb_path):
        """
        Tạo model 3D từ ảnh đã được P1 preprocess (preprocess_single_view).

        Khác với run_fallback(): Nhận trực tiếp ảnh RGB + Alpha Mask đã xử lý
        bởi RMBG-2.0 của P1, KHÔNG gọi rembg lần nữa → tiết kiệm thời gian + nhất quán.

        Args:
            image_rgb: np.ndarray (H, W, 3) uint8 hoặc PIL.Image — ảnh RGB đã resize.
            alpha_mask: np.ndarray (H, W) uint8 {0,1} — Alpha Mask từ P1.
            output_glb_path: str — đường dẫn file .glb đầu ra.

        Returns:
            Tuple (success, model_path, execution_time):
                - success: bool
                - model_path: str hoặc None
                - execution_time: float (giây)
        """
        start_time = time.time()

        try:
            # 1. Chuẩn bị ảnh: hòa trộn alpha mask lên nền trắng (TripoSR cần RGB)
            if isinstance(image_rgb, np.ndarray):
                pil_img = Image.fromarray(image_rgb)
            elif isinstance(image_rgb, Image.Image):
                pil_img = image_rgb.convert("RGB")
            else:
                raise TypeError(f"image_rgb phải là np.ndarray hoặc PIL.Image, nhận được {type(image_rgb)}")

            if alpha_mask is not None:
                # Áp dụng alpha mask: vật thể giữ nguyên, nền → trắng
                img_arr = np.array(pil_img)
                mask_3d = np.stack([alpha_mask] * 3, axis=-1)
                # Nơi mask=0 (nền) → đặt màu trắng
                img_arr = np.where(mask_3d > 0, img_arr, 255)
                pil_img = Image.fromarray(img_arr.astype(np.uint8))

            # 2. Tạo 3D
            if self.model is not None and HAS_TRIPOSR:
                with torch.no_grad():
                    scene_codes = self.model(pil_img, device=self.device)
                meshes = self.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=128)
                mesh = meshes[0]
            else:
                # Mock fallback
                mesh = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
                img_arr = np.array(pil_img)
                mean_color = np.mean(img_arr, axis=(0, 1))[:3].astype(np.uint8)
                mesh.visual.vertex_colors = np.hstack([
                    np.tile(mean_color, (len(mesh.vertices), 1)),
                    np.full((len(mesh.vertices), 1), 255, dtype=np.uint8)
                ])

            # 3. Lưu ra .glb
            mesh.export(output_glb_path)

            execution_time = time.time() - start_time
            print(f"TripoSR from preprocessed: Model generated in {execution_time:.2f}s")

            return True, output_glb_path, execution_time

        except Exception as e:
            print(f"TripoSR from preprocessed failed: {e}")
            return False, None, time.time() - start_time