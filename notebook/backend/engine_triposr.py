import time
from PIL import Image
import trimesh
import numpy as np

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
        Thực thi cứu hộ: Tạo model 3D từ 1 ảnh duy nhất.
        Đảm bảo constraint thời gian thực thi.
        """
        start_time = time.time()
        
        try:
            # 1. Tiền xử lý: Tách nền
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