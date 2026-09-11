import torch
import time
import rembg
from PIL import Image
import trimesh
from tsr.system import TSR

class TripoSREngine:
    def __init__(self):
        # Khởi tạo mô hình ngay khi boot server để không mất thời gian load lại (đảm bảo <= 2s)
        print("Loading TripoSR Fail-safe Model...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.to(self.device)
        self.model.eval()
        print("TripoSR loaded successfully.")

    def preprocess_image(self, image_path):
        """Tách nền và chuẩn hóa về định dạng RGB 3 kênh"""
        # 1. Đọc ảnh gốc
        input_image = Image.open(image_path).convert("RGB")
        
        # 2. Xóa nền (kết quả là ảnh RGBA có nền trong suốt)
        rgba_image = rembg.remove(input_image)
        
        # 3. Tạo một phông nền trắng tinh (RGB) cùng kích thước với ảnh
        white_bg = Image.new("RGB", rgba_image.size, (255, 255, 255))
        
        # 4. Dán vật thể lên nền trắng (dùng kênh thứ 4 - Alpha làm mặt nạ cắt)
        white_bg.paste(rgba_image, mask=rgba_image.split()[3])
        
        return white_bg

    def run_fallback(self, image_path, output_glb_path):
        """
        Thực thi cứu hộ: Tạo model 3D từ 1 ảnh duy nhất.
        Đảm bảo constraint thời gian thực thi.
        """
        start_time = time.time()
        
        try:
            # 1. Tiền xử lý: Tách nền
            img_rgba = self.preprocess_image(image_path)
            
            # 2. Tạo 3D (Feed-forward)
            with torch.no_grad():
                scene_codes = self.model(img_rgba, device=self.device)
            
            # 3. Trích xuất Mesh (Sử dụng marching cubes với độ phân giải tiêu chuẩn)
            meshes = self.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=128)
            mesh = meshes[0] # Lấy mesh đầu tiên
            
            # 4. Lưu ra định dạng .glb
            mesh.export(output_glb_path)
            
            execution_time = time.time() - start_time
            print(f"Rescue successful! Model generated in {execution_time:.2f}s")
            
            return True, output_glb_path, execution_time
            
        except Exception as e:
            print(f"TripoSR Fallback failed: {e}")
            return False, None, time.time() - start_time