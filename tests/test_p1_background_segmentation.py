"""
Kiểm thử Phân hệ 1 (P1): Tách nền đa tầng (Native Alpha -> Lab Chroma Otsu Studio Backdrop).
Xác minh:
1. Ưu tiên giữ nguyên Native Alpha từ ảnh PNG RGBA không suy hao chất lượng.
2. Tách nền phông đơn sắc Studio qua khoảng cách sắc độ Lab + Otsu + Phép đóng/mở hình thái học.
3. Không để lại viền hộp (letterbox border artifacts) ngoài biên ảnh.
"""

import unittest
import numpy as np
import cv2
import sys
import os
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from notebook.backend.preprocess import (
    _detect_solid_background_mask,
    extract_alpha_masks,
    validate_and_load_images,
)


class TestP1BackgroundSegmentation(unittest.TestCase):
    def test_native_alpha_extraction(self):
        """Kiểm tra tách trực tiếp kênh Alpha từ ảnh RGBA tổng hợp."""
        # Tạo ảnh RGBA: hình tròn ở giữa có alpha=255, ngoài viền alpha=0
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        Y, X = np.ogrid[:100, :100]
        dist = np.sqrt((X - 50)**2 + (Y - 50)**2)
        fg_mask = dist <= 30
        rgba[fg_mask, :3] = [200, 80, 40]
        rgba[fg_mask, 3] = 255

        # Lưu tạm
        tmp_path = os.path.join(ROOT_DIR, "tests", "tmp_rgba.png")
        Image.fromarray(rgba, mode="RGBA").save(tmp_path)
        try:
            raw_images, native_masks = validate_and_load_images([tmp_path])
            self.assertIsNotNone(native_masks[0], "Native mask phải được nhận diện từ file RGBA!")

            masks = extract_alpha_masks(raw_images, native_masks=native_masks)
            self.assertEqual(masks[0].shape, (100, 100))
            # Kiểm tra pixel tâm là foreground (255) và góc là background (0)
            self.assertEqual(masks[0][50, 50], 255)
            self.assertEqual(masks[0][0, 0], 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_solid_studio_backdrop_lab_otsu(self):
        """Kiểm tra tách nền studio màu trắng có bóng đổ nhẹ bằng Lab Chroma + Otsu."""
        h, w = 120, 120
        # Nền trắng với bóng đổ gradient (từ 255 ở góc xuống 230 ở gần vật thể)
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                # Gradient bóng mờ
                val = int(255 - 20 * (x / w) * (y / h))
                bg[y, x] = [val, val, val]

        # Đặt một vật thể có màu (quả cầu xanh lục) ở giữa
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - 60)**2 + (Y - 60)**2)
        fg = dist <= 30
        img = bg.copy()
        img[fg] = [30, 180, 60]

        mask = _detect_solid_background_mask(img)
        self.assertIsNotNone(mask, "Phải nhận diện được phông studio trắng!")

        # Tâm vật thể phải thuộc foreground
        self.assertEqual(mask[60, 60], 255)
        # Các góc ảnh phải thuộc background
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[0, -1], 0)
        self.assertEqual(mask[-1, 0], 0)
        self.assertEqual(mask[-1, -1], 0)

        # Đo diện tích: diện tích hình tròn r=30 là pi * 30^2 ~ 2827 pixels
        fg_count = np.sum(mask > 127)
        self.assertGreater(fg_count, 2500)
        self.assertLess(fg_count, 3200)


if __name__ == "__main__":
    unittest.main()
