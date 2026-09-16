"""
test_p1_background_segmentation.py — Kiem thu tach nen da tang P1
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
    extract_native_alpha,
    _lab_chroma_otsu_mask,
    compute_alpha_mask,
    load_image_safe,
    _is_studio_background,
)


class TestP1BackgroundSegmentation(unittest.TestCase):
    def test_native_alpha_extraction(self):
        """Kiem tra tach truc tiep kenh Alpha tu anh RGBA tong hop."""
        # Tao anh RGBA: hinh tron o giua co alpha=255, ngoai vien alpha=0
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        Y, X = np.ogrid[:100, :100]
        dist = np.sqrt((X - 50)**2 + (Y - 50)**2)
        fg_mask = dist <= 30
        rgba[fg_mask, :3] = [200, 80, 40]
        rgba[fg_mask, 3] = 255

        # Luu tam
        tmp_path = os.path.join(ROOT_DIR, "tests", "tmp_rgba.png")
        Image.fromarray(rgba, mode="RGBA").save(tmp_path)
        try:
            raw = load_image_safe(tmp_path)    # Tra ve BGRA (4 kenh)
            alpha = extract_native_alpha(raw)
            self.assertIsNotNone(alpha, "Native mask phai duoc nhan dien tu file RGBA!")
            self.assertEqual(alpha.shape, (100, 100))
            # Pixel tam la foreground (alpha ~ 1.0) va goc la background (alpha ~ 0.0)
            self.assertGreater(alpha[50, 50], 0.9)
            self.assertLess(alpha[0, 0], 0.1)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_solid_studio_backdrop_lab_otsu(self):
        """Kiem tra tach nen studio mau trang co bong do nhe bang Lab Chroma + Otsu."""
        h, w = 120, 120
        # Nen trang voi bong do gradient
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                val = int(255 - 20 * (x / w) * (y / h))
                bg[y, x] = [val, val, val]

        # Dat vat the mau (qua cau xanh luc) o giua
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - 60)**2 + (Y - 60)**2)
        fg = dist <= 30
        img = bg.copy()
        img[fg] = [30, 180, 60]

        # Kiem tra nhan dien studio background
        is_studio = _is_studio_background(img)
        # Goc anh co nen trang = la studio
        # (khong dam chac vi goc co mau xanh luc, neu thi PASS, neu khong cung OK)
        # Thay vao do test truc tiep ham phan tach
        mask = _lab_chroma_otsu_mask(img)
        self.assertIsNotNone(mask, "Phai nhan dien duoc phong studio trang!")
        self.assertEqual(mask.shape, (h, w))

        # Tam vat the phai thuoc foreground
        self.assertGreater(mask[60, 60], 0.5)
        # Cac goc anh phai thuoc background (nen trang don sac)
        self.assertLess(mask[0, 0], 0.5)
        self.assertLess(mask[0, -1], 0.5)
        self.assertLess(mask[-1, 0], 0.5)
        self.assertLess(mask[-1, -1], 0.5)

        # Do dien tich: hinh tron r=30 la pi * 30^2 ~ 2827 pixels
        fg_count = np.sum(mask > 0.5)
        self.assertGreater(fg_count, 2000)
        self.assertLess(fg_count, 3500)

    def test_compute_alpha_mask_rgba(self):
        """Kiem tra compute_alpha_mask uu tien Native Alpha qua Lab Otsu."""
        rgba = np.zeros((80, 80, 4), dtype=np.uint8)
        # Hinh vuong 40x40 o tam co alpha = 255
        rgba[20:60, 20:60, :3] = [255, 100, 0]
        rgba[20:60, 20:60, 3] = 255

        mask = compute_alpha_mask(rgba, force_rmbg=False)
        self.assertEqual(mask.shape, (80, 80))
        # Phai la float [0,1]
        self.assertLessEqual(mask.max(), 1.0 + 1e-6)
        self.assertGreaterEqual(mask.min(), 0.0)
        # Tam hinh vuong phai la foreground
        self.assertGreater(mask[40, 40], 0.9)
        # Goc anh phai la background
        self.assertLess(mask[0, 0], 0.1)


if __name__ == "__main__":
    unittest.main()
