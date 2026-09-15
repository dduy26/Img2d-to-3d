"""
Kiểm thử Phân hệ 1 (P1): Cân bằng quang học trong không gian CIE Lab (Luminance-Only).
Xác minh:
1. Chỉ đồng bộ kênh L (Luminance) theo Anchor View #0.
2. Bảo toàn 100% sắc độ nguyên bản trên kênh a, b của vật thể (không gây lệch màu Albedo).
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Đường dẫn import
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from notebook.backend.preprocess import histogram_match_sequence


class TestP1OpticalNormalization(unittest.TestCase):
    def setUp(self):
        # Tạo 2 ảnh có cùng màu sắc chủ đạo (vật thể đỏ) nhưng khác độ sáng
        # Ảnh 0: Sáng chuẩn (L ~ 180)
        # Ảnh 1: Tối hơn nhiều (L ~ 70) do bóng râm hoặc phơi sáng kém
        self.img_bright = np.zeros((100, 100, 3), dtype=np.uint8)
        self.img_bright[:, :] = [220, 40, 40]  # Đỏ tươi

        self.img_dark = np.zeros((100, 100, 3), dtype=np.uint8)
        self.img_dark[:, :] = [100, 15, 15]   # Đỏ thẫm (cùng tông màu)

    def test_luminance_matching_adjusts_brightness(self):
        """Kiểm tra độ sáng kênh L của ảnh 1 được kéo về gần ảnh 0."""
        matched = histogram_match_sequence([self.img_bright, self.img_dark], anchor_idx=0)
        self.assertEqual(len(matched), 2)

        # Chuyển sang Lab để đo
        lab_anchor = cv2.cvtColor(matched[0], cv2.COLOR_RGB2LAB)
        lab_matched = cv2.cvtColor(matched[1], cv2.COLOR_RGB2LAB)
        lab_original_dark = cv2.cvtColor(self.img_dark, cv2.COLOR_RGB2LAB)

        # Kênh L của ảnh tối ban đầu
        l_orig = np.mean(lab_original_dark[:, :, 0])
        l_anchor = np.mean(lab_anchor[:, :, 0])
        l_matched = np.mean(lab_matched[:, :, 0])

        # l_matched phải gần l_anchor hơn rất nhiều so với l_orig
        diff_before = abs(l_orig - l_anchor)
        diff_after = abs(l_matched - l_anchor)
        self.assertLess(diff_after, diff_before, "Luminance matching phải giảm chênh lệch độ sáng!")
        self.assertLess(diff_after, 5.0, "Độ sáng sau khi match phải bám sát anchor view trong vòng 5 đơn vị!")

    def test_chrominance_preservation(self):
        """Kiểm tra góc sắc tướng (Hue angle) và kênh a, b sau khi match không bị đổi màu."""
        # Tạo ảnh có hoa văn nhiều màu
        test_img = np.zeros((64, 64, 3), dtype=np.uint8)
        test_img[:32, :32] = [200, 70, 70]    # Đỏ
        test_img[:32, 32:] = [70, 200, 70]    # Xanh lá
        test_img[32:, :32] = [70, 70, 200]    # Xanh dương
        test_img[32:, 32:] = [200, 200, 70]   # Vàng

        # Anchor view với độ phơi sáng khác (~25% sáng hơn như trong môi trường chụp thật)
        anchor_img = np.clip(test_img.astype(float) * 1.25, 0, 255).astype(np.uint8)

        matched = histogram_match_sequence([anchor_img, test_img], anchor_idx=0)
        orig_lab = cv2.cvtColor(test_img, cv2.COLOR_RGB2LAB)
        matched_lab = cv2.cvtColor(matched[1], cv2.COLOR_RGB2LAB)

        # Tính góc sắc tướng Hue angle = arctan2(b - 128, a - 128)
        hue_orig = np.degrees(np.arctan2(orig_lab[:, :, 2].astype(float) - 128.0, orig_lab[:, :, 1].astype(float) - 128.0)) % 360.0
        hue_matched = np.degrees(np.arctan2(matched_lab[:, :, 2].astype(float) - 128.0, matched_lab[:, :, 1].astype(float) - 128.0)) % 360.0
        hue_diff = np.abs((hue_orig - hue_matched + 180.0) % 360.0 - 180.0)

        # Góc sắc tướng bảo toàn tuyệt đối (sai lệch < 2 độ)
        max_hue_diff = np.max(hue_diff)
        self.assertLessEqual(max_hue_diff, 2.0, f"Góc sắc tướng Hue bị lệch: {max_hue_diff:.2f}°!")

        # Kênh a và b lệch không quá 3 đơn vị
        diff_a = np.max(np.abs(orig_lab[:, :, 1].astype(float) - matched_lab[:, :, 1].astype(float)))
        diff_b = np.max(np.abs(orig_lab[:, :, 2].astype(float) - matched_lab[:, :, 2].astype(float)))
        self.assertLessEqual(diff_a, 3.0, f"Kênh a bị lệch: {diff_a}!")
        self.assertLessEqual(diff_b, 3.0, f"Kênh b bị lệch: {diff_b}!")


if __name__ == "__main__":
    unittest.main()
