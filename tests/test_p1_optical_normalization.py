"""
test_p1_optical_normalization.py — Kiem thu can bang quang hoc CIE Lab P1
"""

import unittest
import numpy as np
import cv2
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# API moi: normalize_exposure va match_l_channel thay the histogram_match_sequence
from notebook.backend.preprocess import normalize_exposure, match_l_channel


class TestP1OpticalNormalization(unittest.TestCase):
    def setUp(self):
        # Tao 2 anh co cung mau sac chu dao (vat the do) nhung khac do sang
        # Anh 0: Sang chuan (L ~ 180)
        # Anh 1: Toi hon nhieu (L ~ 70) do bong ram hoac phoi sang kem
        self.img_bright = np.zeros((100, 100, 3), dtype=np.uint8)
        self.img_bright[:, :] = [220, 40, 40]  # Do tuoi

        self.img_dark = np.zeros((100, 100, 3), dtype=np.uint8)
        self.img_dark[:, :] = [100, 15, 15]   # Do tham (cung tong mau)

    def test_luminance_matching_adjusts_brightness(self):
        """Kiem tra do sang kenh L cua anh 1 duoc keo ve gan anh 0."""
        matched = normalize_exposure([self.img_bright, self.img_dark], reference_idx=0)
        self.assertEqual(len(matched), 2)

        # Chuyen sang Lab de do
        lab_anchor  = cv2.cvtColor(matched[0], cv2.COLOR_RGB2Lab)
        lab_matched = cv2.cvtColor(matched[1], cv2.COLOR_RGB2Lab)
        lab_orig_dark = cv2.cvtColor(self.img_dark, cv2.COLOR_RGB2Lab)

        l_orig    = np.mean(lab_orig_dark[:, :, 0])
        l_anchor  = np.mean(lab_anchor[:, :, 0])
        l_matched = np.mean(lab_matched[:, :, 0])

        diff_before = abs(l_orig - l_anchor)
        diff_after  = abs(l_matched - l_anchor)
        self.assertLess(diff_after, diff_before,
                        "Luminance matching phai giam chenh lech do sang!")
        self.assertLess(diff_after, 5.0,
                        "Do sang sau khi match phai bam sat anchor view trong vong 5 don vi!")

    def test_chrominance_preservation(self):
        """Kiem tra goc sac tuong (Hue angle) va kenh a, b sau khi match khong bi doi mau."""
        # Tao anh co hoa van nhieu mau
        test_img = np.zeros((64, 64, 3), dtype=np.uint8)
        test_img[:32, :32] = [200, 70,  70]   # Do
        test_img[:32, 32:] = [ 70, 200, 70]   # Xanh la
        test_img[32:, :32] = [ 70,  70, 200]  # Xanh duong
        test_img[32:, 32:] = [200, 200, 70]   # Vang

        # Anchor view voi do phoi sang khac (~25% sang hon)
        anchor_img = np.clip(test_img.astype(float) * 1.25, 0, 255).astype(np.uint8)

        matched = normalize_exposure([anchor_img, test_img], reference_idx=0)
        orig_lab    = cv2.cvtColor(test_img, cv2.COLOR_RGB2Lab)
        matched_lab = cv2.cvtColor(matched[1], cv2.COLOR_RGB2Lab)

        # Tinh goc sac tuong Hue angle = arctan2(b - 128, a - 128)
        hue_orig = np.degrees(
            np.arctan2(orig_lab[:, :, 2].astype(float) - 128.0,
                       orig_lab[:, :, 1].astype(float) - 128.0)
        ) % 360.0
        hue_matched = np.degrees(
            np.arctan2(matched_lab[:, :, 2].astype(float) - 128.0,
                       matched_lab[:, :, 1].astype(float) - 128.0)
        ) % 360.0
        hue_diff = np.abs((hue_orig - hue_matched + 180.0) % 360.0 - 180.0)

        max_hue_diff = np.max(hue_diff)
        self.assertLessEqual(max_hue_diff, 2.0,
                             f"Goc sac tuong Hue bi lech: {max_hue_diff:.2f} do!")

        diff_a = np.max(np.abs(orig_lab[:, :, 1].astype(float) - matched_lab[:, :, 1].astype(float)))
        diff_b = np.max(np.abs(orig_lab[:, :, 2].astype(float) - matched_lab[:, :, 2].astype(float)))
        self.assertLessEqual(diff_a, 3.0, f"Kenh a bi lech: {diff_a}!")
        self.assertLessEqual(diff_b, 3.0, f"Kenh b bi lech: {diff_b}!")


if __name__ == "__main__":
    unittest.main()
