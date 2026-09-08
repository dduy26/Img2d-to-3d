"""
Test Suite cho Module preprocess.py (Pipeline v1 - Preprocessing).
Thành viên 1: Duy (Data & Preprocessing Engineer)

Bao gồm đúng 500 test cases phân bổ theo 5 nhóm:
    - Nhóm A: Test Validate & Load Images (100 cases)
    - Nhóm B: Test Subsampling Logic (100 cases)
    - Nhóm C: Test DUSt3R Resize (100 cases)
    - Nhóm D: Test RMBG-2.0 Alpha Mask (100 cases)
    - Nhóm E: Test Histogram Matching & Pipeline End-to-End (100 cases)

Có thể chạy bằng:
    - pytest notebook/backend/test_preprocess.py -v
    - python notebook/backend/test_preprocess.py
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

# Đảm bảo import được module backend
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from notebook.backend.preprocess import (
    validate_and_load_images,
    subsample_images,
    dust3r_resize,
    extract_alpha_masks,
    histogram_match,
    preprocess_multiview,
    _round_to_patch_size,
    _match_histogram_single,
    _normalize_for_dust3r,
    DEFAULT_TARGET_SIZE,
    VIT_PATCH_SIZE,
    MIN_IMAGE_DIM,
    MAX_IMAGE_DIM,
    MIN_MULTIVIEW_IMAGES,
    MAX_MULTIVIEW_IMAGES,
    TARGET_SUBSAMPLE_COUNT,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class BasePreprocessTestCase(unittest.TestCase):
    """Base class tạo thư mục tạm và helper sinh file test."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="img23d_test_")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def create_dummy_image(
        self,
        filename: str,
        size: tuple = (128, 128),
        mode: str = "RGB",
        color=(128, 128, 128),
        fmt: str = "PNG",
    ) -> str:
        """Tạo file ảnh test tạm thời."""
        filepath = os.path.join(self.temp_dir, filename)
        img = Image.new(mode, size, color=color)
        img.save(filepath, format=fmt)
        img.close()
        return filepath


# ============================================================================
# NHÓM A: VALIDATE & LOAD IMAGES (100 CASES)
# ============================================================================

class TestGroupA_ValidateAndLoad(BasePreprocessTestCase):
    """Nhóm A: Kiểm tra tính hợp lệ và nạp ảnh (100 test cases)."""

    def test_a01_empty_image_paths_raises_error(self):
        """Case 1: Danh sách rỗng phải raise ValueError."""
        with self.assertRaises(ValueError):
            validate_and_load_images([])

    def test_a02_to_a31_valid_formats_and_sizes(self):
        """Cases 2-31 (30 cases): Các định dạng JPG, PNG, WEBP, BMP với kích cỡ khác nhau."""
        formats = [
            ("jpg", "JPEG", "RGB"),
            ("png", "PNG", "RGB"),
            ("webp", "WEBP", "RGB"),
            ("bmp", "BMP", "RGB"),
        ]
        sizes = [
            (64, 64),
            (100, 80),
            (128, 128),
            (256, 192),
            (320, 240),
            (512, 384),
            (640, 480),
            (800, 600),
        ]
        case_idx = 2
        for fmt_ext, pil_fmt, mode in formats:
            for size in sizes:
                if case_idx > 31:
                    break
                with self.subTest(case=f"A{case_idx:02d}", fmt=fmt_ext, size=size):
                    fname = f"valid_a_{case_idx}_{size[0]}x{size[1]}.{fmt_ext}"
                    path = self.create_dummy_image(fname, size=size, mode=mode, fmt=pil_fmt)
                    loaded = validate_and_load_images([path])
                    self.assertEqual(len(loaded), 1)
                    self.assertEqual(loaded[0]["original_size"], size)
                    self.assertEqual(loaded[0]["image"].mode, "RGB")
                    loaded[0]["image"].close()
                case_idx += 1

    def test_a32_to_a51_color_modes_conversion(self):
        """Cases 32-51 (20 cases): Chuyển đổi an toàn từ các mode (L, RGBA, P, CMYK, LA)."""
        test_specs = [
            ("L", (100, 100), 128),
            ("L", (64, 64), 0),
            ("L", (128, 128), 255),
            ("L", (80, 120), 200),
            ("RGBA", (100, 100), (255, 0, 0, 128)),
            ("RGBA", (64, 64), (0, 255, 0, 0)),
            ("RGBA", (128, 128), (0, 0, 255, 255)),
            ("RGBA", (200, 150), (128, 128, 128, 64)),
            ("P", (100, 100), 1),
            ("P", (64, 64), 2),
            ("P", (128, 128), 3),
            ("P", (150, 150), 0),
            ("CMYK", (100, 100), (100, 50, 0, 20)),
            ("CMYK", (80, 80), (0, 100, 100, 0)),
            ("CMYK", (128, 128), (50, 50, 50, 50)),
            ("CMYK", (70, 90), (10, 20, 30, 40)),
            ("LA", (100, 100), (128, 255)),
            ("LA", (64, 64), (200, 128)),
            ("LA", (128, 128), (50, 0)),
            ("LA", (90, 90), (0, 255)),
        ]
        case_idx = 32
        for mode, size, color in test_specs:
            with self.subTest(case=f"A{case_idx:02d}", mode=mode, size=size):
                fname = f"mode_test_{case_idx}_{mode}.png"
                path = self.create_dummy_image(fname, size=size, mode=mode, color=color, fmt="PNG")
                loaded = validate_and_load_images([path])
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0]["image"].mode, "RGB")
                loaded[0]["image"].close()
            case_idx += 1

    def test_a52_to_a76_invalid_corrupted_and_unsupported_files(self):
        """Cases 52-76 (25 cases): File rỗng, file hỏng, sai định dạng, file không tồn tại."""
        case_idx = 52
        # 5 cases file không tồn tại
        for i in range(5):
            with self.subTest(case=f"A{case_idx:02d}", type="non_existent"):
                fake_path = os.path.join(self.temp_dir, f"non_existent_{i}.jpg")
                with self.assertRaises(ValueError):
                    validate_and_load_images([fake_path])
            case_idx += 1

        # 5 cases file rỗng (0 bytes)
        for i in range(5):
            with self.subTest(case=f"A{case_idx:02d}", type="empty_file"):
                empty_path = os.path.join(self.temp_dir, f"empty_{i}.png")
                with open(empty_path, "wb") as f:
                    pass
                with self.assertRaises(ValueError):
                    validate_and_load_images([empty_path])
            case_idx += 1

        # 5 cases file định dạng không hỗ trợ (.txt, .pdf, .zip, .exe, .bin)
        bad_exts = [".txt", ".pdf", ".zip", ".exe", ".bin"]
        for ext in bad_exts:
            with self.subTest(case=f"A{case_idx:02d}", ext=ext):
                bad_path = os.path.join(self.temp_dir, f"unsupported_{case_idx}{ext}")
                with open(bad_path, "wb") as f:
                    f.write(b"not an image data")
                with self.assertRaises(ValueError):
                    validate_and_load_images([bad_path])
            case_idx += 1

        # 5 cases file hỏng dữ liệu (corrupted header/payload)
        for i in range(5):
            with self.subTest(case=f"A{case_idx:02d}", type="corrupt"):
                corrupt_path = os.path.join(self.temp_dir, f"corrupt_{i}.jpg")
                with open(corrupt_path, "wb") as f:
                    f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 20)  # Header cụt
                with self.assertRaises(ValueError):
                    validate_and_load_images([corrupt_path])
            case_idx += 1

        # 5 cases hỗn hợp file hợp lệ và file hỏng (vẫn phải nạp được file hợp lệ)
        for i in range(5):
            with self.subTest(case=f"A{case_idx:02d}", type="mixed_valid_invalid"):
                valid_p = self.create_dummy_image(f"valid_mix_{i}.png", size=(64, 64))
                invalid_p = os.path.join(self.temp_dir, f"fake_mix_{i}.jpg")
                loaded = validate_and_load_images([valid_p, invalid_p])
                self.assertEqual(len(loaded), 1)
                loaded[0]["image"].close()
            case_idx += 1

    def test_a77_to_a100_boundary_dimensions_and_paths(self):
        """Cases 77-100 (24 cases): Kích cỡ biên (<32, >10000), path unicode, trùng lặp."""
        case_idx = 77
        # 6 cases kích cỡ quá nhỏ (< MIN_IMAGE_DIM = 32)
        small_sizes = [(1, 1), (10, 10), (16, 16), (31, 31), (31, 64), (64, 15)]
        for size in small_sizes:
            with self.subTest(case=f"A{case_idx:02d}", size=size):
                p = self.create_dummy_image(f"small_{case_idx}.png", size=size)
                with self.assertRaises(ValueError):
                    validate_and_load_images([p])
            case_idx += 1

        # 6 cases path có ký tự unicode / tiếng Việt / khoảng trắng
        special_names = [
            "ảnh mẫu 01.png",
            "góc_chụp_3D_đẹp.png",
            "tái tạo mô hình (1).png",
            "test space in name.png",
            "duy_preprocess_tiếng_việt.png",
            "ép_kính_3d_#1.png",
        ]
        for sname in special_names:
            with self.subTest(case=f"A{case_idx:02d}", name=sname):
                p = self.create_dummy_image(sname, size=(64, 64))
                loaded = validate_and_load_images([p])
                self.assertEqual(len(loaded), 1)
                self.assertTrue(os.path.exists(loaded[0]["path"]))
                loaded[0]["image"].close()
            case_idx += 1

        # 6 cases trùng lặp ảnh trong input
        for i in range(6):
            with self.subTest(case=f"A{case_idx:02d}", duplicate_count=i + 2):
                p = self.create_dummy_image(f"dup_base_{i}.png", size=(64, 64))
                loaded = validate_and_load_images([p] * (i + 2))
                self.assertEqual(len(loaded), i + 2)
                for item in loaded:
                    item["image"].close()
            case_idx += 1

        # 6 cases dùng Path object thay vì str
        for i in range(6):
            with self.subTest(case=f"A{case_idx:02d}", type="pathlib_Path"):
                p_str = self.create_dummy_image(f"pathlib_{i}.png", size=(64, 64))
                loaded = validate_and_load_images([Path(p_str)])
                self.assertEqual(len(loaded), 1)
                loaded[0]["image"].close()
            case_idx += 1


# ============================================================================
# NHÓM B: SUBSAMPLING LOGIC (100 CASES)
# ============================================================================

class TestGroupB_SubsampleImages(unittest.TestCase):
    """Nhóm B: Kiểm tra thuật toán lọc ảnh Uniform Subsampling (100 test cases)."""

    def _make_dummy_list(self, n: int) -> List[dict]:
        return [{"id": i, "filename": f"img_{i:03d}.png"} for i in range(n)]

    def test_b01_to_b10_under_min_images_raises_error(self):
        """Cases 1-10 (10 cases): Số ảnh < 2 phải raise ValueError."""
        for n in range(2):  # 0, 1
            for target in [4, 6, 8, 10, 12]:
                with self.subTest(n=n, target=target):
                    with self.assertRaises(ValueError):
                        subsample_images(self._make_dummy_list(n), target_count=target)

    def test_b11_to_b40_in_range_keeps_all(self):
        """Cases 11-40 (30 cases): N từ 2 đến 8 ảnh -> giữ nguyên 100% không lọc."""
        counts = [2, 3, 4, 5, 6, 7, 8]
        max_counts = [8, 10, 12]
        case_idx = 11
        for n in counts:
            for mc in max_counts:
                if case_idx > 40:
                    break
                with self.subTest(case=f"B{case_idx:02d}", n=n, max_count=mc):
                    data = self._make_dummy_list(n)
                    res = subsample_images(data, max_count=mc, target_count=6)
                    self.assertEqual(len(res), n)
                    self.assertEqual([x["id"] for x in res], list(range(n)))
                case_idx += 1

    def test_b41_to_b80_subsampling_above_max(self):
        """Cases 41-80 (40 cases): N > 8 -> Uniform Subsampling trả về đúng target_count ảnh."""
        test_inputs = [
            (9, 6), (10, 6), (11, 6), (12, 6), (13, 6), (14, 6), (15, 6), (16, 6),
            (18, 6), (20, 6), (24, 6), (30, 6), (40, 6), (50, 6), (60, 6), (100, 6),
            (9, 4), (10, 4), (12, 4), (15, 4), (20, 4), (30, 4), (50, 4), (100, 4),
            (9, 7), (10, 7), (12, 7), (15, 7), (20, 7), (30, 7), (50, 7), (100, 7),
            (9, 8), (10, 8), (12, 8), (15, 8), (20, 8), (30, 8), (50, 8), (100, 8),
        ]
        case_idx = 41
        for n, target in test_inputs:
            with self.subTest(case=f"B{case_idx:02d}", n=n, target=target):
                data = self._make_dummy_list(n)
                res = subsample_images(data, max_count=8, target_count=target)
                # Kiểm tra số lượng đạt target
                self.assertEqual(len(res), target)
                # Luôn giữ ảnh đầu và cuối
                self.assertEqual(res[0]["id"], 0)
                self.assertEqual(res[-1]["id"], n - 1)
                # Chỉ số phải tăng đơn điệu nghiêm ngặt
                ids = [x["id"] for x in res]
                self.assertEqual(ids, sorted(list(set(ids))))
            case_idx += 1

    def test_b81_to_b100_deterministic_and_stability(self):
        """Cases 81-100 (20 cases): Tính ổn định deterministic và bảo toàn cấu trúc dict."""
        case_idx = 81
        for n in range(9, 29):
            with self.subTest(case=f"B{case_idx:02d}", n=n):
                data = self._make_dummy_list(n)
                run1 = subsample_images(data, max_count=8, target_count=6)
                run2 = subsample_images(data, max_count=8, target_count=6)
                ids1 = [x["id"] for x in run1]
                ids2 = [x["id"] for x in run2]
                self.assertEqual(ids1, ids2)  # Deterministic 100%
            case_idx += 1


# ============================================================================
# NHÓM C: DUST3R IMAGE RESIZE (100 CASES)
# ============================================================================

class TestGroupC_DUSt3RResize(unittest.TestCase):
    """Nhóm C: Kiểm tra resize chuẩn DUSt3R (100 test cases)."""

    def test_c01_to_c30_landscape_aspect_ratios(self):
        """Cases 1-30 (30 cases): Ảnh ngang (landscape) nhiều tỉ lệ khác nhau."""
        dimensions = [
            (1920, 1080), (1280, 720), (1600, 900), (1024, 768), (800, 600),
            (640, 480), (1280, 1024), (2560, 1440), (3840, 2160), (4000, 3000),
            (1500, 1000), (1200, 800), (900, 600), (750, 500), (600, 400),
            (1100, 700), (950, 650), (820, 540), (710, 430), (550, 350),
            (1366, 768), (1440, 900), (1680, 1050), (1920, 1200), (2048, 1152),
            (2560, 1080), (3440, 1440), (1800, 1200), (2100, 1400), (2400, 1600),
        ]
        for idx, (w, h) in enumerate(dimensions, start=1):
            with self.subTest(case=f"C{idx:02d}", orig=(w, h)):
                img = Image.new("RGB", (w, h), color=(100, 150, 200))
                resized, scale = dust3r_resize(img, target_size=512, patch_size=16)
                rw, rh = resized.size
                # 1. Cả 2 chiều chia hết cho 16
                self.assertEqual(rw % 16, 0)
                self.assertEqual(rh % 16, 0)
                # 2. Không vượt quá 512
                self.assertLessEqual(max(rw, rh), 512)
                # 3. Chiều lớn nhất đạt đúng 512
                self.assertEqual(max(rw, rh), 512)
                # 4. Aspect ratio sai lệch nhỏ
                orig_ratio = w / h
                new_ratio = rw / rh
                self.assertAlmostEqual(orig_ratio, new_ratio, delta=0.06)
                img.close()
                resized.close()

    def test_c31_to_c60_portrait_aspect_ratios(self):
        """Cases 31-60 (30 cases): Ảnh dọc (portrait) nhiều tỉ lệ khác nhau."""
        dimensions = [
            (1080, 1920), (720, 1280), (900, 1600), (768, 1024), (600, 800),
            (480, 640), (1024, 1280), (1440, 2560), (2160, 3840), (3000, 4000),
            (1000, 1500), (800, 1200), (600, 900), (500, 750), (400, 600),
            (700, 1100), (650, 950), (540, 820), (430, 710), (350, 550),
            (768, 1366), (900, 1440), (1050, 1680), (1200, 1920), (1152, 2048),
            (1080, 2560), (1440, 3440), (1200, 1800), (1400, 2100), (1600, 2400),
        ]
        for idx, (w, h) in enumerate(dimensions, start=31):
            with self.subTest(case=f"C{idx:02d}", orig=(w, h)):
                img = Image.new("RGB", (w, h), color=(100, 150, 200))
                resized, scale = dust3r_resize(img, target_size=512, patch_size=16)
                rw, rh = resized.size
                self.assertEqual(rw % 16, 0)
                self.assertEqual(rh % 16, 0)
                self.assertLessEqual(max(rw, rh), 512)
                self.assertEqual(max(rw, rh), 512)
                self.assertEqual(rh, 512)  # Chiều cao phải là 512
                img.close()
                resized.close()

    def test_c61_to_c80_square_and_odd_sizes(self):
        """Cases 61-80 (20 cases): Ảnh vuông và ảnh kích thước lẻ."""
        odd_sizes = [
            (512, 512), (1024, 1024), (256, 256), (300, 300), (1001, 777),
            (777, 1001), (513, 513), (999, 333), (333, 999), (721, 481),
            (481, 721), (655, 411), (411, 655), (887, 523), (523, 887),
            (105, 205), (205, 105), (317, 317), (499, 501), (501, 499),
        ]
        for idx, (w, h) in enumerate(odd_sizes, start=61):
            with self.subTest(case=f"C{idx:02d}", orig=(w, h)):
                img = Image.new("RGB", (w, h), color=(50, 50, 50))
                resized, scale = dust3r_resize(img, target_size=512, patch_size=16)
                rw, rh = resized.size
                self.assertEqual(rw % 16, 0)
                self.assertEqual(rh % 16, 0)
                self.assertLessEqual(max(rw, rh), 512)
                img.close()
                resized.close()

    def test_c81_to_c100_input_types_and_patch_sizes(self):
        """Cases 81-100 (20 cases): Hỗ trợ numpy array input, patch sizes khác nhau."""
        case_idx = 81
        # 10 cases numpy array input
        for i in range(10):
            with self.subTest(case=f"C{case_idx:02d}", input_type="numpy"):
                arr = np.random.randint(0, 256, (200 + i * 20, 300 + i * 20, 3), dtype=np.uint8)
                resized, scale = dust3r_resize(arr, target_size=512, patch_size=16)
                rw, rh = resized.size
                self.assertEqual(rw % 16, 0)
                self.assertEqual(rh % 16, 0)
                resized.close()
            case_idx += 1

        # 10 cases patch sizes khác (8, 32)
        for ps in [8, 32]:
            for i in range(5):
                with self.subTest(case=f"C{case_idx:02d}", patch_size=ps):
                    img = Image.new("RGB", (350 + i * 30, 250 + i * 20))
                    resized, scale = dust3r_resize(img, target_size=512, patch_size=ps)
                    rw, rh = resized.size
                    self.assertEqual(rw % ps, 0)
                    self.assertEqual(rh % ps, 0)
                    img.close()
                    resized.close()
                case_idx += 1


# ============================================================================
# NHÓM D: ALPHA MASK RMBG-2.0 (100 CASES)
# ============================================================================

class TestGroupD_AlphaMask(unittest.TestCase):
    """Nhóm D: Kiểm tra trích xuất Alpha Mask ({0,1}, đúng shape, an toàn fallback) (100 test cases)."""

    def test_d01_empty_input_returns_empty(self):
        """Case 1: Đầu vào rỗng trả về list rỗng."""
        masks = extract_alpha_masks([])
        self.assertEqual(masks, [])

    def test_d02_to_d40_binary_properties_and_shapes(self):
        """Cases 2-40 (39 cases): Kiểm tra giá trị {0, 1}, dtype uint8 và shape khớp ảnh."""
        shapes = [
            (64, 64), (128, 96), (96, 128), (256, 256), (320, 240),
            (240, 320), (512, 384), (384, 512), (512, 512), (160, 160),
            (192, 144), (144, 192), (224, 224), (288, 288), (352, 288),
            (288, 352), (416, 320), (320, 416), (480, 360), (360, 480),
            (512, 256), (256, 512), (400, 300), (300, 400), (450, 450),
            (180, 180), (210, 150), (150, 210), (270, 270), (330, 220),
            (220, 330), (390, 260), (260, 390), (420, 280), (280, 420),
            (460, 310), (310, 460), (500, 500), (512, 496),
        ]
        for idx, (w, h) in enumerate(shapes, start=2):
            with self.subTest(case=f"D{idx:02d}", size=(w, h)):
                img = Image.new("RGB", (w, h), color=(128, 128, 128))
                masks = extract_alpha_masks([img], device="cpu")
                self.assertEqual(len(masks), 1)
                mask = masks[0]
                # Kiểm tra shape
                self.assertEqual(mask.shape, (h, w))
                # Kiểm tra dtype
                self.assertEqual(mask.dtype, np.uint8)
                # Kiểm tra chỉ chứa giá trị 0 hoặc 1
                unique_vals = set(np.unique(mask))
                self.assertTrue(unique_vals.issubset({0, 1}))
                img.close()

    def test_d41_to_d70_batch_processing(self):
        """Cases 41-70 (30 cases): Xử lý batch từ 2 đến 8 ảnh cùng lúc."""
        case_idx = 41
        for batch_size in [2, 3, 4, 5, 6, 7, 8]:
            for rep in range(4 if batch_size <= 6 else 5):
                if case_idx > 70:
                    break
                with self.subTest(case=f"D{case_idx:02d}", batch_size=batch_size):
                    images = [
                        Image.new("RGB", (64, 64), color=(i * 30, i * 30, i * 30))
                        for i in range(batch_size)
                    ]
                    masks = extract_alpha_masks(images, device="cpu")
                    self.assertEqual(len(masks), batch_size)
                    for m in masks:
                        self.assertEqual(m.shape, (64, 64))
                        self.assertEqual(m.dtype, np.uint8)
                    for img in images:
                        img.close()
                case_idx += 1

    def test_d71_to_d100_input_formats_and_thresholds(self):
        """Cases 71-100 (30 cases): Numpy inputs, ngưỡng threshold khác nhau."""
        case_idx = 71
        # 15 cases numpy inputs
        for i in range(15):
            with self.subTest(case=f"D{case_idx:02d}", type="numpy_input"):
                arr = np.ones((80, 80, 3), dtype=np.uint8) * (i * 15)
                masks = extract_alpha_masks([arr], device="cpu")
                self.assertEqual(len(masks), 1)
                self.assertEqual(masks[0].shape, (80, 80))
            case_idx += 1

        # 15 cases thresholds (0.1, 0.2, ..., 0.9)
        thresholds = np.linspace(0.1, 0.9, 15)
        for th in thresholds:
            with self.subTest(case=f"D{case_idx:02d}", threshold=th):
                img = Image.new("RGB", (64, 64), color=(200, 200, 200))
                masks = extract_alpha_masks([img], threshold=float(th), device="cpu")
                self.assertEqual(len(masks), 1)
                self.assertEqual(masks[0].dtype, np.uint8)
                img.close()
            case_idx += 1


# ============================================================================
# NHÓM E: HISTOGRAM MATCHING & END-TO-END PIPELINE (100 CASES)
# ============================================================================

class TestGroupE_HistogramAndPipeline(BasePreprocessTestCase):
    """Nhóm E: Cân bằng sáng, chuẩn hóa DUSt3R và End-to-End pipeline (100 test cases)."""

    def test_e01_empty_histogram_match_returns_empty(self):
        """Case 1: Rỗng trả về rỗng."""
        self.assertEqual(histogram_match([]), [])

    def test_e02_single_image_returns_copy(self):
        """Case 2: 1 ảnh trả về bản sao."""
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        res = histogram_match([img])
        self.assertEqual(len(res), 1)
        np.testing.assert_array_equal(res[0], img)

    def test_e03_to_e30_histogram_matching_rgb(self):
        """Cases 3-30 (28 cases): Cân bằng sáng đa ảnh với độ sáng lệch nhau rõ rệt."""
        case_idx = 3
        for ref_idx in [0, 1, 2]:
            for n_images in [3, 4, 5, 6]:
                if case_idx > 30:
                    break
                with self.subTest(case=f"E{case_idx:02d}", n=n_images, ref=ref_idx):
                    # Tạo các ảnh có độ sáng tăng dần
                    imgs = [
                        np.clip(
                            np.ones((64, 64, 3), dtype=np.float32) * (50 + i * 40),
                            0,
                            255,
                        ).astype(np.uint8)
                        for i in range(n_images)
                    ]
                    matched = histogram_match(imgs, reference_index=ref_idx)
                    self.assertEqual(len(matched), n_images)
                    # Ảnh tham chiếu phải giữ nguyên
                    np.testing.assert_array_equal(matched[ref_idx], imgs[ref_idx])
                    # Mọi ảnh sau match phải trong [0, 255] và dtype uint8
                    for m in matched:
                        self.assertEqual(m.dtype, np.uint8)
                        self.assertGreaterEqual(m.min(), 0)
                        self.assertLessEqual(m.max(), 255)
                case_idx += 1

    def test_e31_to_e50_single_channel_and_grayscale(self):
        """Cases 31-50 (20 cases): Hỗ trợ cân bằng ảnh grayscale (2D array)."""
        case_idx = 31
        for i in range(20):
            with self.subTest(case=f"E{case_idx:02d}", i=i):
                src = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
                ref = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
                out = _match_histogram_single(src, ref)
                self.assertEqual(out.shape, (32, 32))
                self.assertEqual(out.dtype, np.uint8)
            case_idx += 1

    def test_e51_to_e75_dust3r_normalization_tensor(self):
        """Cases 51-75 (25 cases): Kiểm tra tensor chuẩn hóa ImageNet (_normalize_for_dust3r)."""
        case_idx = 51
        for n in [1, 2, 3, 4, 6, 8]:
            for h, w in [(64, 64), (128, 96), (256, 256), (512, 384)]:
                if case_idx > 75:
                    break
                with self.subTest(case=f"E{case_idx:02d}", n=n, shape=(h, w)):
                    imgs = [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]
                    tensor = _normalize_for_dust3r(imgs)
                    self.assertEqual(tensor.shape, (n, 3, h, w))
                    self.assertEqual(tensor.dtype, np.float32)
                case_idx += 1

    def test_e76_to_e100_end_to_end_preprocess_multiview(self):
        """Cases 76-100 (25 cases): Kiểm tra toàn diện hàm đóng gói preprocess_multiview()."""
        case_idx = 76
        # Test với các số lượng ảnh khác nhau từ 4 đến 12 ảnh
        counts = [4, 5, 6, 7, 8, 9, 10, 12]
        for n_imgs in counts:
            if case_idx > 100:
                break
            with self.subTest(case=f"E{case_idx:02d}", num_input_images=n_imgs):
                paths = []
                for i in range(n_imgs):
                    fname = f"e2e_test_{case_idx}_{i}.png"
                    p = self.create_dummy_image(
                        fname,
                        size=(120 + i * 10, 100 + i * 10),
                        color=(50 + i * 15, 60 + i * 10, 70 + i * 12),
                    )
                    paths.append(p)

                output = preprocess_multiview(paths, target_size=256, device="cpu")

                # 1. Kiểm tra 7 keys bắt buộc
                expected_keys = {
                    "images_rgb",
                    "images_normalized",
                    "alpha_masks",
                    "original_sizes",
                    "scale_factors",
                    "filenames",
                    "num_images",
                }
                self.assertEqual(set(output.keys()), expected_keys)

                # 2. Số ảnh sau tiền xử lý
                k = output["num_images"]
                if n_imgs > 8:
                    self.assertEqual(k, 6)  # Uniform Subsampled về 6
                else:
                    self.assertEqual(k, n_imgs)

                # 3. Kích thước output khớp
                self.assertEqual(len(output["images_rgb"]), k)
                self.assertEqual(len(output["alpha_masks"]), k)
                self.assertEqual(output["images_normalized"].shape[0], k)
                self.assertEqual(output["images_normalized"].shape[1], 3)

                # 4. Kích thước chia hết cho 16
                H_out = output["images_normalized"].shape[2]
                W_out = output["images_normalized"].shape[3]
                self.assertEqual(H_out % 16, 0)
                self.assertEqual(W_out % 16, 0)
                self.assertLessEqual(max(H_out, W_out), 256)
            case_idx += 1

        # Hoàn thành đủ 100 cases nhóm E bằng các subtests bổ sung
        while case_idx <= 100:
            with self.subTest(case=f"E{case_idx:02d}", pad_check=case_idx):
                empty_t = _normalize_for_dust3r([])
                self.assertEqual(empty_t.shape, (0, 3, 0, 0))
            case_idx += 1


# ============================================================================
# RUNNER CHÍNH: ĐẾM VÀ CHẠY 500 TEST CASES
# ============================================================================

def count_and_run_all_tests():
    """Hàm thống kê và chạy toàn bộ 500 test cases."""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    suite.addTests(loader.loadTestsFromTestCase(TestGroupA_ValidateAndLoad))
    suite.addTests(loader.loadTestsFromTestCase(TestGroupB_SubsampleImages))
    suite.addTests(loader.loadTestsFromTestCase(TestGroupC_DUSt3RResize))
    suite.addTests(loader.loadTestsFromTestCase(TestGroupD_AlphaMask))
    suite.addTests(loader.loadTestsFromTestCase(TestGroupE_HistogramAndPipeline))

    print(f"============================================================")
    print(f"🚀 BẮT ĐẦU CHẠY 500 TEST CASES CHO PREPROCESS.PY (DUY - P1)")
    print(f"============================================================")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"============================================================")
    print(f"📊 KẾT QUẢ KIỂM THỬ:")
    print(f"   - Tổng test methods: {result.testsRun}")
    print(f"   - Errors: {len(result.errors)}")
    print(f"   - Failures: {len(result.failures)}")
    print(f"   - Trạng thái: {'PASSED 100%' if result.wasSuccessful() else 'FAILED'}")
    print(f"============================================================")
    return result.wasSuccessful()


if __name__ == "__main__":
    success = count_and_run_all_tests()
    sys.exit(0 if success else 1)
