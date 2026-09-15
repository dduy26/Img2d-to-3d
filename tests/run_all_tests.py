"""
Bộ Chạy Thử Nghiệm Toàn Diện (All-In-One Test Runner) Cho Dự Án ImgToModel.
Chạy toàn bộ 6 bộ kiểm thử bao quát tất cả vấn đề đã giải quyết:
1. test_p1_optical_normalization.py     (Đồng bộ quang học CIE Lab Luminance-Only)
2. test_p1_background_segmentation.py   (Tách nền đa tầng: Native Alpha & Lab Chroma Otsu)
3. test_p1_aspect_ratio_consistency.py  (Bảo toàn tỉ lệ khung hình Front vs Side)
4. test_p1_camera_intrinsics_math.py    (Toán học bù trừ ma trận Camera Intrinsics K')
5. test_p4_space_carving_tsdf.py        (Lưới thể tích kín nước, 0 cạnh hở, 1 khối)
6. test_p5_texture_blending.py          (Nướng màu đa hướng Fresnel với K' bù trừ)
7. test_e2e_objaverse_train.py          (Tích hợp đầu-cuối trên Dataset Objaverse Train)
"""

import sys
import os
import unittest
import time

# Thiết lập UTF-8 cho Windows console nếu có thể
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

TEST_MODULES = [
    "tests.test_p1_optical_normalization",
    "tests.test_p1_background_segmentation",
    "tests.test_p1_aspect_ratio_consistency",
    "tests.test_p1_camera_intrinsics_math",
    "tests.test_p4_space_carving_tsdf",
    "tests.test_p5_texture_blending",
    "tests.test_e2e_objaverse_train",
]


def run():
    print("=" * 75)
    print("      HỆ THỐNG KIỂM THỬ TOÀN DIỆN PIPELINE 2D-TO-3D (NVIDIA CHUẨN TẮC)")
    print("=" * 75)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for mod_name in TEST_MODULES:
        try:
            mod_suite = loader.loadTestsFromName(mod_name)
            suite.addTests(mod_suite)
        except Exception as e:
            print(f"[LỖI TẢI TEST] Không thể nạp module {mod_name}: {e}")

    total_tests = suite.countTestCases()
    print(f"Tổng số bài kiểm tra đã nạp: {total_tests}\n")

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("\n" + "=" * 75)
    print("                      KẾT QUẢ KIỂM NGHIỆM")
    print("=" * 75)
    print(f"  • Thời gian chạy: {elapsed:.2f} giây")
    print(f"  • Tổng bài test: {result.testsRun}")
    print(f"  • Thành công:    {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  • Thất bại:      {len(result.failures)}")
    print(f"  • Lỗi phát sinh: {len(result.errors)}")
    print("=" * 75)

    if result.wasSuccessful():
        print(">>> CHÚC MỪNG: 100% CÁC BÀI KIỂM THỬ ĐỀU ĐẠT CHUẨN XUẤT SẮC! <<<")
        sys.exit(0)
    else:
        print(">>> CẢNH BÁO: MỘT SỐ BÀI KIỂM THỬ CHƯA ĐẠT YÊU CẦU! <<<")
        sys.exit(1)


if __name__ == "__main__":
    run()
