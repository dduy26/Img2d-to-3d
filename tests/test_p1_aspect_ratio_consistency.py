"""
Kiểm thử Phân hệ 1 (P1): Tính toàn vẹn tỉ lệ hình học (Aspect Ratio Consistency).
Giải quyết triệt để lỗi: Mặt bên (Side views Left/Right) bị co dãn ngang/crop theo mặt trước (Front).
Xác minh:
1. Sử dụng DUY NHẤT một Global Uniform Scale cho tất cả N góc nhìn.
2. Tỉ lệ khung hình (w / h) của từng mặt được bảo toàn nguyên vẹn 1:1 trên canvas 512x512.
3. Thử nghiệm trên cả hình học tổng hợp bất đối xứng và Dataset Objaverse Train thực tế.
"""

import unittest
import numpy as np
import sys
import os
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from notebook.backend.preprocess import preprocess_multiview


class TestP1AspectRatioConsistency(unittest.TestCase):
    def test_synthetic_elongated_box_preserves_aspect_ratios(self):
        """Vật thể dạng hộp dài (chiếc giày / toa tàu / ô tô): Front hẹp, Side rộng gấp 3.3 lần."""
        tmp_dir = os.path.join(ROOT_DIR, "tests", "tmp_elongated")
        os.makedirs(tmp_dir, exist_ok=True)

        views_spec = [
            ("front.png", 120, 200),  # w=120, h=200 -> w/h = 0.60
            ("right.png", 400, 200),  # w=400, h=200 -> w/h = 2.00
            ("back.png",  120, 200),  # w=120, h=200 -> w/h = 0.60
            ("left.png",  400, 200),  # w=400, h=200 -> w/h = 2.00
        ]

        img_paths = []
        try:
            for name, w, h in views_spec:
                canvas = np.zeros((600, 600, 4), dtype=np.uint8)
                cy, cx = 300, 300
                y1, y2 = cy - h//2, cy + h//2
                x1, x2 = cx - w//2, cx + w//2
                canvas[y1:y2, x1:x2, :3] = [180, 70, 70]
                canvas[y1:y2, x1:x2, 3] = 255

                fpath = os.path.join(tmp_dir, name)
                Image.fromarray(canvas, mode="RGBA").save(fpath)
                img_paths.append(fpath)

            res = preprocess_multiview(img_paths, target_size=512)

            # 1. Kiểm tra Global Scale đồng nhất
            scales = res["scale_factors"]
            self.assertEqual(len(scales), 4)
            self.assertAlmostEqual(scales[0], scales[1], places=4)
            self.assertAlmostEqual(scales[1], scales[2], places=4)
            self.assertAlmostEqual(scales[2], scales[3], places=4)

            # 2. Đo tỉ lệ thực tế trên mask đã tiền xử lý
            for i, (name, expected_w, expected_h) in enumerate(views_spec):
                m = res["clean_mask_list"][i]
                coords = np.argwhere(m > 127)
                ymin, xmin = coords.min(axis=0)
                ymax, xmax = coords.max(axis=0)
                w_px = xmax - xmin
                h_px = ymax - ymin
                measured_ratio = float(w_px) / float(h_px)
                expected_ratio = float(expected_w) / float(expected_h)

                # Lệch không quá 5% do làm tròn pixel
                self.assertAlmostEqual(
                    measured_ratio, expected_ratio, delta=0.10,
                    msg=f"{name} bị sai lệch tỉ lệ: đo được {measured_ratio:.2f}, kỳ vọng {expected_ratio:.2f}"
                )

        finally:
            for p in img_paths:
                if os.path.exists(p):
                    os.remove(p)
            if os.path.exists(tmp_dir):
                os.rmdir(tmp_dir)

    def test_real_objaverse_train_aspect_ratio(self):
        """Kiểm tra trên bộ dữ liệu Objaverse train.zip đã tải về tests/data/objaverse_train."""
        data_dir = os.path.join(ROOT_DIR, "tests", "data", "objaverse_train", "images")
        self.assertTrue(os.path.exists(data_dir), "Dataset Objaverse train chưa sẵn sàng!")

        paths = [
            os.path.join(data_dir, "front.png"),
            os.path.join(data_dir, "right.png"),
            os.path.join(data_dir, "back.png"),
            os.path.join(data_dir, "left.png"),
        ]
        for p in paths:
            self.assertTrue(os.path.exists(p), f"Thiếu ảnh {p}")

        res = preprocess_multiview(paths, target_size=512)
        masks = res["clean_mask_list"]

        # Toa tàu nhìn ngang (Right/Left) phải dài hơn rất nhiều so với nhìn thẳng (Front/Back)
        coords_front = np.argwhere(masks[0] > 127)
        w_front = coords_front[:, 1].max() - coords_front[:, 1].min()

        coords_right = np.argwhere(masks[1] > 127)
        w_right = coords_right[:, 1].max() - coords_right[:, 1].min()

        # Toa tàu góc nghiêng/ngang có chiều rộng lớn hơn mặt trước
        self.assertGreater(w_right, w_front, "Mặt ngang của tàu phải rộng hơn mặt trước!")
        # Hệ số phóng đại phải giống nhau
        self.assertEqual(len(set(round(s, 5) for s in res["scale_factors"])), 1)


if __name__ == "__main__":
    unittest.main()
