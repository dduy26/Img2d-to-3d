"""
test_p1_aspect_ratio_consistency.py — Kiem thu tinh toan ven ti le hinh hoc P1
"""

import unittest
import numpy as np
import sys
import os
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# API moi: preprocess_images + place_on_canvas thay the preprocess_multiview
from notebook.backend.preprocess import preprocess_images, place_on_canvas, compute_alpha_mask, ensure_rgb, load_image_safe
from notebook.backend.utils_3d import compute_uniform_scale_params


class TestP1AspectRatioConsistency(unittest.TestCase):
    def test_synthetic_elongated_box_preserves_aspect_ratios(self):
        """Vat the dang hop dai (chiiec giay / toa tau / o to): Front hep, Side rong gap 3.3 lan."""
        tmp_dir = os.path.join(ROOT_DIR, "tests", "tmp_elongated")
        os.makedirs(tmp_dir, exist_ok=True)

        views_spec = [
            ("front.png", 120, 200),  # w=120, h=200 -> w/h = 0.60
            ("right.png", 400, 200),  # w=400, h=200 -> w/h = 2.00
            ("back.png",  120, 200),
            ("left.png",  400, 200),
        ]

        img_paths = []
        try:
            for name, w, h in views_spec:
                # Tao anh RGBA: hinh chu nhat mau do o tam nen trang
                canvas = np.full((600, 600, 4), 255, dtype=np.uint8)
                canvas[:, :, 3] = 0   # Nen trong suot
                cy, cx = 300, 300
                y1, y2 = cy - h//2, cy + h//2
                x1, x2 = cx - w//2, cx + w//2
                canvas[y1:y2, x1:x2, :3] = [180, 70, 70]
                canvas[y1:y2, x1:x2, 3]  = 255

                fpath = os.path.join(tmp_dir, name)
                Image.fromarray(canvas, mode="RGBA").save(fpath)
                img_paths.append(fpath)

            results = preprocess_images(img_paths, canvas_size=512)
            self.assertEqual(len(results), 4)

            # 1. Kiem tra scale: pipeline dung uniform scale = 512 / max(orig_w, orig_h)
            # Moi anh co kich thuoc goc 600x600 -> scale = 512/600
            for r in results:
                expected_scale = 512 / 600
                self.assertAlmostEqual(r["scale"], expected_scale, places=2,
                                       msg="Scale phai dong nhat cho moi anh!")

            # 2. Kiem tra ti le tren mask: foreground object trong canvas
            # Vat the nhin tu mat truoc (120x200) => ty le khoan 0.60
            # Vat the nhin tu mat ben (400x200) => ty le khoan 2.00
            # Chi test view 0 (front) va view 1 (right)
            m0 = results[0]["alpha_mask"]
            m1 = results[1]["alpha_mask"]

            coords0 = np.argwhere(m0 > 0.5)
            coords1 = np.argwhere(m1 > 0.5)

            if len(coords0) > 0 and len(coords1) > 0:
                w0 = coords0[:, 1].max() - coords0[:, 1].min()
                h0 = coords0[:, 0].max() - coords0[:, 0].min()
                w1 = coords1[:, 1].max() - coords1[:, 1].min()
                h1 = coords1[:, 0].max() - coords1[:, 0].min()

                ratio0 = float(w0) / (float(h0) + 1e-9)
                ratio1 = float(w1) / (float(h1) + 1e-9)

                # View 1 (right: 400x200) phai rong hon view 0 (front: 120x200)
                self.assertGreater(ratio1, ratio0,
                                   f"Mat ben phai co ti le rong hon mat truoc! r0={ratio0:.2f}, r1={ratio1:.2f}")

                # Ti le mat truoc ngan: w/h < 1
                self.assertLess(ratio0, 1.0, f"Mat truoc phai hep: {ratio0:.2f}")
                # Ti le mat ben rong: w/h > 1
                self.assertGreater(ratio1, 1.0, f"Mat ben phai rong: {ratio1:.2f}")

        finally:
            for p in img_paths:
                if os.path.exists(p):
                    os.remove(p)
            if os.path.exists(tmp_dir):
                try:
                    os.rmdir(tmp_dir)
                except OSError:
                    pass

    def test_real_objaverse_train_aspect_ratio(self):
        """Kiem tra tren bo du lieu Objaverse train da tai ve tests/data/objaverse_train."""
        data_dir = os.path.join(ROOT_DIR, "tests", "data", "objaverse_train", "images")
        if not os.path.exists(data_dir):
            self.skipTest("Dataset Objaverse train chua san sang!")

        paths = [
            os.path.join(data_dir, "front.png"),
            os.path.join(data_dir, "right.png"),
            os.path.join(data_dir, "back.png"),
            os.path.join(data_dir, "left.png"),
        ]
        for p in paths:
            self.assertTrue(os.path.exists(p), f"Thieu anh {p}")

        results = preprocess_images(paths, canvas_size=512)
        masks = [r["alpha_mask"] for r in results]

        # Toa tau nhin ngang (Right/Left) phai dai hon rat nhieu so voi nhin thang (Front/Back)
        coords_front = np.argwhere(masks[0] > 0.5)
        coords_right = np.argwhere(masks[1] > 0.5)

        if len(coords_front) > 0 and len(coords_right) > 0:
            w_front = coords_front[:, 1].max() - coords_front[:, 1].min()
            w_right = coords_right[:, 1].max() - coords_right[:, 1].min()
            self.assertGreater(w_right, w_front,
                               "Mat ngang cua tau phai rong hon mat truoc!")

        # Scale phai dong nhat
        scales = [r["scale"] for r in results]
        self.assertAlmostEqual(scales[0], scales[1], places=4)
        self.assertAlmostEqual(scales[1], scales[2], places=4)


if __name__ == "__main__":
    unittest.main()
