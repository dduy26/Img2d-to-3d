"""
P6 TỰ KIỂM: mask nền có tới được chỗ LẤY MÀU không + đo độ phủ gốc chụp.
Chạy: python notebook/backend/test_p6_texture_mask.py

VÌ SAO CẦN: hai lỗi này không làm pipeline đỏ, chỉ làm kết quả SAI mà trông như thành công.
  1. P5 gọi sample_rgb_nearest() KHÔNG truyền mask -> đỉnh mesh chiếu trúng nền lấy màu nền
     bake lên vật (nền trắng -> texture bạc màu). Test dưới đây đo thẳng: bật mask thì hết lẫn.
  2. P3 chỉ kiểm đồ thị camera LIÊN THÔNG, mà 5 camera cùng một mặt phẳng vẫn liên thông
     hoàn hảo -> quality_passed=true cho bộ ảnh không ai thấy mặt đáy. Test dưới đây đòi
     bộ ảnh một vòng ngang phải bị BÁO THIẾU, bộ ảnh phủ mặt cầu phải ĐẠT.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from PIL import Image

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import trimesh  # noqa: E402
from preprocess import masks_from_alpha, validate_and_load_images  # noqa: E402
from texture_blender import TextureBlender  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("✅ " if condition else "❌ ") + name + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_masks_from_alpha() -> None:
    """Kênh alpha có sẵn phải thành mask {0,1} đúng hình dạng."""
    rgba = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
    rgba.paste((0, 128, 255, 255), (16, 16, 48, 48))
    mask = masks_from_alpha([rgba.getchannel("A")], [rgba.convert("RGB")])[0]
    check("masks_from_alpha: uint8, chỉ {0,1}",
          mask.dtype == np.uint8 and set(np.unique(mask).tolist()) == {0, 1})
    check("masks_from_alpha: vật=1, nền=0", mask[32, 32] == 1 and mask[0, 0] == 0)
    check("masks_from_alpha: đúng tỉ lệ diện tích vật",
          abs(float(mask.mean()) - 0.25) < 0.01, f"đo được {float(mask.mean()):.3f}")


def test_preprocess_keeps_alpha() -> None:
    """P1 phải GIỮ kênh alpha (và không bịa alpha cho ảnh JPG)."""
    with tempfile.TemporaryDirectory() as tmp:
        Image.new("RGBA", (64, 64), (1, 2, 3, 255)).save(f"{tmp}/a.png")
        Image.new("RGB", (64, 64), (10, 20, 30)).save(f"{tmp}/b.jpg")
        loaded = validate_and_load_images([f"{tmp}/a.png", f"{tmp}/b.jpg"])
        check("P1 giữ kênh alpha của PNG", loaded[0].get("alpha") is not None)
        check("P1 không bịa alpha cho JPG", loaded[1].get("alpha") is None)
        check("P1 vẫn trả ảnh RGB 3 kênh",
              np.array(loaded[0]["image"]).shape == (64, 64, 3))


def test_mask_blocks_background_color() -> None:
    """Lõi của sửa đổi: có mask thì màu NỀN không vào được đỉnh mesh.

    Dựng cầu bán kính 0.5 cách camera 2m, f=200 -> chiếu ra bán kính 50px.
    Ảnh: nền TRẮNG, đĩa ĐỎ bán kính 50px. Mask cố ý nhỏ hơn (30px) để mô phỏng đúng ca
    thật: mesh/pose hơi lệch nên một vành đỉnh chiếu ra NGOÀI mask -> rơi vào nền.
    """
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    pose = np.eye(4)
    pose[:3, 0] = (1, 0, 0)      # right
    pose[:3, 1] = (0, -1, 0)     # down
    pose[:3, 2] = (0, 0, -1)     # forward: nhìn về gốc toạ độ
    pose[:3, 3] = (0, 0, 2)
    focal = (200.0, 200.0)

    size, radius_object, radius_mask = 200, 50, 30
    image = np.full((size, size, 3), 255, dtype=np.uint8)          # nền TRẮNG
    yy, xx = np.mgrid[0:size, 0:size]
    in_object = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= radius_object ** 2
    image[in_object] = (255, 0, 0)                                 # vật ĐỎ
    alpha = (((xx - size / 2) ** 2 + (yy - size / 2) ** 2) <= radius_mask ** 2).astype(np.uint8)

    blender = TextureBlender()
    without = blender.blend_colors_for_vertices(mesh, [image], [pose], [focal])
    with_mask = blender.blend_colors_for_vertices(mesh, [image], [pose], [focal], [alpha])

    green_without = float(without[:, 1].mean())
    green_with = float(with_mask[:, 1].mean())
    check("không mask: màu nền TRẮNG lẫn vào đỉnh mesh (tái hiện lỗi cũ)",
          green_without > 20, f"G trung bình = {green_without:.1f}")
    check("có mask: hết lẫn nền trắng", green_with < 2, f"G trung bình = {green_with:.1f}")
    check("có mask: vẫn giữ đúng màu ĐỎ của vật",
          float(with_mask[:, 0].mean()) > 240, f"R trung bình = {float(with_mask[:, 0].mean()):.1f}")


def test_audit_view_coverage() -> None:
    """Bộ ảnh một vòng ngang phải bị báo thiếu; bộ phủ mặt cầu phải đạt."""
    try:
        from app import audit_view_coverage
    except Exception as exc:      # app.py nạp engine -> thiếu dep thì bỏ qua, không tính là lỗi
        print(f"⚠️  bỏ qua kiểm độ phủ (không import được app: {type(exc).__name__}: {exc})")
        return

    rng = np.random.default_rng(0)
    points = rng.normal(size=(5000, 3)) * np.array([0.10, 0.033, 0.04])   # vật dẹt như giày
    confidence = np.ones((len(points), 1))

    def poses(directions, radius: float = 0.6):
        out = []
        for direction in directions:
            pose = np.eye(4)
            pose[:3, 3] = np.asarray(direction) * radius
            out.append(pose)
        return out

    i = np.arange(14) + 0.5
    phi, theta = np.arccos(1 - 2 * i / 14), np.pi * (1 + 5 ** 0.5) * i
    sphere = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], -1)

    n, elevation = 6, 20.0
    ring = np.stack([
        [np.cos(2 * np.pi * k / n) * np.cos(np.deg2rad(elevation)),
         np.sin(2 * np.pi * k / n) * np.cos(np.deg2rad(elevation)),
         np.sin(np.deg2rad(elevation))]
        for k in range(n)
    ])

    good = audit_view_coverage(points, confidence, poses(sphere))
    bad = audit_view_coverage(points, confidence, poses(ring))
    check("độ phủ: 14 camera phủ mặt cầu -> ĐẠT", good["ok"] is True,
          f"góc tốt nhất theo trục = {good['axis_best_angles_deg']}°")
    check("độ phủ: 6 camera một vòng ngang 20° -> BÁO THIẾU", bad["ok"] is False,
          f"góc tốt nhất theo trục = {bad['axis_best_angles_deg']}°")
    print("    lý do:", bad["reason"])


def main() -> int:
    print("=" * 74)
    print("P6 TỰ KIỂM: mask nền + độ phủ gốc chụp")
    print("=" * 74)
    test_masks_from_alpha()
    test_preprocess_keeps_alpha()
    test_mask_blocks_background_color()
    test_audit_view_coverage()
    print("=" * 74)
    print("KẾT QUẢ:", "TẤT CẢ ĐẠT" if not FAILURES else f"HỎNG {len(FAILURES)}: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
