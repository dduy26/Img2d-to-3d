"""
Script sinh ảnh giả lập (synthetic) cho mục đích kiểm thử module preprocess.py.
Tạo 6 ảnh RGB với các đặc điểm khác nhau mô phỏng ảnh chụp đa góc quanh vật thể.
Bao gồm cả các ảnh edge-case (quá sáng, quá tối, blur, kích thước lạ).

Chạy: python data/input/generate_test_images.py
"""

import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    raise ImportError("Cần cài Pillow: pip install Pillow")


def create_object_on_background(
    width: int, height: int,
    bg_color: tuple, obj_color: tuple,
    obj_center: tuple, obj_radius: int,
    noise_level: float = 10.0,
    brightness_shift: int = 0
) -> Image.Image:
    """Tạo ảnh RGB có vật thể hình tròn trên nền có vân (texture)."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Vẽ vân nền (giả lập mặt bàn) bằng các đường kẻ ngang
    for y in range(0, height, 20):
        line_color = tuple(max(0, min(255, c + np.random.randint(-15, 15))) for c in bg_color)
        draw.line([(0, y), (width, y)], fill=line_color, width=1)

    # Vẽ vật thể hình tròn
    x0 = obj_center[0] - obj_radius
    y0 = obj_center[1] - obj_radius
    x1 = obj_center[0] + obj_radius
    y1 = obj_center[1] + obj_radius
    draw.ellipse([x0, y0, x1, y1], fill=obj_color, outline=(0, 0, 0), width=2)

    # Thêm chi tiết trên vật thể (hình tam giác nhỏ giả lập vân bề mặt)
    tri_size = obj_radius // 3
    cx, cy = obj_center
    draw.polygon([
        (cx, cy - tri_size),
        (cx - tri_size, cy + tri_size),
        (cx + tri_size, cy + tri_size)
    ], fill=tuple(max(0, min(255, c - 40)) for c in obj_color))

    # Thêm nhiễu Gaussian nhẹ
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, noise_level, arr.shape)
    arr = np.clip(arr + noise + brightness_shift, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def generate_multiview_set(output_dir: Path):
    """Sinh bộ 6 ảnh giả lập đa góc nhìn (vật thể ở các vị trí hơi khác nhau)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    base_w, base_h = 1920, 1080
    bg_color = (180, 160, 140)  # Màu nền giả lập mặt bàn gỗ
    obj_color = (60, 120, 200)  # Màu vật thể xanh dương

    # 6 góc nhìn: vật thể dịch chuyển nhẹ giả lập camera xoay quanh
    configs = [
        {"center": (960, 540), "brightness": 0, "label": "view_01_front"},
        {"center": (880, 520), "brightness": 15, "label": "view_02_front_left"},
        {"center": (750, 500), "brightness": 30, "label": "view_03_left"},
        {"center": (1050, 530), "brightness": -10, "label": "view_04_front_right"},
        {"center": (1150, 510), "brightness": -25, "label": "view_05_right"},
        {"center": (960, 480), "brightness": -40, "label": "view_06_back"},
    ]

    for cfg in configs:
        img = create_object_on_background(
            width=base_w, height=base_h,
            bg_color=bg_color, obj_color=obj_color,
            obj_center=cfg["center"], obj_radius=180,
            noise_level=8.0,
            brightness_shift=cfg["brightness"]
        )
        filepath = output_dir / f"{cfg['label']}.jpg"
        img.save(str(filepath), "JPEG", quality=92)
        print(f"  ✅ Đã tạo: {filepath.name} ({base_w}x{base_h})")


def generate_edge_case_images(output_dir: Path):
    """Sinh các ảnh edge-case để kiểm thử lỗi biên."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ảnh quá sáng (overexposed)
    bright = Image.new("RGB", (800, 600), (250, 250, 245))
    draw = ImageDraw.Draw(bright)
    draw.ellipse([300, 200, 500, 400], fill=(255, 255, 255))
    bright.save(str(output_dir / "edge_overexposed.jpg"), "JPEG")
    print("  ✅ edge_overexposed.jpg")

    # 2. Ảnh quá tối (underexposed)
    dark = Image.new("RGB", (800, 600), (15, 10, 8))
    draw = ImageDraw.Draw(dark)
    draw.ellipse([300, 200, 500, 400], fill=(30, 25, 20))
    dark.save(str(output_dir / "edge_underexposed.jpg"), "JPEG")
    print("  ✅ edge_underexposed.jpg")

    # 3. Ảnh bị blur (mờ)
    normal = create_object_on_background(800, 600, (180, 160, 140), (60, 120, 200), (400, 300), 120)
    blurred = normal.filter(ImageFilter.GaussianBlur(radius=8))
    blurred.save(str(output_dir / "edge_blurred.jpg"), "JPEG")
    print("  ✅ edge_blurred.jpg")

    # 4. Ảnh rất nhỏ (tiny)
    tiny = Image.new("RGB", (64, 48), (100, 150, 200))
    tiny.save(str(output_dir / "edge_tiny_64x48.jpg"), "JPEG")
    print("  ✅ edge_tiny_64x48.jpg")

    # 5. Ảnh rất lớn (large)
    large = Image.new("RGB", (4000, 3000), (200, 180, 160))
    draw = ImageDraw.Draw(large)
    draw.ellipse([1500, 1000, 2500, 2000], fill=(80, 140, 220))
    large.save(str(output_dir / "edge_large_4000x3000.jpg"), "JPEG")
    print("  ✅ edge_large_4000x3000.jpg")

    # 6. Ảnh vuông (square)
    square = Image.new("RGB", (1024, 1024), (170, 170, 170))
    draw = ImageDraw.Draw(square)
    draw.ellipse([312, 312, 712, 712], fill=(50, 100, 180))
    square.save(str(output_dir / "edge_square_1024.jpg"), "JPEG")
    print("  ✅ edge_square_1024.jpg")

    # 7. Ảnh dọc (portrait)
    portrait = Image.new("RGB", (600, 1080), (190, 175, 155))
    draw = ImageDraw.Draw(portrait)
    draw.ellipse([150, 340, 450, 640], fill=(70, 130, 210))
    portrait.save(str(output_dir / "edge_portrait_600x1080.jpg"), "JPEG")
    print("  ✅ edge_portrait_600x1080.jpg")

    # 8. Ảnh grayscale lưu dưới dạng JPG (1 channel nhưng lưu RGB)
    gray = Image.new("L", (800, 600), 128)
    gray_rgb = gray.convert("RGB")
    gray_rgb.save(str(output_dir / "edge_grayscale.jpg"), "JPEG")
    print("  ✅ edge_grayscale.jpg")

    # 9. File PNG (định dạng khác JPG)
    png_img = Image.new("RGB", (800, 600), (180, 160, 140))
    draw = ImageDraw.Draw(png_img)
    draw.ellipse([300, 200, 500, 400], fill=(60, 120, 200))
    png_img.save(str(output_dir / "edge_format.png"), "PNG")
    print("  ✅ edge_format.png")

    # 10. File RGBA (4 kênh)
    rgba = Image.new("RGBA", (800, 600), (180, 160, 140, 255))
    draw = ImageDraw.Draw(rgba)
    draw.ellipse([300, 200, 500, 400], fill=(60, 120, 200, 200))
    rgba.save(str(output_dir / "edge_rgba.png"), "PNG")
    print("  ✅ edge_rgba.png")

    # 11. File không phải ảnh (text file giả mạo extension)
    with open(str(output_dir / "edge_fake_image.jpg"), "w") as f:
        f.write("This is not an image file!")
    print("  ✅ edge_fake_image.jpg (file giả)")

    # 12. File rỗng
    with open(str(output_dir / "edge_empty.jpg"), "wb") as f:
        pass
    print("  ✅ edge_empty.jpg (file rỗng)")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent

    print("\n📸 [1/2] Đang sinh bộ 6 ảnh multi-view benchmark...")
    generate_multiview_set(base / "multi_view")

    print("\n⚠️  [2/2] Đang sinh ảnh edge-case...")
    generate_edge_case_images(base / "multi_view")

    print("\n✅ Hoàn thành! Toàn bộ ảnh đã được lưu vào data/input/multi_view/")
    print(f"   Tổng số file: {len(list((base / 'multi_view').iterdir()))}")
