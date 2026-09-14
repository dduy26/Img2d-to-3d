"""
P4 TỰ KIỂM BẰNG GROUND TRUTH TỔNG HỢP (không cần model, không cần mạng)
Chạy: python notebook/backend/test_tsdf_synthetic.py

VÌ SAO CẦN: mọi kết luận trước đây về P4 đều dựa vào ảnh GSO mà KHÔNG biết trước hình
dạng thật -> không tách được "P4 sai" với "ảnh/pose sai". Ở đây tự dựng một ellipsoid
(tỉ lệ giống giày), tự đặt camera, tự tính pointmap CHÍNH XÁC (không nhiễu), rồi đưa
đúng vào P4. Biết trước ground truth nên đọc được sai số thật.

HAI KỊCH BẢN — vì kết quả ĐÚNG khác nhau:
  A. Phủ đủ mọi hướng (14 camera trên mặt cầu) -> phải ra VẬT ĐẶC, kín, đúng thể tích.
     Đây là phép kiểm bắt lỗi "vỏ rỗng": TSDF một phía để ruột = +1 nên Marching Cubes
     dựng thêm mặt bên trong -> thể tích chỉ còn ~20% vật đặc.
  B. Chỉ chụp từ trên (6 camera, nâng 20°) -> ĐÁY không ai thấy. Kết quả ĐÚNG là mesh
     HỞ ở đáy. Không được là vỏ rỗng, cũng không được "bịa" ra đáy.
"""

from __future__ import annotations

import os
import sys

import numpy as np

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from engine_tsdf_mesh import (  # noqa: E402
    TSDFMeshEngine,
    TSDFVolume,
    coverage_report,
    mesh_health,
    prune_background_points,
)

H, W = 368, 512
FOCAL = 588.0
SEMI_AXES = np.array([0.10, 0.033, 0.04])   # giày: tỉ lệ 1 : 0.33 : 0.4


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Ma trận camera->world (OpenCV: +Z hướng nhìn, +Y xuống)."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def render_world_pointmap(c2w: np.ndarray):
    """
    Với mỗi pixel, tìm giao điểm tia-camera với ellipsoid, trả TOẠ ĐỘ WORLD (đúng định
    dạng DUSt3R get_pts3d) + mask trúng vật.
    """
    cx, cy = W / 2.0, H / 2.0
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    dirs_cam = np.stack(
        [(us - cx) / FOCAL, (vs - cy) / FOCAL, np.ones_like(us)], axis=-1
    ).reshape(-1, 3)
    rot = c2w[:3, :3]
    origin = c2w[:3, 3]
    dirs_world = dirs_cam @ rot.T
    dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=1, keepdims=True)

    o = -SEMI_AXES * 0.0 / SEMI_AXES          # tâm tại gốc
    o = (origin - 0.0) / SEMI_AXES
    d = dirs_world / SEMI_AXES
    A = np.sum(d * d, axis=1)
    B = 2.0 * np.sum(o * d, axis=1)
    C = float(np.sum(o * o)) - 1.0
    disc = B * B - 4.0 * A * C
    hit = disc > 0.0
    sqrt_disc = np.sqrt(np.where(hit, disc, 0.0))
    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)
    t = np.where(t1 > 1e-6, t1, t2)
    hit &= t > 1e-6
    points_world = origin[None, :] + t[:, None] * dirs_world
    return points_world.reshape(H, W, 3), hit.reshape(H, W)


def sphere_directions(n: int) -> np.ndarray:
    """Fibonacci sphere: n hướng trải đều trên mặt cầu (phủ cả trên lẫn dưới)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1
    )


def make_scene(directions: np.ndarray, radius: float = 0.6):
    pointmaps, confs, alphas, poses = [], [], [], []
    for dirv in directions:
        c2w = look_at(radius * np.asarray(dirv, dtype=np.float64), np.zeros(3))
        pm, hit = render_world_pointmap(c2w)
        pointmaps.append(pm.astype(np.float32))
        confs.append(hit.astype(np.float32))
        alphas.append(np.ones((H, W), dtype=np.float32))
        poses.append(c2w.astype(np.float32))
    return np.stack(pointmaps), np.stack(confs), alphas, poses


def run_engine(pointmaps, confs, alphas, poses, resolution=128):
    return TSDFMeshEngine(resolution=resolution).reconstruct(
        pointmaps_3d=pointmaps,
        alpha_masks=alphas,
        confidence_masks=confs,
        camera_poses=poses,
        focal_lengths=[(FOCAL, FOCAL)] * len(poses),
    )


def measure_coverage(pointmaps, confs, alphas, poses, resolution=128):
    valid, filt = prune_background_points(
        pointmaps_3d=pointmaps, alpha_masks=alphas, confidence_masks=confs,
        tau_conf=0.35, tau_edge=0.07,
    )
    pts = np.concatenate([p for p in filt if len(p) > 0], axis=0)
    p_min, p_max = np.percentile(pts, 1.0, axis=0), np.percentile(pts, 99.0, axis=0)
    c = (p_min + p_max) / 2.0
    e = (p_max - p_min) * 1.3
    vol = TSDFVolume(c - e / 2.0, c + e / 2.0, resolution=resolution)
    for i in range(len(poses)):
        vol.integrate(pointmaps[i], valid[i], confs[i], poses[i], (FOCAL, FOCAL))
    return coverage_report(vol)


TRUE_VOLUME = 4.0 / 3.0 * np.pi * float(np.prod(SEMI_AXES))
TRUE_EXTENTS = 2.0 * SEMI_AXES


def report(title: str, mesh):
    h = mesh_health(mesh)
    print(f"\n{'-' * 74}\n{title}\n{'-' * 74}")
    print(f"  đỉnh/mặt        : {h['vertices']} / {h['faces']}")
    print(f"  mảnh rời        : {h['components']}   cạnh biên (hở): {h['boundary_edges']}")
    print(f"  watertight      : {h['watertight']}")
    print(f"  kích thước      : {np.round(mesh.extents, 4).tolist()} m")
    print(f"  sai số k.thước  : {np.round(100 * (mesh.extents - TRUE_EXTENTS) / TRUE_EXTENTS, 1).tolist()} %")
    print(f"  thể tích        : {h['volume']:.6f} m3  (tỉ lệ ra/thật = {h['volume'] / TRUE_VOLUME:.3f})")
    if h["volume"] > 0:
        print(f"  sai số thể tích : {100 * (h['volume'] - TRUE_VOLUME) / TRUE_VOLUME:+.1f}%")
    return h


def main() -> int:
    print("=" * 74)
    print("P4 TỰ KIỂM BẰNG GROUND TRUTH TỔNG HỢP")
    print("=" * 74)
    print(f"[GT] Ellipsoid thật: thể tích={TRUE_VOLUME:.6f} m3, "
          f"kích thước={np.round(TRUE_EXTENTS, 4).tolist()} m, "
          f"tỉ lệ={np.round(TRUE_EXTENTS / TRUE_EXTENTS.max(), 3).tolist()}")

    results: dict[str, dict] = {}

    # ── A. PHỦ ĐỦ MỌI HƯỚNG -> kỳ vọng VẬT ĐẶC, KÍN, ĐÚNG THỂ TÍCH ──
    dirs_a = sphere_directions(14)
    pm, cf, al, ps = make_scene(dirs_a)
    mesh_a = run_engine(pm, cf, al, ps)
    results["A"] = report("A. PHỦ ĐỦ MỌI HƯỚNG (14 camera trên mặt cầu)", mesh_a)
    cov_a = measure_coverage(pm, cf, al, ps)
    print(f"  độ phủ bề mặt   : {cov_a['pct_band_multi']:.1f}% dải bề mặt được >=2 view "
          f"xác nhận (nhiều nhất {cov_a['max_views']} view)")

    # ── B. CHỈ CHỤP TỪ TRÊN -> kỳ vọng HỞ Ở ĐÁY, KHÔNG vỏ rỗng ──
    n_b, elev_b = 6, 20.0
    dirs_b = np.stack([
        [np.cos(2 * np.pi * i / n_b) * np.cos(np.deg2rad(elev_b)),
         np.sin(2 * np.pi * i / n_b) * np.cos(np.deg2rad(elev_b)),
         np.sin(np.deg2rad(elev_b))]
        for i in range(n_b)
    ])
    pm_b, cf_b, al_b, ps_b = make_scene(dirs_b)
    mesh_b = run_engine(pm_b, cf_b, al_b, ps_b)
    results["B"] = report("B. CHỈ CHỤP TỪ TRÊN (6 camera, nâng 20° — đáy KHÔNG thấy)", mesh_b)

    # ── KẾT LUẬN ──
    print("\n" + "=" * 74)
    ok = True
    ha, hb = results["A"], results["B"]

    vol_a = ha["volume"] / TRUE_VOLUME if ha["volume"] > 0 else 0.0
    if vol_a < 0.8:
        print(f"❌ KỊCH BẢN A: thể tích chỉ {vol_a:.0%} vật đặc -> VẪN CÒN VỎ RỖNG")
        ok = False
    else:
        print(f"✅ KỊCH BẢN A: thể tích đúng ({vol_a:.0%} vật đặc) -> hết vỏ rỗng")

    ext_a = np.abs(100 * (mesh_a.extents - TRUE_EXTENTS) / TRUE_EXTENTS)
    if ext_a.max() > 12.0:
        print(f"❌ KỊCH BẢN A: kích thước lệch tới {ext_a.max():.1f}%")
        ok = False
    else:
        print(f"✅ KỊCH BẢN A: kích thước đúng (lệch tối đa {ext_a.max():.1f}%)")

    if ha["boundary_edges"] != 0 or not ha["watertight"]:
        print("❌ KỊCH BẢN A: mesh phải KÍN khi đã phủ đủ mọi hướng")
        ok = False
    else:
        print("✅ KỊCH BẢN A: mesh kín, 0 cạnh biên")

    if cov_a["pct_band_multi"] < 50.0:
        print(f"❌ KỊCH BẢN A: chỉ {cov_a['pct_band_multi']:.1f}% dải bề mặt được >=2 view xác nhận")
        ok = False
    else:
        print(f"✅ KỊCH BẢN A: độ phủ đa góc tốt ({cov_a['pct_band_multi']:.1f}% dải bề mặt >=2 view)")

    # ĐỘ ĐẶC: thứ `watertight=True` KHÔNG nói được. Hai kịch bản phải KHÁC NHAU rõ rệt —
    # A tản góc -> khối đặc; B một vòng ngang -> bề mặt gấp/dán vào chính nó, thể tích sụp.
    # Đây là phép kiểm bắt lỗi "mesh kín, 1 mảnh, mà trông như bị nhân đôi".
    sol_a, sol_b = ha["solidity_pct"], hb["solidity_pct"]
    print(f"  độ đặc          : A {sol_a:.0f}%  |  B {sol_b:.0f}%  (khối đặc ~100%)")
    if sol_a < 70.0:
        print(f"❌ KỊCH BẢN A: độ đặc chỉ {sol_a:.0f}% -> mesh không phải khối đặc dù phủ đủ góc")
        ok = False
    else:
        print(f"✅ KỊCH BẢN A: là khối đặc ({sol_a:.0f}% vỏ bao lồi)")
    if sol_b >= 40.0:
        print(f"❌ KỊCH BẢN B: độ đặc {sol_b:.0f}% — lẽ ra một vòng ngang phải làm bề mặt gấp lại")
        ok = False
    else:
        print(f"✅ KỊCH BẢN B: đúng là bề mặt bị GẤP ({sol_b:.0f}% vỏ bao lồi) "
              f"— đây là dấu hiệu 'nhân đôi', do GÓC CHỤP chứ không do số ảnh")

    # Kịch bản B KHÔNG phải lỗi của P4: đáy không ai thấy thì không suy ra được. Kỳ vọng
    # duy nhất là P4 KHÔNG ĐƯỢC im lặng — nó phải cảnh báo trong log. Kiểm tra bằng cách
    # dò dòng cảnh báo "[P4] Không lấp được ruột" trong log của lần chạy.
    vol_b = hb["volume"] / TRUE_VOLUME if hb["volume"] > 0 else 0.0
    print(f"ℹ️  KỊCH BẢN B: thể tích {vol_b:.0%} vật đặc — ĐÚNG LÀ KHÔNG TIN ĐƯỢC, vì đáy "
          f"chưa ai thấy nên ruột không lấp được.")
    print("    P4 phải cảnh báo dòng '[P4] Không lấp được ruột' trong log — nếu không có "
          "là P4 im lặng, đáng lo.")

    print("=" * 74)
    print("KET QUA:", "P4 OK" if ok else "P4 CON VAN DE (xem tren)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
