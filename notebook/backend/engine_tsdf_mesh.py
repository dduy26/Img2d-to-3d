"""
engine_tsdf_mesh.py — P4: True Space Carving + TSDF Fusion + Mesh
==================================================================
Chuc nang:
  1. True Silhouette Space Carving (Visual Hull):
       Chieu tia 3D qua K' -> loai bo voxel nam ngoai Alpha Mask.
  2. Ray-TSDF Volumetric Fusion:
       Tich luy gia tri TSDF (Truncated Signed Distance Function) tren luoi voxel
       cho moi depth map va view.
  3. Marching Cubes:
       Trich xuat be mat dang Isosurface (isovalue = 0) tu truong TSDF.
  4. Taubin Smoothing:
       Lam muot be mat 2 buoc lien tiep (lambda, mu) de giu the tich.
  5. Quadric Decimation:
       Giam so mat xuong ~35,000 de tang toc unwrap UV (P5).

Rang buoc phan cung:
  - CPU: i5-12450HX, RAM 16GB.
  - Kich thuoc luoi voxel toi da: 128^3 (cho RTX 3050 6GB VRAM an toan).
  - Toi uu: Su dung numpy vectorized, tranh vong lap Python thuon.

Ly thuyet:
  - TSDF Fusion: Curless & Levoy, 1996.
  - Space Carving (Visual Hull): Laurentini, 1994.
  - Taubin Smoothing: Taubin, 1995.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .utils_3d import (
    CameraIntrinsics,
    MeshHealth,
    check_mesh_health,
    get_orthographic_camera_poses,
    compute_normals,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VOXEL_RES  = 96     # Do phan giai luoi voxel (96^3 ~ 884K voxels)
TSDF_TRUNCATION    = 0.04   # Truncation distance (don vi: fraction of voxel grid)
MIN_WEIGHT         = 2      # Trong so toi thieu de voxel duoc tin cay
TARGET_FACE_COUNT  = 35_000 # So mat muc tieu sau Quadric Decimation
TAUBIN_LAMBDA      = 0.5    # Buoc lam muot dau (positive)
TAUBIN_MU          = -0.53  # Buoc lam muot thu hai (negative, |mu| > lambda)
TAUBIN_ITERS       = 10     # So vong lap Taubin

# ---------------------------------------------------------------------------
# 1. VOXEL GRID — Khoi toa do voxel
# ---------------------------------------------------------------------------

def create_voxel_grid(
    resolution: int = DEFAULT_VOXEL_RES,
    extent: float = 1.0,
) -> tuple:
    """Tao luoi voxel deu deu trong hop don vi [-extent, +extent]^3.

    Args:
        resolution: So voxel moi chieu.
        extent:     Ban kinh hop voxel (don vi tuong doi).

    Returns:
        (voxel_coords, voxel_size)
        - voxel_coords: (res, res, res, 3) float32 toa do trung tam voxel.
        - voxel_size:   float — canh mot voxel.
    """
    lin = np.linspace(-extent, extent, resolution, dtype=np.float32)
    xs, ys, zs = np.meshgrid(lin, lin, lin, indexing='ij')
    voxel_coords = np.stack([xs, ys, zs], axis=-1)   # (R, R, R, 3)
    voxel_size   = float(2.0 * extent / (resolution - 1))
    return voxel_coords, voxel_size


# ---------------------------------------------------------------------------
# 2. TRUE SPACE CARVING (VISUAL HULL)
# ---------------------------------------------------------------------------

def space_carving(
    voxel_coords: np.ndarray,
    alpha_masks:  list,
    K_primes:     list,
    Rs:           list,
    ts:           list,
    canvas_size:  int = 512,
) -> np.ndarray:
    """Goi voxel bang Visual Hull Space Carving.

    Thuat toan:
      For each voxel X:
        For each view i:
          Chieu X vao anh i: uv = pi(K'_i, R_i, t_i, X)
          Neu uv nam trong canvas:
            Neu alpha_mask_i[v, u] < 0.5: danh dau voxel = voi (loai bo)
      Giu lai voxel chua bi loai boi bat ky view nao.

    Args:
        voxel_coords: (R, R, R, 3) float32.
        alpha_masks:  Danh sach N mang (H, W) float32 [0, 1].
        K_primes:     Danh sach N CameraIntrinsics da bu tru.
        Rs:           Danh sach N ma tran quay (3, 3).
        ts:           Danh sach N vector tinh tien (3,).
        canvas_size:  Kich thuoc canvas (mac dinh 512).

    Returns:
        carved_mask: (R, R, R) bool — True = con lai (foreground voxel).
    """
    R_res = voxel_coords.shape[0]
    carved_mask = np.ones((R_res, R_res, R_res), dtype=bool)

    # Flatten voxels de vectorize
    pts_flat = voxel_coords.reshape(-1, 3).T   # (3, N_vox)

    for alpha, K, Rm, t in zip(alpha_masks, K_primes, Rs, ts):
        # Chuyen sang he toa do camera
        Xc = Rm @ pts_flat + t[:, None]          # (3, N_vox)
        depth = Xc[2]                              # (N_vox,)
        valid_depth = depth > 0

        Km = K.as_matrix()
        # Chieu phoi canh
        uvh = Km @ Xc                              # (3, N_vox)
        u = uvh[0] / (uvh[2] + 1e-9)
        v = uvh[1] / (uvh[2] + 1e-9)

        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)

        in_canvas = (
            valid_depth &
            (u_int >= 0) & (u_int < canvas_size) &
            (v_int >= 0) & (v_int < canvas_size)
        )

        # Lay gia tri alpha mask tai toa do chieu
        alpha_vals = np.zeros(pts_flat.shape[1], dtype=np.float32)
        valid_idx  = np.where(in_canvas)[0]
        if len(valid_idx) > 0:
            u_v = u_int[valid_idx]
            v_v = v_int[valid_idx]
            alpha_vals[valid_idx] = alpha[v_v, u_v]

        # Voxel chieu ra ngoai vat the (alpha < 0.5) va trong canvas -> loai bo
        outside = in_canvas & (alpha_vals < 0.5)
        carved_flat = carved_mask.reshape(-1)
        carved_flat[outside] = False
        carved_mask = carved_flat.reshape(R_res, R_res, R_res)

    return carved_mask


# ---------------------------------------------------------------------------
# 3. TSDF FUSION
# ---------------------------------------------------------------------------

def tsdf_fusion(
    voxel_coords:    np.ndarray,
    depth_maps:      list,
    valid_masks:     list,
    K_primes:        list,
    Rs:              list,
    ts:              list,
    voxel_size:      float,
    trunc_factor:    float = TSDF_TRUNCATION,
    carved_mask:     Optional[np.ndarray] = None,
    canvas_size:     int = 512,
) -> tuple:
    """Tich luy truong TSDF tren luoi voxel tu nhieu depth map.

    Thuat toan (Curless & Levoy 1996):
      For each voxel X, each view i:
        depth_obs = depth_map_i[project(X, K'_i, R_i, t_i)]
        sdf_i     = depth_obs - depth_camera(X)    # Signed Distance
        tsdf_i    = clip(sdf_i / trunc, -1, 1)
        Tich luy: tsdf += w * tsdf_i; weight += w

    Args:
        voxel_coords: (R, R, R, 3) float32.
        depth_maps:   Danh sach N (H, W) float32.
        valid_masks:  Danh sach N (H, W) float32.
        K_primes:     Danh sach N CameraIntrinsics.
        Rs, ts:       Danh sach N poses.
        voxel_size:   Canh mot voxel.
        trunc_factor: He so truncation (tinh theo voxel_size * resolution).
        carved_mask:  (R, R, R) bool tu Space Carving (co the None).
        canvas_size:  Kich thuoc canvas.

    Returns:
        (tsdf_vol, weight_vol)
        - tsdf_vol:   (R, R, R) float32 — gia tri TSDF.
        - weight_vol: (R, R, R) float32 — trong so tich luy.
    """
    R_res = voxel_coords.shape[0]
    tsdf_vol   = np.zeros((R_res, R_res, R_res), dtype=np.float32)
    weight_vol = np.zeros((R_res, R_res, R_res), dtype=np.float32)

    trunc_dist = trunc_factor * R_res * voxel_size

    pts_flat = voxel_coords.reshape(-1, 3).T   # (3, N_vox)
    N_vox = pts_flat.shape[1]

    for depth_map, valid_mask, K, Rm, t in zip(depth_maps, valid_masks, K_primes, Rs, ts):
        h, w = depth_map.shape

        # Chuyen sang he toa do camera
        Xc    = Rm @ pts_flat + t[:, None]
        depth_cam = Xc[2]                          # Chieu sau thuc te trong camera
        valid_depth = depth_cam > 0

        # Chieu phoi canh
        Km  = K.as_matrix()
        uvh = Km @ Xc
        u   = uvh[0] / (uvh[2] + 1e-9)
        v   = uvh[1] / (uvh[2] + 1e-9)

        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)

        in_canvas = (
            valid_depth &
            (u_int >= 0) & (u_int < w) &
            (v_int >= 0) & (v_int < h)
        )

        # Lay gia tri depth va valid tai toa do chieu
        depth_obs = np.zeros(N_vox, dtype=np.float32)
        valid_px  = np.zeros(N_vox, dtype=bool)
        idx_valid = np.where(in_canvas)[0]
        if len(idx_valid) > 0:
            depth_obs[idx_valid] = depth_map[v_int[idx_valid], u_int[idx_valid]]
            valid_px[idx_valid]  = valid_mask[v_int[idx_valid], u_int[idx_valid]] > 0.5

        # TSDF calculation
        # depth_obs la [0,1] tuong doi -> scale theo depth_cam range
        depth_obs_abs = depth_obs * depth_cam.max()
        sdf    = depth_obs_abs - depth_cam
        tsdf_v = np.clip(sdf / (trunc_dist + 1e-9), -1.0, 1.0)

        # Chi tich luy cho voxel hop le
        active = in_canvas & valid_px
        tsdf_flat   = tsdf_vol.reshape(-1)
        weight_flat = weight_vol.reshape(-1)
        w_inc       = active.astype(np.float32)
        tsdf_flat   += tsdf_v * w_inc
        weight_flat += w_inc
        tsdf_vol   = tsdf_flat.reshape(R_res, R_res, R_res)
        weight_vol = weight_flat.reshape(R_res, R_res, R_res)

    # Trung binh trong so
    safe_w = np.where(weight_vol > 0, weight_vol, 1.0)
    tsdf_vol = tsdf_vol / safe_w

    # Ap dung Space Carving mask (voxel bi khac = set tsdf = +1 -> ben ngoai)
    if carved_mask is not None:
        tsdf_vol[~carved_mask] = 1.0
        weight_vol[~carved_mask] = 0.0

    return tsdf_vol, weight_vol


# ---------------------------------------------------------------------------
# 4. MARCHING CUBES — Trich xuat be mat
# ---------------------------------------------------------------------------

def extract_mesh_marching_cubes(
    tsdf_vol:   np.ndarray,
    weight_vol: np.ndarray,
    voxel_size: float,
    extent:     float = 1.0,
    min_weight: int   = MIN_WEIGHT,
) -> tuple:
    """Trich xuat be mat tu truong TSDF bang Marching Cubes.

    Args:
        tsdf_vol:   (R, R, R) float32.
        weight_vol: (R, R, R) float32.
        voxel_size: Canh voxel.
        extent:     Ban kinh hop voxel.
        min_weight: Trong so toi thieu de voxel co gia tri hop le.

    Returns:
        (vertices, faces) float32 / int32, hoac (None, None) neu khong co mat.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError("Can cai dat scikit-image: pip install scikit-image")

    # Mask voxel co trong so qua thap -> set tsdf = 1 (loai bo)
    low_weight_mask = weight_vol < min_weight
    tsdf_clean = tsdf_vol.copy()
    tsdf_clean[low_weight_mask] = 1.0

    # Chon level phu hop: uu tien 0.0, fallback ve median cua vung co weight
    d_min, d_max = tsdf_clean.min(), tsdf_clean.max()
    if d_min < 0.0 < d_max:
        level = 0.0
    else:
        # TSDF khong cat qua 0 (e.g., depth phang) -> dung median lam isosurface
        weighted_vals = tsdf_clean[weight_vol >= min_weight]
        if len(weighted_vals) > 0:
            level = float(np.median(weighted_vals))
        else:
            level = float((d_min + d_max) / 2.0)

    # Pad 1 voxel voi gia tri ngoai bien (> level) de dam bao luoi kin nuoc 100%
    pad_val = max(1.0, level + 0.5)
    padded_tsdf = np.pad(tsdf_clean, pad_width=1, mode="constant", constant_values=pad_val)

    try:
        try:
            verts, fcs, normals, values = marching_cubes(
                padded_tsdf,
                level=level,
                spacing=(voxel_size, voxel_size, voxel_size),
                method="lorensen",
            )
        except (TypeError, ValueError):
            # Fallback ve method mac dinh neu scikit-image cu khong co lorensen
            verts, fcs, normals, values = marching_cubes(
                padded_tsdf,
                level=level,
                spacing=(voxel_size, voxel_size, voxel_size),
            )
    except Exception as e:
        warnings.warn(f"Marching Cubes that bai: {e}", RuntimeWarning)
        return None, None

    if len(verts) == 0 or len(fcs) == 0:
        warnings.warn("Marching Cubes tra ve mesh rong.", RuntimeWarning)
        return None, None

    # Bu tru toa do do da pad 1 voxel o moi chieu
    verts = verts - voxel_size

    # Chuyen doi toa do voxel -> the gioi (dich ve trung tam [-extent, extent])
    origin = np.array([-extent, -extent, -extent], dtype=np.float32)
    verts  = verts.astype(np.float32) + origin
    fcs    = fcs.astype(np.int32)

    # Loc loai bo cac dao vun roi rac (chi giu khoi lien thong chinh lon nhat)
    try:
        import trimesh
        tm = trimesh.Trimesh(verts, fcs, process=False)
        comps = tm.split(only_watertight=False)
        if len(comps) > 1:
            main_comp = max(comps, key=lambda c: len(c.faces))
            verts = main_comp.vertices.astype(np.float32)
            fcs   = main_comp.faces.astype(np.int32)
    except Exception:
        pass

    return verts, fcs



# ---------------------------------------------------------------------------
# 5. TAUBIN SMOOTHING — Lam muot giu the tich
# ---------------------------------------------------------------------------

def taubin_smooth(
    vertices: np.ndarray,
    faces:    np.ndarray,
    lam:      float = TAUBIN_LAMBDA,
    mu:       float = TAUBIN_MU,
    iters:    int   = TAUBIN_ITERS,
) -> np.ndarray:
    """Lam muot be mat bang thuat toan Taubin (1995).

    Buoc 1: dich chuyen dinh theo huong Laplacian * lambda (tren mat phang).
    Buoc 2: dich chuyen theo huong nguoc lai * |mu| (bu khoi the tich).
    Lap lai 'iters' lan.

    Args:
        vertices: (V, 3) float32.
        faces:    (F, 3) int32.
        lam, mu:  He so lam muot (lam > 0, mu < 0, |mu| > lam).
        iters:    So vong lap.

    Returns:
        vertices_smoothed: (V, 3) float32.
    """
    verts = vertices.astype(np.float64)
    V = len(verts)

    # Xay dung danh sach hang xom (adjacency)
    adj: list = [set() for _ in range(V)]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])

    def laplacian_step(v: np.ndarray, factor: float) -> np.ndarray:
        new_v = v.copy()
        for i in range(V):
            nb = list(adj[i])
            if nb:
                centroid = v[nb].mean(axis=0)
                new_v[i] = v[i] + factor * (centroid - v[i])
        return new_v

    for _ in range(iters):
        verts = laplacian_step(verts, lam)
        verts = laplacian_step(verts, mu)

    return verts.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. QUADRIC DECIMATION — Giam so mat
# ---------------------------------------------------------------------------

def quadric_decimate(
    vertices: np.ndarray,
    faces:    np.ndarray,
    target_faces: int = TARGET_FACE_COUNT,
) -> tuple:
    """Giam so mat bang Quadric Error Metrics Decimation.

    Thu tu uu tien:
      1. open3d SimplifymeshQuadricDecimation.
      2. pymeshlab QEM.
      3. Fallback: giu nguyen neu ca hai that bai.

    Args:
        vertices:     (V, 3) float32.
        faces:        (F, 3) int32.
        target_faces: So mat muc tieu.

    Returns:
        (verts_dec, faces_dec) sau decimation.
    """
    F = len(faces)
    if F <= target_faces:
        return vertices, faces

    # Method 1: open3d
    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh_dec = mesh.simplify_quadric_decimation(target_faces)
        verts_dec = np.asarray(mesh_dec.vertices, dtype=np.float32)
        faces_dec = np.asarray(mesh_dec.triangles, dtype=np.int32)
        return verts_dec, faces_dec
    except Exception:
        pass

    # Method 2: pymeshlab
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        m  = pymeshlab.Mesh(
            vertex_matrix=vertices.astype(np.float64),
            face_matrix=faces.astype(np.int32),
        )
        ms.add_mesh(m)
        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=target_faces,
        )
        out = ms.current_mesh()
        return out.vertex_matrix().astype(np.float32), out.face_matrix().astype(np.int32)
    except Exception:
        pass

    warnings.warn(
        "Quadric Decimation khong kha dung (can open3d hoac pymeshlab). "
        "Giu nguyen mesh.",
        RuntimeWarning,
    )
    return vertices, faces


# ---------------------------------------------------------------------------
# 7. PIPELINE P4 — ENTRY POINT
# ---------------------------------------------------------------------------

def reconstruct_mesh(
    depth_views:    list,
    voxel_resolution: int   = DEFAULT_VOXEL_RES,
    extent:           float = 1.0,
    use_carving:      bool  = True,
    smooth:           bool  = True,
    decimate:         bool  = True,
) -> tuple:
    """Chay toan bo pipeline P4: Space Carving + TSDF Fusion + Mesh.

    Args:
        depth_views:      Ket qua tu estimate_depth_pipeline() (da qua Quality Gate).
        voxel_resolution: Do phan giai luoi voxel.
        extent:           Ban kinh hop voxel.
        use_carving:      Co dung Space Carving hay khong.
        smooth:           Co dung Taubin Smoothing hay khong.
        decimate:         Co dung Quadric Decimation hay khong.

    Returns:
        (vertices, faces, mesh_health)
        - vertices:    (V, 3) float32.
        - faces:       (F, 3) int32.
        - mesh_health: MeshHealth.

    Raises:
        RuntimeError: Neu mesh dau ra rong.
    """
    N = len(depth_views)
    if N == 0:
        raise RuntimeError("Khong co view nao de reconstruct.")

    # Trich du lieu tu views
    alphas     = [v["alpha_mask"] for v in depth_views]
    depths     = [v["depth_map"]  for v in depth_views]
    valids     = [v["valid_mask"] for v in depth_views]
    K_primes   = [v["K_prime"]    for v in depth_views]
    canvas_sz  = depth_views[0]["canvas_rgb"].shape[1]

    # Sinh camera poses (su dung 4-view ortho convention)
    poses = get_orthographic_camera_poses(n_views=N)
    Rs = [p[0] for p in poses]
    ts = [p[1] for p in poses]

    # Tao luoi voxel
    voxel_coords, voxel_size = create_voxel_grid(voxel_resolution, extent)

    # Buoc 1: Space Carving
    carved_mask = None
    if use_carving:
        carved_mask = space_carving(
            voxel_coords, alphas, K_primes, Rs, ts, canvas_size=canvas_sz
        )

    # Buoc 2: TSDF Fusion
    tsdf_vol, weight_vol = tsdf_fusion(
        voxel_coords, depths, valids, K_primes, Rs, ts,
        voxel_size, carved_mask=carved_mask, canvas_size=canvas_sz
    )

    # Buoc 3: Marching Cubes
    vertices, faces = extract_mesh_marching_cubes(
        tsdf_vol, weight_vol, voxel_size, extent
    )
    if vertices is None:
        raise RuntimeError("Marching Cubes tra ve mesh rong — kiem tra du lieu dau vao.")

    # Buoc 4: Taubin Smoothing
    if smooth:
        vertices = taubin_smooth(vertices, faces)

    # Buoc 5: Quadric Decimation
    if decimate:
        vertices, faces = quadric_decimate(vertices, faces)

    # Kiem dinh chat luong
    health = check_mesh_health(vertices, faces)

    return vertices, faces, health


def mesh_health(mesh_or_verts, faces=None) -> dict:
    """Helper tra ve dict mesh health tu Trimesh object hoac (vertices, faces)."""
    if hasattr(mesh_or_verts, "vertices") and hasattr(mesh_or_verts, "faces"):
        v = mesh_or_verts.vertices
        f = mesh_or_verts.faces
        vol = float(mesh_or_verts.volume) if getattr(mesh_or_verts, "is_watertight", False) else 0.0
    else:
        v = mesh_or_verts
        f = faces
        vol = 0.0
    h = check_mesh_health(v, f)
    return {
        "watertight": h.is_watertight,
        "boundary_edges": h.boundary_edges,
        "components": h.components,
        "faces": h.face_count,
        "vertices": h.vertex_count,
        "volume": h.volume_m3 if h.volume_m3 > 0 else vol,
        "euler_number": h.euler_number,
    }
