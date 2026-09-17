"""
utils_3d.py — Tiện Ích Hình Học 3D Cốt Lõi
============================================
Cung cấp:
  - Tính toán ma trận Camera Intrinsics bù trừ K' sau khi Uniform Scale + Letterbox
  - Chiếu tia phối cảnh điểm 3D lên canvas chuẩn (với K' đã bù trừ)
  - Kiểm định tính kín nước Watertight của mesh (boundary_edges, components)
  - Xuất file .glb (GLTF 2.0 Binary) chuẩn hóa

Lý thuyết tham chiếu: docs/lythuyet.md §1 — Bản Chất Quang Học & Hình Học Epipolar
"""

from __future__ import annotations

import math
import struct
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# 1. CAMERA INTRINSICS — K và K' (bù trừ sau Uniform Scale)
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """Ma trận nội thông camera 3x3 dạng tường minh."""
    fx: float
    fy: float
    cx: float
    cy: float
    width:  int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float = 60.0) -> "CameraIntrinsics":
        """Xây dựng K từ góc nhìn (FoV) ngang, tâm quang học ở tâm ảnh."""
        fov_rad = math.radians(fov_deg)
        fx = fy = (width / 2.0) / math.tan(fov_rad / 2.0)
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    @classmethod
    def default_ortho(cls, width: int, height: int) -> "CameraIntrinsics":
        """K mặc định cho ảnh render trực giao (Objaverse-style 4 góc 90°)."""
        f = max(width, height) / 2.0
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    def as_matrix(self) -> np.ndarray:
        """Trả về ma trận K dạng numpy (3, 3) float64."""
        return np.array([
            [self.fx,      0.0, self.cx],
            [    0.0, self.fy,  self.cy],
            [    0.0,     0.0,      1.0],
        ], dtype=np.float64)


def compensate_intrinsics(
    K: CameraIntrinsics,
    orig_w: int,
    orig_h: int,
    canvas_size: int = 512,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> tuple:
    """Tính K' sau khi Uniform Scale + đặt ảnh vào canvas vuong.

    Cong thuc (lythuyet.md §1.3):
        s   = canvas_size / max(orig_h, orig_w)
        fx' = s * fx
        cx' = s * cx + offset_x
        cy' = s * cy + offset_y

    Returns:
        (K_prime, scale_factor)
    """
    s = canvas_size / max(orig_h, orig_w)
    k_prime = CameraIntrinsics(
        fx=s * K.fx,
        fy=s * K.fy,
        cx=s * K.cx + offset_x,
        cy=s * K.cy + offset_y,
        width=canvas_size,
        height=canvas_size,
    )
    return k_prime, float(s)


def project_point(
    point_3d: np.ndarray,
    K: CameraIntrinsics,
    R: np.ndarray,
    t: np.ndarray,
) -> Optional[np.ndarray]:
    """Chieu tia phoi canh: diem 3D -> toa do pixel tren canvas.

    Cong thuc:  [u, v, 1]^T = K' * (R * X + t) / depth

    Returns:
        (u, v) pixel neu diem nam phia truoc camera (depth > 0), else None.
    """
    X    = np.asarray(point_3d, dtype=np.float64)
    Xc   = R @ X + t
    depth = Xc[2]
    if depth <= 0:
        return None
    uv = K.as_matrix() @ Xc
    return uv[:2] / uv[2]


def project_points_batch(
    points: np.ndarray,
    K: CameraIntrinsics,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple:
    """Chieu hang loat diem 3D (N, 3) len anh.

    Returns:
        uvs:   (N, 2) toa do pixel.
        valid: (N,) bool — True neu depth > 0.
    """
    pts   = np.asarray(points, dtype=np.float64)
    Xc    = (R @ pts.T).T + t
    depth = Xc[:, 2]
    valid = depth > 0
    safe_depth = np.where(valid, depth, 1.0)
    Km    = K.as_matrix()
    uvs_h = (Km @ (Xc / safe_depth[:, None]).T).T
    return uvs_h[:, :2], valid


# ---------------------------------------------------------------------------
# 2. CAMERA POSES — Convention Objaverse (4-view ortho)
# ---------------------------------------------------------------------------

def get_orthographic_camera_poses(
    n_views: int = 4,
    radius: float = 2.5,
    elevation_deg: float = 20.0,
) -> list:
    """Sinh poses camera phan phoi deu quanh vat the.

    Returns:
        Danh sach (R, t) cho tung view.
    """
    poses = []
    elev_rad = math.radians(elevation_deg)
    for i in range(n_views):
        azim_rad = 2 * math.pi * i / n_views
        cx_ = radius * math.cos(elev_rad) * math.cos(azim_rad)
        cy_ = radius * math.sin(elev_rad)
        cz_ = radius * math.cos(elev_rad) * math.sin(azim_rad)
        cam_pos = np.array([cx_, cy_, cz_], dtype=np.float64)

        z_axis = -cam_pos / (np.linalg.norm(cam_pos) + 1e-9)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(world_up, z_axis)
        x_axis /= (np.linalg.norm(x_axis) + 1e-9)
        y_axis = np.cross(z_axis, x_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        t = -R @ cam_pos
        poses.append((R.copy(), t.copy()))
    return poses


# ---------------------------------------------------------------------------
# 3. MESH HEALTH — Kiem dinh tinh kin nuoc Watertight
# ---------------------------------------------------------------------------

@dataclass
class MeshHealth:
    """Ket qua kiem dinh chat luong hinh hoc cua mesh."""
    vertex_count:    int
    face_count:      int
    boundary_edges:  int   # == 0 -> kin nuoc
    components:      int   # == 1 -> mot khoi lien thong
    is_watertight:   bool
    euler_number:    int   # == 2 cho sphere-topology watertight
    volume_m3:       float = 0.0
    notes:           list = field(default_factory=list)


def check_mesh_health(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> MeshHealth:
    """Kiem dinh chat luong hinh hoc cua mesh (vertices, faces).

    Thuat toan:
      - Dem boundary edges: canh chi thuoc 1 mat tam giac.
      - Dem components: BFS tren do thi mat-mat.
      - Tinh Euler characteristic: V - E + F.
      - Tinh the tich (Divergence Theorem).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    fcs   = np.asarray(faces,   dtype=np.int64)
    V, F  = len(verts), len(fcs)
    notes = []

    # --- Dem boundary edges --------------------------------------------------
    edge_count: dict = {}
    for tri in fcs:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            e = (min(a, b), max(a, b))
            edge_count[e] = edge_count.get(e, 0) + 1
    E = len(edge_count)
    boundary_edges = sum(1 for cnt in edge_count.values() if cnt == 1)

    # --- Dem components (BFS tren face-adjacency) ----------------------------
    edge_to_faces: dict = {}
    for fi, tri in enumerate(fcs):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            e = (min(a, b), max(a, b))
            if e not in edge_to_faces:
                edge_to_faces[e] = []
            edge_to_faces[e].append(fi)

    adj: dict = {i: set() for i in range(F)}
    for fl in edge_to_faces.values():
        if len(fl) == 2:
            adj[fl[0]].add(fl[1])
            adj[fl[1]].add(fl[0])

    visited = [False] * F
    components = 0
    for start in range(F):
        if not visited[start]:
            components += 1
            stack = [start]
            while stack:
                fi = stack.pop()
                if visited[fi]:
                    continue
                visited[fi] = True
                for nb in adj[fi]:
                    if not visited[nb]:
                        stack.append(nb)

    # --- Euler characteristic ------------------------------------------------
    euler = V - E + F

    # --- The tich co dau (Divergence Theorem) --------------------------------
    v0 = verts[fcs[:, 0]]
    v1 = verts[fcs[:, 1]]
    v2 = verts[fcs[:, 2]]
    volume = float(np.abs(np.sum(v0 * np.cross(v1, v2)) / 6.0))

    is_watertight = (boundary_edges == 0) and (components == 1)
    if boundary_edges > 0:
        notes.append(f"WARNING: {boundary_edges} canh bien ho (boundary edges).")
    if components > 1:
        notes.append(f"WARNING: Mesh co {components} khoi lien thong roi nhau.")
    if euler != 2 and is_watertight:
        notes.append(f"INFO: Euler characteristic = {euler} (!=2, co the co lo topo).")
    if is_watertight and not notes:
        notes.append("OK: Mesh kin nuoc hoan toan (Watertight).")

    return MeshHealth(
        vertex_count=V,
        face_count=F,
        boundary_edges=boundary_edges,
        components=components,
        is_watertight=is_watertight,
        euler_number=euler,
        volume_m3=volume,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 4. GLB EXPORT — GLTF 2.0 Binary
# ---------------------------------------------------------------------------

def export_glb(
    vertices:  np.ndarray,
    faces:     np.ndarray,
    colors:    Optional[np.ndarray] = None,
    normals:   Optional[np.ndarray] = None,
    out_path:  str = "output.glb",
) -> str:
    """Xuat mesh sang dinh dang GLTF 2.0 Binary (.glb).

    Args:
        vertices: (V, 3) float32.
        faces:    (F, 3) uint32.
        colors:   (V, 3) uint8 hoac float32 (tuy chon).
        normals:  (V, 3) float32 (tuy chon).
        out_path: Duong dan file dau ra.

    Returns:
        Duong dan file da xuat.
    """
    verts  = np.asarray(vertices, dtype=np.float32)
    fcs    = np.asarray(faces,    dtype=np.uint32)
    V = len(verts)
    F = len(fcs)

    buf_parts:   list = []
    accessors:   list = []
    buffer_views: list = []
    byte_offset = 0

    def add_buffer(data_bytes: bytes, target=None) -> int:
        nonlocal byte_offset
        pad = (4 - len(data_bytes) % 4) % 4
        data_bytes = data_bytes + b"\x00" * pad
        bv_idx = len(buffer_views)
        bv = {
            "buffer":     0,
            "byteOffset": byte_offset,
            "byteLength": len(data_bytes),
        }
        if target is not None:
            bv["target"] = target
        buffer_views.append(bv)
        buf_parts.append(data_bytes)
        byte_offset += len(data_bytes)
        return bv_idx

    ARRAY_BUFFER         = 34962
    ELEMENT_ARRAY_BUFFER = 34963

    # --- POSITION ---
    bv_verts = add_buffer(verts.tobytes(), ARRAY_BUFFER)
    acc_pos = len(accessors)
    accessors.append({
        "bufferView":    bv_verts,
        "componentType": 5126,
        "count":         V,
        "type":          "VEC3",
        "min":           verts.min(axis=0).tolist(),
        "max":           verts.max(axis=0).tolist(),
    })

    # --- INDICES ---
    bv_faces = add_buffer(fcs.tobytes(), ELEMENT_ARRAY_BUFFER)
    acc_idx = len(accessors)
    accessors.append({
        "bufferView":    bv_faces,
        "componentType": 5125,
        "count":         F * 3,
        "type":          "SCALAR",
    })

    primitive = {
        "attributes": {"POSITION": acc_pos},
        "indices":    acc_idx,
        "mode":       4,
    }

    # --- NORMAL (optional) ---
    if normals is not None:
        norms = np.asarray(normals, dtype=np.float32)
        bv_nrm = add_buffer(norms.tobytes(), ARRAY_BUFFER)
        acc_nrm = len(accessors)
        accessors.append({
            "bufferView":    bv_nrm,
            "componentType": 5126,
            "count":         V,
            "type":          "VEC3",
        })
        primitive["attributes"]["NORMAL"] = acc_nrm

    # --- COLOR_0 (optional) ---
    if colors is not None:
        cols = np.asarray(colors, dtype=np.float32)
        if cols.max() > 1.5:
            cols = cols / 255.0
        cols = np.clip(cols, 0.0, 1.0).astype(np.float32)
        bv_col = add_buffer(cols.tobytes(), ARRAY_BUFFER)
        acc_col = len(accessors)
        accessors.append({
            "bufferView":    bv_col,
            "componentType": 5126,
            "count":         V,
            "type":          "VEC3",
        })
        primitive["attributes"]["COLOR_0"] = acc_col

    # --- GLTF JSON ---
    total_bin = b"".join(buf_parts)
    gltf = {
        "asset": {"version": "2.0", "generator": "ImgToModel-Pipeline-v2"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [primitive]}],
        "accessors":   accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(total_bin)}],
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad_json = (4 - len(json_bytes) % 4) % 4
    json_bytes = json_bytes + b" " * pad_json

    # --- GLB Container ---
    MAGIC           = 0x46546C67   # "glTF"
    VERSION         = 2
    JSON_CHUNK_TYPE = 0x4E4F534A   # "JSON"
    BIN_CHUNK_TYPE  = 0x004E4942   # "BIN\0"

    json_chunk   = struct.pack("<II", len(json_bytes), JSON_CHUNK_TYPE) + json_bytes
    bin_chunk    = struct.pack("<II", len(total_bin),  BIN_CHUNK_TYPE)  + total_bin
    total_length = 12 + len(json_chunk) + len(bin_chunk)
    header       = struct.pack("<III", MAGIC, VERSION, total_length)

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(json_chunk)
        f.write(bin_chunk)

    return out_path


# ---------------------------------------------------------------------------
# 5. TIEN ICH PHU TRO
# ---------------------------------------------------------------------------

def round_to_16(x: float) -> int:
    """Lam tron x len boi so cua 16 (chuan ViT patch-size).

    Cong thuc: floor((x + 8) / 16) * 16  (lythuyet.md §1.3)
    """
    return int(math.floor((x + 8) / 16)) * 16


def compute_uniform_scale_params(
    orig_w: int,
    orig_h: int,
    target: int = 512,
) -> tuple:
    """Tinh tham so Uniform Scale de dat anh vao canvas vuong target x target.

    Returns:
        (scale, new_w, new_h, offset_x, offset_y)
    """
    s = target / max(orig_w, orig_h)
    new_w    = round_to_16(orig_w * s)
    new_h    = round_to_16(orig_h * s)
    offset_x = (target - new_w) // 2
    offset_y = (target - new_h) // 2
    return float(s), new_w, new_h, offset_x, offset_y


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Tinh phap tuyen dinh theo trung binh co trong so dien tich mat.

    Returns:
        normals: (V, 3) float32 — phap tuyen don vi moi dinh.
    """
    verts   = np.asarray(vertices, dtype=np.float64)
    fcs     = np.asarray(faces,   dtype=np.int64)
    normals = np.zeros_like(verts)
    v0 = verts[fcs[:, 0]]
    v1 = verts[fcs[:, 1]]
    v2 = verts[fcs[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, fcs[:, i], fn)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norms > 1e-9, norms, 1.0)
    return normals.astype(np.float32)
