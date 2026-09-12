"""Shared 3D helpers for the Member 5 texture pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import trimesh


def as_numpy(value: object) -> np.ndarray:
    """Convert NumPy-like or torch-like values to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def as_camera_pose(pose: object) -> np.ndarray:
    """Normalize a camera-to-world pose to a finite 4x4 matrix."""
    matrix = as_numpy(pose).astype(np.float32, copy=False)
    if matrix.shape == (3, 4):
        homogeneous = np.eye(4, dtype=np.float32)
        homogeneous[:3, :4] = matrix
        matrix = homogeneous
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("camera pose must be a finite 3x4 or 4x4 matrix")
    return matrix


def validate_multiview_inputs(
    images_rgb: Sequence[np.ndarray],
    camera_poses: Sequence[object],
    focal_lengths: Sequence[Tuple[float, float]],
) -> None:
    """Validate the P1/P2 data contract consumed by P5."""
    if not images_rgb:
        raise ValueError("at least one RGB image is required")
    if not (len(images_rgb) == len(camera_poses) == len(focal_lengths)):
        raise ValueError("images, camera poses and focal lengths must have equal lengths")

    for index, image in enumerate(images_rgb):
        array = as_numpy(image)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError(f"image {index} must have shape (H, W, 3) and dtype uint8")
        if array.shape[0] < 2 or array.shape[1] < 2:
            raise ValueError(f"image {index} is too small for projection")
        fx, fy = focal_lengths[index]
        if not np.isfinite([fx, fy]).all() or fx <= 0 or fy <= 0:
            raise ValueError(f"focal length {index} must contain positive values")
        as_camera_pose(camera_poses[index])


def project_vertices(
    vertices: np.ndarray,
    camera_pose: object,
    focal_length: Tuple[float, float],
    image_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-space vertices using the P2 camera-to-world convention."""
    pose = as_camera_pose(camera_pose)
    points = as_numpy(vertices).astype(np.float32, copy=False)
    camera_center = pose[:3, 3]
    camera_points = (points - camera_center) @ pose[:3, :3]
    depth = camera_points[:, 2]

    height, width = image_shape[:2]
    fx, fy = focal_length
    cx, cy = width / 2.0, height / 2.0
    safe_depth = np.maximum(depth, 1e-6)
    pixels = np.column_stack(
        (
            fx * camera_points[:, 0] / safe_depth + cx,
            fy * camera_points[:, 1] / safe_depth + cy,
        )
    )
    in_image = (
        (depth > 1e-5)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels, depth, in_image


def visible_vertex_mask(
    pixels: np.ndarray,
    depth: np.ndarray,
    in_image: np.ndarray,
    image_shape: Tuple[int, int, int],
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Approximate visibility with a projected-vertex depth buffer."""
    height, width = image_shape[:2]
    integer_pixels = np.rint(pixels).astype(np.int32)
    visible = np.zeros(len(pixels), dtype=bool)
    valid_indices = np.flatnonzero(in_image)
    if len(valid_indices) == 0:
        return visible

    keys = integer_pixels[valid_indices, 1] * width + integer_pixels[valid_indices, 0]
    nearest_depth = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(nearest_depth, keys, depth[valid_indices])
    visible[valid_indices] = depth[valid_indices] <= nearest_depth[keys] + tolerance
    return visible


def rasterize_depth_buffer(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera_pose: object,
    focal_length: Tuple[float, float],
    image_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Rasterize the nearest mesh-triangle depth for one camera."""
    pixels, depths, _ = project_vertices(vertices, camera_pose, focal_length, image_shape)
    height, width = image_shape[:2]
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

    for face in np.asarray(faces, dtype=np.int64):
        triangle = pixels[face]
        triangle_depth = depths[face]
        if np.any(triangle_depth <= 1e-5):
            continue
        x_min = max(int(np.floor(triangle[:, 0].min())), 0)
        x_max = min(int(np.ceil(triangle[:, 0].max())), width - 1)
        y_min = max(int(np.floor(triangle[:, 1].min())), 0)
        y_max = min(int(np.ceil(triangle[:, 1].max())), height - 1)
        if x_min > x_max or y_min > y_max:
            continue

        grid_x, grid_y = np.meshgrid(
            np.arange(x_min, x_max + 1, dtype=np.float32),
            np.arange(y_min, y_max + 1, dtype=np.float32),
        )
        denominator = (
            (triangle[1, 1] - triangle[2, 1]) * (triangle[0, 0] - triangle[2, 0])
            + (triangle[2, 0] - triangle[1, 0]) * (triangle[0, 1] - triangle[2, 1])
        )
        if abs(float(denominator)) < 1e-8:
            continue
        bary_a = (
            (triangle[1, 1] - triangle[2, 1]) * (grid_x - triangle[2, 0])
            + (triangle[2, 0] - triangle[1, 0]) * (grid_y - triangle[2, 1])
        ) / denominator
        bary_b = (
            (triangle[2, 1] - triangle[0, 1]) * (grid_x - triangle[2, 0])
            + (triangle[0, 0] - triangle[2, 0]) * (grid_y - triangle[2, 1])
        ) / denominator
        inside = (bary_a >= 0) & (bary_b >= 0) & (bary_a + bary_b <= 1)
        interpolated_depth = (
            bary_a * triangle_depth[0]
            + bary_b * triangle_depth[1]
            + (1.0 - bary_a - bary_b) * triangle_depth[2]
        )
        region = depth_buffer[y_min:y_max + 1, x_min:x_max + 1]
        region[inside] = np.minimum(region[inside], interpolated_depth[inside])

    return depth_buffer


def visible_projected_points(
    pixels: np.ndarray,
    depths: np.ndarray,
    depth_buffer: np.ndarray,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Compare projected samples against a triangle-rasterized depth buffer."""
    height, width = depth_buffer.shape
    integer_pixels = np.rint(pixels).astype(np.int32)
    in_image = (
        (depths > 1e-5)
        & (integer_pixels[:, 0] >= 0)
        & (integer_pixels[:, 0] < width)
        & (integer_pixels[:, 1] >= 0)
        & (integer_pixels[:, 1] < height)
    )
    visible = np.zeros(len(pixels), dtype=bool)
    indices = np.flatnonzero(in_image)
    if len(indices):
        nearest = depth_buffer[integer_pixels[indices, 1], integer_pixels[indices, 0]]
        visible[indices] = depths[indices] <= nearest + tolerance
    return visible


def sample_rgb_nearest(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Sample RGB values at projected pixels using nearest-neighbor lookup."""
    height, width = image.shape[:2]
    x = np.clip(np.rint(pixels[:, 0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(pixels[:, 1]).astype(np.int32), 0, height - 1)
    return image[y, x].astype(np.float64)


def export_glb(mesh: trimesh.Trimesh, output_path: str | os.PathLike[str]) -> str:
    """Export one binary GLB and verify that the file exists."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="glb")
    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"GLB export did not create a file: {path}")
    return str(path)