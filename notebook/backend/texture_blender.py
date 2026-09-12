"""Member 5: XAtlas UV, multi-view Base-Color blending and GLB export."""

from __future__ import annotations

import logging
import time
from typing import Sequence, Tuple

import numpy as np
import trimesh
from PIL import Image

from utils_3d import (
    export_glb,
    project_vertices,
    rasterize_depth_buffer,
    sample_rgb_nearest,
    validate_multiview_inputs,
    visible_projected_points,
    visible_vertex_mask,
)

try:
    import xatlas
except ImportError:
    xatlas = None

logger = logging.getLogger(__name__)

DEFAULT_TEXTURE_SIZE = 1024
DEFAULT_ANGLE_GAMMA = 3.0


class TextureBlender:
    """Create a textured GLB from P1/P2 views and the P4 mesh."""

    def __init__(self, texture_size: int = DEFAULT_TEXTURE_SIZE, gamma: float = DEFAULT_ANGLE_GAMMA):
        if texture_size < 16 or texture_size > 4096:
            raise ValueError("texture_size must be between 16 and 4096")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.texture_size = int(texture_size)
        self.gamma = float(gamma)

    def unwrap_uv(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray]:
        """Use XAtlas when available and a deterministic spherical fallback otherwise."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("mesh must contain vertices and triangular faces")
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        if xatlas is not None:
            try:
                atlas = xatlas.Atlas()
                atlas.add_mesh(vertices, faces)
                atlas.generate()
                vertex_mapping, atlas_faces, uvs = atlas.get_mesh(0)
                remapped_vertices = vertices[np.asarray(vertex_mapping, dtype=np.int32)]
                unwrapped = trimesh.Trimesh(
                    vertices=remapped_vertices,
                    faces=np.asarray(atlas_faces, dtype=np.int64),
                    process=False,
                )
                return unwrapped, np.asarray(uvs, dtype=np.float32)
            except Exception as error:
                logger.warning("XAtlas failed; using spherical fallback: %s", error)

        centered = vertices - np.asarray(mesh.centroid, dtype=np.float32)
        theta = np.arctan2(centered[:, 0], centered[:, 2])
        u = (theta + np.pi) / (2.0 * np.pi)
        y_min = float(centered[:, 1].min())
        y_range = max(float(centered[:, 1].max()) - y_min, 1e-6)
        v = (centered[:, 1] - y_min) / y_range
        uvs = np.clip(np.column_stack((u, v)), 0.0, 1.0).astype(np.float32)
        return mesh.copy(), uvs

    def blend_colors_for_vertices(
        self,
        mesh: trimesh.Trimesh,
        images_rgb: Sequence[np.ndarray],
        camera_poses: Sequence[object],
        focal_lengths: Sequence[Tuple[float, float]],
    ) -> np.ndarray:
        """Blend visible samples with the angle weight max(0, n dot v)^gamma."""
        validate_multiview_inputs(images_rgb, camera_poses, focal_lengths)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        accumulated = np.zeros((len(vertices), 3), dtype=np.float64)
        accumulated_weights = np.zeros(len(vertices), dtype=np.float64)

        for image, pose, focal in zip(images_rgb, camera_poses, focal_lengths):
            pixels, depth, in_image = project_vertices(vertices, pose, focal, image.shape)
            depth_buffer = rasterize_depth_buffer(
                vertices, mesh.faces, pose, focal, image.shape
            )
            visible = visible_projected_points(pixels, depth, depth_buffer)
            camera_center = np.asarray(pose, dtype=np.float32)[:3, 3]
            to_camera = camera_center - vertices
            to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-6)
            cosine = np.clip(np.sum(normals * to_camera, axis=1), 0.0, 1.0)
            selected = visible & (cosine > 0.0)
            if not np.any(selected):
                continue
            view_weights = np.power(cosine[selected], self.gamma)
            samples = sample_rgb_nearest(image, pixels[selected])
            accumulated[selected] += samples * view_weights[:, None]
            accumulated_weights[selected] += view_weights

        colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        observed = accumulated_weights > 1e-8
        if np.any(observed):
            colors[observed, :3] = np.clip(
                accumulated[observed] / accumulated_weights[observed, None], 0, 255
            ).astype(np.uint8)
            colors[~observed, :3] = np.mean(colors[observed, :3], axis=0).astype(np.uint8)
        else:
            colors[:, :3] = (180, 180, 180)
        colors[:, 3] = 255
        return colors

    def bake_texture_map(
        self,
        mesh: trimesh.Trimesh,
        uvs: np.ndarray,
        vertex_colors: np.ndarray,
    ) -> Image.Image:
        """Rasterize barycentrically interpolated vertex colors into the UV atlas."""
        size = self.texture_size
        texture = np.full((size, size, 3), 180, dtype=np.uint8)
        covered = np.zeros((size, size), dtype=bool)
        uv_pixels = np.column_stack(
            (uvs[:, 0] * (size - 1), (1.0 - uvs[:, 1]) * (size - 1))
        )

        for face in np.asarray(mesh.faces, dtype=np.int64):
            triangle = uv_pixels[face]
            x_min = max(int(np.floor(triangle[:, 0].min())), 0)
            x_max = min(int(np.ceil(triangle[:, 0].max())), size - 1)
            y_min = max(int(np.floor(triangle[:, 1].min())), 0)
            y_max = min(int(np.ceil(triangle[:, 1].max())), size - 1)
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
            bary_c = 1.0 - bary_a - bary_b
            inside = (bary_a >= 0) & (bary_b >= 0) & (bary_c >= 0)
            if not np.any(inside):
                continue
            interpolated = (
                bary_a[..., None] * vertex_colors[face[0], :3]
                + bary_b[..., None] * vertex_colors[face[1], :3]
                + bary_c[..., None] * vertex_colors[face[2], :3]
            )
            texture_region = texture[y_min:y_max + 1, x_min:x_max + 1]
            texture_region[inside] = np.clip(interpolated[inside], 0, 255).astype(np.uint8)
            coverage_region = covered[y_min:y_max + 1, x_min:x_max + 1]
            coverage_region[inside] = True

        try:
            from scipy import ndimage
            _, indices = ndimage.distance_transform_edt(~covered, return_indices=True)
            texture = texture[indices[0], indices[1]]
        except ImportError:
            pass
        return Image.fromarray(texture, mode="RGB")

    def bake_texture_from_views(
        self,
        mesh: trimesh.Trimesh,
        uvs: np.ndarray,
        images_rgb: Sequence[np.ndarray],
        camera_poses: Sequence[object],
        focal_lengths: Sequence[Tuple[float, float]],
    ) -> Image.Image:
        """Bake colors per UV texel from visible projected mesh triangles."""
        validate_multiview_inputs(images_rgb, camera_poses, focal_lengths)
        size = self.texture_size
        texture = np.full((size, size, 3), 180, dtype=np.uint8)
        covered = np.zeros((size, size), dtype=bool)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        uv_pixels = np.column_stack(
            (uvs[:, 0] * (size - 1), (1.0 - uvs[:, 1]) * (size - 1))
        )
        depth_buffers = [
            rasterize_depth_buffer(vertices, faces, pose, focal, image.shape)
            for image, pose, focal in zip(images_rgb, camera_poses, focal_lengths)
        ]
        face_normals = np.asarray(mesh.face_normals, dtype=np.float32)

        for face_index, face in enumerate(faces):
            triangle_uv = uv_pixels[face]
            x_min = max(int(np.floor(triangle_uv[:, 0].min())), 0)
            x_max = min(int(np.ceil(triangle_uv[:, 0].max())), size - 1)
            y_min = max(int(np.floor(triangle_uv[:, 1].min())), 0)
            y_max = min(int(np.ceil(triangle_uv[:, 1].max())), size - 1)
            if x_min > x_max or y_min > y_max:
                continue
            grid_x, grid_y = np.meshgrid(
                np.arange(x_min, x_max + 1, dtype=np.float32),
                np.arange(y_min, y_max + 1, dtype=np.float32),
            )
            denominator = (
                (triangle_uv[1, 1] - triangle_uv[2, 1]) * (triangle_uv[0, 0] - triangle_uv[2, 0])
                + (triangle_uv[2, 0] - triangle_uv[1, 0]) * (triangle_uv[0, 1] - triangle_uv[2, 1])
            )
            if abs(float(denominator)) < 1e-8:
                continue
            bary_a = (
                (triangle_uv[1, 1] - triangle_uv[2, 1]) * (grid_x - triangle_uv[2, 0])
                + (triangle_uv[2, 0] - triangle_uv[1, 0]) * (grid_y - triangle_uv[2, 1])
            ) / denominator
            bary_b = (
                (triangle_uv[2, 1] - triangle_uv[0, 1]) * (grid_x - triangle_uv[2, 0])
                + (triangle_uv[0, 0] - triangle_uv[2, 0]) * (grid_y - triangle_uv[2, 1])
            ) / denominator
            bary_c = 1.0 - bary_a - bary_b
            inside = (bary_a >= 0) & (bary_b >= 0) & (bary_c >= 0)
            if not np.any(inside):
                continue
            world_points = (
                bary_a[..., None] * vertices[face[0]]
                + bary_b[..., None] * vertices[face[1]]
                + bary_c[..., None] * vertices[face[2]]
            )
            flat_points = world_points.reshape(-1, 3)
            flat_inside = inside.reshape(-1)
            blended = np.zeros((len(flat_points), 3), dtype=np.float64)
            blend_weights = np.zeros(len(flat_points), dtype=np.float64)
            for view_index, (image, pose, focal) in enumerate(
                zip(images_rgb, camera_poses, focal_lengths)
            ):
                pixels, depths, _ = project_vertices(flat_points, pose, focal, image.shape)
                visible = visible_projected_points(
                    pixels, depths, depth_buffers[view_index]
                )
                camera_center = np.asarray(pose, dtype=np.float32)[:3, 3]
                view_direction = camera_center - flat_points
                view_direction /= np.maximum(
                    np.linalg.norm(view_direction, axis=1, keepdims=True), 1e-6
                )
                selected = visible & flat_inside
                cosine = np.clip(
                    np.sum(view_direction * face_normals[face_index], axis=1), 0.0, 1.0
                )
                selected &= cosine > 0
                if not np.any(selected):
                    continue
                weights = np.power(cosine[selected], self.gamma)
                blended[selected] += sample_rgb_nearest(image, pixels[selected]) * weights[:, None]
                blend_weights[selected] += weights
            observed = blend_weights > 1e-8
            flat_colors = np.full((len(flat_points), 3), 180, dtype=np.uint8)
            flat_colors[observed] = np.clip(
                blended[observed] / blend_weights[observed, None], 0, 255
            ).astype(np.uint8)
            region = texture[y_min:y_max + 1, x_min:x_max + 1]
            region[inside] = flat_colors.reshape(region.shape[0], region.shape[1], 3)[inside]
            coverage_region = covered[y_min:y_max + 1, x_min:x_max + 1]
            coverage_region[inside] = True

        try:
            from scipy import ndimage
            _, indices = ndimage.distance_transform_edt(~covered, return_indices=True)
            texture = texture[indices[0], indices[1]]
        except ImportError:
            pass
        return Image.fromarray(texture, mode="RGB")

    def process_and_export(
        self,
        mesh: trimesh.Trimesh,
        images_rgb: Sequence[np.ndarray],
        camera_poses: Sequence[object],
        focal_lengths: Sequence[Tuple[float, float]],
        output_path: str,
    ) -> Tuple[bool, str]:
        """Run P5: unwrap, blend, bake and export one binary GLB."""
        started = time.time()
        try:
            validate_multiview_inputs(images_rgb, camera_poses, focal_lengths)
            unwrapped_mesh, uvs = self.unwrap_uv(mesh)
            vertex_colors = self.blend_colors_for_vertices(
                unwrapped_mesh, images_rgb, camera_poses, focal_lengths
            )
            texture = self.bake_texture_from_views(
                unwrapped_mesh, uvs, images_rgb, camera_poses, focal_lengths
            )
            unwrapped_mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=uvs,
                image=texture,
            )
            unwrapped_mesh.visual.vertex_colors = vertex_colors
            path = export_glb(unwrapped_mesh, output_path)
            logger.info("P5 exported %s in %.2fs", path, time.time() - started)
            return True, path
        except Exception:
            logger.exception("P5 texture export failed")
            return False, ""