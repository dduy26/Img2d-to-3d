"""Focused tests for the Member 5 contract."""

import sys
from pathlib import Path

import numpy as np
import trimesh

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from texture_blender import TextureBlender
from utils_3d import project_vertices, visible_vertex_mask


def _camera(angle: float) -> np.ndarray:
    position = np.array([2.0 * np.sin(angle), 0.0, 2.0 * np.cos(angle)], dtype=np.float32)
    forward = -position / np.linalg.norm(position)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = forward
    pose[:3, 3] = position
    return pose


def _p1_p2_views(count: int = 4):
    images = []
    poses = []
    focals = []
    colors = [(220, 20, 20), (20, 220, 20), (20, 20, 220), (220, 220, 20)]
    for index in range(count):
        images.append(np.full((64, 64, 3), colors[index], dtype=np.uint8))
        poses.append(_camera(index * 2.0 * np.pi / count))
        focals.append((55.0, 55.0))
    return images, poses, focals


def test_projection_and_visibility_contract():
    mesh = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
    pixels, depth, in_image = project_vertices(
        mesh.vertices, _camera(0.0), (55.0, 55.0), (64, 64, 3)
    )
    visible = visible_vertex_mask(pixels, depth, in_image, (64, 64, 3))
    assert pixels.shape == (len(mesh.vertices), 2)
    assert np.count_nonzero(visible) > 0


def test_p4_mesh_to_p5_uv_blend_and_bake():
    mesh = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
    images, poses, focals = _p1_p2_views()
    blender = TextureBlender(texture_size=64)
    unwrapped, uvs = blender.unwrap_uv(mesh)
    colors = blender.blend_colors_for_vertices(unwrapped, images, poses, focals)
    texture = blender.bake_texture_map(unwrapped, uvs, colors)
    assert uvs.shape == (len(unwrapped.vertices), 2)
    assert np.all((uvs >= 0.0) & (uvs <= 1.0))
    assert colors.shape == (len(unwrapped.vertices), 4)
    assert texture.size == (64, 64)
    assert texture.mode == "RGB"


def test_glb_contains_uv_and_base_color_texture(tmp_path):
    mesh = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
    images, poses, focals = _p1_p2_views()
    output_path = tmp_path / "p5_textured.glb"
    success, result_path = TextureBlender(texture_size=64).process_and_export(
        mesh, images, poses, focals, str(output_path)
    )
    assert success
    assert result_path == str(output_path)
    loaded = trimesh.load(str(output_path), file_type="glb", force="scene")
    geometry = next(iter(loaded.geometry.values()))
    assert geometry.visual.uv is not None
    assert geometry.visual.material.baseColorTexture is not None
    assert geometry.visual.material.baseColorTexture.size == (64, 64)


def test_invalid_p1_p2_contract_fails_without_export(tmp_path):
    mesh = trimesh.creation.box()
    success, result_path = TextureBlender(texture_size=32).process_and_export(
        mesh, [np.zeros((8, 8, 3), dtype=np.uint8)], [], [], str(tmp_path / "bad.glb")
    )
    assert not success
    assert result_path == ""