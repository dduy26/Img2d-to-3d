"""
__init__.py — Package Interface voi Lazy Loading
================================================
Su dung __getattr__ de chi import module khi can thiet,
tranh import cascade (cascade import lam tang thoi gian khoi dong).

Module nay hoat dong nhu cong vao chung cua package 'backend'.
"""

from __future__ import annotations

# Version cua pipeline
__version__ = "2.0.0"
__author__  = "ImgToModel Team"

# Cac module chinh
_MODULES = {
    "utils_3d":         "notebook.backend.utils_3d",
    "preprocess":       "notebook.backend.preprocess",
    "engine_depth":     "notebook.backend.engine_depth",
    "quality_gate":     "notebook.backend.quality_gate",
    "engine_tsdf_mesh": "notebook.backend.engine_tsdf_mesh",
    "texture_blender":  "notebook.backend.texture_blender",
    "app":              "notebook.backend.app",
}

def __getattr__(name: str):
    """Lazy load module khi duoc truy cap lan dau."""
    import importlib
    if name in _MODULES:
        mod = importlib.import_module(_MODULES[name])
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'notebook.backend' khong co thuoc tinh '{name}'")


def __dir__():
    return list(_MODULES.keys()) + ["__version__", "__author__"]
