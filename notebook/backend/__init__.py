# notebook/backend/__init__.py
"""
Backend package cho ImgToModel Pipeline chuẩn NVIDIA.
Cung cấp toàn bộ các phân hệ P1 - P6 cho người dùng:
- P1: Preprocessing & Viewpoint Assignment
- P2: Depth & Surface Mesh Engine (Depth-Anything-V2)
- P3: Quality Gate
- P4: Volumetric TSDF Mesh Engine (True Space Carving)
- P5: Texture Blender & PBR Exporter
- P6: FastAPI Application & execute_3d_pipeline
"""

# Đảm bảo thư mục hiện tại có trong sys.path để hỗ trợ cả import tuyệt đối và tương đối
import os, sys
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

try:
    from .preprocess import (
        preprocess_multiview,
        preprocess_single_view,
        validate_and_load_images,
        histogram_match_sequence,
        extract_alpha_masks,
        refine_alpha_mask,
        vit_geometric_resize,
        normalize_multiview_scales_and_canvas,
        classify_viewpoints,
    )
    from .engine_depth import (
        DepthReconstructionEngine,
        SurfaceMeshEngine,
    )
    from .quality_gate import (
        QualityGate,
    )
    from .engine_tsdf_mesh import (
        TSDFMeshEngine,
        mesh_health,
        log_mesh_health,
        generate_camera_poses,
    )
    from .texture_blender import (
        TextureBlender,
    )
    from .utils_3d import (
        export_glb,
    )
    from .app import (
        execute_3d_pipeline,
        app,
    )
except ImportError:
    from preprocess import (
        preprocess_multiview,
        preprocess_single_view,
        validate_and_load_images,
        histogram_match_sequence,
        extract_alpha_masks,
        refine_alpha_mask,
        vit_geometric_resize,
        normalize_multiview_scales_and_canvas,
        classify_viewpoints,
    )
    from engine_depth import (
        DepthReconstructionEngine,
        SurfaceMeshEngine,
    )
    from quality_gate import (
        QualityGate,
    )
    from engine_tsdf_mesh import (
        TSDFMeshEngine,
        mesh_health,
        log_mesh_health,
        generate_camera_poses,
    )
    from texture_blender import (
        TextureBlender,
    )
    from utils_3d import (
        export_glb,
    )
    from app import (
        execute_3d_pipeline,
        app,
    )

# Backward-compatibility aliases
dust3r_resize = vit_geometric_resize
histogram_match = histogram_match_sequence

__all__ = [
    # P1
    "preprocess_multiview",
    "preprocess_single_view",
    "validate_and_load_images",
    "histogram_match_sequence",
    "histogram_match",
    "extract_alpha_masks",
    "refine_alpha_mask",
    "vit_geometric_resize",
    "normalize_multiview_scales_and_canvas",
    "dust3r_resize",
    "classify_viewpoints",
    # P2
    "DepthReconstructionEngine",
    "SurfaceMeshEngine",
    # P3
    "QualityGate",
    # P4
    "TSDFMeshEngine",
    "mesh_health",
    "log_mesh_health",
    "generate_camera_poses",
    # P5
    "TextureBlender",
    "export_glb",
    # P6
    "execute_3d_pipeline",
    "app",
]
