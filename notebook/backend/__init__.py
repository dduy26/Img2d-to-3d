# notebook/backend/__init__.py
"""Backend package cho ImgToModel Pipeline chuẩn NVIDIA."""

try:
    from .preprocess import (
        preprocess_multiview,
        preprocess_single_view,
        validate_and_load_images,
        histogram_match_sequence,
        extract_alpha_masks,
        refine_alpha_mask,
        vit_geometric_resize,
        classify_viewpoints,
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
        classify_viewpoints,
    )

# Backward-compatibility aliases
dust3r_resize = vit_geometric_resize
histogram_match = histogram_match_sequence

__all__ = [
    "preprocess_multiview",
    "preprocess_single_view",
    "validate_and_load_images",
    "histogram_match_sequence",
    "histogram_match",
    "extract_alpha_masks",
    "refine_alpha_mask",
    "vit_geometric_resize",
    "dust3r_resize",
    "classify_viewpoints",
]
