"""
Backward-compatibility alias module.
Re-exports data and preprocessing functions from src.preprocessing.
"""

from src.preprocessing import (
    SquatPoseDataset,
    load_manifest,
    load_cached_features,
    get_grouped_splits,
    get_grouped_kfold,
    create_dataloader,
    compute_class_weights,
    verify_zero_leakage,
)

__all__ = [
    "SquatPoseDataset",
    "load_manifest",
    "load_cached_features",
    "get_grouped_splits",
    "get_grouped_kfold",
    "create_dataloader",
    "compute_class_weights",
    "verify_zero_leakage",
]
