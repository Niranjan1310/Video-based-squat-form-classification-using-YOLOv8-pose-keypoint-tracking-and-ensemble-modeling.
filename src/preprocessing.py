"""
Preprocessing and participant-grouped dataset splitting for squat pose sequences.
"""

from pathlib import Path
from typing import Tuple, List, Generator, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, GroupKFold


class SquatPoseDataset(Dataset):
    """PyTorch Dataset wrapping pose sequence tensors and classification labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Array of shape (N, T, 51) containing normalized keypoint sequences.
            y: Array of shape (N,) containing binary integer labels (0=bad, 1=good).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_manifest(manifest_path: str | Path, exercise_filter: str = "squat") -> pd.DataFrame:
    """
    Load and validate metadata manifest CSV.

    Args:
        manifest_path: Path to metadata.csv
        exercise_filter: Substring to filter exercises by (case-insensitive)

    Returns:
        Filtered DataFrame with standardized columns
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    df = pd.read_csv(manifest_path, dtype=str).fillna("")
    required_cols = {"video_path", "label", "subject_id", "exercise", "view"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    df["label"] = df["label"].str.lower().str.strip()
    df["exercise"] = df["exercise"].str.lower().str.strip()

    if exercise_filter:
        df = df[df["exercise"].str.contains(exercise_filter.lower(), regex=False)].copy()

    return df


def load_cached_features(cache_path: str | Path) -> Dict[str, np.ndarray]:
    """
    Load extracted feature arrays from compressed .npz archive.

    Returns dict containing keys: 'X', 'y', 'groups', 'views', 'paths'.
    'X' has shape (N_clips, N_frames, 51).
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached features not found at {cache_path}")

    data = np.load(cache_path, allow_pickle=True)
    return {
        "X": data["X"],
        "y": data["y"],
        "groups": data["groups"],
        "views": data["views"],
        "paths": data["paths"],
    }


def verify_zero_leakage(train_groups: np.ndarray, val_groups: np.ndarray, test_groups: np.ndarray) -> None:
    """Assert that there is no participant overlap between train, val, and test groups."""
    train_subj = set(train_groups)
    val_subj = set(val_groups)
    test_subj = set(test_groups)

    overlap_tv = train_subj & val_subj
    overlap_tt = train_subj & test_subj
    overlap_vt = val_subj & test_subj

    assert not overlap_tv, f"Leakage detected: train & val overlap on {overlap_tv}"
    assert not overlap_tt, f"Leakage detected: train & test overlap on {overlap_tt}"
    assert not overlap_vt, f"Leakage detected: val & test overlap on {overlap_vt}"


def get_grouped_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.25,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train, validation, and test sets grouped by participant.
    Ensures zero subject overlap across splits.

    Returns:
        (train_idx, val_idx, test_idx) as integer numpy arrays.
    """
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups))

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
    rel_train_idx, rel_val_idx = next(
        gss_val.split(X[trainval_idx], y[trainval_idx], groups[trainval_idx])
    )

    train_idx = trainval_idx[rel_train_idx]
    val_idx = trainval_idx[rel_val_idx]

    verify_zero_leakage(groups[train_idx], groups[val_idx], groups[test_idx])

    return train_idx, val_idx, test_idx


def get_grouped_kfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Generator[Tuple[int, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Generate grouped K-fold splits. For each fold, rotates test subjects and splits
    trainval into train (80%) and validation (20%) by subject.

    Yields:
        (fold_num, train_idx, val_idx, test_idx)
    """
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (trainval_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold)
        rel_tr, rel_va = next(
            gss.split(X[trainval_idx], y[trainval_idx], groups[trainval_idx])
        )
        train_idx = trainval_idx[rel_tr]
        val_idx = trainval_idx[rel_va]

        # Verify zero leakage
        assert not (set(groups[train_idx]) & set(groups[test_idx])), f"Fold {fold+1}: train/test leak"
        assert not (set(groups[val_idx]) & set(groups[test_idx])), f"Fold {fold+1}: val/test leak"

        yield fold + 1, train_idx, val_idx, test_idx


def compute_class_weights(y_train: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """Compute balanced class weights: N / (num_classes * bincount)."""
    class_counts = np.bincount(y_train, minlength=2)
    weights = len(y_train) / (2.0 * np.maximum(class_counts, 1))
    tensor_weights = torch.tensor(weights, dtype=torch.float32)
    return tensor_weights.to(device) if device is not None else tensor_weights


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    indices: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader from arrays and optional index subset."""
    if indices is not None:
        X_sub = X[indices]
        y_sub = y[indices]
    else:
        X_sub = X
        y_sub = y
    dataset = SquatPoseDataset(X_sub, y_sub)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
