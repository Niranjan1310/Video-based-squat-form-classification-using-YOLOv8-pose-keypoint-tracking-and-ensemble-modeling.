"""
Evaluation metrics and diagnostic analysis for squat form classification.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    balanced_accuracy_score,
)

from src.models import MODEL_REGISTRY
from src.data import load_cached_features, get_grouped_splits, create_dataloader


def predict(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on DataLoader batches.

    Returns:
        (y_true, y_pred, y_prob_good)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_preds, all_probs, all_true = [], [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            logits = model(xb.to(device))
            probs = logits.softmax(dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_true.extend(yb.numpy())

    return np.array(all_true), np.array(all_preds), np.array(all_probs)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None,
) -> Dict[str, Any]:
    """Calculate and return evaluation metrics dictionary."""
    if target_names is None:
        target_names = ["bad", "good"]

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "report": report_dict,
        "report_str": classification_report(y_true, y_pred, target_names=target_names, zero_division=0),
    }


def analyze_views(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    views: np.ndarray,
) -> pd.DataFrame:
    """Breakdown classification performance by camera viewing angle."""
    rows = []
    for view in sorted(np.unique(views)):
        mask = views == view
        vf1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        ba = balanced_accuracy_score(y_true[mask], y_pred[mask])
        rows.append({
            "view": view,
            "clips": int(mask.sum()),
            "macro_f1": round(vf1, 4),
            "bal_accuracy": round(ba, 4),
        })
    return pd.DataFrame(rows)


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save confusion matrix figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["bad", "good"]).plot(
        cmap="Blues", values_format="d", ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {out_path}")


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    cache_path: str | Path = "data/processed/squat_pose_16f_v1.npz",
    out_dir: str | Path = "outputs",
) -> None:
    """Evaluate a saved model checkpoint on the held-out test split."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", "TCN")
    state_dict = ckpt["state_dict"]

    model = MODEL_REGISTRY[model_name]()
    model.load_state_dict(state_dict)

    data = load_cached_features(cache_path)
    X, y, groups, views, paths = (
        data["X"],
        data["y"],
        data["groups"],
        data["views"],
        data["paths"],
    )

    _, _, test_idx = get_grouped_splits(X, y, groups, seed=42)
    test_loader = create_dataloader(X, y, test_idx, batch_size=32, shuffle=False)

    y_true, y_pred, y_prob = predict(model, test_loader)
    metrics = evaluate_predictions(y_true, y_pred)

    print(f"\n==================================================")
    print(f"  Evaluation on Test Split: {model_name}")
    print(f"==================================================")
    print(f"Macro-F1:          {metrics['macro_f1']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
    print(metrics["report_str"])

    view_df = analyze_views(y_true, y_pred, views[test_idx])
    print("\nPer-View Breakdown:")
    print(view_df.to_string(index=False))

    out_dir = Path(out_dir)
    plot_and_save_confusion_matrix(
        y_true, y_pred, out_dir / f"{model_name.lower()}_confusion_matrix.png", f"{model_name} Confusion Matrix"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate squat form classification checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="outputs/tcn_squat.pt", help="Path to checkpoint .pt")
    parser.add_argument("--cache", type=str, default="data/processed/squat_pose_16f_v1.npz", help="Path to cached .npz")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory for output plots")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.cache, args.out_dir)
