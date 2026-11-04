"""
Training pipeline for squat form classification models.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score

# Ensure clean UTF-8 printing on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.models import MODEL_REGISTRY, MLP, GRU, TCN
from src.preprocessing import (
    load_cached_features,
    get_grouped_splits,
    get_grouped_kfold,
    create_dataloader,
    compute_class_weights,
)


def get_training_device() -> torch.device:
    """Return PyTorch CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    max_epochs: int = 60,
    patience: int = 10,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, float]:
    """
    Train a model with early stopping on validation macro-F1.

    Returns:
        (best_model, best_val_macro_f1)
    """
    if device is None:
        device = get_training_device()

    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -1.0
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        # Training step
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation step
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb.to(device)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())

        val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

        # Checkpoint & early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_f1


def run_cross_validation(
    cache_path: str | Path = "data/processed/squat_pose_16f_v1.npz",
    n_folds: int = 5,
    out_dir: str | Path = "outputs",
    suffix: str = "",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Execute full 5-fold grouped cross-validation across all 25 subjects.
    Computes Macro-F1, Accuracy, and Balanced Accuracy.
    """
    data = load_cached_features(cache_path)
    X, y, groups = data["X"], data["y"], data["groups"]
    n_frames = X.shape[1]
    dev = get_training_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = suffix if suffix else f"{n_frames}f"
    print(f"Starting {n_folds}-Fold Grouped CV on {len(np.unique(groups))} subjects ({n_frames} frames per clip)...")
    cv_results = []

    for fold, train_idx, val_idx, test_idx in get_grouped_kfold(X, y, groups, n_splits=n_folds, seed=seed):
        print(f"\n{'='*60}")
        print(f"  Fold {fold}/{n_folds} | Test subjects: {sorted(set(groups[test_idx]))}")
        print(f"{'='*60}")

        train_loader = create_dataloader(X, y, train_idx, batch_size=32, shuffle=True)
        val_loader = create_dataloader(X, y, val_idx, batch_size=32, shuffle=False)
        test_loader = create_dataloader(X, y, test_idx, batch_size=32, shuffle=False)

        weights = compute_class_weights(y[train_idx], device=dev)

        for name, ModelCls in MODEL_REGISTRY.items():
            torch.manual_seed(seed + fold)
            model = ModelCls()
            trained_model, val_f1 = train_model(
                model,
                train_loader,
                val_loader,
                class_weights=weights,
                device=dev,
                verbose=False,
            )

            # Evaluate on held-out test fold
            trained_model.eval()
            test_preds, test_true = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    preds = trained_model(xb.to(dev)).argmax(dim=1).cpu().numpy()
                    test_preds.extend(preds)
                    test_true.extend(yb.numpy())

            test_true = np.array(test_true)
            test_preds = np.array(test_preds)

            acc = float((test_true == test_preds).mean())
            mf1 = float(f1_score(test_true, test_preds, average="macro", zero_division=0))
            ba = float(balanced_accuracy_score(test_true, test_preds))

            cv_results.append({
                "fold": fold,
                "model": name,
                "test_accuracy": round(acc, 4),
                "test_macro_f1": round(mf1, 4),
                "test_bal_acc": round(ba, 4),
                "val_f1": round(val_f1, 4),
            })
            print(f"  {name:5s} | Acc={acc:.3f} | Macro-F1={mf1:.3f} | Bal-Acc={ba:.3f} | val-F1={val_f1:.3f}")

        # Majority class baseline
        majority_class = np.bincount(y[train_idx]).argmax()
        maj_preds = np.full(len(test_idx), majority_class)
        maj_acc = float((y[test_idx] == maj_preds).mean())
        maj_f1 = float(f1_score(y[test_idx], maj_preds, average="macro", zero_division=0))
        maj_ba = float(balanced_accuracy_score(y[test_idx], maj_preds))
        cv_results.append({
            "fold": fold,
            "model": "Majority",
            "test_accuracy": round(maj_acc, 4),
            "test_macro_f1": round(maj_f1, 4),
            "test_bal_acc": round(maj_ba, 4),
            "val_f1": 0.0,
        })

    cv_df = pd.DataFrame(cv_results)
    summary = cv_df.groupby("model").agg(
        mean_acc=("test_accuracy", "mean"),
        std_acc=("test_accuracy", "std"),
        mean_f1=("test_macro_f1", "mean"),
        std_f1=("test_macro_f1", "std"),
        mean_ba=("test_bal_acc", "mean"),
        std_ba=("test_bal_acc", "std"),
    ).round(4)

    summary["Accuracy"] = summary.apply(lambda r: f"{r['mean_acc']:.3f} ± {r['std_acc']:.3f}", axis=1)
    summary["Macro-F1"] = summary.apply(lambda r: f"{r['mean_f1']:.3f} ± {r['std_f1']:.3f}", axis=1)
    summary["Bal Acc"] = summary.apply(lambda r: f"{r['mean_ba']:.3f} ± {r['std_ba']:.3f}", axis=1)

    print("\n" + "=" * 65)
    print(f"[RESULTS] Grouped 5-Fold Cross-Validation Summary ({tag}):")
    print("=" * 65)
    print(summary[["Accuracy", "Macro-F1", "Bal Acc"]].sort_values("Macro-F1", ascending=False).to_string())

    cv_df.to_csv(out_dir / f"cv_results_per_fold_{tag}.csv", index=False)
    summary.to_csv(out_dir / f"cv_summary_{tag}.csv")
    print(f"\nSaved CV metrics to {out_dir}/cv_summary_{tag}.csv")
    return summary


def train_single_split(
    cache_path: str | Path = "data/processed/squat_pose_16f_v1.npz",
    out_dir: str | Path = "outputs",
    model_name: str = "all",
    suffix: str = "",
    seed: int = 42,
) -> None:
    """Train models on the standard 60/20/20 subject split and save checkpoints."""
    data = load_cached_features(cache_path)
    X, y, groups = data["X"], data["y"], data["groups"]
    n_frames = X.shape[1]
    dev = get_training_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = suffix if suffix else f"{n_frames}f"

    train_idx, val_idx, test_idx = get_grouped_splits(X, y, groups, seed=seed)
    train_loader = create_dataloader(X, y, train_idx, batch_size=32, shuffle=True)
    val_loader = create_dataloader(X, y, val_idx, batch_size=32, shuffle=False)
    test_loader = create_dataloader(X, y, test_idx, batch_size=32, shuffle=False)

    weights = compute_class_weights(y[train_idx], device=dev)
    models_to_train = MODEL_REGISTRY.keys() if model_name.lower() == "all" else [model_name.upper()]

    for name in models_to_train:
        print(f"\nTraining {name} on {len(train_idx)} clips ({len(set(groups[train_idx]))} subjects, {n_frames} frames)...")
        model = MODEL_REGISTRY[name]()
        trained_model, best_val_f1 = train_model(
            model, train_loader, val_loader, class_weights=weights, device=dev
        )

        # Evaluate on test split
        trained_model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                p = trained_model(xb.to(dev)).argmax(dim=1).cpu().numpy()
                preds.extend(p)
                trues.extend(yb.numpy())

        test_acc = float((np.array(trues) == np.array(preds)).mean())
        test_f1 = float(f1_score(trues, preds, average="macro", zero_division=0))

        print(f"  {name} Best Val Macro-F1: {best_val_f1:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

        # Save checkpoint
        file_name = f"{name.lower()}_squat_{tag}.pt" if tag != "16f" else f"{name.lower()}_squat.pt"
        ckpt_path = out_dir / file_name
        torch.save({
            "model_name": name,
            "state_dict": trained_model.state_dict(),
            "frames": n_frames,
            "labels": {"0": "bad", "1": "good"},
            "val_f1": best_val_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train exercise form classifiers.")
    parser.add_argument("--cache", type=str, default="data/processed/squat_pose_16f_v1.npz", help="Path to cached .npz")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory for checkpoints and metrics")
    parser.add_argument("--cv", action="store_true", help="Run 5-Fold Grouped Cross-Validation")
    parser.add_argument("--model", type=str, default="all", choices=["MLP", "GRU", "TCN", "all"], help="Model architecture")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for checkpoint/results (e.g. 32f)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.cv:
        run_cross_validation(cache_path=args.cache, out_dir=args.out_dir, suffix=args.suffix, seed=args.seed)
    else:
        train_single_split(cache_path=args.cache, out_dir=args.out_dir, model_name=args.model, suffix=args.suffix, seed=args.seed)
