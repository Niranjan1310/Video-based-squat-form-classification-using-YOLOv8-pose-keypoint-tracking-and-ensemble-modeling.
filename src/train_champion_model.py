"""
Biomechanical feature extraction and voting ensemble for squat form classification.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
import joblib


def compute_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.sum(v1 * v2, axis=-1)
    norm = np.maximum(np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1), 1e-6)
    return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))


def extract_champion_features(X_raw, views):
    """Compute biomechanical and kinematic features from raw pose sequences."""
    N, T, _ = X_raw.shape
    p = X_raw[..., :34].reshape(N, T, 17, 2)
    conf = X_raw[..., 2::3]

    mid_shoulder = (p[:, :, 5] + p[:, :, 6]) * 0.5
    mid_hip = (p[:, :, 11] + p[:, :, 12]) * 0.5
    mid_knee = (p[:, :, 13] + p[:, :, 14]) * 0.5
    mid_ankle = (p[:, :, 15] + p[:, :, 16]) * 0.5

    # Torso length (scale factor)
    torso_len = np.maximum(np.linalg.norm(mid_shoulder - mid_hip, axis=-1), 1e-3)

    # 1. Joint angles
    l_knee_deg = compute_angle(p[:, :, 11], p[:, :, 13], p[:, :, 15])
    r_knee_deg = compute_angle(p[:, :, 12], p[:, :, 14], p[:, :, 16])
    mean_knee_deg = (l_knee_deg + r_knee_deg) * 0.5
    knee_diff = np.abs(l_knee_deg - r_knee_deg)

    l_hip_deg = compute_angle(p[:, :, 5], p[:, :, 11], p[:, :, 13])
    r_hip_deg = compute_angle(p[:, :, 6], p[:, :, 12], p[:, :, 14])
    mean_hip_deg = (l_hip_deg + r_hip_deg) * 0.5

    torso_vec = mid_shoulder - mid_hip
    torso_lean = np.degrees(np.arctan2(np.abs(torso_vec[:, :, 0]), np.abs(torso_vec[:, :, 1]) + 1e-6))

    # Biomechanical cues normalized by torso length
    hip_knee_depth = (mid_hip[:, :, 1] - mid_knee[:, :, 1]) / torso_len
    stance_w = np.abs(p[:, :, 15, 0] - p[:, :, 16, 0]) / torso_len
    knee_w = np.abs(p[:, :, 13, 0] - p[:, :, 14, 0]) / torso_len
    valgus_ratio = knee_w / np.maximum(stance_w, 1e-3)
    pelvic_tilt = (p[:, :, 11, 1] - p[:, :, 12, 1]) / torso_len

    # Standing reference (first 3 frames)
    p_stand = p[:, :3].mean(axis=1)
    hip_stand_y = mid_hip[:, :3, 1].mean(axis=1, keepdims=True)
    hip_drop = (mid_hip[:, :, 1] - hip_stand_y) / torso_len

    # Velocity features
    knee_vel = np.diff(mean_knee_deg, axis=1, prepend=mean_knee_deg[:, :1])
    hip_vel = np.diff(mean_hip_deg, axis=1, prepend=mean_hip_deg[:, :1])
    depth_vel = np.diff(hip_knee_depth, axis=1, prepend=hip_knee_depth[:, :1])

    all_features = []
    for i in range(N):
        f = []
        k_arr = mean_knee_deg[i]
        bottom_idx = int(np.argmin(k_arr))

        # Bottom inflection features
        f.append(k_arr[bottom_idx])
        f.append(mean_hip_deg[i, bottom_idx])
        f.append(hip_knee_depth[i, bottom_idx])
        f.append(torso_lean[i, bottom_idx])
        f.append(valgus_ratio[i, bottom_idx])
        f.append(pelvic_tilt[i, bottom_idx])
        f.append(knee_diff[i, bottom_idx])
        f.append(hip_drop[i, bottom_idx])
        f.append(np.max(k_arr) - np.min(k_arr))  # Knee ROM
        f.append(float(bottom_idx) / max(T - 1, 1))

        # Trajectory statistics
        trajs = [
            mean_knee_deg[i], mean_hip_deg[i], torso_lean[i],
            hip_knee_depth[i], valgus_ratio[i], pelvic_tilt[i],
            knee_diff[i], hip_drop[i], knee_vel[i], depth_vel[i]
        ]
        for t_arr in trajs:
            diff = np.abs(np.diff(t_arr))
            f.extend([
                t_arr.mean(), t_arr.std(), t_arr.min(), t_arr.max(),
                np.quantile(t_arr, 0.1), np.quantile(t_arr, 0.5), np.quantile(t_arr, 0.9),
                diff.mean()
            ])

        # Raw keypoints summary statistics
        raw = X_raw[i]  # (32, 51)
        f.extend(raw.mean(axis=0))
        f.extend(raw.std(axis=0))
        f.extend(raw.min(axis=0))
        f.extend(raw.max(axis=0))
        f.extend(np.abs(np.diff(raw, axis=0)).mean(axis=0))

        # View one-hot
        v = views[i].lower()
        f.extend([
            1.0 if v == "front" else 0.0,
            1.0 if v == "side" else 0.0,
            1.0 if v == "diagonal" else 0.0,
        ])

        all_features.append(f)

    return np.asarray(all_features, dtype=np.float32)


def create_champion_ensemble(random_state=42):
    """
    Tuned ensemble combining ExtraTrees (450 trees) + HistGradientBoosting (150 trees)
    + Random Forest (350 trees) with soft voting.
    """
    et = ExtraTreesClassifier(
        n_estimators=450, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=150, min_samples_leaf=3, l2_regularization=1.5,
        class_weight="balanced", random_state=random_state
    )
    rf = RandomForestClassifier(
        n_estimators=350, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    return VotingClassifier(
        estimators=[("et", et), ("hgb", hgb), ("rf", rf)],
        voting="soft",
        weights=[0.50, 0.30, 0.20],
    )


def evaluate_and_save():
    cache_path = Path("data/processed/squat_pose_32f_v1.npz")
    d = np.load(cache_path, allow_pickle=True)
    X_raw, y, groups, views, paths = d["X"], d["y"], d["groups"], d["views"], d["paths"]

    print("Computing champion feature representations...")
    X = extract_champion_features(X_raw, views)
    print(f"Feature matrix: {X.shape}")

    # Load split
    split_path = Path("outputs/final_split.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_subjs = set(split["test_subjects"])

    dev_mask = np.array([str(g) not in test_subjs for g in groups])
    test_mask = np.array([str(g) in test_subjs for g in groups])

    # 1. 5-Fold Grouped CV on Development Set
    print("\nRunning 5-Fold Grouped CV on Development Set (120 clips, 20 subjects)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    dev_X, dev_y, dev_groups, dev_views = X[dev_mask], y[dev_mask], groups[dev_mask], views[dev_mask]

    cv_scores = []
    for fold, (tr, val) in enumerate(sgkf.split(dev_X, dev_y, dev_groups), 1):
        ens = create_champion_ensemble(random_state=42 + fold)
        ens.fit(dev_X[tr], dev_y[tr])
        probs = ens.predict_proba(dev_X[val])[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(dev_y[val], preds)
        f1 = f1_score(dev_y[val], preds, average="macro", zero_division=0)
        ba = balanced_accuracy_score(dev_y[val], preds)
        cv_scores.append({"fold": fold, "accuracy": acc, "macro_f1": f1, "balanced_accuracy": ba})
        print(f"  Fold {fold}: Accuracy = {acc:.3f} | Macro-F1 = {f1:.3f} | Bal-Acc = {ba:.3f}")

    cv_df = pd.DataFrame(cv_scores)
    print("\n" + "=" * 60)
    print(f"CV Summary: Accuracy = {cv_df['accuracy'].mean():.3f} +/- {cv_df['accuracy'].std():.3f} | "
          f"Macro-F1 = {cv_df['macro_f1'].mean():.3f} +/- {cv_df['macro_f1'].std():.3f}")
    print("=" * 60)

    # 2. Train Champion Model on all Development Data
    print("\nTraining final champion model on all 20 development subjects...")
    champion = create_champion_ensemble(random_state=42)
    champion.fit(dev_X, dev_y)

    # 3. Evaluate on Held-Out Test Set
    test_X, test_y, test_views, test_groups = X[test_mask], y[test_mask], views[test_mask], groups[test_mask]
    test_probs = champion.predict_proba(test_X)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    test_acc = accuracy_score(test_y, test_preds)
    test_f1 = f1_score(test_y, test_preds, average="macro")
    test_ba = balanced_accuracy_score(test_y, test_preds)

    print("\n" + "=" * 60)
    print("HELD-OUT TEST EVALUATION (5 Reserved Subjects, 30 Clips):")
    print("=" * 60)
    print(f"  Test Accuracy:     {test_acc:.4f} (up from 0.433 GRU baseline)")
    print(f"  Test Macro-F1:     {test_f1:.4f} (up from 0.401 GRU baseline)")
    print(f"  Balanced Accuracy: {test_ba:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_y, test_preds, target_names=["bad", "good"]))

    for v in ["front", "side", "diagonal"]:
        m = (test_views == v)
        print(f"  View {v:8s}: Accuracy = {accuracy_score(test_y[m], test_preds[m]):.3f} ({m.sum()} clips)")

    # 4. Multi-View Consensus Evaluation
    df_cons = pd.DataFrame({"subject": test_groups, "y_true": test_y, "prob": test_probs})
    cons_agg = df_cons.groupby(["subject", "y_true"])["prob"].mean().reset_index()
    cons_pred = (cons_agg["prob"].values >= 0.5).astype(int)
    mv_acc = accuracy_score(cons_agg["y_true"].values, cons_pred)
    mv_f1 = f1_score(cons_agg["y_true"].values, cons_pred, average="macro")

    print(f"\nMulti-View Consensus (subject-exercise level):")
    print(f"  Consensus Accuracy: {mv_acc:.4f}")
    print(f"  Consensus Macro-F1: {mv_f1:.4f}")

    # 5. Save model checkpoint for web app & inference
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "champion_ensemble.joblib"
    joblib.dump({
        "model": champion,
        "cv_accuracy": cv_df["accuracy"].mean(),
        "cv_macro_f1": cv_df["macro_f1"].mean(),
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "features": "champion_biomech_kinematic_348",
    }, model_path)
    print(f"\nChampion model saved to {model_path}!")


if __name__ == "__main__":
    evaluate_and_save()

