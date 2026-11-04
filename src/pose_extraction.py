"""
Pose extraction module using pretrained YOLOv8n-pose.

Extracts 17 COCO body keypoints across uniformly sampled video frames,
normalizes coordinates relative to the person's bounding box, interpolates
missing frames, and flattens to (N_FRAMES, 51) feature vectors.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def get_default_device() -> str | int:
    """Select CUDA device 0 if available, otherwise 'cpu'."""
    return 0 if torch.cuda.is_available() else "cpu"


def extract_pose_sequence(
    video_path: str | Path,
    n_frames: int = 16,
    imgsz: int = 384,
    device: Optional[str | int] = None,
    pose_model: Optional[YOLO] = None,
) -> Optional[np.ndarray]:
    """
    Extract a normalized pose sequence of shape (n_frames, 51) from a video.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return None

    if device is None:
        device = get_default_device()

    if pose_model is None:
        pose_model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    # Pre-allocate sequence array
    seq = np.zeros((n_frames, 17, 3), dtype=np.float32)

    # Sample frame indices
    sample_indices = np.linspace(0, total_frames - 1, n_frames).round().astype(int)

    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # YOLO pose inference
        result = pose_model.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            continue  # No person detected in this frame

        xy = result.keypoints.xy.cpu().numpy()  # (n_people, 17, 2)
        conf = result.keypoints.conf.cpu().numpy()  # (n_people, 17)

        if len(conf) == 0:
            continue

        # Select person with highest mean confidence
        best_person = int(conf.mean(axis=1).argmax())
        box = result.boxes.xyxy[best_person].cpu().numpy()  # [x1, y1, x2, y2]
        pts = xy[best_person]  # (17, 2)

        # Normalize coordinates by bbox
        box_w = max(float(box[2] - box[0]), 1.0)
        box_h = max(float(box[3] - box[1]), 1.0)

        seq[i, :, 0] = (pts[:, 0] - box[0]) / box_w
        seq[i, :, 1] = (pts[:, 1] - box[1]) / box_h
        seq[i, :, 2] = conf[best_person]

    cap.release()

    # Interpolate missing frames
    for j in range(17):
        valid = seq[:, j, 2] > 0
        if valid.any():
            for k in range(3):
                seq[:, j, k] = np.interp(
                    np.arange(n_frames),
                    np.flatnonzero(valid),
                    seq[valid, j, k],
                )

    # Flatten: (n_frames, 17, 3) -> (n_frames, 51)
    return seq.reshape(n_frames, 51)


def batch_extract_and_cache(
    manifest_csv: str | Path,
    root_dir: str | Path,
    output_npz: str | Path,
    n_frames: int = 16,
    imgsz: int = 384,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features from all videos listed in manifest and save to compressed .npz.
    """
    import pandas as pd
    from src.data import load_manifest

    root_dir = Path(root_dir)
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    df = load_manifest(manifest_csv, exercise_filter="squat")
    device = get_default_device()
    pose_model = YOLO("yolov8n-pose.pt")

    X_list, y_list, group_list, view_list, path_list = [], [], [], [], []
    failures = []

    print(f"Extracting features from {len(df)} clips using {device}...")
    for idx, row in enumerate(df.itertuples(index=False), 1):
        full_path = root_dir / row.video_path
        features = extract_pose_sequence(
            full_path, n_frames=n_frames, imgsz=imgsz, device=device, pose_model=pose_model
        )

        if features is None or features[:, 2::3].mean() < 0.05:
            failures.append(row.video_path)
            continue

        X_list.append(features)
        y_list.append(int(row.label == "good"))
        group_list.append(row.subject_id)
        view_list.append(row.view)
        path_list.append(row.video_path)

        if idx % 25 == 0 or idx == len(df):
            print(f"  Processed {idx}/{len(df)} clips...")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(group_list)
    views = np.asarray(view_list)
    paths = np.asarray(path_list)

    np.savez_compressed(output_npz, X=X, y=y, groups=groups, views=views, paths=paths)
    print(f"\nExtraction complete! Saved {len(X)} clips to {output_npz}")
    if failures:
        print(f"Failures ({len(failures)}): {failures}")

    return X, y, groups, views, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pose sequences from video dataset.")
    parser.add_argument("--manifest", type=str, default="data/metadata.csv", help="Path to manifest CSV")
    parser.add_argument("--root", type=str, default=".", help="Project root directory")
    parser.add_argument("--output", type=str, default="data/processed/squat_pose_16f_v1.npz", help="Output .npz path")
    parser.add_argument("--frames", type=int, default=16, help="Frames to sample per video")
    parser.add_argument("--imgsz", type=int, default=384, help="YOLO image resolution")
    args = parser.parse_args()

    batch_extract_and_cache(args.manifest, args.root, args.output, args.frames, args.imgsz)
