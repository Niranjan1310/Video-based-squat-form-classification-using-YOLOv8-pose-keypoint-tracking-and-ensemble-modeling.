"""
Inference script for squat form classification on video files.
"""

from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Optional, Tuple, Any
import argparse
import cv2
import numpy as np
import torch
import joblib
from ultralytics import YOLO

from src.models import MODEL_REGISTRY, TCN, GRU, MLP
from src.pose_extraction import extract_pose_sequence, get_default_device
from src.train_champion_model import extract_champion_features


# COCO 17 Keypoint Skeleton Connections (indices for drawing lines)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6),                                     # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # Arms
    (5, 11), (6, 12), (11, 12),                 # Torso / Hips
    (11, 13), (13, 15), (12, 14), (14, 16)     # Legs
]


def detect_camera_view(seq: np.ndarray) -> str:
    """Estimate camera view ('front', 'side', 'diagonal') from keypoint spatial distribution."""
    p = seq[:, :34].reshape(-1, 17, 2)
    ankle_w = np.abs(p[:, 15, 0] - p[:, 16, 0]).mean()
    hip_w = np.abs(p[:, 11, 0] - p[:, 12, 0]).mean()
    if ankle_w < 0.40 and hip_w < 0.20:
        return "side"
    elif ankle_w > 0.58 and hip_w > 0.22:
        return "front"
    else:
        return "diagonal"


def predict_video(
    video_path: str | Path,
    checkpoint_path: Optional[str | Path] = None,
    imgsz: int = 384,
) -> dict:
    """
    Run full end-to-end prediction pipeline on a single video file.

    Returns:
        Dict containing prediction, probabilities, detected view, and biomechanics.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at {video_path}")

    # Auto-select checkpoint if not provided
    if checkpoint_path is None:
        if Path("outputs/champion_ensemble.joblib").exists():
            checkpoint_path = Path("outputs/champion_ensemble.joblib")
        elif Path("outputs/gru_squat_32f.pt").exists():
            checkpoint_path = Path("outputs/gru_squat_32f.pt")
        elif Path("outputs/gru_squat.pt").exists():
            checkpoint_path = Path("outputs/gru_squat.pt")
        else:
            raise FileNotFoundError("No checkpoint found in outputs/")
    else:
        checkpoint_path = Path(checkpoint_path)

    is_joblib = checkpoint_path.suffix == ".joblib"
    n_frames = 32 if is_joblib else 16

    # 1. Pose sequence extraction
    seq = extract_pose_sequence(video_path, n_frames=n_frames, imgsz=imgsz)
    if seq is None or seq[:, 2::3].mean() < 0.05:
        raise ValueError(f"Could not read or extract clear pose sequence from {video_path}")

    # 2. Detect camera view
    detected_view = detect_camera_view(seq)

    # 3. Model Inference
    if is_joblib:
        data = joblib.load(checkpoint_path)
        model = data["model"]
        model_name = "Champion Ensemble (ET+HGB+RF)"
        feat_vec = extract_champion_features(seq[None, ...], np.array([detected_view]))
        probs = model.predict_proba(feat_vec)[0]
        prob_bad, prob_good = float(probs[0]), float(probs[1])
        pred_idx = 1 if prob_good >= 0.5 else 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_name = ckpt.get("model_name", "GRU")
        n_frames = ckpt.get("frames", n_frames)
        model = MODEL_REGISTRY[model_name]()
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()

        tensor_input = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor_input)
            probs = logits.softmax(dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            prob_bad, prob_good = float(probs[0]), float(probs[1])

    label_map = {0: "Needs Correction (Bad Form)", 1: "Good Form"}
    return {
        "prediction": label_map[pred_idx],
        "is_good": bool(pred_idx == 1),
        "label_idx": pred_idx,
        "confidence": prob_good if pred_idx == 1 else prob_bad,
        "confidence_bad": prob_bad,
        "confidence_good": prob_good,
        "detected_view": detected_view,
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "n_frames": n_frames,
    }


def annotate_and_save_video(
    video_path: str | Path,
    output_path: str | Path,
    prediction_result: dict,
    imgsz: int = 384,
) -> None:
    """Render pose skeleton and prediction overlay onto video frames and save to file."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pose_model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening {video_path} for annotation")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    pred_text = prediction_result["prediction"]
    is_good = prediction_result["is_good"]
    conf = prediction_result["confidence"]
    box_color = (0, 200, 0) if is_good else (0, 0, 220)

    device = get_default_device()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        results = pose_model.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]

        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            kpts = results.keypoints.xy[0].cpu().numpy().astype(int)
            confs = results.keypoints.conf[0].cpu().numpy()

            for p1_idx, p2_idx in SKELETON_EDGES:
                if confs[p1_idx] > 0.3 and confs[p2_idx] > 0.3:
                    p1 = tuple(kpts[p1_idx])
                    p2 = tuple(kpts[p2_idx])
                    cv2.line(frame, p1, p2, (255, 128, 0), 2)

            for j, (x, y) in enumerate(kpts):
                if confs[j] > 0.3:
                    cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

        # Overlay UI card
        cv2.rectangle(frame, (20, 20), (450, 110), (20, 20, 20), -1)
        cv2.rectangle(frame, (20, 20), (450, 110), box_color, 2)
        cv2.putText(frame, f"Form: {pred_text}", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {conf*100:.1f}% ({prediction_result['detected_view']} view)", (35, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Annotated video saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single squat video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.joblib or .pt)")
    parser.add_argument("--annotate", action="store_true", help="Save annotated video with skeleton overlay")
    parser.add_argument("--output", type=str, default="outputs/annotated_squat.mp4", help="Path to annotated video")
    args = parser.parse_args()

    result = predict_video(args.video, args.checkpoint)
    print("\n" + "=" * 55)
    print("SQUAT FORM INFERENCE RESULTS")
    print("=" * 55)
    print(f"Video:            {args.video}")
    print(f"Prediction:       {result['prediction']}")
    print(f"Confidence:       {result['confidence']*100:.1f}%")
    print(f"Detected View:    {result['detected_view']}")
    print(f"Probabilities:    Good={result['confidence_good']*100:.1f}% | Bad={result['confidence_bad']*100:.1f}%")
    print(f"Model:            {result['model_name']}")
    print("=" * 55)

    if args.annotate:
        annotate_and_save_video(args.video, args.output, result)
