"""
FastAPI backend for squat form classification.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import os
import tempfile
import numpy as np
import torch
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.models import MODEL_REGISTRY, GRU, TCN, MLP
from src.pose_extraction import extract_pose_sequence, get_default_device
from src.train_champion_model import extract_champion_features


# FastAPI app
app = FastAPI(
    title="Squat Form Classifier API",
    description="Squat exercise form quality classification from video clips.",
    version="1.0.0",
)

# Enable CORS for web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
WEB_DIR = BASE_DIR / "web"

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model state
loaded_model: Optional[Any] = None
loaded_model_type: str = "ensemble"  # "ensemble" or "pytorch"
loaded_model_info: Dict[str, Any] = {
    "name": "Champion Ensemble",
    "frames": 32,
    "checkpoint": "outputs/champion_ensemble.joblib",
    "device": str(device),
}


def detect_camera_view(seq: np.ndarray) -> str:
    """
    Estimate camera view ('front', 'side', 'diagonal') from keypoint spatial distribution.
    Front view has wide ankle/hip separation; side view has small lateral separation.
    """
    p = seq[:, :34].reshape(-1, 17, 2)
    # Lateral width of ankles (15, 16) and hips (11, 12)
    ankle_w = np.abs(p[:, 15, 0] - p[:, 16, 0]).mean()
    hip_w = np.abs(p[:, 11, 0] - p[:, 12, 0]).mean()

    if ankle_w < 0.40 and hip_w < 0.20:
        return "side"
    elif ankle_w > 0.58 and hip_w > 0.22:
        return "front"
    else:
        return "diagonal"


def load_model_on_startup():
    """Load champion joblib ensemble or best available PyTorch checkpoint."""
    global loaded_model, loaded_model_type, loaded_model_info

    joblib_path = OUTPUTS_DIR / "champion_ensemble.joblib"
    if joblib_path.exists():
        try:
            data = joblib.load(joblib_path)
            loaded_model = data["model"]
            loaded_model_type = "ensemble"
            loaded_model_info = {
                "name": "Champion Ensemble (ET+HGB+RF)",
                "frames": 32,
                "checkpoint": str(joblib_path.relative_to(BASE_DIR)),
                "device": "CPU (Multi-thread)",
                "features": data.get("features", "biomechanical_kinematic"),
            }
            print(f"[API] Loaded Champion Ensemble from {joblib_path}")
            return
        except Exception as e:
            print(f"[API] Warning: Could not load joblib ensemble: {e}")

    # Fallback to PyTorch checkpoint
    candidates = [
        OUTPUTS_DIR / "gru_squat_32f.pt",
        OUTPUTS_DIR / "gru_squat.pt",
        OUTPUTS_DIR / "tcn_squat_32f.pt",
        OUTPUTS_DIR / "tcn_squat.pt",
    ]
    for ckpt_path in candidates:
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model_name = ckpt.get("model_name", "GRU")
                frames = ckpt.get("frames", 16)
                model = MODEL_REGISTRY[model_name]()
                model.load_state_dict(ckpt["state_dict"])
                model.to(device)
                model.eval()

                loaded_model = model
                loaded_model_type = "pytorch"
                loaded_model_info = {
                    "name": model_name,
                    "frames": frames,
                    "checkpoint": str(ckpt_path.relative_to(BASE_DIR)),
                    "device": str(device),
                }
                print(f"[API] Loaded PyTorch {model_name} ({frames}f) from {ckpt_path}")
                return
            except Exception as e:
                print(f"[API] Warning: Failed loading {ckpt_path}: {e}")

    print("[API] Warning: No models found during startup.")


@app.on_event("startup")
async def startup_event():
    load_model_on_startup()


@app.get("/api/health")
async def health_check():
    """Return service health and loaded model status."""
    return {
        "status": "healthy" if loaded_model is not None else "degraded",
        "model": loaded_model_info,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/api/predict")
async def predict_squat_form(file: UploadFile = File(...)):
    """
    Accept an uploaded squat video, run pose sequence extraction,
    and return predicted good/bad form classification with confidence.
    """
    if loaded_model is None:
        load_model_on_startup()
        if loaded_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model classifier is not loaded.",
            )

    # 1. Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Save uploaded bytes to a safe temporary file
    temp_dir = Path(tempfile.gettempdir()) / "squat_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"upload_{os.getpid()}_{file.filename}"

    try:
        total_size = 0
        with open(temp_file, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Video file size exceeds maximum 100 MB limit.",
                    )
                buffer.write(chunk)

        # 3. Extract pose sequence (32 frames for champion model)
        n_frames = loaded_model_info.get("frames", 32)
        seq = extract_pose_sequence(temp_file, n_frames=n_frames, imgsz=384)
        if seq is None or seq[:, 2::3].mean() < 0.05:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not detect a clear person or pose sequence in the video.",
            )

        # 4. Detect camera view
        detected_view = detect_camera_view(seq)

        # 5. Run Model Inference
        if loaded_model_type == "ensemble":
            # Extract champion biomechanical & kinematic feature vector (1, 348)
            seq_batch = seq[None, ...]  # (1, 32, 51)
            views_arr = np.array([detected_view])
            feat_vec = extract_champion_features(seq_batch, views_arr)

            probs = loaded_model.predict_proba(feat_vec)[0]  # [prob_bad, prob_good]
            prob_bad = float(probs[0])
            prob_good = float(probs[1])
            pred_idx = 1 if prob_good >= 0.5 else 0

        else:
            # PyTorch sequential fallback
            tensor_input = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = loaded_model(tensor_input)
                probs = logits.softmax(dim=1).cpu().numpy()[0]
                pred_idx = int(logits.argmax(dim=1).cpu().numpy()[0])
            prob_bad = float(probs[0])
            prob_good = float(probs[1])

        prediction_label = "Good Form" if pred_idx == 1 else "Needs Correction"
        confidence = prob_good if pred_idx == 1 else prob_bad

        return {
            "prediction": prediction_label,
            "is_good": bool(pred_idx == 1),
            "confidence": round(confidence, 4),
            "confidence_percent": f"{confidence * 100:.1f}%",
            "probabilities": {
                "good": round(prob_good, 4),
                "bad": round(prob_bad, 4),
            },
            "model": loaded_model_info["name"],
            "detected_view": detected_view,
            "frames_analyzed": n_frames,
            "filename": file.filename,
        }

    finally:
        # Guaranteed cleanup of temporary upload file
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


# Serve frontend static assets if web/ directory exists
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
