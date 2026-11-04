# Squat Form Classification — Project Plan

## Overview
Build a computer vision system that classifies squat exercise form quality (good form vs. needs correction) from multi-view video clips. The system uses a pretrained 2D pose estimator to track body keypoints over time, extracts kinematic and biomechanical features, and trains classification models evaluated strictly on held-out participants.

## Dataset
- **Source**: [Mendeley Data — Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad Form Labels](https://data.mendeley.com/datasets/kgbb3yn47p/1) (DOI: `10.17632/kgbb3yn47p.1`, CC BY 4.0).
- **Scope**: Squat clips from 25 participants across 3 camera views (front, side, diagonal).
- **Labels**: Binary (`good` vs. `bad` form).
- **Manifest**: `data/metadata.csv` mapping video paths to subject ID, exercise, camera view, and form label.

## Technical Approach
1. **Pose Extraction**:
   - Sample frames uniformly across each video clip (16 or 32 frames).
   - Use pretrained YOLOv8n-pose to detect 17 COCO body keypoints per frame.
   - Normalize (x, y) coordinates relative to person bounding box; track detection confidence.
   - Linearly interpolate any missed frames and cache feature arrays to `data/processed/`.

2. **Validation Strategy**:
   - Strict participant-grouped splits (60% train / 20% validation / 20% test) using `GroupShuffleSplit`.
   - 5-fold grouped cross-validation (`GroupKFold`) so all participants are evaluated on held-out folds.
   - Assert zero subject overlap between splits to ensure models generalize to unseen people.

3. **Modeling & Benchmarks**:
   - **Majority baseline**: Class prior baseline.
   - **MLP Baseline**: Middle-frame static pose classification.
   - **Sequence Models**: Bidirectional GRU and Temporal Convolutional Network (TCN).
   - **Champion Biomechanical Ensemble**: 348D feature vector (knee angle, hip depth, torso lean, valgus ratio, velocities, bottom-inflection frame) with soft-voting ensemble (`ExtraTrees` + `HistGradientBoosting` + `RandomForest`).

4. **Evaluation Metrics**:
   - Primary: Macro-F1 and Balanced Accuracy (accounts for class distribution).
   - Secondary: Precision, Recall, Confusion Matrix, and camera view breakdown.

5. **Deployment & Interface**:
   - FastAPI backend service (`src/api.py`) with `/api/predict` and `/api/health`.
   - Web application (`web/`) for video drag-and-drop and real-time form feedback.
   - CLI inference tool (`src/inference.py`) with optional skeleton video annotation.

## Project Structure
```text
DL/
├── data/
│   ├── metadata.csv          # Clip annotations and labels
│   ├── raw/                  # Video clips
│   └── processed/            # Cached pose feature tensors (.npz)
├── notebooks/                # Modular exploration & tuning notebooks
│   ├── 01_data_inspection.ipynb
│   ├── 02_pose_extraction.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── exercise_form_tcn.ipynb
├── src/                      # Modular Python package
│   ├── models.py             # MLP, GRU, TCN architectures
│   ├── pose_extraction.py    # YOLOv8n-pose feature extraction
│   ├── preprocessing.py      # Splits, DataLoaders, class weights
│   ├── train.py              # PyTorch training & cross-validation
│   ├── train_champion_model.py # Biomechanical feature extraction & ensemble
│   ├── evaluate.py           # Evaluation metrics & diagnostics
│   ├── inference.py          # CLI prediction and video overlay
│   └── api.py                # FastAPI backend service
├── web/                      # Web frontend (HTML, CSS, JS)
├── outputs/                  # Saved models, metrics, and figures
├── requirements.txt
└── README.md
```

## Milestones
- [x] Download & inspect dataset; generate metadata manifest.
- [x] Pose extraction pipeline with YOLOv8n-pose and feature caching.
- [x] Participant-grouped splits and zero-leakage verification.
- [x] Implement and benchmark baseline MLP, GRU, and TCN models.
- [x] Engineer 348D biomechanical features and train Champion voting ensemble.
- [x] 5-fold cross-validation and held-out participant evaluation.
- [x] Build CLI inference demo with skeleton video export.
- [x] Build and test FastAPI backend and web interface.
