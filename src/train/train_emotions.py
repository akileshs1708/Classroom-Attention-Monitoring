"""
Train YOLOv5s for emotion detection.

Uses AffectNet (5 classes) with MaskTheFace augmentation,
as described in Section 2.4.2 of the paper.

Target mAP@0.5 ≈ 0.877 (paper Figure 13-14, page 13-14).
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config

CFG = get_config()


def check_prerequisites():
    """Verify dataset and YOLOv5 are ready."""
    yolov5_dir = CFG.paths.yolov5_repo

    if not yolov5_dir.exists():
        print(f"Cloning YOLOv5...")
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/ultralytics/yolov5.git",
                str(yolov5_dir)
            ],
            check=True,
        )

    data_yaml = CFG.paths.configs / "yolov5_emotions.yaml"
    if not data_yaml.exists():
        CFG._write_emotion_yaml()

    train_dir = CFG.paths.emotions_images / "train"
    n_images = len(list(train_dir.rglob("*.*"))) if train_dir.exists() else 0
    print(f"[INFO] Emotion training images: {n_images}")

    if n_images == 0:
        raise FileNotFoundError(
            f"No images in {train_dir}. Run prepare_emotions_dataset.py first."
        )


def train_emotion_model(epochs: int = None, batch_size: int = None):
    """Train YOLOv5s for emotion detection."""
    if epochs is None:
        epochs = CFG.yolo.epochs_emotions
    if batch_size is None:
        batch_size = CFG.yolo.batch_size

    yolov5_dir = CFG.paths.yolov5_repo
    data_yaml = CFG.paths.configs / "yolov5_emotions.yaml"
    run_name = f"emotions_yolov5s_{datetime.now().strftime('%Y%m%d_%H%M')}"

    print(f"\n{'='*60}")
    print(f"Training YOLOv5s for Emotion Detection")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Classes: {CFG.emotions.classes}")
    print(f"{'='*60}")

    cmd = [
        "python", str(yolov5_dir / "train.py"),
        "--img", str(CFG.yolo.img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", str(data_yaml),
        "--weights", "yolov5s.pt",
        "--name", run_name,
        "--project", str(CFG.paths.outputs / "training"),
        "--patience", str(CFG.yolo.patience),
        "--exist-ok",
        "--cache",
    ]

    print(f"\nCommand: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        best_weights = (
            CFG.paths.outputs / "training" / run_name / "weights" / "best.pt"
        )
        if best_weights.exists():
            dst = CFG.paths.weights / "yolov5s_emotions.pt"
            shutil.copy2(str(best_weights), str(dst))
            print(f"\n[OK] Emotion model saved to: {dst}")
    else:
        print("[ERROR] Emotion model training failed.")

    return result.returncode == 0


def evaluate_emotion_model():
    """Evaluate the trained emotion model."""
    yolov5_dir = CFG.paths.yolov5_repo
    weights = CFG.paths.weights / "yolov5s_emotions.pt"
    data_yaml = CFG.paths.configs / "yolov5_emotions.yaml"

    if not weights.exists():
        print(f"[ERROR] Emotion model not found: {weights}")
        return

    cmd = [
        "python", str(yolov5_dir / "val.py"),
        "--data", str(data_yaml),
        "--weights", str(weights),
        "--img", str(CFG.yolo.img_size),
        "--batch", "16",
        "--task", "test",
        "--name", "eval_emotions",
        "--project", str(CFG.paths.outputs / "evaluations"),
        "--exist-ok",
        "--verbose",
    ]

    print("\nEvaluating emotion model...")
    subprocess.run(cmd, capture_output=False)


def main():
    print("=" * 60)
    print("YOLOv5 EMOTION DETECTION TRAINING")
    print("=" * 60)

    check_prerequisites()
    train_emotion_model()
    evaluate_emotion_model()

    print("\n[DONE] Emotion model training complete.")


if __name__ == "__main__":
    main()