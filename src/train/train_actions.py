"""
Train YOLOv5 models for student behavior/action detection.

Trains all 5 variants (n, s, m, l, x) as described in the paper,
then evaluates and selects the best model.

Reference: Paper Table 2 (page 11) shows mAP results for each variant.
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config

CFG = get_config()


def check_prerequisites():
    """Verify YOLOv5 repo and dataset are ready."""
    yolov5_dir = CFG.paths.yolov5_repo

    if not yolov5_dir.exists():
        print(f"[INFO] Cloning YOLOv5 to {yolov5_dir}...")
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/ultralytics/yolov5.git",
                str(yolov5_dir)
            ],
            check=True,
        )
        # Install requirements
        subprocess.run(
            ["pip", "install", "-r", str(yolov5_dir / "requirements.txt")],
            check=True,
        )
    else:
        print(f"[OK] YOLOv5 found at {yolov5_dir}")

    # Check dataset config
    data_yaml = CFG.paths.configs / "yolov5_actions.yaml"
    if not data_yaml.exists():
        CFG._write_action_yaml()

    # Check for training images
    train_dir = CFG.paths.actions_images / "train"
    n_images = len(list(train_dir.rglob("*.*"))) if train_dir.exists() else 0
    print(f"[INFO] Training images: {n_images}")
    if n_images == 0:
        raise FileNotFoundError(
            f"No training images found in {train_dir}. "
            f"Run prepare_actions_dataset.py first."
        )

    return True


def train_single_variant(variant: str, epochs: int = None, batch_size: int = None):
    """
    Train a single YOLOv5 variant for action detection.

    Args:
        variant: one of 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
        epochs: override number of epochs
        batch_size: override batch size
    """
    if epochs is None:
        epochs = CFG.yolo.epochs_actions
    if batch_size is None:
        batch_size = CFG.yolo.batch_size

    yolov5_dir = CFG.paths.yolov5_repo
    data_yaml = CFG.paths.configs / "yolov5_actions.yaml"
    run_name = f"actions_{variant}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    print(f"\n{'='*60}")
    print(f"Training {variant} for Action Detection")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {CFG.yolo.img_size}")
    print(f"  Data: {data_yaml}")
    print(f"{'='*60}")

    # Build training command
    cmd = [
        "python", str(yolov5_dir / "train.py"),
        "--img", str(CFG.yolo.img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", str(data_yaml),
        "--weights", f"{variant}.pt",
        "--name", run_name,
        "--project", str(CFG.paths.outputs / "training"),
        "--patience", str(CFG.yolo.patience),
        "--exist-ok",
        "--cache",
        "--device 0",
    ]

    # Add augmentation flags
    if CFG.yolo.augment:
        cmd.extend([
            "--hyp", str(yolov5_dir / "data" / "hyps" / "hyp.scratch-low.yaml"),
        ])

    print(f"\nCommand: {' '.join(cmd)}")

    # Run training
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        # Copy best weights
        best_weights = (
            CFG.paths.outputs / "training" / run_name / "weights" / "best.pt"
        )
        if best_weights.exists():
            dst = CFG.paths.weights / f"{variant}_actions.pt"
            shutil.copy2(str(best_weights), str(dst))
            print(f"\n[OK] Best weights saved to: {dst}")
        else:
            print(f"[WARN] best.pt not found at {best_weights}")
    else:
        print(f"[ERROR] Training failed for {variant}")

    return result.returncode == 0


def train_all_variants():
    """Train all YOLOv5 variants as described in the paper."""
    results = {}

    # Adjust batch sizes for larger models
    batch_sizes = {
        "yolov5n": 2,
        "yolov5s": 2,
        "yolov5m": 2,
        "yolov5l": 2,
        "yolov5x": 2,
    }

    for variant in CFG.yolo.variants:
        batch_size = batch_sizes.get(variant, CFG.yolo.batch_size)
        success = train_single_variant(variant, batch_size=batch_size)
        results[variant] = success

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for variant, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {variant}: {status}")

    return results


def evaluate_model(variant: str):
    """Evaluate a trained model on the test set."""
    yolov5_dir = CFG.paths.yolov5_repo
    weights = CFG.paths.weights / f"{variant}_actions.pt"
    data_yaml = CFG.paths.configs / "yolov5_actions.yaml"

    if not weights.exists():
        print(f"[WARN] Weights not found: {weights}")
        return None

    cmd = [
        "python", str(yolov5_dir / "val.py"),
        "--data", str(data_yaml),
        "--weights", str(weights),
        "--img", str(CFG.yolo.img_size),
        "--batch", "16",
        "--task", "test",
        "--name", f"eval_{variant}_actions",
        "--project", str(CFG.paths.outputs / "evaluations"),
        "--exist-ok",
        "--verbose",
    ]

    print(f"\nEvaluating {variant}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"[OK] Evaluation complete for {variant}")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    else:
        print(f"[ERROR] Evaluation failed: {result.stderr}")

    return result.returncode == 0


def compare_variants():
    """
    Compare all trained variants and produce a summary table
    similar to Table 2 in the paper.
    """
    print(f"\n{'='*60}")
    print("MODEL COMPARISON (Action Detection)")
    print(f"{'='*60}")
    print(f"{'Variant':<12} {'Weights (MB)':<14} {'Status':<10}")
    print("-" * 40)

    for variant in CFG.yolo.variants:
        weights = CFG.paths.weights / f"{variant}_actions.pt"
        if weights.exists():
            size_mb = weights.stat().st_size / (1024 * 1024)
            print(f"{variant:<12} {size_mb:<14.1f} {'OK':<10}")
        else:
            print(f"{variant:<12} {'N/A':<14} {'NOT FOUND':<10}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("YOLOv5 ACTION DETECTION TRAINING")
    print("=" * 60)

    # Check prerequisites
    check_prerequisites()

    # Ask user what to do
    print("\nOptions:")
    print("  1. Train all variants (n, s, m, l, x)")
    print("  2. Train only YOLOv5s (recommended for testing)")
    print("  3. Evaluate existing models")
    print("  4. Compare variants")

    choice = input("\nChoice [2]: ").strip() or "2"

    if choice == "1":
        train_all_variants()
    elif choice == "2":
        train_single_variant("yolov5s")
    elif choice == "3":
        for variant in CFG.yolo.variants:
            evaluate_model(variant)
    elif choice == "4":
        compare_variants()

    compare_variants()
    print("\n[DONE] Training pipeline complete.")


if __name__ == "__main__":
    main()