"""
Model comparison and evaluation script.
Reproduces Table 2 from the paper (page 11) with mAP scores for each YOLOv5 variant.
"""

import sys
import subprocess
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config

CFG = get_config()


def evaluate_all_action_models():
    """Evaluate all trained action models and compare."""
    yolov5_dir = CFG.paths.yolov5_repo
    data_yaml = CFG.paths.configs / "yolov5_actions.yaml"

    results = {}

    for variant in CFG.yolo.variants:
        weights = CFG.paths.weights / f"{variant}_actions.pt"

        if not weights.exists():
            print(f"  {variant}: weights not found, skipping.")
            results[variant] = {"status": "not_trained"}
            continue

        print(f"\nEvaluating {variant}...")

        cmd = [
            "python", str(yolov5_dir / "val.py"),
            "--data", str(data_yaml),
            "--weights", str(weights),
            "--img", str(CFG.yolo.img_size),
            "--batch", "16",
            "--task", "test",
            "--name", f"compare_{variant}",
            "--project", str(CFG.paths.outputs / "comparisons"),
            "--exist-ok",
            "--save-json",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse results
        if result.returncode == 0:
            # Try to extract metrics from output
            output = result.stdout
            metrics = _parse_val_output(output)
            metrics["status"] = "success"
            metrics["model_size_mb"] = round(
                weights.stat().st_size / (1024 * 1024), 1
            )
            results[variant] = metrics
            print(f"  {variant}: mAP@0.5={metrics.get('mAP50', 'N/A')}")
        else:
            results[variant] = {"status": "failed"}
            print(f"  {variant}: evaluation failed")

    # Create comparison table
    rows = []
    for variant, metrics in results.items():
        rows.append({
            "Model": variant,
            "Status": metrics.get("status", "unknown"),
            "mAP@0.5": metrics.get("mAP50", "-"),
            "mAP@0.5:0.95": metrics.get("mAP50_95", "-"),
            "Precision": metrics.get("precision", "-"),
            "Recall": metrics.get("recall", "-"),
            "Size (MB)": metrics.get("model_size_mb", "-"),
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print("MODEL COMPARISON TABLE (Similar to Paper Table 2)")
    print(f"{'='*70}")
    print(df.to_string(index=False))

    # Save
    csv_path = CFG.paths.outputs / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # Create visualization
    _plot_comparison(df)

    return df


def _parse_val_output(output: str) -> dict:
    """Parse YOLOv5 val.py output to extract metrics."""
    metrics = {}

    for line in output.split("\n"):
        line = line.strip()
        # Look for "all" summary line
        if line.startswith("all") or "all" in line:
            parts = line.split()
            try:
                # Typical format: all  images  labels  P  R  mAP@.5  mAP@.5:.95
                if len(parts) >= 7:
                    metrics["precision"] = float(parts[3])
                    metrics["recall"] = float(parts[4])
                    metrics["mAP50"] = float(parts[5])
                    metrics["mAP50_95"] = float(parts[6])
            except (ValueError, IndexError):
                pass

    return metrics


def _plot_comparison(df: pd.DataFrame):
    """Create comparison plots."""
    df_valid = df[df["Status"] == "success"].copy()
    if df_valid.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # mAP comparison
    if "mAP@0.5" in df_valid.columns:
        try:
            map_values = pd.to_numeric(df_valid["mAP@0.5"], errors="coerce")
            axes[0].bar(
                df_valid["Model"],
                map_values,
                color=["#2196F3", "#4CAF50", "#FF9800", "#f44336", "#9C27B0"],
            )
            axes[0].set_title("mAP@0.5 Comparison")
            axes[0].set_ylabel("mAP@0.5")
            axes[0].set_ylim(0, 1)

            for i, v in enumerate(map_values):
                if pd.notna(v):
                    axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center")
        except Exception:
            pass

    # Size vs mAP trade-off
    if "Size (MB)" in df_valid.columns:
        try:
            sizes = pd.to_numeric(df_valid["Size (MB)"], errors="coerce")
            map_values = pd.to_numeric(
                df_valid["mAP@0.5"], errors="coerce"
            )

            axes[1].scatter(sizes, map_values, s=100, c="#1E88E5", zorder=5)
            for i, row in df_valid.iterrows():
                axes[1].annotate(
                    row["Model"],
                    (
                        float(row["Size (MB)"]) if row["Size (MB)"] != "-" else 0,
                        float(row["mAP@0.5"]) if row["mAP@0.5"] != "-" else 0,
                    ),
                    textcoords="offset points",
                    xytext=(5, 5),
                )
            axes[1].set_title("Model Size vs mAP Trade-off")
            axes[1].set_xlabel("Model Size (MB)")
            axes[1].set_ylabel("mAP@0.5")
        except Exception:
            pass

    plt.tight_layout()
    plot_path = CFG.paths.outputs / "charts" / "model_comparison.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {plot_path}")


if __name__ == "__main__":
    evaluate_all_action_models()