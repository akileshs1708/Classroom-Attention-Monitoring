"""
Download and organize all three datasets:
1. SCB-05 (Student Classroom Behavior) - for action detection
2. AffectNet (YOLO format) - for emotion detection
3. FER2013 - supplementary emotion data

Requires: kaggle API configured (~/.kaggle/kaggle.json)
Alternatively, download manually and place in data/ folders.
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config

CFG = get_config()


def download_kaggle_dataset(dataset_slug: str, output_dir: Path):
    """Download a dataset from Kaggle using the kaggle CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_slug}")
    print(f"To: {output_dir}")
    print(f"{'='*60}")

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", dataset_slug,
                "-p", str(output_dir),
                "--unzip"
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"[OK] Downloaded {dataset_slug}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Kaggle download failed: {e.stderr}")
        print(
            f"[INFO] Please download manually from "
            f"https://www.kaggle.com/datasets/{dataset_slug}"
        )
        print(f"[INFO] Extract to: {output_dir}")
    except FileNotFoundError:
        print("[ERROR] kaggle CLI not found. Install with: pip install kaggle")
        print(
            f"[INFO] Download manually: "
            f"https://www.kaggle.com/datasets/{dataset_slug}"
        )
        print(f"[INFO] Extract to: {output_dir}")


def download_scb05():
    """Download SCB-05 Student Classroom Behavior dataset."""
    download_kaggle_dataset(
        "shreyasudaya/scb-05-dataset",
        CFG.paths.scb05_raw
    )


def download_affectnet():
    """Download AffectNet in YOLO format."""
    download_kaggle_dataset(
        "fatihkgg/affectnet-yolo-format",
        CFG.paths.affectnet_raw
    )


def download_fer2013():
    """Download FER2013 emotion dataset."""
    download_kaggle_dataset(
        "msambare/fer2013",
        CFG.paths.fer2013_raw
    )


def verify_downloads():
    """Check that datasets exist and print summaries."""
    datasets = {
        "SCB-05": CFG.paths.scb05_raw,
        "AffectNet": CFG.paths.affectnet_raw,
        "FER2013": CFG.paths.fer2013_raw,
    }

    print(f"\n{'='*60}")
    print("Dataset Verification")
    print(f"{'='*60}")

    for name, path in datasets.items():
        if path.exists():
            # Count files
            files = list(path.rglob("*"))
            image_files = [
                f for f in files
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
            label_files = [
                f for f in files
                if f.suffix.lower() in {".txt", ".csv", ".xml"}
            ]
            print(
                f"  {name}: {len(image_files)} images, "
                f"{len(label_files)} label files"
            )
        else:
            print(f"  {name}: NOT FOUND at {path}")


def download_haar_cascade():
    """Download OpenCV Haar Cascade XML if not present."""
    cascade_path = CFG.paths.haar_cascade
    if cascade_path.exists():
        print(f"[OK] Haar cascade already exists: {cascade_path}")
        return

    import urllib.request

    url = (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "data/haarcascades/haarcascade_frontalface_default.xml"
    )
    cascade_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Haar cascade from {url}...")
    urllib.request.urlretrieve(url, str(cascade_path))
    print(f"[OK] Saved to {cascade_path}")


def main():
    """Download all datasets."""
    CFG.paths.create_all()

    print("=" * 60)
    print("CLASSROOM MONITORING SYSTEM - DATASET DOWNLOADER")
    print("=" * 60)

    # Download datasets
    download_scb05()
    download_affectnet()
    download_fer2013()

    # Download Haar cascade
    download_haar_cascade()

    # Verify
    verify_downloads()

    print("\n[DONE] Dataset download complete.")
    print("Next step: Run prepare_actions_dataset.py")


if __name__ == "__main__":
    main()