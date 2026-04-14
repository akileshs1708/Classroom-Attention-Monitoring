"""
Prepare the emotion dataset for YOLOv5 training.

FIXED SAMPLE SIZES PER CLASS:
  - Train: 500 images per class
  - Val:   125 images per class
  - Test:   75 images per class
  - Total: 700 images per class × 5 classes = 3,500 images

Uses two sources:
  1. AffectNet (YOLO format from Kaggle) — primary source
  2. FER2013 (classification format) — supplementary

Target emotions (5 classes):
  [0] angry
  [1] happy
  [2] neutral
  [3] sad
  [4] surprise

Steps:
  1. Load AffectNet YOLO-format data (already has bounding boxes)
  2. Convert FER2013 from classification to detection format
  3. Filter to 5 target emotions
  4. Pool all data per class, sample exactly 500/125/75
  5. Apply MaskTheFace augmentation for masked faces
  6. Verify final dataset
"""

import os
import sys
import shutil
import random
import csv
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config

CFG = get_config()


# ══════════════════════════════════════════════
# FIXED SAMPLE SIZES PER CLASS
# ══════════════════════════════════════════════
SAMPLES_PER_CLASS = {
    "train": 500,
    "val": 125,
    "test": 75,
}
# Total per class = 700
# Total dataset = 700 × 5 classes = 3,500 images


# ══════════════════════════════════════════════
# Dataset Exploration
# ══════════════════════════════════════════════
def explore_affectnet() -> dict:
    """
    Explore the AffectNet YOLO format dataset structure.
    Returns info dict or None if not found.
    """
    root = CFG.paths.affectnet_raw
    print(f"\n{'='*70}")
    print(f"EXPLORING AFFECTNET at: {root}")
    print(f"{'='*70}")

    if not root.exists():
        print("[ERROR] AffectNet not found.")
        print("  Download: https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format")
        return None

    info = {
        "root": root,
        "original_classes": None,
        "yaml_data": None,
        "images_by_split": {},
        "labels_by_split": {},
    }

    # Print folder structure
    for item in sorted(root.iterdir()):
        if item.is_dir():
            sub_count = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  DIR:  {item.name}/ ({sub_count} files)")
            for sub in sorted(item.iterdir())[:5]:
                if sub.is_dir():
                    print(f"    DIR:  {sub.name}/ ({sum(1 for _ in sub.rglob('*') if _.is_file())} files)")
                else:
                    print(f"    FILE: {sub.name}")
        else:
            print(f"  FILE: {item.name} ({item.stat().st_size // 1024}KB)")

    # Find YAML/classes config
    yaml_files = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))
    classes_files = list(root.rglob("classes.txt"))

    original_classes = None

    for yf in yaml_files:
        try:
            with open(yf, "r") as f:
                data = yaml.safe_load(f)
            if data and "names" in data:
                names = data["names"]
                if isinstance(names, dict):
                    original_classes = {int(k): v for k, v in names.items()}
                elif isinstance(names, list):
                    original_classes = {i: n for i, n in enumerate(names)}
                info["yaml_data"] = data
                print(f"\n  YAML ({yf.name}): {original_classes}")
                break
        except Exception:
            pass

    if original_classes is None and classes_files:
        with open(classes_files[0], "r") as f:
            classes = [line.strip() for line in f if line.strip()]
        original_classes = {i: c for i, c in enumerate(classes)}
        print(f"\n  classes.txt: {original_classes}")

    if original_classes is None:
        # Default AffectNet classes
        original_classes = {
            0: "angry", 1: "contempt", 2: "disgust", 3: "fear",
            4: "happy", 5: "sad", 6: "surprise", 7: "neutral",
        }
        print(f"\n  Using default AffectNet classes: {original_classes}")

    info["original_classes"] = original_classes

    # Find images and labels by split
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in ["train", "val", "test", "valid"]:
        # Try multiple directory structures
        possible_img_dirs = [
            root / "images" / split,
            root / split / "images",
            root / split,
            root / "images",
        ]
        possible_lbl_dirs = [
            root / "labels" / split,
            root / split / "labels",
            root / split,
            root / "labels",
        ]

        for img_dir in possible_img_dirs:
            if img_dir.exists():
                images = [f for f in img_dir.rglob("*") if f.suffix.lower() in image_exts]
                if images:
                    mapped_split = "val" if split == "valid" else split
                    info["images_by_split"][mapped_split] = (img_dir, images)
                    print(f"  Found {len(images)} images for {mapped_split} in {img_dir}")
                    break

        for lbl_dir in possible_lbl_dirs:
            if lbl_dir.exists():
                labels = [
                    f for f in lbl_dir.rglob("*.txt")
                    if f.name not in ("classes.txt", "notes.txt")
                ]
                if labels:
                    mapped_split = "val" if split == "valid" else split
                    info["labels_by_split"][mapped_split] = (lbl_dir, labels)
                    print(f"  Found {len(labels)} labels for {mapped_split} in {lbl_dir}")
                    break

    # If no splits found, check for flat structure
    if not info["images_by_split"]:
        all_images = [f for f in root.rglob("*") if f.suffix.lower() in image_exts]
        all_labels = [
            f for f in root.rglob("*.txt")
            if f.name not in ("classes.txt", "notes.txt")
        ]
        if all_images:
            info["images_by_split"]["all"] = (root, all_images)
            print(f"  Found {len(all_images)} images (flat/unsplit)")
        if all_labels:
            info["labels_by_split"]["all"] = (root, all_labels)
            print(f"  Found {len(all_labels)} labels (flat/unsplit)")

    # Sample label content
    for split, (lbl_dir, labels) in info["labels_by_split"].items():
        if labels:
            sample = labels[0]
            with open(sample, "r") as f:
                lines = [l.strip() for l in f.readlines()[:3]]
            print(f"  Sample label ({split}, {sample.name}): {lines}")

            # Count class IDs
            class_ids = set()
            for lbl in labels[:200]:
                try:
                    with open(lbl, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
                except Exception:
                    pass
            print(f"  Class IDs in {split}: {sorted(class_ids)}")

    return info


def explore_fer2013() -> dict:
    """
    Explore FER2013 dataset structure.
    FER2013 is typically: train/<emotion_name>/*.jpg (48×48 grayscale)
    """
    root = CFG.paths.fer2013_raw
    print(f"\n{'='*70}")
    print(f"EXPLORING FER2013 at: {root}")
    print(f"{'='*70}")

    if not root.exists():
        print("[ERROR] FER2013 not found.")
        print("  Download: https://www.kaggle.com/datasets/msambare/fer2013")
        return None

    info = {
        "root": root,
        "classes_by_split": {},
    }

    for item in sorted(root.iterdir()):
        if item.is_dir():
            print(f"  DIR: {item.name}/")
            for sub in sorted(item.iterdir()):
                if sub.is_dir():
                    n_files = len([
                        f for f in sub.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    ])
                    print(f"    {sub.name}/: {n_files} images")

                    split = item.name.lower()
                    if split not in info["classes_by_split"]:
                        info["classes_by_split"][split] = {}
                    info["classes_by_split"][split][sub.name.lower()] = n_files
        else:
            print(f"  FILE: {item.name}")

    return info


# ══════════════════════════════════════════════
# Class Mapping
# ══════════════════════════════════════════════
def build_affectnet_remapping(original_classes: dict) -> dict:
    """
    Build mapping from AffectNet original class IDs to our 5-class emotion IDs.

    Our classes: [angry(0), happy(1), neutral(2), sad(3), surprise(4)]

    Merge rules:
      angry    → angry(0)
      contempt → angry(0)
      disgust  → angry(0)
      fear     → surprise(4)
      happy    → happy(1)
      sad      → sad(3)
      surprise → surprise(4)
      neutral  → neutral(2)
    """
    our_classes = CFG.emotions.classes

    merge_rules = {
        "angry": "angry",
        "contempt": "angry",
        "disgust": "angry",
        "fear": "surprise",
        "happy": "happy",
        "sad": "sad",
        "surprise": "surprise",
        "neutral": "neutral",
    }

    remapping = {}

    for orig_id, orig_name in original_classes.items():
        orig_lower = orig_name.lower().strip()
        if orig_lower in merge_rules:
            target_name = merge_rules[orig_lower]
            target_idx = our_classes.index(target_name)
            remapping[int(orig_id)] = target_idx
            print(f"    MAP: {orig_id}:{orig_name} → {target_idx}:{target_name}")
        else:
            # Try fuzzy match
            matched = False
            for key, val in merge_rules.items():
                if key in orig_lower or orig_lower in key:
                    target_idx = our_classes.index(val)
                    remapping[int(orig_id)] = target_idx
                    print(f"    MAP (fuzzy): {orig_id}:{orig_name} → {target_idx}:{val}")
                    matched = True
                    break
            if not matched:
                # Default to neutral
                remapping[int(orig_id)] = our_classes.index("neutral")
                print(f"    MAP (default): {orig_id}:{orig_name} → neutral")

    return remapping


# ══════════════════════════════════════════════
# Data Collection from AffectNet
# ══════════════════════════════════════════════
def collect_affectnet_pairs(affectnet_info: dict) -> dict:
    """
    Collect all image-label pairs from AffectNet, grouped by our 5 target classes.

    Returns:
        dict mapping class_name -> list of (img_path, lbl_path, remapping) tuples
    """
    if affectnet_info is None:
        return {}

    original_classes = affectnet_info["original_classes"]
    remapping = build_affectnet_remapping(original_classes)

    our_classes = CFG.emotions.classes
    class_pairs = {cls: [] for cls in our_classes}  # class_name -> [(img, lbl, remap)]

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    print(f"\n  Collecting AffectNet pairs by class...")

    # Process each available split
    for split, (img_dir, images) in affectnet_info["images_by_split"].items():
        # Find corresponding labels
        lbl_info = affectnet_info["labels_by_split"].get(split)
        if lbl_info is None:
            print(f"    [WARN] No labels for split '{split}', skipping.")
            continue

        lbl_dir, labels = lbl_info

        # Build label lookup by stem
        labels_by_stem = {}
        for lbl in labels:
            labels_by_stem[lbl.stem] = lbl

        paired_count = 0
        for img_path in images:
            lbl_path = labels_by_stem.get(img_path.stem)
            if lbl_path is None:
                continue

            # Read label to determine which of our classes it maps to
            try:
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            orig_cls = int(parts[0])
                            if orig_cls in remapping:
                                target_idx = remapping[orig_cls]
                                target_name = our_classes[target_idx]
                                class_pairs[target_name].append(
                                    (img_path, lbl_path, remapping)
                                )
                                paired_count += 1
                            break  # only use first annotation per file
            except Exception:
                pass

        print(f"    Split '{split}': {paired_count} paired")

    # Print per-class counts
    print(f"\n  AffectNet pairs by class:")
    for cls_name in our_classes:
        print(f"    {cls_name:12s}: {len(class_pairs[cls_name]):6d}")

    return class_pairs


# ══════════════════════════════════════════════
# Data Collection from FER2013
# ══════════════════════════════════════════════
def collect_fer2013_pairs(fer_info: dict) -> dict:
    """
    Collect FER2013 images grouped by our 5 target classes.
    FER2013 is classification format (folder per class, 48×48 images).
    We treat the entire image as one face (YOLO: cx=0.5, cy=0.5, w=0.9, h=0.9).

    Returns:
        dict mapping class_name -> list of (img_path, None, remap) tuples
        (lbl_path=None because we generate labels on-the-fly)
    """
    if fer_info is None:
        return {}

    our_classes = CFG.emotions.classes
    fer_mapping = CFG.emotions.fer2013_to_emotion

    class_pairs = {cls: [] for cls in our_classes}

    print(f"\n  Collecting FER2013 pairs by class...")

    root = fer_info["root"]

    for split_dir_name in ["train", "test"]:
        split_dir = root / split_dir_name
        if not split_dir.exists():
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            fer_class = class_dir.name.lower()
            if fer_class not in fer_mapping:
                print(f"    [SKIP] FER class '{fer_class}' not in mapping")
                continue

            target_class = fer_mapping[fer_class]
            target_idx = our_classes.index(target_class)
            remap = {0: target_idx}  # dummy remap for full-image label

            images = sorted([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ])

            for img_path in images:
                class_pairs[target_class].append(
                    (img_path, None, remap)
                )

        print(f"    Split '{split_dir_name}' processed")

    # Print counts
    print(f"\n  FER2013 pairs by class:")
    for cls_name in our_classes:
        print(f"    {cls_name:12s}: {len(class_pairs[cls_name]):6d}")

    return class_pairs


# ══════════════════════════════════════════════
# Pool and Sample
# ══════════════════════════════════════════════
def pool_and_sample(
    affectnet_pairs: dict,
    fer2013_pairs: dict,
) -> dict:
    """
    Pool data from all sources and sample exactly
    SAMPLES_PER_CLASS per class per split.

    Priority: AffectNet first (has real YOLO labels), FER2013 as supplement.

    Returns:
        dict mapping split_name -> list of (img_path, lbl_path_or_None, remap, class_name, source)
    """
    our_classes = CFG.emotions.classes
    total_needed = sum(SAMPLES_PER_CLASS.values())  # 700 per class

    random.seed(42)

    print(f"\n{'='*70}")
    print(f"POOLING AND SAMPLING EMOTION DATA")
    print(f"  Per class: train={SAMPLES_PER_CLASS['train']}, "
          f"val={SAMPLES_PER_CLASS['val']}, test={SAMPLES_PER_CLASS['test']}")
    print(f"  Total per class: {total_needed}")
    print(f"  Total dataset: {total_needed * len(our_classes)}")
    print(f"{'='*70}")

    split_data = {"train": [], "val": [], "test": []}

    for cls_name in our_classes:
        # Pool from all sources, prioritize AffectNet
        affectnet_items = affectnet_pairs.get(cls_name, [])
        fer_items = fer2013_pairs.get(cls_name, [])

        # Tag with source
        tagged_affectnet = [
            (img, lbl, remap, cls_name, "affectnet")
            for img, lbl, remap in affectnet_items
        ]
        tagged_fer = [
            (img, lbl, remap, cls_name, "fer2013")
            for img, lbl, remap in fer_items
        ]

        # AffectNet first, then FER2013
        pool = tagged_affectnet + tagged_fer
        random.shuffle(pool)

        available = len(pool)

        # Calculate actual counts
        n_train = min(SAMPLES_PER_CLASS["train"], available)
        remaining = available - n_train
        n_val = min(SAMPLES_PER_CLASS["val"], remaining)
        remaining -= n_val
        n_test = min(SAMPLES_PER_CLASS["test"], remaining)

        # If shortage, redistribute proportionally
        if available < total_needed:
            ratio_train = SAMPLES_PER_CLASS["train"] / total_needed
            ratio_val = SAMPLES_PER_CLASS["val"] / total_needed
            n_train = int(available * ratio_train)
            n_val = int(available * ratio_val)
            n_test = available - n_train - n_val

            print(f"  ⚠ {cls_name:12s}: only {available:5d}/{total_needed} available "
                  f"→ train={n_train}, val={n_val}, test={n_test}")
        else:
            print(f"  ✓ {cls_name:12s}: {available:5d} available "
                  f"→ train={n_train}, val={n_val}, test={n_test}")

        # Split
        idx = 0
        split_data["train"].extend(pool[idx: idx + n_train])
        idx += n_train
        split_data["val"].extend(pool[idx: idx + n_val])
        idx += n_val
        split_data["test"].extend(pool[idx: idx + n_test])

        # Report source distribution
        affectnet_used = sum(1 for item in pool[:idx] if item[4] == "affectnet")
        fer_used = sum(1 for item in pool[:idx] if item[4] == "fer2013")
        if fer_used > 0:
            print(f"           Sources: AffectNet={affectnet_used}, FER2013={fer_used}")

    # Shuffle each split
    for split in split_data:
        random.shuffle(split_data[split])

    # Print summary
    print(f"\n  Final split sizes:")
    for split, data in split_data.items():
        cls_counts = Counter(item[3] for item in data)
        src_counts = Counter(item[4] for item in data)
        total = len(data)
        print(f"\n    {split:6s}: {total:5d} images")
        print(f"      Sources: {dict(src_counts)}")
        for cls_name in our_classes:
            count = cls_counts.get(cls_name, 0)
            expected = SAMPLES_PER_CLASS[split]
            marker = "✓" if count == expected else f"⚠({count}/{expected})"
            print(f"      {cls_name:12s}: {count:4d}  {marker}")

    return split_data


# ══════════════════════════════════════════════
# Write to Disk
# ══════════════════════════════════════════════
def write_emotion_dataset(split_data: dict) -> dict:
    """
    Write the sampled emotion data to disk.

    For each image:
      1. Read image
      2. Resize to 128×128 (emotions are face crops, small is fine)
         — AffectNet images are kept as-is if larger
         — FER2013 images (48×48) are upscaled to 128×128
      3. Save to data/emotions/images/{split}/
      4. Remap label or create YOLO label, save to data/emotions/labels/{split}/

    Returns:
        dict of class counts per split
    """
    dst_root = CFG.paths.emotions_dataset

    # Clean existing data
    for split in ["train", "val", "test"]:
        img_dir = dst_root / "images" / split
        lbl_dir = dst_root / "labels" / split
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    class_counts_per_split = {}

    for split_name, items in split_data.items():
        print(f"\n  Writing {split_name}: {len(items)} images")
        class_counter = Counter()

        for i, (img_path, lbl_path, remapping, class_name, source) in enumerate(
            tqdm(items, desc=f"  {split_name}")
        ):
            # Unique filename
            safe_stem = (
                img_path.stem
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
            )
            new_name = f"{class_name}_{source}_{safe_stem}"

            dst_img = dst_root / "images" / split_name / f"{new_name}.jpg"
            if dst_img.exists():
                new_name = f"{class_name}_{source}_{i:05d}_{safe_stem}"
                dst_img = dst_root / "images" / split_name / f"{new_name}.jpg"

            dst_lbl = dst_root / "labels" / split_name / f"{new_name}.txt"

            # Read and process image
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h_img, w_img = img.shape[:2]

                    if source == "fer2013" and max(h_img, w_img) < 100:
                        # FER2013 is 48×48, upscale to 128×128
                        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
                    elif max(h_img, w_img) > 640:
                        # Large images, resize down
                        scale = 640 / max(h_img, w_img)
                        img = cv2.resize(
                            img,
                            (int(w_img * scale), int(h_img * scale)),
                            interpolation=cv2.INTER_AREA,
                        )

                    cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    shutil.copy2(str(img_path), str(dst_img))
            except Exception as e:
                try:
                    shutil.copy2(str(img_path), str(dst_img))
                except Exception:
                    continue

            # Handle label
            target_idx = CFG.emotions.classes.index(class_name)

            if lbl_path is not None:
                # AffectNet: remap existing YOLO label
                lines_out = []
                try:
                    with open(lbl_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                orig_cls = int(parts[0])
                                if orig_cls in remapping:
                                    parts[0] = str(remapping[orig_cls])
                                else:
                                    parts[0] = str(target_idx)
                                lines_out.append(" ".join(parts))
                except Exception:
                    pass

                if lines_out:
                    with open(dst_lbl, "w") as f:
                        f.write("\n".join(lines_out) + "\n")
                else:
                    # Fallback: full-image label
                    with open(dst_lbl, "w") as f:
                        f.write(f"{target_idx} 0.5 0.5 0.9 0.9\n")
            else:
                # FER2013: create full-image YOLO label
                with open(dst_lbl, "w") as f:
                    f.write(f"{target_idx} 0.5 0.5 0.9 0.9\n")

            class_counter[class_name] += 1

        class_counts_per_split[split_name] = class_counter

    return class_counts_per_split


# ══════════════════════════════════════════════
# MaskTheFace Augmentation
# ══════════════════════════════════════════════
def apply_masktheface():
    """
    Apply MaskTheFace to create masked face variants.
    As per Section 2.4.2 of the paper.

    Adds masked copies to the training set (does NOT count toward the 500 per class).
    """
    train_img_dir = CFG.paths.emotions_images / "train"
    train_lbl_dir = CFG.paths.emotions_labels / "train"

    if not train_img_dir.exists():
        print("[WARN] Emotion train dir not found, skipping mask augmentation.")
        return

    images = sorted(
        list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    )

    if not images:
        print("[WARN] No training images, skipping mask augmentation.")
        return

    # Try MaskTheFace
    try:
        sys.path.insert(0, str(CFG.paths.root / "MaskTheFace"))
        from utils.aux_functions import mask_image
        use_masktheface = True
        print("\n[OK] MaskTheFace found. Using real mask augmentation.")
    except ImportError:
        use_masktheface = False
        print("\n[INFO] MaskTheFace not installed. Using synthetic mask overlay.")
        print("  To install: git clone https://github.com/aqeelanwar/MaskTheFace.git")

    # Create ~30% extra masked images
    n_masked = min(int(len(images) * 0.3), 1500)

    print(f"\n{'='*70}")
    print(f"MASK AUGMENTATION")
    print(f"  Training images: {len(images)}")
    print(f"  Creating {n_masked} masked variants")
    print(f"{'='*70}")

    created = 0
    for i in tqdm(range(n_masked), desc="  Masking faces"):
        img_path = random.choice(images)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if use_masktheface:
            try:
                masked_img, _ = mask_image(
                    str(img_path),
                    mask_type=random.choice(["surgical", "N95", "cloth", "KN95"]),
                )
                if masked_img is not None:
                    img = masked_img
                else:
                    img = _apply_synthetic_mask(img)
            except Exception:
                img = _apply_synthetic_mask(img)
        else:
            img = _apply_synthetic_mask(img)

        # Save
        mask_name = f"masked_{i:05d}_{img_path.stem}"
        cv2.imwrite(
            str(train_img_dir / f"{mask_name}.jpg"),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

        # Copy label
        src_lbl = train_lbl_dir / (img_path.stem + ".txt")
        dst_lbl = train_lbl_dir / f"{mask_name}.txt"
        if src_lbl.exists():
            shutil.copy2(str(src_lbl), str(dst_lbl))

        created += 1

    print(f"[OK] Created {created} masked face images.")
    print(f"     New training total: {len(images) + created}")


def _apply_synthetic_mask(img: np.ndarray) -> np.ndarray:
    """
    Apply a synthetic mask overlay to a face image.
    """
    h, w = img.shape[:2]

    mask_color = random.choice([
        (200, 200, 200),
        (100, 100, 100),
        (0, 0, 0),
        (0, 100, 200),
        (200, 200, 150),
    ])

    mask_top = int(h * random.uniform(0.4, 0.5))
    mask_bottom = int(h * random.uniform(0.85, 0.95))
    mask_left = int(w * random.uniform(0.05, 0.15))
    mask_right = int(w * random.uniform(0.85, 0.95))

    pts = np.array([
        [mask_left, mask_top],
        [mask_right, mask_top],
        [mask_right, mask_bottom],
        [int(w * 0.5), int(h * 0.92)],
        [mask_left, mask_bottom],
    ], np.int32)

    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], mask_color)

    alpha = random.uniform(0.7, 0.9)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.polylines(img, [pts], True, (150, 150, 150), 1)

    return img


# ══════════════════════════════════════════════
# Synthetic Fallback Dataset
# ══════════════════════════════════════════════
def create_synthetic_emotion_dataset():
    """
    Create a synthetic emotion dataset with exact counts per class.
    Used only when real data is not available.
    """
    print(f"\n{'='*70}")
    print("CREATING SYNTHETIC EMOTION DATASET")
    print(f"{'='*70}")

    dst_root = CFG.paths.emotions_dataset
    classes = CFG.emotions.classes

    emotion_features = {
        "angry": {"eyebrow": "v", "mouth": "line"},
        "happy": {"eyebrow": "arc", "mouth": "smile"},
        "neutral": {"eyebrow": "line", "mouth": "line"},
        "sad": {"eyebrow": "sad", "mouth": "frown"},
        "surprise": {"eyebrow": "high", "mouth": "o"},
    }

    for split in ["train", "val", "test"]:
        n_per_class = SAMPLES_PER_CLASS[split]
        n_total = n_per_class * len(classes)

        img_dir = dst_root / "images" / split
        lbl_dir = dst_root / "labels" / split

        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  {split}: {n_per_class} per class × {len(classes)} = {n_total}")

        for cls_idx, cls_name in enumerate(classes):
            for j in tqdm(
                range(n_per_class),
                desc=f"    {cls_name:12s}",
                leave=False,
            ):
                # Create face-like image
                size = 128
                skin_color = (
                    random.randint(170, 230),
                    random.randint(140, 200),
                    random.randint(120, 180),
                )
                img = np.full((size, size, 3), 220, dtype=np.uint8)

                # Head
                cv2.ellipse(
                    img, (64, 64), (50, 60), 0, 0, 360,
                    skin_color, -1,
                )

                # Eyes
                eye_color = (50, 50, 50)
                cv2.circle(img, (45, 48), 5 + random.randint(-1, 1), eye_color, -1)
                cv2.circle(img, (83, 48), 5 + random.randint(-1, 1), eye_color, -1)

                # Emotion-specific features
                features = emotion_features.get(cls_name, {})

                if features.get("mouth") == "smile":
                    cv2.ellipse(img, (64, 78), (18, 10), 0, 0, 180, (50, 50, 50), 2)
                elif features.get("mouth") == "frown":
                    cv2.ellipse(img, (64, 88), (18, 10), 0, 180, 360, (50, 50, 50), 2)
                elif features.get("mouth") == "o":
                    cv2.circle(img, (64, 82), 10, (50, 50, 50), 2)
                elif features.get("mouth") == "line":
                    cv2.line(img, (50, 82), (78, 82), (50, 50, 50), 2)

                if features.get("eyebrow") == "v":
                    cv2.line(img, (38, 40), (48, 36), (50, 50, 50), 2)
                    cv2.line(img, (90, 40), (80, 36), (50, 50, 50), 2)
                elif features.get("eyebrow") == "high":
                    cv2.line(img, (38, 32), (52, 34), (50, 50, 50), 2)
                    cv2.line(img, (90, 32), (76, 34), (50, 50, 50), 2)
                elif features.get("eyebrow") == "sad":
                    cv2.line(img, (38, 36), (52, 40), (50, 50, 50), 2)
                    cv2.line(img, (90, 36), (76, 40), (50, 50, 50), 2)

                # Add noise
                noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                fname = f"synth_{cls_name}_{j:04d}"
                cv2.imwrite(str(img_dir / f"{fname}.jpg"), img)
                with open(lbl_dir / f"{fname}.txt", "w") as f:
                    f.write(f"{cls_idx} 0.5 0.5 0.85 0.9\n")

    print(f"\n[OK] Synthetic emotion dataset created:")
    for split, n in SAMPLES_PER_CLASS.items():
        print(f"  {split:6s}: {n} per class × {len(classes)} = {n * len(classes)}")


# ══════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════
def verify_emotion_dataset():
    """Verify the prepared emotion dataset with detailed table."""
    dst_root = CFG.paths.emotions_dataset
    our_classes = CFG.emotions.classes

    print(f"\n{'='*70}")
    print("FINAL EMOTION DATASET VERIFICATION")
    print(f"{'='*70}")

    all_ok = True

    # ─── Per-split verification ───
    for split in ["train", "val", "test"]:
        img_dir = dst_root / "images" / split
        lbl_dir = dst_root / "labels" / split

        n_img = len(list(img_dir.rglob("*.*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0

        expected = SAMPLES_PER_CLASS[split] * CFG.emotions.num_classes

        # Count classes
        class_counts = Counter()
        if lbl_dir.exists():
            for lbl_file in lbl_dir.glob("*.txt"):
                with open(lbl_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                cls_idx = int(parts[0])
                                if 0 <= cls_idx < len(our_classes):
                                    class_counts[our_classes[cls_idx]] += 1
                            except ValueError:
                                pass

        # Orphaned images
        orphaned = 0
        if img_dir.exists() and lbl_dir.exists():
            for img_file in img_dir.iterdir():
                if not (lbl_dir / (img_file.stem + ".txt")).exists():
                    orphaned += 1

        status = "✓" if n_img >= expected * 0.9 else "⚠"
        print(f"\n  {status} {split.upper()}: {n_img} images, {n_lbl} labels (expected ~{expected})")

        if orphaned > 0:
            print(f"    ⚠ {orphaned} images without labels")
            all_ok = False

        for cls_name in our_classes:
            count = class_counts.get(cls_name, 0)
            expected_cls = SAMPLES_PER_CLASS[split]
            cls_status = "✓" if count >= expected_cls * 0.9 else "⚠"
            print(f"    {cls_status} {cls_name:12s}: {count:4d} / {expected_cls}")

    # ─── YAML config ───
    yaml_path = CFG.paths.configs / "yolov5_emotions.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            yaml_content = yaml.safe_load(f)
        print(f"\n  YAML config: {yaml_path}")
        print(f"    nc: {yaml_content.get('nc')}")
        print(f"    names: {yaml_content.get('names')}")

        if yaml_content.get("nc") != CFG.emotions.num_classes:
            print(f"    ⚠ nc mismatch!")
            all_ok = False
    else:
        print(f"\n  ⚠ YAML not found: {yaml_path}")
        all_ok = False

    if all_ok:
        print(f"\n  ✓ Emotion dataset verification PASSED")
    else:
        print(f"\n  ⚠ Verification has warnings")

    # ─── Summary Table ───
    print(f"\n  ┌─────────────────┬───────┬───────┬───────┐")
    print(f"  │ {'Emotion':15s} │ Train │  Val  │ Test  │")
    print(f"  ├─────────────────┼───────┼───────┼───────┤")
    print(f"  │ {'EXPECTED':15s} │ {SAMPLES_PER_CLASS['train']:5d} │ {SAMPLES_PER_CLASS['val']:5d} │ {SAMPLES_PER_CLASS['test']:5d} │")
    print(f"  ├─────────────────┼───────┼───────┼───────┤")

    for cls_name in our_classes:
        counts = []
        for split in ["train", "val", "test"]:
            lbl_dir = dst_root / "labels" / split
            count = 0
            if lbl_dir.exists():
                cls_idx = our_classes.index(cls_name)
                for lbl_file in lbl_dir.glob("*.txt"):
                    try:
                        with open(lbl_file, "r") as f:
                            first_line = f.readline().strip().split()
                            if first_line and int(first_line[0]) == cls_idx:
                                count += 1
                    except Exception:
                        pass
            counts.append(count)

        print(f"  │ {cls_name:15s} │ {counts[0]:5d} │ {counts[1]:5d} │ {counts[2]:5d} │")

    print(f"  └─────────────────┴───────┴───────┴───────┘")

    # Count masked images in training
    train_img_dir = dst_root / "images" / "train"
    if train_img_dir.exists():
        masked = len(list(train_img_dir.glob("masked_*")))
        total = len(list(train_img_dir.rglob("*.*")))
        print(f"\n  Training set composition:")
        print(f"    Original: {total - masked}")
        print(f"    Masked:   {masked}")
        print(f"    Total:    {total}")


# ══════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════
def main():
    """Main pipeline for emotion dataset preparation."""
    CFG.paths.create_all()

    print("=" * 70)
    print("EMOTION DATASET PREPARATION")
    print(f"  Fixed samples: train={SAMPLES_PER_CLASS['train']}, "
          f"val={SAMPLES_PER_CLASS['val']}, test={SAMPLES_PER_CLASS['test']} per class")
    print(f"  Classes ({CFG.emotions.num_classes}): {CFG.emotions.classes}")
    print(f"  Expected total: {sum(SAMPLES_PER_CLASS.values()) * CFG.emotions.num_classes}")
    print("=" * 70)

    # Step 1: Explore datasets
    affectnet_info = explore_affectnet()
    fer_info = explore_fer2013()

    # Step 2: Collect pairs by class from each source
    affectnet_pairs = collect_affectnet_pairs(affectnet_info)
    fer2013_pairs = collect_fer2013_pairs(fer_info)

    # Check if we have any data
    total_affectnet = sum(len(v) for v in affectnet_pairs.values())
    total_fer = sum(len(v) for v in fer2013_pairs.values())
    total = total_affectnet + total_fer

    print(f"\n  Total available: {total} (AffectNet={total_affectnet}, FER2013={total_fer})")

    if total == 0:
        print("\n[WARN] No emotion data found. Creating synthetic dataset...")
        create_synthetic_emotion_dataset()
        # Write YAML
        CFG._write_emotion_yaml()
        verify_emotion_dataset()
        print("\n[DONE] Synthetic emotion dataset created.")
        return

    # Step 3: Pool and sample fixed counts
    split_data = pool_and_sample(affectnet_pairs, fer2013_pairs)

    # Step 4: Write YAML config
    CFG._write_emotion_yaml()

    # Step 5: Write to disk
    print(f"\n{'='*70}")
    print("WRITING EMOTION DATASET TO DISK")
    print(f"{'='*70}")

    class_counts = write_emotion_dataset(split_data)

    # Print write summary
    print(f"\n  Write summary:")
    grand_total = 0
    for split_name, counts in class_counts.items():
        split_total = sum(counts.values())
        grand_total += split_total
        expected_total = SAMPLES_PER_CLASS[split_name] * CFG.emotions.num_classes
        print(f"    {split_name:6s}: {split_total:5d} / {expected_total} images")

    print(f"    TOTAL: {grand_total}")

    # Step 6: Apply MaskTheFace augmentation
    apply_masktheface()

    # Step 7: Full verification
    verify_emotion_dataset()

    print(f"\n{'='*70}")
    print("[DONE] Emotion dataset is ready for YOLOv5 training.")
    print(f"  Dataset: {CFG.paths.emotions_dataset}")
    print(f"  Config:  {CFG.paths.configs / 'yolov5_emotions.yaml'}")
    print(f"\n  Next: python run.py train-emotions")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()