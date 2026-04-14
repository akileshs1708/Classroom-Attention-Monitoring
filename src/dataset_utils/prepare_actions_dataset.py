"""
Prepare the SCB-05 dataset for YOLOv5 action/behavior detection training.

FIXED SAMPLE SIZES PER CLASS:
  - Train: 500 images per class
  - Val:   125 images per class
  - Test:   75 images per class
  - Total: 700 images per class × 7 classes = 4,900 images

ACTUAL SCB-05 STRUCTURE:
  scb05/SCB-Dataset/
    SCB5-BlackBoard-Screen/               → blackboard_screen    (HIGH)
    SCB5-BlackBoard-Sreen-Teacher/        → blackboard_teacher   (HIGH)
    SCB5-Discuss-2024-9-17/               → discussing           (MEDIUM)
    SCB5-Handrise-Read-write-2024-9-17/   → handrise_read_write  (HIGH)
    SCB5-Stand-2024-9-17/                 → standing             (LOW)
    SCB5-Talk-2024-9-17/                  → talking              (LOW)
    SCB5-Talk-Teacher-Behavior-2024-9-17/ → talk_teacher         (HIGH)
    SCB5-Teacher-2024-9-17/               → EXCLUDED
    SCB5-Teacher-Behavior-2024-9-17/      → EXCLUDED
"""

import os
import sys
import shutil
import random
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
# Total dataset = 700 × 7 classes = 4,900 images


def find_scb_dataset_root() -> Path:
    """
    Find the actual SCB-Dataset root directory.
    Searches multiple possible locations.
    """
    base = CFG.paths.scb05_raw

    candidates = [
        base / "SCB-Dataset",
        base / "scb05" / "SCB-Dataset",
        base / "scb05",
        base,
    ]

    for candidate in candidates:
        if candidate.exists():
            has_scb_folders = any(
                (candidate / folder_name).exists()
                for folder_name in CFG.actions.scb05_folder_to_class.keys()
            )
            if has_scb_folders:
                print(f"[OK] Found SCB-Dataset at: {candidate}")
                return candidate

    # Recursive search
    for item in base.rglob("SCB5-BlackBoard-Screen"):
        if item.is_dir():
            parent = item.parent
            print(f"[OK] Found SCB-Dataset at: {parent}")
            return parent

    print(f"[ERROR] Could not find SCB-Dataset under {base}")
    print(f"  Expected folders: {list(CFG.actions.scb05_folder_to_class.keys())}")
    return base


def explore_scb05_structure():
    """
    Explore the actual SCB-05 dataset structure in detail.
    Reports available image-label pairs per folder.
    """
    scb_root = find_scb_dataset_root()

    print(f"\n{'='*70}")
    print(f"EXPLORING SCB-05 DATASET")
    print(f"Root: {scb_root}")
    print(f"{'='*70}")

    folder_info = {}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for item in sorted(scb_root.iterdir()):
        if not item.is_dir():
            continue

        folder_name = item.name
        is_target = folder_name in CFG.actions.scb05_folder_to_class
        is_excluded = folder_name in CFG.actions.excluded_folders
        status = (
            "✓ TARGET"
            if is_target
            else ("✗ EXCLUDED" if is_excluded else "? UNKNOWN")
        )

        images = [f for f in item.rglob("*") if f.suffix.lower() in image_exts]
        labels = [
            f for f in item.rglob("*.txt")
            if f.name != "classes.txt" and f.name != "notes.txt"
        ]

        # Find classes.txt or data.yaml
        classes_files = list(item.rglob("classes.txt"))
        yaml_files = list(item.rglob("*.yaml")) + list(item.rglob("*.yml"))

        print(f"\n  [{status}] {folder_name}/")
        print(f"    Images: {len(images)}")
        print(f"    Labels: {len(labels)}")

        if classes_files:
            with open(classes_files[0], "r") as f:
                classes = [line.strip() for line in f if line.strip()]
            print(f"    classes.txt: {classes}")

        if yaml_files:
            for yf in yaml_files:
                try:
                    with open(yf, "r") as f:
                        data = yaml.safe_load(f)
                    if data and "names" in data:
                        print(f"    yaml names: {data['names']}")
                except Exception:
                    pass

        if labels:
            sample = labels[0]
            with open(sample, "r") as f:
                lines = [l.strip() for l in f.readlines()[:3]]
            print(f"    Sample label: {lines}")

            class_ids = set()
            for lbl in labels[:100]:
                try:
                    with open(lbl, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
                except Exception:
                    pass
            print(f"    Class IDs in labels: {sorted(class_ids)}")

        folder_info[folder_name] = {
            "path": item,
            "n_images": len(images),
            "n_labels": len(labels),
            "is_target": is_target,
            "is_excluded": is_excluded,
        }

    # Print availability summary for required samples
    print(f"\n{'='*70}")
    print(f"AVAILABILITY CHECK (need {sum(SAMPLES_PER_CLASS.values())} per class)")
    print(f"{'='*70}")

    total_needed = sum(SAMPLES_PER_CLASS.values())
    for folder_name, class_name in CFG.actions.scb05_folder_to_class.items():
        info = folder_info.get(folder_name, {})
        available = min(info.get("n_images", 0), info.get("n_labels", 0))
        status = "✓ OK" if available >= total_needed else f"⚠ ONLY {available}"
        print(f"  {class_name:25s} ({folder_name}): {available:5d} available  {status}")

    return folder_info


def detect_folder_layout(folder_path: Path) -> dict:
    """
    Detect the internal layout of a single SCB-05 behavior folder.

    Possible layouts:
    A) folder/images/ + folder/labels/
    B) folder/train/images/ + folder/train/labels/ + folder/val/...
    C) folder/*.jpg + folder/*.txt (flat)
    D) folder/images/train/ + folder/labels/train/
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    layout = {
        "type": "unknown",
        "image_label_pairs": [],  # list of (img_path, lbl_path)
    }

    # Collect ALL image files
    all_images = sorted([
        f for f in folder_path.rglob("*")
        if f.suffix.lower() in image_exts
    ])

    # Collect ALL label files
    all_labels_by_stem = {}
    for f in folder_path.rglob("*.txt"):
        if f.name not in ("classes.txt", "notes.txt"):
            all_labels_by_stem[f.stem] = f

    # Match images to labels by filename stem
    pairs = []
    for img_path in all_images:
        lbl_path = all_labels_by_stem.get(img_path.stem)
        if lbl_path is not None:
            pairs.append((img_path, lbl_path))

    layout["image_label_pairs"] = pairs

    if len(pairs) > 0:
        layout["type"] = "paired"
    else:
        layout["type"] = "images_only"

    return layout


def read_folder_classes(folder_path: Path) -> dict:
    """
    Read the original class names/indices from a folder.
    Returns: {original_class_id: original_class_name}
    """
    original_classes = {}

    # Try classes.txt
    for cf in folder_path.rglob("classes.txt"):
        with open(cf, "r") as f:
            for idx, line in enumerate(f):
                name = line.strip()
                if name:
                    original_classes[idx] = name
        if original_classes:
            return original_classes

    # Try data.yaml
    for yf in list(folder_path.rglob("*.yaml")) + list(folder_path.rglob("*.yml")):
        try:
            with open(yf, "r") as f:
                data = yaml.safe_load(f)
            if data and "names" in data:
                names = data["names"]
                if isinstance(names, dict):
                    return {int(k): v for k, v in names.items()}
                elif isinstance(names, list):
                    return {i: n for i, n in enumerate(names)}
        except Exception:
            pass

    return original_classes


def build_class_remapping(
    folder_name: str,
    original_classes: dict,
    target_class_name: str,
) -> dict:
    """
    Build remapping from original class IDs to our unified class ID.
    """
    target_idx = CFG.actions.get_class_index(target_class_name)
    remapping = {}

    if original_classes:
        for orig_id, orig_name in original_classes.items():
            orig_lower = orig_name.lower().strip().replace(" ", "_").replace("-", "_")

            matched = False
            for our_name in CFG.actions.classes:
                if our_name in orig_lower or orig_lower in our_name:
                    remapping[orig_id] = CFG.actions.get_class_index(our_name)
                    matched = True
                    break

            if not matched:
                remapping[orig_id] = target_idx
    else:
        # No class info — map IDs 0-20 to target class
        for i in range(20):
            remapping[i] = target_idx

    return remapping


def remap_and_save_label(
    src_label_path: Path,
    dst_label_path: Path,
    remapping: dict,
) -> bool:
    """
    Read YOLO label, remap class indices, save to destination.
    If src is None, creates full-image detection label.
    """
    dst_label_path.parent.mkdir(parents=True, exist_ok=True)

    if src_label_path is None:
        target_idx = next(iter(remapping.values()))
        with open(dst_label_path, "w") as f:
            f.write(f"{target_idx} 0.5 0.5 0.9 0.9\n")
        return True

    lines_out = []
    with open(src_label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    orig_class = int(parts[0])
                    if orig_class in remapping:
                        parts[0] = str(remapping[orig_class])
                    else:
                        default = next(iter(remapping.values()))
                        parts[0] = str(default)
                    lines_out.append(" ".join(parts))
                except ValueError:
                    continue

    if lines_out:
        with open(dst_label_path, "w") as f:
            f.write("\n".join(lines_out) + "\n")
        return True

    return False


def collect_pairs_from_folder(
    folder_path: Path,
    folder_name: str,
    target_class_name: str,
) -> list:
    """
    Collect all valid (image, label, remapping) tuples from one folder.

    Returns:
        list of (img_path, lbl_path_or_None, remapping_dict)
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    original_classes = read_folder_classes(folder_path)
    target_idx = CFG.actions.get_class_index(target_class_name)
    remapping = build_class_remapping(folder_name, original_classes, target_class_name)

    print(f"\n  Processing: {folder_name}")
    print(f"    → Class: {target_class_name} (idx={target_idx})")
    print(f"    → Remapping: {remapping}")

    layout = detect_folder_layout(folder_path)
    print(f"    → Layout: {layout['type']}, {len(layout['image_label_pairs'])} paired")

    pairs = []

    if layout["image_label_pairs"]:
        for img_path, lbl_path in layout["image_label_pairs"]:
            pairs.append((img_path, lbl_path, remapping))
    else:
        # No labels found — use images with auto-generated labels
        all_images = sorted([
            f for f in folder_path.rglob("*")
            if f.suffix.lower() in image_exts
        ])
        print(f"    [WARN] No labels found. Using {len(all_images)} images with auto-labels.")
        for img_path in all_images:
            pairs.append((img_path, None, {0: target_idx}))

    print(f"    → Total pairs: {len(pairs)}")
    return pairs


def sample_fixed_per_class(all_folder_pairs: dict) -> dict:
    """
    Sample exactly SAMPLES_PER_CLASS images from each class folder.

    Args:
        all_folder_pairs: dict mapping class_name -> list of (img, lbl, remap) tuples

    Returns:
        dict mapping split_name -> list of (img, lbl, remap, class_name) tuples

    Sampling strategy:
      1. Shuffle all pairs within each class
      2. Take first 500 for train, next 125 for val, next 75 for test
      3. If not enough data, take what's available and print warning
    """
    random.seed(42)

    split_data = {
        "train": [],
        "val": [],
        "test": [],
    }

    total_needed = sum(SAMPLES_PER_CLASS.values())  # 700

    print(f"\n{'='*70}")
    print(f"SAMPLING FIXED COUNTS PER CLASS")
    print(f"  Train: {SAMPLES_PER_CLASS['train']} per class")
    print(f"  Val:   {SAMPLES_PER_CLASS['val']} per class")
    print(f"  Test:  {SAMPLES_PER_CLASS['test']} per class")
    print(f"  Total: {total_needed} per class × {len(all_folder_pairs)} classes")
    print(f"{'='*70}")

    for class_name, pairs in sorted(all_folder_pairs.items()):
        available = len(pairs)

        # Shuffle
        shuffled = pairs.copy()
        random.shuffle(shuffled)

        # Calculate actual counts
        n_train = min(SAMPLES_PER_CLASS["train"], available)
        remaining = available - n_train
        n_val = min(SAMPLES_PER_CLASS["val"], remaining)
        remaining -= n_val
        n_test = min(SAMPLES_PER_CLASS["test"], remaining)

        # If we don't have enough, redistribute proportionally
        if available < total_needed:
            ratio_train = SAMPLES_PER_CLASS["train"] / total_needed
            ratio_val = SAMPLES_PER_CLASS["val"] / total_needed
            ratio_test = SAMPLES_PER_CLASS["test"] / total_needed

            n_train = int(available * ratio_train)
            n_val = int(available * ratio_val)
            n_test = available - n_train - n_val

            print(f"  ⚠ {class_name:25s}: only {available}/{total_needed} available "
                  f"→ train={n_train}, val={n_val}, test={n_test}")
        else:
            print(f"  ✓ {class_name:25s}: {available} available "
                  f"→ train={n_train}, val={n_val}, test={n_test}")

        # Split
        idx = 0
        train_pairs = shuffled[idx: idx + n_train]
        idx += n_train
        val_pairs = shuffled[idx: idx + n_val]
        idx += n_val
        test_pairs = shuffled[idx: idx + n_test]

        # Add class_name to each tuple for tracking
        for img, lbl, remap in train_pairs:
            split_data["train"].append((img, lbl, remap, class_name))
        for img, lbl, remap in val_pairs:
            split_data["val"].append((img, lbl, remap, class_name))
        for img, lbl, remap in test_pairs:
            split_data["test"].append((img, lbl, remap, class_name))

    # Shuffle each split (so classes are interleaved)
    for split in split_data:
        random.shuffle(split_data[split])

    # Print summary
    print(f"\n  Final split sizes:")
    for split, data in split_data.items():
        class_counts = Counter(item[3] for item in data)
        total = len(data)
        print(f"    {split:6s}: {total:5d} images")
        for cls_name in CFG.actions.classes:
            count = class_counts.get(cls_name, 0)
            expected = SAMPLES_PER_CLASS[split]
            marker = "✓" if count == expected else f"⚠({count}/{expected})"
            print(f"      {cls_name:25s}: {count:4d}  {marker}")

    return split_data


def write_split_to_disk(split_data: dict):
    """
    Write the sampled split data to the actions dataset directory.

    For each image:
      1. Read and resize to 640×640
      2. Save to data/actions/images/{split}/
      3. Remap labels and save to data/actions/labels/{split}/
    """
    dst_root = CFG.paths.actions_dataset

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

        for i, (img_path, lbl_path, remapping, class_name) in enumerate(
            tqdm(items, desc=f"  {split_name}")
        ):
            # Create unique filename: classname_originalname
            safe_stem = (
                img_path.stem
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
            )
            new_name = f"{class_name}_{safe_stem}"

            # Ensure uniqueness
            dst_img = dst_root / "images" / split_name / f"{new_name}.jpg"
            if dst_img.exists():
                new_name = f"{class_name}_{i:05d}_{safe_stem}"
                dst_img = dst_root / "images" / split_name / f"{new_name}.jpg"

            dst_lbl = dst_root / "labels" / split_name / f"{new_name}.txt"

            # Read and resize image to 640×640
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(dst_img), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    # Fallback: just copy
                    shutil.copy2(str(img_path), str(dst_img))
            except Exception as e:
                print(f"    [WARN] Could not process {img_path.name}: {e}")
                try:
                    shutil.copy2(str(img_path), str(dst_img))
                except Exception:
                    continue

            # Remap and save label
            success = remap_and_save_label(lbl_path, dst_lbl, remapping)

            if success:
                class_counter[class_name] += 1

        class_counts_per_split[split_name] = class_counter

    return class_counts_per_split


def prepare_dataset():
    """
    Main function: process all SCB-05 folders and create balanced dataset.

    Steps:
      1. Scan each target folder and collect all image-label pairs
      2. Sample exactly 500/125/75 per class for train/val/test
      3. Resize all images to 640×640
      4. Remap all labels to unified 7-class scheme
      5. Write to data/actions/images/{split}/ and data/actions/labels/{split}/
    """
    scb_root = find_scb_dataset_root()
    dst_root = CFG.paths.actions_dataset

    print(f"\n{'='*70}")
    print("PREPARING BALANCED ACTION DATASET FROM SCB-05")
    print(f"{'='*70}")
    print(f"  Source: {scb_root}")
    print(f"  Destination: {dst_root}")
    print(f"  Classes ({CFG.actions.num_classes}): {CFG.actions.classes}")
    print(f"  Per-class samples: train={SAMPLES_PER_CLASS['train']}, "
          f"val={SAMPLES_PER_CLASS['val']}, test={SAMPLES_PER_CLASS['test']}")
    print(f"  Expected total: {sum(SAMPLES_PER_CLASS.values()) * CFG.actions.num_classes}")

    # ─── Step 1: Collect all pairs from each folder ───
    all_folder_pairs = {}  # class_name -> [(img, lbl, remap), ...]

    for folder_name, class_name in CFG.actions.scb05_folder_to_class.items():
        folder_path = scb_root / folder_name

        if not folder_path.exists():
            # Try case-insensitive search
            found = False
            for item in scb_root.iterdir():
                if item.is_dir() and item.name.lower() == folder_name.lower():
                    folder_path = item
                    found = True
                    break

            if not found:
                print(f"\n  [ERROR] Folder not found: {folder_name}")
                print(f"    Available folders: {[d.name for d in scb_root.iterdir() if d.is_dir()]}")
                all_folder_pairs[class_name] = []
                continue

        pairs = collect_pairs_from_folder(folder_path, folder_name, class_name)
        all_folder_pairs[class_name] = pairs

    # Check if we have any data
    total_pairs = sum(len(p) for p in all_folder_pairs.values())
    if total_pairs == 0:
        print("\n[ERROR] No data found! Creating synthetic dataset...")
        create_synthetic_action_dataset()
        return

    # ─── Step 2: Sample fixed counts per class ───
    split_data = sample_fixed_per_class(all_folder_pairs)

    # ─── Step 3: Write to disk ───
    print(f"\n{'='*70}")
    print("WRITING DATASET TO DISK")
    print(f"{'='*70}")

    class_counts = write_split_to_disk(split_data)

    # ─── Step 4: Print final summary ───
    print(f"\n{'='*70}")
    print("DATASET PREPARATION COMPLETE")
    print(f"{'='*70}")

    grand_total = 0
    for split_name, counts in class_counts.items():
        split_total = sum(counts.values())
        grand_total += split_total
        expected_per_class = SAMPLES_PER_CLASS[split_name]
        expected_total = expected_per_class * CFG.actions.num_classes

        print(f"\n  {split_name.upper()} ({split_total}/{expected_total} images):")
        for cls_name in CFG.actions.classes:
            count = counts.get(cls_name, 0)
            attn = CFG.actions.get_attention_level(cls_name)
            bar_len = min(count // 5, 50)
            bar = "█" * bar_len
            marker = "✓" if count == expected_per_class else f"({count}/{expected_per_class})"
            print(f"    [{attn:6s}] {cls_name:25s}: {count:4d}  {marker}  {bar}")

    print(f"\n  GRAND TOTAL: {grand_total} images")
    expected_grand = sum(SAMPLES_PER_CLASS.values()) * CFG.actions.num_classes
    print(f"  EXPECTED:    {expected_grand} images")

    if grand_total < expected_grand:
        deficit = expected_grand - grand_total
        print(f"  DEFICIT:     {deficit} images (some classes had fewer samples)")


def create_synthetic_action_dataset():
    """
    Create a synthetic dataset with exact sample counts per class.
    Used only when real data is not available.
    """
    print("\nCreating synthetic action dataset with fixed counts per class...")

    dst_root = CFG.paths.actions_dataset
    classes = CFG.actions.classes
    img_size = 640

    behavior_colors = {
        "blackboard_screen": (50, 50, 50),
        "blackboard_teacher": (80, 120, 80),
        "discussing": (100, 100, 200),
        "handrise_read_write": (200, 200, 100),
        "standing": (150, 100, 100),
        "talking": (100, 150, 200),
        "talk_teacher": (120, 200, 120),
    }

    for split in ["train", "val", "test"]:
        n_per_class = SAMPLES_PER_CLASS[split]
        n_total = n_per_class * len(classes)

        img_dir = dst_root / "images" / split
        lbl_dir = dst_root / "labels" / split

        # Clean
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  {split}: {n_per_class} per class × {len(classes)} classes = {n_total}")

        img_idx = 0
        for cls_idx, cls_name in enumerate(classes):
            for j in tqdm(
                range(n_per_class),
                desc=f"    {cls_name:25s}",
                leave=False,
            ):
                bg = behavior_colors.get(cls_name, (128, 128, 128))
                img = np.full((img_size, img_size, 3), bg, dtype=np.uint8)
                noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                n_objects = random.randint(1, 4)
                lines = []

                for _ in range(n_objects):
                    cx = random.uniform(0.15, 0.85)
                    cy = random.uniform(0.15, 0.85)
                    w = random.uniform(0.08, 0.25)
                    h = random.uniform(0.15, 0.45)

                    lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                    x1 = int((cx - w / 2) * img_size)
                    y1 = int((cy - h / 2) * img_size)
                    x2 = int((cx + w / 2) * img_size)
                    y2 = int((cy + h / 2) * img_size)
                    color = [random.randint(50, 255) for _ in range(3)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    img, cls_name.replace("_", " "), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

                fname = f"synth_{cls_name}_{j:04d}"
                cv2.imwrite(str(img_dir / f"{fname}.jpg"), img)
                with open(lbl_dir / f"{fname}.txt", "w") as f:
                    f.write("\n".join(lines) + "\n")

                img_idx += 1

    print(f"\n[OK] Synthetic dataset created:")
    for split, n in SAMPLES_PER_CLASS.items():
        total = n * len(classes)
        print(f"  {split:6s}: {n:4d} per class × {len(classes)} classes = {total}")


def augment_actions_dataset():
    """
    Apply augmentations as in the paper:
    - 10% horizontal flip
    - Grayscale
    - Brightness variation

    NOTE: Augmentation adds ~20% more images to training set.
          This is ON TOP of the 500 per class.
    """
    train_img_dir = CFG.paths.actions_images / "train"
    train_lbl_dir = CFG.paths.actions_labels / "train"

    if not train_img_dir.exists():
        print("[WARN] Train directory not found, skipping augmentation.")
        return

    images = sorted(
        list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    )

    if not images:
        print("[WARN] No training images found, skipping augmentation.")
        return

    # Augment ~20% of images
    n_to_augment = max(int(len(images) * 0.2), 10)

    print(f"\n{'='*70}")
    print(f"AUGMENTING TRAINING DATA")
    print(f"  Existing images: {len(images)}")
    print(f"  Adding: ~{n_to_augment} augmented images")
    print(f"{'='*70}")

    aug_count = 0
    for i in tqdm(range(n_to_augment), desc="  Augmenting"):
        src_img_path = random.choice(images)
        img = cv2.imread(str(src_img_path))
        if img is None:
            continue

        src_label_path = train_lbl_dir / (src_img_path.stem + ".txt")

        # Already resized to 640×640, but ensure
        img = cv2.resize(img, (640, 640))

        augmentation = random.choice(["flip", "grayscale", "brightness"])
        new_lines = None

        if augmentation == "flip":
            img = cv2.flip(img, 1)
            if src_label_path.exists():
                with open(src_label_path, "r") as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[1] = f"{1.0 - float(parts[1]):.6f}"
                        new_lines.append(" ".join(parts))

        elif augmentation == "grayscale":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        else:  # brightness
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        aug_name = f"aug_{augmentation}_{i:05d}"
        cv2.imwrite(str(train_img_dir / f"{aug_name}.jpg"), img)

        aug_label = train_lbl_dir / f"{aug_name}.txt"
        if new_lines is not None:
            with open(aug_label, "w") as f:
                f.write("\n".join(new_lines) + "\n")
        elif src_label_path.exists():
            shutil.copy2(str(src_label_path), str(aug_label))

        aug_count += 1

    print(f"[OK] Added {aug_count} augmented images to training set.")
    print(f"     New training total: {len(images) + aug_count}")


def verify_dataset():
    """Verify the prepared dataset and print detailed statistics."""
    dst_root = CFG.paths.actions_dataset

    print(f"\n{'='*70}")
    print("FINAL DATASET VERIFICATION")
    print(f"{'='*70}")

    all_ok = True

    for split in ["train", "val", "test"]:
        img_dir = dst_root / "images" / split
        lbl_dir = dst_root / "labels" / split

        n_img = len(list(img_dir.rglob("*.*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0

        expected = SAMPLES_PER_CLASS[split] * CFG.actions.num_classes

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
                                if 0 <= cls_idx < len(CFG.actions.classes):
                                    class_counts[CFG.actions.classes[cls_idx]] += 1
                            except ValueError:
                                pass

        # Check for orphaned images (no label)
        orphaned = 0
        if img_dir.exists() and lbl_dir.exists():
            for img_file in img_dir.iterdir():
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if not lbl_file.exists():
                    orphaned += 1

        status = "✓" if n_img >= expected * 0.9 else "⚠"  # allow 10% tolerance
        print(f"\n  {status} {split.upper()}: {n_img} images, {n_lbl} labels (expected ~{expected})")

        if orphaned > 0:
            print(f"    ⚠ {orphaned} images without labels")
            all_ok = False

        for cls_name in CFG.actions.classes:
            count = class_counts.get(cls_name, 0)
            expected_cls = SAMPLES_PER_CLASS[split]
            cls_status = "✓" if count >= expected_cls * 0.9 else "⚠"
            attn = CFG.actions.get_attention_level(cls_name)
            print(f"    {cls_status} [{attn:6s}] {cls_name:25s}: {count:4d} / {expected_cls}")

    # Verify YAML config
    yaml_path = CFG.paths.configs / "yolov5_actions.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            yaml_content = yaml.safe_load(f)
        print(f"\n  YAML config: {yaml_path}")
        print(f"    nc: {yaml_content.get('nc')}")
        print(f"    names: {yaml_content.get('names')}")

        if yaml_content.get("nc") != CFG.actions.num_classes:
            print(f"    ⚠ nc mismatch! Config says {CFG.actions.num_classes}")
            all_ok = False
    else:
        print(f"\n  ⚠ YAML config not found at {yaml_path}")
        all_ok = False

    if all_ok:
        print(f"\n  ✓ Dataset verification PASSED")
    else:
        print(f"\n  ⚠ Dataset verification has warnings (may still work)")

    # Print sample counts table
    print(f"\n  ┌───────────────────────────┬───────┬───────┬───────┐")
    print(f"  │ {'Class':25s} │ Train │  Val  │ Test  │")
    print(f"  ├───────────────────────────┼───────┼───────┼───────┤")
    print(f"  │ {'EXPECTED':25s} │ {SAMPLES_PER_CLASS['train']:5d} │ {SAMPLES_PER_CLASS['val']:5d} │ {SAMPLES_PER_CLASS['test']:5d} │")
    print(f"  ├───────────────────────────┼───────┼───────┼───────┤")

    for cls_name in CFG.actions.classes:
        counts = []
        for split in ["train", "val", "test"]:
            lbl_dir = dst_root / "labels" / split
            count = 0
            if lbl_dir.exists():
                for lbl_file in lbl_dir.glob("*.txt"):
                    with open(lbl_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and int(parts[0]) == CFG.actions.get_class_index(cls_name):
                                count += 1
                                break  # count file, not annotations
            counts.append(count)

        print(f"  │ {cls_name:25s} │ {counts[0]:5d} │ {counts[1]:5d} │ {counts[2]:5d} │")

    print(f"  └───────────────────────────┴───────┴───────┴───────┘")


def main():
    """Main pipeline for action dataset preparation."""
    CFG.paths.create_all()

    print("=" * 70)
    print("SCB-05 ACTION/BEHAVIOR DATASET PREPARATION")
    print(f"  Fixed samples: train={SAMPLES_PER_CLASS['train']}, "
          f"val={SAMPLES_PER_CLASS['val']}, test={SAMPLES_PER_CLASS['test']} per class")
    print(f"  Classes: {CFG.actions.num_classes}")
    print(f"  Expected total: {sum(SAMPLES_PER_CLASS.values()) * CFG.actions.num_classes}")
    print("=" * 70)

    # Step 1: Explore dataset structure
    folder_info = explore_scb05_structure()

    # Step 2: Write YAML config
    CFG._write_action_yaml()

    # Step 3: Process all folders → balanced dataset
    prepare_dataset()

    # Step 4: Augment training set
    augment_actions_dataset()

    # Step 5: Full verification
    verify_dataset()

    print(f"\n{'='*70}")
    print("[DONE] Action dataset is ready for YOLOv5 training.")
    print(f"  Dataset: {CFG.paths.actions_dataset}")
    print(f"  Config:  {CFG.paths.configs / 'yolov5_actions.yaml'}")
    print(f"\n  Next: python run.py train-actions --variant yolov5s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()