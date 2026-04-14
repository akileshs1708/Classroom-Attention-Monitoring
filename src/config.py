"""
Central configuration for the Classroom Monitoring System.
All paths, hyperparameters, class definitions, and model settings.

UPDATED: Remapped to actual SCB-05 dataset folder structure.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import yaml


# ──────────────────────────────────────────────
# Project Root
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────
# Directory Structure
# ──────────────────────────────────────────────
@dataclass
class Paths:
    # Root directories
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    models: Path = PROJECT_ROOT / "models"
    logs: Path = PROJECT_ROOT / "logs"
    outputs: Path = PROJECT_ROOT / "outputs"
    src: Path = PROJECT_ROOT / "src"

    # Data subdirectories
    actions_dataset: Path = PROJECT_ROOT / "data" / "actions"
    actions_images: Path = PROJECT_ROOT / "data" / "actions" / "images"
    actions_labels: Path = PROJECT_ROOT / "data" / "actions" / "labels"

    emotions_dataset: Path = PROJECT_ROOT / "data" / "emotions"
    emotions_images: Path = PROJECT_ROOT / "data" / "emotions" / "images"
    emotions_labels: Path = PROJECT_ROOT / "data" / "emotions" / "labels"

    fer2013_raw: Path = PROJECT_ROOT / "data" / "fer2013"
    affectnet_raw: Path = PROJECT_ROOT / "data" / "affectnet"
    scb05_raw: Path = PROJECT_ROOT / "data" / "scb05"

    attendance_data: Path = PROJECT_ROOT / "data" / "attendance"
    enrolled_faces: Path = PROJECT_ROOT / "data" / "attendance" / "enrolled_faces"
    haar_cascade: Path = (
        PROJECT_ROOT
        / "data"
        / "attendance"
        / "haarcascade_frontalface_default.xml"
    )

    configs: Path = PROJECT_ROOT / "data" / "configs"

    # Model weights
    weights: Path = PROJECT_ROOT / "models" / "weights"
    yolov5_repo: Path = PROJECT_ROOT / "models" / "yolov5"

    # Log files
    detections_csv: Path = PROJECT_ROOT / "logs" / "detections.csv"
    attention_csv: Path = PROJECT_ROOT / "logs" / "attention_scores.csv"
    attendance_csv: Path = PROJECT_ROOT / "logs" / "attendance.csv"
    session_log: Path = PROJECT_ROOT / "logs" / "session_log.csv"

    # Dashboard
    dashboard_data: Path = PROJECT_ROOT / "logs" / "dashboard"

    def create_all(self):
        """Create all directories if they don't exist."""
        dirs = [
            self.data, self.models, self.logs, self.outputs,
            self.actions_images / "train",
            self.actions_images / "val",
            self.actions_images / "test",
            self.actions_labels / "train",
            self.actions_labels / "val",
            self.actions_labels / "test",
            self.emotions_images / "train",
            self.emotions_images / "val",
            self.emotions_images / "test",
            self.emotions_labels / "train",
            self.emotions_labels / "val",
            self.emotions_labels / "test",
            self.fer2013_raw,
            self.affectnet_raw,
            self.scb05_raw,
            self.enrolled_faces,
            self.configs,
            self.weights,
            self.dashboard_data,
            self.outputs / "annotated_frames",
            self.outputs / "charts",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        print(f"[CONFIG] Created {len(dirs)} directories under {self.root}")


# ──────────────────────────────────────────────
# Behavior / Action Classes
# UPDATED FOR ACTUAL SCB-05 DATASET
# ──────────────────────────────────────────────
@dataclass
class ActionConfig:
    """
    7 student behavior classes derived from the SCB-05 folder structure.

    Excluded folders:
      - SCB5-Teacher-2024-9-17         (teacher only, not student behavior)
      - SCB5-Teacher-Behavior-2024-9-17 (teacher only, not student behavior)

    Folder-to-class mapping:
      SCB5-BlackBoard-Screen              → blackboard_screen
      SCB5-BlackBoard-Sreen-Teacher       → blackboard_teacher
      SCB5-Discuss-2024-9-17              → discussing
      SCB5-Handrise-Read-write-2024-9-17  → handrise_read_write
      SCB5-Stand-2024-9-17                → standing
      SCB5-Talk-2024-9-17                 → talking
      SCB5-Talk-Teacher-Behavior-2024-9-17→ talk_teacher
    """

    classes: List[str] = field(default_factory=lambda: [
        "blackboard_screen",      # 0 - HIGH attention (looking at board/screen)
        "blackboard_teacher",     # 1 - HIGH attention (looking at teacher + board)
        "discussing",             # 2 - MEDIUM attention (group discussion)
        "handrise_read_write",    # 3 - HIGH attention (raising hand / reading / writing)
        "standing",               # 4 - LOW attention (standing, not seated)
        "talking",                # 5 - LOW attention (talking to peers, off-task)
        "talk_teacher",           # 6 - HIGH attention (interacting with teacher)
    ])

    high_attention_classes: List[str] = field(default_factory=lambda: [
        "blackboard_screen",      # looking at board/screen
        "blackboard_teacher",     # looking at teacher + board
        "handrise_read_write",    # raising hand, reading, writing
        "talk_teacher",           # interacting with teacher
    ])

    medium_attention_classes: List[str] = field(default_factory=lambda: [
        "discussing",             # could be on-task group work
    ])

    low_attention_classes: List[str] = field(default_factory=lambda: [
        "standing",               # not in seat
        "talking",                # off-task peer talking
    ])

    # ── SCB-05 folder name → our class name ──
    # This maps the EXACT folder names in your dataset to class names
    scb05_folder_to_class: Dict[str, str] = field(default_factory=lambda: {
        "SCB5-BlackBoard-Screen":                   "blackboard_screen",
        "SCB5-BlackBoard-Sreen-Teacher":            "blackboard_teacher",
        "SCB5-Discuss-2024-9-17":                   "discussing",
        "SCB5-Handrise-Read-write-2024-9-17":       "handrise_read_write",
        "SCB5-Stand-2024-9-17":                     "standing",
        "SCB5-Talk-2024-9-17":                      "talking",
        "SCB5-Talk-Teacher-Behavior-2024-9-17":     "talk_teacher",
    })

    # Folders to EXCLUDE (teacher-only, not student behavior)
    excluded_folders: List[str] = field(default_factory=lambda: [
        "SCB5-Teacher-2024-9-17",
        "SCB5-Teacher-Behavior-2024-9-17",
    ])

    num_classes: int = 7

    def get_class_index(self, class_name: str) -> int:
        """Get the integer index for a class name."""
        return self.classes.index(class_name)

    def get_attention_level(self, class_name: str) -> str:
        """Get attention level for a class name."""
        if class_name in self.high_attention_classes:
            return "HIGH"
        elif class_name in self.medium_attention_classes:
            return "MEDIUM"
        elif class_name in self.low_attention_classes:
            return "LOW"
        return "UNKNOWN"


# ──────────────────────────────────────────────
# Emotion Classes
# ──────────────────────────────────────────────
@dataclass
class EmotionConfig:
    classes: List[str] = field(default_factory=lambda: [
        "angry",      # 0
        "happy",      # 1
        "neutral",    # 2
        "sad",        # 3
        "surprise",   # 4
    ])

    fer2013_to_emotion: Dict[str, str] = field(default_factory=lambda: {
        "angry": "angry",
        "disgust": "angry",
        "fear": "surprise",
        "happy": "happy",
        "sad": "sad",
        "surprise": "surprise",
        "neutral": "neutral",
    })

    affectnet_to_emotion: Dict[int, str] = field(default_factory=lambda: {
        0: "angry",
        1: "angry",
        2: "angry",
        3: "surprise",
        4: "happy",
        5: "sad",
        6: "surprise",
        7: "neutral",
    })

    num_classes: int = 5


# ──────────────────────────────────────────────
# YOLOv5 Training Hyperparameters
# ──────────────────────────────────────────────
@dataclass
class YOLOTrainConfig:
    img_size: int = 512
    batch_size: int = 2
    epochs_actions: int = 30
    epochs_emotions: int = 30
    patience: int = 20

    variants: List[str] = field(default_factory=lambda: [
        "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"
    ])

    default_variant: str = "yolov5s"

    conf_threshold: float = 0.45
    iou_threshold: float = 0.5

    augment: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flipud: float = 0.0
    fliplr: float = 0.1
    grayscale_prob: float = 0.1


# ──────────────────────────────────────────────
# DeepSORT Configuration
# ──────────────────────────────────────────────
@dataclass
class DeepSORTConfig:
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.3
    nn_budget: int = 100
    embedder: str = "mobilenet"
    embedder_gpu: bool = True


# ──────────────────────────────────────────────
# Attention Scoring
# UPDATED: 3-tier scoring for HIGH / MEDIUM / LOW
# ──────────────────────────────────────────────
@dataclass
class AttentionConfig:
    high_attention_increment: float = 1.5   # strong positive signal
    medium_attention_increment: float = 0.5  # slight positive signal
    low_attention_decrement: float = 1.5     # negative signal
    min_score: float = 0.0
    max_score: float = 100.0
    initial_score: float = 50.0
    smoothing_window: int = 30
    report_interval_sec: float = 5.0


# ──────────────────────────────────────────────
# Attendance Configuration
# ──────────────────────────────────────────────
@dataclass
class AttendanceConfig:
    haar_scale_factor: float = 1.3
    haar_min_neighbors: int = 5
    haar_min_size: Tuple[int, int] = (60, 60)
    lbph_radius: int = 1
    lbph_neighbors: int = 8
    lbph_grid_x: int = 8
    lbph_grid_y: int = 8
    recognition_threshold: float = 80.0
    min_frames_for_presence: int = 10
    capture_images_per_student: int = 30


# ──────────────────────────────────────────────
# Camera / Input Configuration
# ──────────────────────────────────────────────
@dataclass
class CameraConfig:
    source: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    display_width: int = 1280
    display_height: int = 720


# ──────────────────────────────────────────────
# Dashboard Configuration
# ──────────────────────────────────────────────
@dataclass
class DashboardConfig:
    refresh_interval_sec: int = 2
    chart_history_minutes: int = 30
    port: int = 8501
    theme: str = "dark"


# ──────────────────────────────────────────────
# Master Config
# ──────────────────────────────────────────────
@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    actions: ActionConfig = field(default_factory=ActionConfig)
    emotions: EmotionConfig = field(default_factory=EmotionConfig)
    yolo: YOLOTrainConfig = field(default_factory=YOLOTrainConfig)
    deepsort: DeepSORTConfig = field(default_factory=DeepSORTConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    attendance: AttendanceConfig = field(default_factory=AttendanceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    def initialize(self):
        """Create directories and write YAML configs."""
        self.paths.create_all()
        self._write_action_yaml()
        self._write_emotion_yaml()
        print("[CONFIG] Initialization complete.")
        self._print_class_summary()

    def _write_action_yaml(self):
        """Write YOLOv5 dataset YAML for action detection."""
        yaml_content = {
            "path": str(self.paths.actions_dataset),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": self.actions.num_classes,
            "names": self.actions.classes,
        }
        yaml_path = self.paths.configs / "yolov5_actions.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        print(f"[CONFIG] Wrote {yaml_path}")

    def _write_emotion_yaml(self):
        """Write YOLOv5 dataset YAML for emotion detection."""
        yaml_content = {
            "path": str(self.paths.emotions_dataset),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": self.emotions.num_classes,
            "names": self.emotions.classes,
        }
        yaml_path = self.paths.configs / "yolov5_emotions.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        print(f"[CONFIG] Wrote {yaml_path}")

    def _print_class_summary(self):
        """Print a summary of the class configuration."""
        print(f"\n{'='*60}")
        print("ACTION CLASS CONFIGURATION (from SCB-05)")
        print(f"{'='*60}")
        print(f"  Total classes: {self.actions.num_classes}")
        print(f"\n  HIGH attention ({len(self.actions.high_attention_classes)}):")
        for c in self.actions.high_attention_classes:
            idx = self.actions.classes.index(c)
            print(f"    [{idx}] {c}")
        print(f"\n  MEDIUM attention ({len(self.actions.medium_attention_classes)}):")
        for c in self.actions.medium_attention_classes:
            idx = self.actions.classes.index(c)
            print(f"    [{idx}] {c}")
        print(f"\n  LOW attention ({len(self.actions.low_attention_classes)}):")
        for c in self.actions.low_attention_classes:
            idx = self.actions.classes.index(c)
            print(f"    [{idx}] {c}")
        print(f"\n  Excluded folders (teacher-only):")
        for f in self.actions.excluded_folders:
            print(f"    ✗ {f}")
        print(f"{'='*60}")


# ──────────────────────────────────────────────
# Global singleton
# ──────────────────────────────────────────────
CFG = Config()


def get_config() -> Config:
    return CFG


def init_project():
    cfg = get_config()
    cfg.initialize()
    return cfg


if __name__ == "__main__":
    init_project()