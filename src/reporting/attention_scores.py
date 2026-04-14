"""
Attention scoring system.
UPDATED for 3-tier scoring: HIGH / MEDIUM / LOW attention.

7 SCB-05 behavior classes:
  HIGH:   blackboard_screen, blackboard_teacher, handrise_read_write, talk_teacher
  MEDIUM: discussing
  LOW:    standing, talking
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config

CFG = get_config()


class AttentionScorer:
    """
    Tracks and computes attention scores for each student/track.
    Uses 3-tier scoring based on the SCB-05 behavior classes.
    """

    def __init__(self):
        self.cfg = CFG.attention
        self.action_cfg = CFG.actions

        # Per-track raw score (cumulative)
        self.raw_scores: Dict[int, float] = defaultdict(
            lambda: self.cfg.initial_score
        )

        # Per-track score history for smoothing
        self.score_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.cfg.smoothing_window)
        )

        # Per-track behavior log
        self.behavior_log: Dict[int, List[dict]] = defaultdict(list)

        # Time series for reporting
        self.time_series: List[dict] = []

        # Frame counter
        self.frame_count = 0

        # Student name mapping (from face recognition)
        self.track_to_name: Dict[int, str] = {}

        print("[AttentionScorer] Initialized with 3-tier scoring")
        print(f"  Initial score: {self.cfg.initial_score}")
        print(f"  HIGH  (+{self.cfg.high_attention_increment}): "
              f"{self.action_cfg.high_attention_classes}")
        print(f"  MEDIUM (+{self.cfg.medium_attention_increment}): "
              f"{self.action_cfg.medium_attention_classes}")
        print(f"  LOW   (-{self.cfg.low_attention_decrement}): "
              f"{self.action_cfg.low_attention_classes}")

    def update(self, tracked_objects: List[dict], timestamp: float = None):
        """
        Update attention scores based on current frame detections.

        3-tier scoring:
          HIGH behaviors   → score += high_increment   (1.5)
          MEDIUM behaviors → score += medium_increment  (0.5)
          LOW behaviors    → score -= low_decrement      (1.5)
        """
        self.frame_count += 1

        if timestamp is None:
            timestamp = self.frame_count / 30.0

        for obj in tracked_objects:
            track_id = obj["track_id"]
            behavior = obj.get("class", "unknown")
            confidence = obj.get("confidence", 0.0)

            # Determine attention level and score delta
            attention_level = self.action_cfg.get_attention_level(behavior)

            if attention_level == "HIGH":
                delta = self.cfg.high_attention_increment
            elif attention_level == "MEDIUM":
                delta = self.cfg.medium_attention_increment
            elif attention_level == "LOW":
                delta = -self.cfg.low_attention_decrement
            else:
                delta = 0
                attention_level = "UNKNOWN"

            # Update raw score with clamping
            self.raw_scores[track_id] = max(
                self.cfg.min_score,
                min(
                    self.cfg.max_score,
                    self.raw_scores[track_id] + delta,
                ),
            )

            # Add to history for smoothing
            self.score_history[track_id].append(self.raw_scores[track_id])

            # Log behavior
            self.behavior_log[track_id].append({
                "frame": self.frame_count,
                "timestamp": timestamp,
                "behavior": behavior,
                "confidence": confidence,
                "attention_level": attention_level,
                "delta": delta,
                "raw_score": self.raw_scores[track_id],
                "smoothed_score": self.get_smoothed_score(track_id),
            })

        # Periodic time series logging (every ~1 second at 30fps)
        if self.frame_count % 30 == 0:
            for track_id in self.raw_scores:
                student_name = self.track_to_name.get(
                    track_id, f"Student_{track_id}"
                )
                self.time_series.append({
                    "timestamp": timestamp,
                    "frame": self.frame_count,
                    "track_id": track_id,
                    "student_name": student_name,
                    "attention_score": self.get_smoothed_score(track_id),
                    "raw_score": self.raw_scores[track_id],
                })

    def get_smoothed_score(self, track_id: int) -> float:
        """Get the smoothed (rolling average) score for a track."""
        history = self.score_history.get(track_id)
        if history and len(history) > 0:
            return float(np.mean(list(history)))
        return self.cfg.initial_score

    def get_all_scores(self) -> Dict[int, float]:
        """Get current smoothed scores for all tracks."""
        return {
            tid: self.get_smoothed_score(tid)
            for tid in self.raw_scores
        }

    def get_score_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of all students' attention scores."""
        rows = []
        for track_id in self.raw_scores:
            student_name = self.track_to_name.get(
                track_id, f"Student_{track_id}"
            )
            log = self.behavior_log[track_id]

            # Count behaviors
            behavior_counts = defaultdict(int)
            for entry in log:
                behavior_counts[entry["behavior"]] += 1

            # Most common behavior
            if behavior_counts:
                most_common = max(behavior_counts.items(), key=lambda x: x[1])
            else:
                most_common = ("unknown", 0)

            # Count attention levels
            total = max(len(log), 1)
            high_count = sum(1 for e in log if e["attention_level"] == "HIGH")
            medium_count = sum(1 for e in log if e["attention_level"] == "MEDIUM")
            low_count = sum(1 for e in log if e["attention_level"] == "LOW")

            rows.append({
                "track_id": track_id,
                "student_name": student_name,
                "current_score": round(self.get_smoothed_score(track_id), 1),
                "raw_score": round(self.raw_scores[track_id], 1),
                "total_frames": len(log),
                "dominant_behavior": most_common[0],
                "high_attention_pct": round(high_count / total * 100, 1),
                "medium_attention_pct": round(medium_count / total * 100, 1),
                "low_attention_pct": round(low_count / total * 100, 1),
            })

        return pd.DataFrame(rows)

    def get_time_series_df(self) -> pd.DataFrame:
        """Get time series data for dashboard visualization."""
        if not self.time_series:
            return pd.DataFrame()
        return pd.DataFrame(self.time_series)

    def get_behavior_distribution(self, track_id: int = None) -> Dict[str, int]:
        """Get behavior distribution for a specific track or all tracks."""
        counts = defaultdict(int)

        if track_id is not None:
            for entry in self.behavior_log.get(track_id, []):
                counts[entry["behavior"]] += 1
        else:
            for tid_log in self.behavior_log.values():
                for entry in tid_log:
                    counts[entry["behavior"]] += 1

        return dict(counts)

    def set_student_name(self, track_id: int, name: str):
        """Associate a face-recognized name with a track ID."""
        self.track_to_name[track_id] = name

    def save_reports(self):
        """Save all reports to CSV files."""
        # Attention scores summary
        summary_df = self.get_score_summary()
        summary_df.to_csv(CFG.paths.attention_csv, index=False)
        print(f"[Attention] Saved scores to {CFG.paths.attention_csv}")

        # Time series
        ts_df = self.get_time_series_df()
        if not ts_df.empty:
            ts_path = CFG.paths.dashboard_data / "attention_timeseries.csv"
            ts_df.to_csv(ts_path, index=False)
            print(f"[Attention] Saved time series to {ts_path}")

        # Detailed behavior log
        all_logs = []
        for tid, log in self.behavior_log.items():
            for entry in log:
                entry_copy = entry.copy()
                entry_copy["track_id"] = tid
                entry_copy["student_name"] = self.track_to_name.get(
                    tid, f"Student_{tid}"
                )
                all_logs.append(entry_copy)

        if all_logs:
            log_df = pd.DataFrame(all_logs)
            log_df.to_csv(CFG.paths.detections_csv, index=False)
            print(f"[Attention] Saved detections to {CFG.paths.detections_csv}")

    def get_class_attention_summary(self) -> dict:
        """Get overall class-level attention statistics."""
        all_scores = self.get_all_scores()
        if not all_scores:
            return {
                "average_attention": 0,
                "min_attention": 0,
                "max_attention": 0,
                "num_students": 0,
                "high_attention_count": 0,
                "medium_attention_count": 0,
                "low_attention_count": 0,
            }

        scores = list(all_scores.values())
        return {
            "average_attention": round(np.mean(scores), 1),
            "min_attention": round(min(scores), 1),
            "max_attention": round(max(scores), 1),
            "num_students": len(scores),
            "high_attention_count": sum(1 for s in scores if s >= 60),
            "medium_attention_count": sum(1 for s in scores if 40 <= s < 60),
            "low_attention_count": sum(1 for s in scores if s < 40),
        }