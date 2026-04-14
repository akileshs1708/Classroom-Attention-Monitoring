"""
Main real-time classroom monitoring pipeline.
UPDATED: Includes shared state for Streamlit dashboard integration.

Writes real-time data to shared files that the dashboard reads:
  - logs/dashboard/live_frame.jpg          (latest annotated frame)
  - logs/dashboard/live_detections.json    (current frame detections)
  - logs/dashboard/live_scores.json        (current attention scores)
  - logs/dashboard/live_status.json        (system status)
  - logs/dashboard/attention_timeseries.csv (rolling time series)
"""

import os
import sys
import time
import json
import threading
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
import base64

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config

CFG = get_config()


class SharedState:
    """
    Thread-safe shared state between the monitoring pipeline
    and the Streamlit dashboard.

    Writes JSON/CSV/image files to logs/dashboard/ so the
    dashboard can read them without direct memory sharing.
    """

    def __init__(self):
        self.dashboard_dir = CFG.paths.dashboard_data
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)

        # File paths for live data exchange
        self.live_frame_path = self.dashboard_dir / "live_frame.jpg"
        self.live_frame_small_path = self.dashboard_dir / "live_frame_small.jpg"
        self.live_detections_path = self.dashboard_dir / "live_detections.json"
        self.live_scores_path = self.dashboard_dir / "live_scores.json"
        self.live_status_path = self.dashboard_dir / "live_status.json"
        self.live_behaviors_path = self.dashboard_dir / "live_behaviors.json"
        self.live_emotions_path = self.dashboard_dir / "live_emotions.json"
        self.live_attendance_path = self.dashboard_dir / "live_attendance.json"
        self.timeseries_path = self.dashboard_dir / "attention_timeseries.csv"
        self.behavior_history_path = self.dashboard_dir / "behavior_history.csv"

        # Lock for thread safety
        self._lock = threading.Lock()

        # In-memory buffers
        self._frame_buffer = None
        self._is_running = False
        self._frame_count = 0
        self._fps = 0.0
        self._start_time = None

        print(f"[SharedState] Dashboard data dir: {self.dashboard_dir}")

    def update_frame(self, frame: np.ndarray, annotated: np.ndarray):
        """Save the latest annotated frame for dashboard display."""
        with self._lock:
            self._frame_buffer = annotated.copy()

            # Save full-size frame
            cv2.imwrite(
                str(self.live_frame_path),
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )

            # Save small thumbnail for faster loading
            h, w = annotated.shape[:2]
            scale = min(640 / w, 480 / h)
            small = cv2.resize(
                annotated,
                (int(w * scale), int(h * scale)),
            )
            cv2.imwrite(
                str(self.live_frame_small_path),
                small,
                [cv2.IMWRITE_JPEG_QUALITY, 70],
            )

    def update_detections(self, tracked_objects: List[dict]):
        """Save current frame detections."""
        with self._lock:
            data = {
                "timestamp": time.time(),
                "frame": self._frame_count,
                "num_detections": len(tracked_objects),
                "detections": [],
            }

            for obj in tracked_objects:
                det = {
                    "track_id": int(obj["track_id"]),
                    "behavior": obj.get("class", "unknown"),
                    "confidence": round(float(obj.get("confidence", 0)), 3),
                    "bbox": [int(x) for x in obj["bbox"]],
                    "attention_level": CFG.actions.get_attention_level(
                        obj.get("class", "unknown")
                    ),
                }
                data["detections"].append(det)

            with open(self.live_detections_path, "w") as f:
                json.dump(data, f, indent=2)

    def update_scores(self, scores: Dict[int, float], track_names: Dict[int, str]):
        """Save current attention scores."""
        with self._lock:
            data = {
                "timestamp": time.time(),
                "frame": self._frame_count,
                "scores": {},
            }

            for track_id, score in scores.items():
                name = track_names.get(track_id, f"Student_{track_id}")
                data["scores"][str(track_id)] = {
                    "student_name": name,
                    "score": round(float(score), 1),
                    "level": (
                        "HIGH" if score >= 60
                        else "MEDIUM" if score >= 40
                        else "LOW"
                    ),
                }

            with open(self.live_scores_path, "w") as f:
                json.dump(data, f, indent=2)

    def update_status(
        self,
        is_running: bool,
        frame_count: int,
        fps: float,
        num_students: int,
        avg_attention: float,
        high_count: int,
        medium_count: int,
        low_count: int,
    ):
        """Save system status."""
        with self._lock:
            self._is_running = is_running
            self._frame_count = frame_count
            self._fps = fps

            elapsed = 0
            if self._start_time:
                elapsed = time.time() - self._start_time

            data = {
                "is_running": is_running,
                "timestamp": time.time(),
                "frame_count": frame_count,
                "fps": round(fps, 1),
                "elapsed_sec": round(elapsed, 1),
                "num_students": num_students,
                "avg_attention": round(avg_attention, 1),
                "high_attention_count": high_count,
                "medium_attention_count": medium_count,
                "low_attention_count": low_count,
                "video_source": str(CFG.camera.source),
                "behavior_classes": CFG.actions.classes,
                "start_time": (
                    datetime.fromtimestamp(self._start_time).isoformat()
                    if self._start_time
                    else None
                ),
            }

            with open(self.live_status_path, "w") as f:
                json.dump(data, f, indent=2)

    def update_behaviors(self, behavior_counts: Dict[str, int]):
        """Save current behavior distribution."""
        with self._lock:
            data = {
                "timestamp": time.time(),
                "behaviors": behavior_counts,
            }
            with open(self.live_behaviors_path, "w") as f:
                json.dump(data, f, indent=2)

    def update_emotions(self, emotions: Dict[int, dict], track_names: Dict[int, str]):
        """Save current emotions."""
        with self._lock:
            data = {
                "timestamp": time.time(),
                "emotions": {},
            }
            for track_id, emo in emotions.items():
                name = track_names.get(track_id, f"Student_{track_id}")
                data["emotions"][str(track_id)] = {
                    "student_name": name,
                    "emotion": emo.get("emotion", "unknown"),
                    "confidence": round(float(emo.get("confidence", 0)), 3),
                }

            with open(self.live_emotions_path, "w") as f:
                json.dump(data, f, indent=2)

    def update_attendance_live(self, attendance_data: dict):
        """Save live attendance state."""
        with self._lock:
            with open(self.live_attendance_path, "w") as f:
                json.dump(attendance_data, f, indent=2)

    def set_start_time(self):
        """Record the start time."""
        self._start_time = time.time()

    def mark_stopped(self):
        """Mark the system as stopped."""
        self.update_status(
            is_running=False,
            frame_count=self._frame_count,
            fps=0,
            num_students=0,
            avg_attention=0,
            high_count=0,
            medium_count=0,
            low_count=0,
        )

    def get_frame_as_base64(self) -> Optional[str]:
        """Get the latest frame as base64 string (for embedding in HTML)."""
        with self._lock:
            if self._frame_buffer is not None:
                _, buffer = cv2.imencode(".jpg", self._frame_buffer)
                return base64.b64encode(buffer).decode("utf-8")
        return None


class ClassroomMonitor:
    """
    Complete real-time classroom monitoring system.
    Updated for 7 SCB-05 classes with dashboard integration.
    """

    def __init__(
        self,
        action_weights: str = None,
        emotion_weights: str = None,
        video_source=None,
        enable_attendance: bool = True,
        enable_emotions: bool = True,
        enable_display: bool = True,
        save_output: bool = True,
        shared_state: SharedState = None,
    ):
        CFG.paths.create_all()

        self.enable_attendance = enable_attendance
        self.enable_emotions = enable_emotions
        self.enable_display = enable_display
        self.save_output = save_output

        # Shared state for dashboard
        self.shared_state = shared_state or SharedState()

        # ─── Load Action Detection Model ───
        print("\n[1/5] Loading Action Detection Model (7 SCB-05 classes)...")
        if action_weights is None:
            action_weights = str(CFG.paths.weights / "yolov5s_actions.pt")

        if os.path.exists(action_weights):
            self.action_model = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=action_weights, force_reload=False
            )
            self.using_custom_model = True
            print(f"  Loaded custom model: {action_weights}")
        else:
            print(f"  [WARN] Weights not found: {action_weights}")
            print(f"  Using pretrained YOLOv5s (COCO) with mapping")
            self.action_model = torch.hub.load(
                "ultralytics/yolov5", "yolov5s", pretrained=True
            )
            self.using_custom_model = False

        self.action_model.conf = CFG.yolo.conf_threshold
        self.action_model.iou = CFG.yolo.iou_threshold

        if torch.cuda.is_available():
            self.action_model = self.action_model.cuda()
            print("  Using CUDA")

        # ─── Load Emotion Detection Model ───
        print("\n[2/5] Loading Emotion Detection Model...")
        self.emotion_model = None

        if enable_emotions:
            if emotion_weights is None:
                emotion_weights = str(CFG.paths.weights / "yolov5s_emotions.pt")

            if os.path.exists(emotion_weights):
                self.emotion_model = torch.hub.load(
                    "ultralytics/yolov5", "custom",
                    path=emotion_weights, force_reload=False
                )
                self.emotion_model.conf = 0.3
                if torch.cuda.is_available():
                    self.emotion_model = self.emotion_model.cuda()
                print(f"  Loaded emotion model: {emotion_weights}")
            else:
                print(f"  [WARN] Not found: {emotion_weights}")
                self.enable_emotions = False

        # ─── Initialize Tracker ───
        print("\n[3/5] Initializing DeepSORT Tracker...")
        from tracking.deepsort_wrapper import create_tracker
        self.tracker = create_tracker()

        # ─── Initialize Attendance ───
        print("\n[4/5] Initializing Attendance System...")
        if enable_attendance:
            try:
                from attendance.attendance_system import AttendanceSystem
                self.attendance = AttendanceSystem()
            except Exception as e:
                print(f"  [WARN] Attendance failed: {e}")
                self.attendance = None
                self.enable_attendance = False
        else:
            self.attendance = None

        # ─── Initialize Attention Scorer ───
        print("\n[5/5] Initializing Attention Scorer (3-tier)...")
        from reporting.attention_scores import AttentionScorer
        self.scorer = AttentionScorer()

        # ─── Video Source ───
        if video_source is None:
            video_source = CFG.camera.source
        self.video_source = video_source

        # ─── State ───
        self.frame_count = 0
        self.fps = 0
        self.start_time = None
        self.is_running = False
        self._stop_event = threading.Event()

        # ─── Current frame results ───
        self.current_results = {
            "tracked_objects": [],
            "emotions": {},
            "attendance": [],
            "attention_scores": {},
        }

        # ─── Colors ───
        self.attention_level_colors = {
            "HIGH": (0, 255, 0),
            "MEDIUM": (0, 200, 255),
            "LOW": (0, 0, 255),
            "UNKNOWN": (128, 128, 128),
        }

        self.emotion_colors = {
            "angry": (0, 0, 255),
            "happy": (0, 255, 255),
            "neutral": (200, 200, 200),
            "sad": (255, 0, 0),
            "surprise": (255, 0, 255),
        }

        print("\n[OK] Classroom Monitor initialized!")

    def detect_actions(self, frame: np.ndarray) -> List[dict]:
        """Run YOLOv5 action detection on a frame."""
        results = self.action_model(frame)

        detections = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            class_name = results.names[int(cls)]

            if self.using_custom_model:
                if class_name in CFG.actions.classes:
                    behavior = class_name
                else:
                    continue
            else:
                coco_to_scb05 = {
                    "person": "blackboard_screen",
                    "book": "handrise_read_write",
                    "laptop": "blackboard_screen",
                    "cell phone": "talking",
                    "cup": "standing",
                    "bottle": "standing",
                    "tv": "blackboard_screen",
                    "keyboard": "handrise_read_write",
                }

                if class_name in coco_to_scb05:
                    behavior = coco_to_scb05[class_name]
                else:
                    continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class": behavior,
                "confidence": float(conf),
                "original_class": class_name,
            })

        return detections

    def detect_emotions(
        self, frame: np.ndarray, tracked_objects: List[dict]
    ) -> Dict[int, dict]:
        """Run emotion detection on face regions."""
        if not self.enable_emotions or self.emotion_model is None:
            return {}

        emotions = {}
        for obj in tracked_objects:
            track_id = obj["track_id"]
            x1, y1, x2, y2 = obj["bbox"]

            face_y2 = y1 + int((y2 - y1) * 0.4)
            face_x1 = x1 + int((x2 - x1) * 0.1)
            face_x2 = x2 - int((x2 - x1) * 0.1)

            h, w = frame.shape[:2]
            face_x1 = max(0, face_x1)
            face_y1 = max(0, y1)
            face_x2 = min(w, face_x2)
            face_y2 = min(h, face_y2)

            if face_x2 - face_x1 < 20 or face_y2 - face_y1 < 20:
                continue

            face_crop = frame[face_y1:face_y2, face_x1:face_x2]

            try:
                results = self.emotion_model(face_crop)
                if len(results.xyxy[0]) > 0:
                    best = results.xyxy[0][0].cpu().numpy()
                    emotions[track_id] = {
                        "emotion": results.names[int(best[5])],
                        "confidence": float(best[4]),
                        "face_bbox": [face_x1, face_y1, face_x2, face_y2],
                    }
            except Exception:
                pass

        return emotions

    def draw_annotations(
        self,
        frame: np.ndarray,
        tracked_objects: List[dict],
        emotions: Dict[int, dict],
        attendance_results: List[dict],
    ) -> np.ndarray:
        """Draw all annotations on the frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for obj in tracked_objects:
            track_id = obj["track_id"]
            x1, y1, x2, y2 = obj["bbox"]
            behavior = obj["class"]
            conf = obj["confidence"]

            attention_level = CFG.actions.get_attention_level(behavior)
            color = self.attention_level_colors.get(attention_level, (128, 128, 128))
            thickness = 3 if attention_level == "HIGH" else 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            student_name = self.scorer.track_to_name.get(track_id, f"ID:{track_id}")
            score = self.scorer.get_smoothed_score(track_id)

            level_icon = {"HIGH": "▲", "MEDIUM": "●", "LOW": "▼", "UNKNOWN": "?"}

            label_parts = [
                f"{student_name}",
                f"{behavior.replace('_', ' ').title()} ({conf:.0%})",
                f"{level_icon.get(attention_level, '?')} Attn: {score:.0f}%",
            ]

            if track_id in emotions:
                emo = emotions[track_id]
                label_parts.append(f"Emo: {emo['emotion']} ({emo['confidence']:.0%})")

            for i, text in enumerate(label_parts):
                text_y = y1 - 10 - (len(label_parts) - 1 - i) * 20
                if text_y < 15:
                    text_y = y2 + 20 + i * 20

                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    annotated,
                    (x1, text_y - th - 4),
                    (x1 + tw + 6, text_y + 4),
                    color, -1,
                )
                cv2.putText(
                    annotated, text, (x1 + 3, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

            # Attention bar
            bar_x = x2 + 5
            bar_width = 10
            bar_height = y2 - y1
            if bar_height > 20 and bar_x + bar_width < w:
                bar_fill = int(bar_height * score / 100)
                cv2.rectangle(annotated, (bar_x, y1), (bar_x + bar_width, y2), (80, 80, 80), -1)

                bar_color = (0, 255, 0) if score >= 60 else (0, 200, 255) if score >= 40 else (0, 0, 255)
                cv2.rectangle(annotated, (bar_x, y2 - bar_fill), (bar_x + bar_width, y2), bar_color, -1)
                cv2.rectangle(annotated, (bar_x, y1), (bar_x + bar_width, y2), (200, 200, 200), 1)

        # Emotion boxes
        for track_id, emo in emotions.items():
            if "face_bbox" in emo:
                fx1, fy1, fx2, fy2 = emo["face_bbox"]
                emo_color = self.emotion_colors.get(emo["emotion"], (200, 200, 200))
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), emo_color, 1)

        # Attendance boxes
        for att in attendance_results:
            x, y, aw, ah = att["bbox"]
            if att["recognized"]:
                cv2.rectangle(annotated, (x, y), (x + aw, y + ah), (255, 255, 0), 2)
                cv2.putText(annotated, att["name"], (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # HUD
        self._draw_hud(annotated, tracked_objects)

        return annotated

    def _draw_hud(self, frame: np.ndarray, tracked_objects: List[dict]):
        """Draw heads-up display."""
        h, w = frame.shape[:2]

        panel_h = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (360, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "Classroom Monitor (SCB-05)", (10, 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        summary = self.scorer.get_class_attention_summary()

        # Count current behaviors
        behavior_counts = defaultdict(int)
        for obj in tracked_objects:
            behavior_counts[obj["class"]] += 1

        stats = [
            f"Students: {summary['num_students']}",
            f"Avg Attention: {summary['average_attention']:.0f}%",
            f"HIGH: {summary['high_attention_count']}  "
            f"MED: {summary['medium_attention_count']}  "
            f"LOW: {summary['low_attention_count']}",
            f"FPS: {self.fps:.1f} | Frame: {self.frame_count}",
        ]

        if behavior_counts:
            top = sorted(behavior_counts.items(), key=lambda x: -x[1])[:3]
            beh_str = ", ".join(f"{b.replace('_', ' ')[:15]}:{c}" for b, c in top)
            stats.append(f"Top: {beh_str}")

        # Dashboard indicator
        stats.append("Dashboard: LIVE ●" if self.is_running else "Dashboard: OFF")

        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (10, 42 + i * 22),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Legend
        legend_y = panel_h + 10
        for label, color in [("HIGH", (0, 255, 0)), ("MEDIUM", (0, 200, 255)), ("LOW", (0, 0, 255))]:
            cv2.circle(frame, (15, legend_y), 5, color, -1)
            cv2.putText(frame, label, (25, legend_y + 4),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            legend_y += 18

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the entire pipeline."""
        self.frame_count += 1
        timestamp = time.time()

        # Step 1: Detect behaviors
        detections = self.detect_actions(frame)

        # Step 2: Track
        tracked_objects = self.tracker.update(detections, frame)

        # Step 3: Emotions (every 5 frames)
        emotions = {}
        if self.frame_count % 5 == 0:
            emotions = self.detect_emotions(frame, tracked_objects)
        self.current_results["emotions"].update(emotions)

        # Step 4: Attendance (every 30 frames)
        attendance_results = []
        if self.enable_attendance and self.attendance and self.frame_count % 30 == 0:
            attendance_results = self.attendance.process_frame(frame)
            for att in attendance_results:
                if att["recognized"]:
                    ax, ay, aw, ah = att["bbox"]
                    face_center = (ax + aw // 2, ay + ah // 2)
                    for obj in tracked_objects:
                        tx1, ty1, tx2, ty2 = obj["bbox"]
                        if tx1 <= face_center[0] <= tx2 and ty1 <= face_center[1] <= ty2:
                            self.scorer.set_student_name(obj["track_id"], att["name"])
                            break

        # Step 5: Update scores
        self.scorer.update(tracked_objects, timestamp)

        # Step 6: Draw
        annotated = self.draw_annotations(
            frame, tracked_objects, self.current_results["emotions"], attendance_results
        )

        # Step 7: Update shared state for dashboard
        self._update_shared_state(
            frame, annotated, tracked_objects,
            self.current_results["emotions"], attendance_results
        )

        self.current_results["tracked_objects"] = tracked_objects
        self.current_results["attendance"] = attendance_results
        self.current_results["attention_scores"] = self.scorer.get_all_scores()

        return annotated

    def _update_shared_state(
        self,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        tracked_objects: List[dict],
        emotions: Dict[int, dict],
        attendance_results: List[dict],
    ):
        """Push all current data to shared state for dashboard."""
        # Frame (every 3rd frame to reduce I/O)
        if self.frame_count % 3 == 0:
            self.shared_state.update_frame(raw_frame, annotated_frame)

        # Detections
        self.shared_state.update_detections(tracked_objects)

        # Scores
        self.shared_state.update_scores(
            self.scorer.get_all_scores(),
            self.scorer.track_to_name,
        )

        # Status (every 10 frames)
        if self.frame_count % 10 == 0:
            summary = self.scorer.get_class_attention_summary()
            self.shared_state.update_status(
                is_running=self.is_running,
                frame_count=self.frame_count,
                fps=self.fps,
                num_students=summary["num_students"],
                avg_attention=summary["average_attention"],
                high_count=summary["high_attention_count"],
                medium_count=summary["medium_attention_count"],
                low_count=summary["low_attention_count"],
            )

        # Behaviors
        if self.frame_count % 10 == 0:
            beh_counts = defaultdict(int)
            for obj in tracked_objects:
                beh_counts[obj["class"]] += 1
            self.shared_state.update_behaviors(dict(beh_counts))

        # Emotions
        if emotions:
            self.shared_state.update_emotions(emotions, self.scorer.track_to_name)

        # Attendance
        if attendance_results and self.attendance:
            att_data = {
                "timestamp": time.time(),
                "present": list(self.attendance.attendance_today.keys()),
                "total_enrolled": len(self.attendance.student_names),
                "details": [
                    {
                        "name": att["name"],
                        "recognized": att["recognized"],
                        "confidence": float(att["confidence"]),
                    }
                    for att in attendance_results
                ],
            }
            self.shared_state.update_attendance_live(att_data)

    def run(self, max_frames: int = None, save_every_n: int = 300):
        """Run the complete monitoring pipeline."""
        print(f"\n{'='*60}")
        print("STARTING CLASSROOM MONITORING (SCB-05)")
        print(f"  Dashboard data: {self.shared_state.dashboard_dir}")
        print(f"{'='*60}")

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.video_source}")

        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30

        print(f"  Video: {vid_w}x{vid_h} @ {vid_fps:.0f}fps")

        writer = None
        if self.save_output:
            output_path = str(
                CFG.paths.outputs / f"classroom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, vid_fps, (vid_w, vid_h))
            print(f"  Output: {output_path}")

        self.is_running = True
        self.start_time = time.time()
        self.shared_state.set_start_time()
        fps_timer = time.time()
        fps_frame_count = 0

        try:
            while self.is_running and not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("\n[INFO] End of video.")
                    break

                if max_frames and self.frame_count >= max_frames:
                    break

                annotated = self.process_frame(frame)

                # FPS
                fps_frame_count += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    self.fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_timer = time.time()

                # Display
                if self.enable_display:
                    display = cv2.resize(
                        annotated,
                        (CFG.camera.display_width, CFG.camera.display_height),
                    )
                    cv2.imshow("Classroom Monitor (SCB-05)", display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\n[INFO] User quit.")
                        break
                    elif key == ord("s"):
                        self._save_snapshot(annotated)
                    elif key == ord("r"):
                        self.scorer.save_reports()

                if writer:
                    writer.write(annotated)

                if self.frame_count % save_every_n == 0 and self.frame_count > 0:
                    self.scorer.save_reports()
                    if self.attendance:
                        self.attendance.save_attendance()

                if self.frame_count % 100 == 0:
                    self._print_status()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted.")

        finally:
            self.is_running = False
            self.shared_state.mark_stopped()
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self._final_save()

    def run_headless(self, max_frames: int = None):
        """Run without display (for dashboard-only mode)."""
        self.enable_display = False
        self.run(max_frames=max_frames)

    def run_in_thread(self, max_frames: int = None):
        """Run the monitor in a background thread."""
        self._stop_event.clear()
        thread = threading.Thread(
            target=self.run,
            kwargs={"max_frames": max_frames},
            daemon=True,
        )
        thread.start()
        return thread

    def stop(self):
        """Signal the monitor to stop."""
        self._stop_event.set()
        self.is_running = False

    def _save_snapshot(self, frame: np.ndarray):
        path = (
            CFG.paths.outputs / "annotated_frames"
            / f"snapshot_{datetime.now().strftime('%H%M%S')}.jpg"
        )
        cv2.imwrite(str(path), frame)
        print(f"  [Snapshot] {path}")

    def _print_status(self):
        summary = self.scorer.get_class_attention_summary()
        print(
            f"  Frame {self.frame_count} | FPS: {self.fps:.1f} | "
            f"Students: {summary['num_students']} | "
            f"Attn: {summary['average_attention']:.0f}% | "
            f"H:{summary['high_attention_count']} "
            f"M:{summary['medium_attention_count']} "
            f"L:{summary['low_attention_count']}"
        )

    def _final_save(self):
        print(f"\n{'='*60}")
        print("SAVING FINAL REPORTS")
        print(f"{'='*60}")

        self.scorer.save_reports()

        if self.attendance:
            self.attendance.save_attendance()

        summary = self.scorer.get_score_summary()
        if not summary.empty:
            print(f"\n  Attention Summary:")
            print(summary.to_string(index=False))

        elapsed = time.time() - self.start_time if self.start_time else 0
        session = {
            "date": datetime.now().isoformat(),
            "duration_sec": round(elapsed, 1),
            "total_frames": self.frame_count,
            "avg_fps": round(self.frame_count / elapsed if elapsed > 0 else 0, 1),
            "num_students": len(self.scorer.raw_scores),
            "avg_attention": round(
                np.mean(list(self.scorer.get_all_scores().values()))
                if self.scorer.get_all_scores() else 0, 1
            ),
        }
        pd.DataFrame([session]).to_csv(
            CFG.paths.session_log,
            mode="a",
            header=not CFG.paths.session_log.exists(),
            index=False,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Classroom Monitor (SCB-05)")
    parser.add_argument("--source", default=None)
    parser.add_argument("--action-weights", default=None)
    parser.add_argument("--emotion-weights", default=None)
    parser.add_argument("--no-emotions", action="store_true")
    parser.add_argument("--no-attendance", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true",
                         help="Run without display (dashboard only)")

    args = parser.parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    monitor = ClassroomMonitor(
        action_weights=args.action_weights,
        emotion_weights=args.emotion_weights,
        video_source=source if source is not None else 0,
        enable_attendance=not args.no_attendance,
        enable_emotions=not args.no_emotions,
        enable_display=not (args.no_display or args.headless),
    )

    if args.headless:
        monitor.run_headless(max_frames=args.max_frames)
    else:
        monitor.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()