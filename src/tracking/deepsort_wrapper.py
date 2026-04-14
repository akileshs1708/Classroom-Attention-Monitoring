"""
DeepSORT wrapper for multi-object tracking.

Implements the tracking described in the paper (page 7):
- Kalman filter state: [u, v, a, h, u', v', a', h']
- Matching: Mahalanobis distance + cosine distance
- Track management: confirmed/tentative/deleted

Uses the deep-sort-realtime library for convenience.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config

CFG = get_config()


class DeepSORTTracker:
    """
    Wrapper around deep-sort-realtime for student tracking.
    """

    def __init__(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError:
            raise ImportError(
                "deep-sort-realtime not installed. "
                "Run: pip install deep-sort-realtime"
            )

        self.tracker = DeepSort(
            max_age=CFG.deepsort.max_age,
            n_init=CFG.deepsort.n_init,
            max_iou_distance=CFG.deepsort.max_iou_distance,
            max_cosine_distance=CFG.deepsort.max_cosine_distance,
            nn_budget=CFG.deepsort.nn_budget,
            embedder=CFG.deepsort.embedder,
            embedder_gpu=CFG.deepsort.embedder_gpu,
        )

        # Track history: track_id -> list of (frame_num, behavior, bbox)
        self.track_history: Dict[int, List[dict]] = {}
        self.frame_count = 0

        print("[DeepSORT] Tracker initialized")
        print(f"  Max age: {CFG.deepsort.max_age}")
        print(f"  N_init: {CFG.deepsort.n_init}")
        print(f"  Embedder: {CFG.deepsort.embedder}")

    def update(
        self,
        detections: List[dict],
        frame: np.ndarray,
    ) -> List[dict]:
        """
        Update tracker with new detections.

        Args:
            detections: list of dicts, each with:
                - 'bbox': [x1, y1, x2, y2] in pixel coords
                - 'confidence': float
                - 'class': str (behavior class name)
            frame: current video frame (numpy array, BGR)

        Returns:
            list of tracked objects, each with:
                - 'track_id': int
                - 'bbox': [x1, y1, x2, y2]
                - 'class': str
                - 'confidence': float
                - 'confirmed': bool
        """
        self.frame_count += 1

        if not detections:
            # Update with empty detections (to age tracks)
            self.tracker.update_tracks([], frame=frame)
            return []

        # Convert detections to deep-sort format
        # deep-sort-realtime expects: [([x1, y1, w, h], confidence, class), ...]
        ds_detections = []
        det_classes = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            conf = det.get("confidence", 0.5)
            cls = det.get("class", "unknown")
            ds_detections.append(([x1, y1, w, h], conf, cls))
            det_classes.append(cls)

        # Update tracker
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        # Process results
        tracked_objects = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]

            # Get the detection class (from the most recent detection)
            det_class = track.det_class if hasattr(track, 'det_class') else "unknown"
            det_conf = track.det_conf if hasattr(track, 'det_conf') else 0.0

            tracked_obj = {
                "track_id": track_id,
                "bbox": [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                "class": det_class if det_class else "unknown",
                "confidence": float(det_conf) if det_conf else 0.0,
                "confirmed": True,
            }
            tracked_objects.append(tracked_obj)

            # Update track history
            if track_id not in self.track_history:
                self.track_history[track_id] = []

            self.track_history[track_id].append({
                "frame": self.frame_count,
                "behavior": tracked_obj["class"],
                "bbox": tracked_obj["bbox"],
                "confidence": tracked_obj["confidence"],
            })

        return tracked_objects

    def get_active_track_ids(self) -> List[int]:
        """Get list of currently active track IDs."""
        return [
            track.track_id
            for track in self.tracker.tracker.tracks
            if track.is_confirmed()
        ]

    def get_track_history(self, track_id: int) -> List[dict]:
        """Get the full history for a specific track."""
        return self.track_history.get(track_id, [])

    def get_all_track_ids(self) -> List[int]:
        """Get all track IDs (including inactive)."""
        return list(self.track_history.keys())

    def reset(self):
        """Reset the tracker."""
        self.__init__()


class SimpleTracker:
    """
    Fallback tracker using simple IoU matching.
    Used when deep-sort-realtime is not available.
    """

    def __init__(self, iou_threshold=0.3, max_age=30):
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.frame_count = 0
        self.track_history: Dict[int, List[dict]] = {}
        print("[SimpleTracker] Initialized (fallback IoU-based tracker)")

    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def update(self, detections: List[dict], frame: np.ndarray) -> List[dict]:
        self.frame_count += 1

        # Age all tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        if not detections:
            return [
                {
                    "track_id": tid,
                    "bbox": t["bbox"],
                    "class": t["class"],
                    "confidence": t["confidence"],
                    "confirmed": True,
                }
                for tid, t in self.tracks.items()
                if t["age"] < 5
            ]

        # Match detections to existing tracks
        matched = set()
        results = []

        for det in detections:
            best_iou = 0
            best_tid = None

            for tid, track in self.tracks.items():
                if tid in matched:
                    continue
                iou = self._compute_iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_iou > self.iou_threshold and best_tid is not None:
                # Update existing track
                self.tracks[best_tid]["bbox"] = det["bbox"]
                self.tracks[best_tid]["class"] = det["class"]
                self.tracks[best_tid]["confidence"] = det.get("confidence", 0.5)
                self.tracks[best_tid]["age"] = 0
                matched.add(best_tid)
                tid = best_tid
            else:
                # Create new track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "bbox": det["bbox"],
                    "class": det["class"],
                    "confidence": det.get("confidence", 0.5),
                    "age": 0,
                }

            results.append({
                "track_id": tid,
                "bbox": det["bbox"],
                "class": det["class"],
                "confidence": det.get("confidence", 0.5),
                "confirmed": True,
            })

            # Track history
            if tid not in self.track_history:
                self.track_history[tid] = []
            self.track_history[tid].append({
                "frame": self.frame_count,
                "behavior": det["class"],
                "bbox": det["bbox"],
            })

        return results

    def get_track_history(self, track_id):
        return self.track_history.get(track_id, [])

    def get_all_track_ids(self):
        return list(self.track_history.keys())


def create_tracker() -> object:
    """Factory function to create the best available tracker."""
    try:
        return DeepSORTTracker()
    except ImportError:
        print("[WARN] DeepSORT not available, using SimpleTracker fallback.")
        return SimpleTracker()