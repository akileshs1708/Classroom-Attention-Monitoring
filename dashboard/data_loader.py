import time
import json
import base64
import pandas as pd
from pathlib import Path

# Project root is two levels up from dashboard/data_loader.py
project_root = Path(__file__).resolve().parent.parent
dashboard_data_dir = project_root / "logs" / "dashboard"

# How many seconds before we consider the backend data stale/offline
STALE_THRESHOLD_SEC = 10


def safe_read_json(filepath: Path, default=None):
    """
    Reads a JSON file safely.
    - Uses None sentinel (avoids mutable-default-arg bug with {}).
    - Silently returns default on partial-write collisions (JSONDecodeError/OSError).
    """
    if default is None:
        default = {}
    if not filepath.exists():
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return json.loads(content)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default


def is_stream_stale(data: dict) -> bool:
    """
    Returns True if the data timestamp is older than STALE_THRESHOLD_SEC,
    meaning the backend is no longer writing live data.
    """
    ts = data.get("timestamp")
    if ts is None:
        return True
    return (time.time() - float(ts)) > STALE_THRESHOLD_SEC


def load_status() -> dict:
    """
    Loads live_status.json written by SharedState.update_status().
    Real backend schema: is_running, timestamp, frame_count, fps,
    elapsed_sec, num_students, avg_attention, high_attention_count,
    medium_attention_count, low_attention_count, video_source,
    behavior_classes, start_time.
    """
    data = safe_read_json(dashboard_data_dir / "live_status.json")
    # If no timestamp key, the real backend hasn't started yet
    if not data:
        return {"is_running": False, "fps": 0, "num_students": 0, "avg_attention": 0}
    # Override is_running based on staleness even if backend crashed
    if is_stream_stale(data):
        data["is_running"] = False
    return data


def load_scores() -> dict:
    """
    Loads live_scores.json written by SharedState.update_scores().
    Real backend schema: {timestamp, frame, scores: {track_id: {student_name, score, level}}}
    """
    return safe_read_json(dashboard_data_dir / "live_scores.json")


def load_behaviors() -> dict:
    """
    Loads live_behaviors.json written by SharedState.update_behaviors().
    Real backend schema: {timestamp, behaviors: {class_name: count}}
    """
    return safe_read_json(dashboard_data_dir / "live_behaviors.json")


def load_emotions() -> dict:
    """
    Loads live_emotions.json written by SharedState.update_emotions().
    Real backend schema: {timestamp, emotions: {track_id: {student_name, emotion, confidence}}}
    """
    return safe_read_json(dashboard_data_dir / "live_emotions.json")


def load_detections() -> dict:
    """
    Loads live_detections.json written by SharedState.update_detections().
    Real backend schema:
      {timestamp, frame, num_detections, detections: [{track_id, behavior, confidence, bbox, attention_level}]}
    """
    return safe_read_json(dashboard_data_dir / "live_detections.json")


def get_live_frame_base64() -> str | None:
    """
    Reads the latest annotated JPEG frame from the backend and returns
    it as a base64 string for embedding in an <img> tag.
    Returns None if no frame is available yet.
    """
    video_frame_path = dashboard_data_dir / "live_frame.jpg"
    if not video_frame_path.exists():
        return None
    try:
        with open(video_frame_path, "rb") as f:
            img_bytes = f.read()
        # Reject empty/corrupt files
        if len(img_bytes) < 100:
            return None
        return base64.b64encode(img_bytes).decode("utf-8")
    except OSError:
        return None


def load_attention_timeseries() -> pd.DataFrame:
    """
    Loads the rolling attention timeseries CSV written by AttentionScorer.save_reports().
    Columns: timestamp, frame, track_id, student_name, attention_score, raw_score.
    Returns an empty DataFrame if not yet available.
    """
    ts_path = dashboard_data_dir / "attention_timeseries.csv"
    if not ts_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(ts_path)
        # Keep only last 30 minutes of data to avoid chart slowdown
        if "timestamp" in df.columns and not df.empty:
            cutoff = time.time() - 30 * 60
            df = df[df["timestamp"] >= cutoff]
        return df
    except Exception:
        return pd.DataFrame()


def build_student_dataframe() -> pd.DataFrame:
    """
    Merges scores + emotions + detections into one tidy DataFrame
    for the student attention table.
    """
    scores_data = load_scores()
    emotions_data = load_emotions()
    detections_data = load_detections()

    scores = scores_data.get("scores", {})
    if not scores:
        return pd.DataFrame()

    emotions = emotions_data.get("emotions", {})
    det_list = detections_data.get("detections", [])

    # Build track_id → behavior lookup from detections
    # Real backend uses integer track_id; keys in scores are strings
    track_to_det = {
        str(d["track_id"]): {
            "action": d.get("behavior", "unknown"),
            "attention_level": d.get("attention_level", "UNKNOWN"),
            "confidence": d.get("confidence", 0.0),
        }
        for d in det_list
    }

    rows = []
    for track_id, score_info in scores.items():
        emo_info = emotions.get(str(track_id), {})
        det_info = track_to_det.get(str(track_id), {})

        score_val = score_info.get("score", 0)
        level = score_info.get("level", "UNKNOWN")

        rows.append({
            "ID": track_id,
            "Name": score_info.get("student_name", f"Student_{track_id}"),
            "Attention %": round(float(score_val), 1),
            "Level": level,
            "Action": det_info.get("action", "unknown").replace("_", " ").title(),
            "Conf": f"{det_info.get('confidence', 0.0):.0%}",
            "Emotion": emo_info.get("emotion", "—").title(),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Attention %", ascending=True).reset_index(drop=True)
    return df
