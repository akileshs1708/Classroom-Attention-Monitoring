"""
Attendance system using Haar Cascade face detection + LBPH face recognition.

As described in the paper (page 7, 9):
- Haar Cascade for face detection
- LBPH (Local Binary Pattern Histograms) for face recognition/matching

Pipeline:
1. Enroll students: capture reference face images
2. Train LBPH recognizer
3. Real-time: detect faces → recognize → mark attendance
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config

CFG = get_config()


class AttendanceSystem:
    """
    Complete attendance system with enrollment, training, and recognition.
    """

    def __init__(self):
        # Load Haar Cascade
        cascade_path = str(CFG.paths.haar_cascade)
        if not os.path.exists(cascade_path):
            # Download it
            self._download_haar_cascade()

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from {cascade_path}"
            )

        # LBPH Face Recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=CFG.attendance.lbph_radius,
            neighbors=CFG.attendance.lbph_neighbors,
            grid_x=CFG.attendance.lbph_grid_x,
            grid_y=CFG.attendance.lbph_grid_y,
        )

        # Student database
        self.student_names: Dict[int, str] = {}  # label_id -> name
        self.is_trained = False

        # Attendance tracking
        self.attendance_today: Dict[str, dict] = {}  # name -> info
        self.recognition_counts: Dict[str, int] = defaultdict(int)

        # Model path
        self.model_path = CFG.paths.models / "face_recognition" / "lbph_model.yml"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Names mapping path
        self.names_path = self.model_path.parent / "student_names.txt"

        # Load existing model if available
        self._load_model()

        print("[Attendance] System initialized")
        print(f"  Enrolled faces dir: {CFG.paths.enrolled_faces}")
        print(f"  Model: {self.model_path}")
        print(f"  Trained: {self.is_trained}")

    def _download_haar_cascade(self):
        """Download the Haar cascade XML file."""
        import urllib.request

        url = (
            "https://raw.githubusercontent.com/opencv/opencv/master/"
            "data/haarcascades/haarcascade_frontalface_default.xml"
        )
        cascade_path = CFG.paths.haar_cascade
        cascade_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Haar Cascade...")
        urllib.request.urlretrieve(url, str(cascade_path))
        print(f"[OK] Saved to {cascade_path}")

    def enroll_student(self, student_name: str, camera_source=0):
        """
        Capture reference images for a student using webcam.

        Args:
            student_name: Name/ID of the student
            camera_source: Camera index or video path
        """
        student_dir = CFG.paths.enrolled_faces / student_name
        student_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_source}")
            return False

        n_captured = 0
        target = CFG.attendance.capture_images_per_student

        print(f"\nEnrolling student: {student_name}")
        print(f"  Capturing {target} face images...")
        print(f"  Press 'q' to stop early, 'c' to capture manually")

        while n_captured < target:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=CFG.attendance.haar_scale_factor,
                minNeighbors=CFG.attendance.haar_min_neighbors,
                minSize=CFG.attendance.haar_min_size,
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Auto-capture every few frames
                if n_captured < target:
                    face_roi = gray[y:y + h, x:x + w]
                    face_roi = cv2.resize(face_roi, (200, 200))

                    img_path = (
                        student_dir / f"{student_name}_{n_captured:03d}.jpg"
                    )
                    cv2.imwrite(str(img_path), face_roi)
                    n_captured += 1

                    cv2.putText(
                        frame,
                        f"Captured: {n_captured}/{target}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            cv2.putText(
                frame,
                f"Student: {student_name} | {n_captured}/{target}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Enrollment", frame)

            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"[OK] Enrolled {student_name}: {n_captured} images saved.")
        return True

    def enroll_from_folder(self, folder_path: str, student_name: str):
        """
        Enroll a student from a folder of existing images.
        """
        src_dir = Path(folder_path)
        if not src_dir.exists():
            print(f"[ERROR] Folder not found: {src_dir}")
            return False

        student_dir = CFG.paths.enrolled_faces / student_name
        student_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for img_file in src_dir.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=3,
                    minSize=(30, 30),
                )

                for (x, y, w, h) in faces:
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (200, 200))
                    cv2.imwrite(str(student_dir / f"{student_name}_{n:03d}.jpg"), face)
                    n += 1

                # If no face detected, use whole image
                if len(faces) == 0:
                    face = cv2.resize(gray, (200, 200))
                    cv2.imwrite(str(student_dir / f"{student_name}_{n:03d}.jpg"), face)
                    n += 1

        print(f"[OK] Enrolled {student_name}: {n} face images from {src_dir}")
        return True

    def train_recognizer(self):
        """
        Train the LBPH recognizer on all enrolled students.
        """
        enrolled_dir = CFG.paths.enrolled_faces
        if not enrolled_dir.exists():
            print("[ERROR] No enrolled faces directory found.")
            return False

        faces = []
        labels = []
        label_id = 0
        self.student_names = {}

        for student_dir in sorted(enrolled_dir.iterdir()):
            if not student_dir.is_dir():
                continue

            student_name = student_dir.name
            self.student_names[label_id] = student_name

            for img_file in student_dir.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (200, 200))
                        faces.append(img)
                        labels.append(label_id)

            print(f"  {student_name} (id={label_id}): {labels.count(label_id)} images")
            label_id += 1

        if not faces:
            print("[ERROR] No face images found for training.")
            return False

        print(f"\nTraining LBPH on {len(faces)} images, {len(self.student_names)} students...")

        self.recognizer.train(faces, np.array(labels))

        # Save model
        self.recognizer.write(str(self.model_path))

        # Save names mapping
        with open(self.names_path, "w") as f:
            for lid, name in self.student_names.items():
                f.write(f"{lid},{name}\n")

        self.is_trained = True
        print(f"[OK] LBPH model saved to {self.model_path}")
        return True

    def _load_model(self):
        """Load a previously trained model."""
        if self.model_path.exists() and self.names_path.exists():
            try:
                self.recognizer.read(str(self.model_path))
                self.student_names = {}
                with open(self.names_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(",", 1)
                        if len(parts) == 2:
                            self.student_names[int(parts[0])] = parts[1]
                self.is_trained = True
                print(
                    f"[OK] Loaded LBPH model with "
                    f"{len(self.student_names)} students"
                )
            except Exception as e:
                print(f"[WARN] Could not load model: {e}")
                self.is_trained = False

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame using Haar Cascade.

        Args:
            frame: BGR image

        Returns:
            List of (x, y, w, h) tuples
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=CFG.attendance.haar_scale_factor,
            minNeighbors=CFG.attendance.haar_min_neighbors,
            minSize=CFG.attendance.haar_min_size,
        )

        return faces if len(faces) > 0 else []

    def recognize_face(
        self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]
    ) -> Tuple[Optional[str], float]:
        """
        Recognize a detected face.

        Args:
            frame: BGR image
            face_bbox: (x, y, w, h) from Haar detection

        Returns:
            (student_name or None, confidence)
        """
        if not self.is_trained:
            return None, 0.0

        x, y, w, h = face_bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (200, 200))

        label_id, confidence = self.recognizer.predict(face_roi)

        # Lower confidence = better match in LBPH
        if confidence < CFG.attendance.recognition_threshold:
            name = self.student_names.get(label_id, f"Unknown_{label_id}")
            return name, confidence
        else:
            return None, confidence

    def process_frame(
        self, frame: np.ndarray
    ) -> List[Dict]:
        """
        Process a single frame for attendance.

        Returns:
            List of recognized faces with their info.
        """
        results = []
        faces = self.detect_faces(frame)

        for (x, y, w, h) in faces:
            name, confidence = self.recognize_face(frame, (x, y, w, h))

            result = {
                "bbox": (x, y, w, h),
                "name": name,
                "confidence": confidence,
                "recognized": name is not None,
            }
            results.append(result)

            if name:
                self.recognition_counts[name] += 1

                # Mark as present if recognized enough times
                if (
                    self.recognition_counts[name]
                    >= CFG.attendance.min_frames_for_presence
                    and name not in self.attendance_today
                ):
                    self.attendance_today[name] = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "date": date.today().isoformat(),
                        "confidence": confidence,
                    }
                    print(
                        f"[Attendance] {name} marked PRESENT at "
                        f"{self.attendance_today[name]['time']}"
                    )

        return results

    def get_attendance_df(self) -> pd.DataFrame:
        """Get current attendance as a DataFrame."""
        if not self.attendance_today:
            return pd.DataFrame(
                columns=["student_name", "date", "time", "status", "confidence"]
            )

        rows = []
        # Include present students
        for name, info in self.attendance_today.items():
            rows.append({
                "student_name": name,
                "date": info["date"],
                "time": info["time"],
                "status": "Present",
                "confidence": info["confidence"],
            })

        # Include absent students (enrolled but not recognized)
        for lid, name in self.student_names.items():
            if name not in self.attendance_today:
                rows.append({
                    "student_name": name,
                    "date": date.today().isoformat(),
                    "time": "",
                    "status": "Absent",
                    "confidence": 0,
                })

        return pd.DataFrame(rows)

    def save_attendance(self, filepath: Path = None):
        """Save attendance to CSV."""
        if filepath is None:
            filepath = CFG.paths.attendance_csv

        df = self.get_attendance_df()
        df.to_csv(filepath, index=False)
        print(f"[Attendance] Saved to {filepath}")
        return df

    def reset_daily(self):
        """Reset attendance for a new day."""
        self.attendance_today.clear()
        self.recognition_counts.clear()