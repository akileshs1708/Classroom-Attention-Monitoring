"""
src/inference/detect_emotions.py

Standalone emotion recognition using a trained YOLOv5 model.
Crops faces from the webcam feed (via Haar Cascade) and classifies them.

Usage:
    python src/inference/detect_emotions.py
    python src/inference/detect_emotions.py --source path/to/video.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "yolov5"
))

import src.config as cfg

# Colour palette per emotion class (BGR)
_EMOTION_PALETTE = {
    "angry":    (  0,   0, 220),
    "sad":      (200, 100,  50),
    "happy":    (  0, 210,  80),
    "neutral":  (200, 200, 200),
    "surprise": (  0, 165, 255),
}


def _detect_one_face(face_img: np.ndarray, model, device, names) -> str:
    """Run emotion model on a single face crop and return predicted class name."""
    from utils.general import non_max_suppression
    from utils.augmentations import letterbox

    img = letterbox(face_img, cfg.IMG_SIZE_EMOTIONS, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).to(device).float() / 255.0
    img_t = img_t.unsqueeze(0)

    pred = model(img_t)[0]
    pred = non_max_suppression(pred, cfg.CONF_THRESHOLD, cfg.IOU_THRESHOLD)

    best_label = "neutral"
    best_conf  = 0.0
    for det in pred:
        if det is None or not len(det):
            continue
        for *_, conf, cls in det:
            if conf.item() > best_conf:
                best_conf  = conf.item()
                best_label = names[int(cls.item())]
    return best_label


@torch.no_grad()
def detect_emotions(
    source: str | int = 0,
    weights: str      = cfg.WEIGHTS["emotions"],
) -> None:
    from models.experimental import attempt_load
    from utils.torch_utils import select_device
    from src.attendance.enroll_student import download_haar_cascade

    download_haar_cascade()
    device = select_device("")
    model  = attempt_load(weights, device=device)
    model.eval()
    names  = model.module.names if hasattr(model, "module") else model.names

    face_cascade = cv2.CascadeClassifier(cfg.HAAR_CASCADE)
    cap = cv2.VideoCapture(int(source) if str(source).isnumeric() else source)

    t_prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
        )

        for x, y, w, h in faces:
            face_crop = frame[y : y + h, x : x + w]
            emotion   = _detect_one_face(face_crop, model, device, names)
            color     = _EMOTION_PALETTE.get(emotion, (255, 255, 255))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fps = 1.0 / (time.time() - t_prev + 1e-9)
        t_prev = time.time()
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Emotion detection — q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  default="0")
    parser.add_argument("--weights", default=cfg.WEIGHTS["emotions"])
    args = parser.parse_args()

    src = int(args.source) if args.source.isnumeric() else args.source
    detect_emotions(src, args.weights)
