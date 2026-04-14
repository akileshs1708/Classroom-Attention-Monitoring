"""
src/inference/detect_actions.py

Standalone action/behaviour detection using a trained YOLOv5 model.
Useful for testing the actions model independently before integrating
it into the full pipeline.

Usage:
    # Run on webcam
    python src/inference/detect_actions.py

    # Run on a video file
    python src/inference/detect_actions.py --source path/to/video.mp4

    # Run on an image directory
    python src/inference/detect_actions.py --source data/actions/images/test \
                                            --save_dir runs/detect_actions
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "yolov5"
))

import src.config as cfg

# Colour palette for the 9 action classes (BGR)
_PALETTE = [
    (  0, 200,   0),   # raising_hand  – green
    ( 50, 205, 250),   # focused       – yellow-ish
    (  0, 128, 255),   # eating        – orange
    (180,   0, 180),   # distracted    – magenta
    (255,  80,  80),   # reading_book  – blue
    (  0,   0, 220),   # using_phone   – red
    (200, 200,   0),   # writing       – teal
    (100, 100, 100),   # bored         – grey
    (  0, 165, 255),   # laughing      – amber
]


def load_model(weights: str = cfg.WEIGHTS["actions_s"]) -> torch.nn.Module:
    from models.experimental import attempt_load
    from utils.torch_utils import select_device

    device = select_device("")
    model  = attempt_load(weights, device=device)
    model.eval()
    return model, device


@torch.no_grad()
def detect_actions(
    source: str | int = 0,
    weights: str      = cfg.WEIGHTS["actions_s"],
    conf_thres: float = cfg.CONF_THRESHOLD,
    iou_thres: float  = cfg.IOU_THRESHOLD,
    imgsz: int        = cfg.IMG_SIZE_ACTIONS,
    save_dir: str | None = None,
) -> None:
    from utils.general import non_max_suppression, scale_coords
    from utils.augmentations import letterbox

    model, device = load_model(weights)
    names  = model.module.names if hasattr(model, "module") else model.names

    is_webcam = str(source).isnumeric() or source == 0
    cap       = cv2.VideoCapture(int(source) if is_webcam else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    writer = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "output.mp4")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    t_prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pre-process
        img = letterbox(frame, imgsz, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]        # HWC → CHW, BGR→RGB
        img = np.ascontiguousarray(img)
        img_t = torch.from_numpy(img).to(device)
        img_t = img_t.float() / 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        # Inference
        pred = model(img_t)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_t.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
                    cls_id   = int(cls.item())
                    label    = names[cls_id]
                    color    = _PALETTE[cls_id % len(_PALETTE)]
                    txt      = f"{label} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, txt, (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # FPS overlay
        fps = 1.0 / (time.time() - t_prev + 1e-9)
        t_prev = time.time()
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if writer:
            writer.write(frame)
        cv2.imshow("Action detection — q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   default="0",
                        help="Webcam index (0) or path to video/image dir")
    parser.add_argument("--weights",  default=cfg.WEIGHTS["actions_s"])
    parser.add_argument("--conf",     type=float, default=cfg.CONF_THRESHOLD)
    parser.add_argument("--iou",      type=float, default=cfg.IOU_THRESHOLD)
    parser.add_argument("--imgsz",    type=int,   default=cfg.IMG_SIZE_ACTIONS)
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    src = int(args.source) if args.source.isnumeric() else args.source
    detect_actions(src, args.weights, args.conf, args.iou, args.imgsz, args.save_dir)
