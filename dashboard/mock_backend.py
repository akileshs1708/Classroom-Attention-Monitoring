import os
import time
import json
import random
import cv2
import numpy as np
from pathlib import Path

dashboard_data_dir = Path("logs/dashboard")
dashboard_data_dir.mkdir(parents=True, exist_ok=True)

names = ["Alice", "Bob", "Charlie", "Diana"]
behaviors = ["blackboard_screen", "discussing", "handrise_read_write", "talking"]
emotions = ["happy", "neutral", "sad", "surprise"]

def create_dummy_frame():
    # create black image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (30, 41, 59) # match the background a bit
    
    cv2.putText(img, "Simulation Mode - NO CAMERA DETECTED", (100, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
    for i in range(4):
        x = 50 + i * 150
        y = 300
        # Draw fake bounding box
        color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
        cv2.rectangle(img, (x, y), (x+100, y+100), color, 2)
        cv2.putText(img, names[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(str(dashboard_data_dir / "live_frame.jpg"), img)

def dump_data(frame_count):
    status = {
        "is_running": True,
        "frame_count": frame_count,
        "fps": round(random.uniform(28.0, 32.0), 1),
        "num_students": 4,
        "avg_attention": random.randint(50, 95)
    }
    
    scores = {"scores": {}}
    for i, name in enumerate(names):
        score = random.randint(30, 100)
        scores["scores"][str(i)] = {
            "student_name": name,
            "score": score,
            "level": "HIGH" if score > 60 else "MEDIUM" if score > 40 else "LOW"
        }
        
    behs = {"behaviors": {}}
    for b in behaviors:
        behs["behaviors"][b] = random.randint(0, 3)
        
    emos = {"emotions": {}}
    for i, name in enumerate(names):
        emos["emotions"][str(i)] = {
            "emotion": random.choice(emotions),
            "confidence": round(random.uniform(0.6, 0.99), 2)
        }
        
    dets = {"detections": []}
    for i, name in enumerate(names):
        dets["detections"].append({
            "track_id": i,
            "behavior": random.choice(behaviors),
            "confidence": 0.95
        })

    with open(dashboard_data_dir / "live_status.json", "w") as f: json.dump(status, f)
    with open(dashboard_data_dir / "live_scores.json", "w") as f: json.dump(scores, f)
    with open(dashboard_data_dir / "live_behaviors.json", "w") as f: json.dump(behs, f)
    with open(dashboard_data_dir / "live_emotions.json", "w") as f: json.dump(emos, f)
    with open(dashboard_data_dir / "live_detections.json", "w") as f: json.dump(dets, f)


if __name__ == "__main__":
    count = 0
    while True:
        create_dummy_frame()
        dump_data(count)
        count += 1
        time.sleep(1.0)
