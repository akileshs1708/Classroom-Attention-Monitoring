# Classroom Attention Monitoring System

A real-time vision-based system for monitoring student behaviour, emotions,
attention levels, and attendance in classroom settings.

Implements the full pipeline described in the paper using:
- **YOLOv5n/s/m/l/x** for behaviour and emotion detection
- **DeepSORT** for multi-object tracking
- **Haar Cascade + LBPH** for face-based attendance
- **Streamlit** dashboard for live visualisation

---

## Architecture

```
Camera frame
    │
    ▼
YOLOv5s (9 behaviour classes)
    │
    ├──► DeepSORT → persistent track IDs
    │
    ├──► Face crop → YOLOv5s (5 emotion classes)
    │
    ├──► Haar Cascade + LBPH → attendance CSV
    │
    └──► Attention scoring → per-student score [0–100]
                                    │
                                    ▼
                          detections.csv
                          attention_scores.csv
                          attendance.csv
                                    │
                                    ▼
                          Streamlit dashboard
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <this-repo>
cd classroom-attention
pip install -r requirements.txt
python src/dataset_utils/setup_yolov5.py


# Configure Kaggle API first: https://github.com/Kaggle/kaggle-api
python run.py download

python run.py prepare-actions
python run.py prepare-emotions

# Train YOLOv5s for action detection
python run.py train-actions

# Train all variants (paper comparison)
python run.py train-actions --variant all

# Train emotion model
python run.py train-emotions

# Option 1: LIVE MODE (monitor + dashboard together)
python run.py live --source 0

# Option 2: Separate processes
# Terminal 1:
python run.py monitor-headless --source 0
# Terminal 2:
python run.py dashboard

python run.py dashboard
# Opens at http://localhost:8501

python run.py enroll --name "Alice"
python run.py enroll --name "Bob"
```