"""
Classroom Attention Monitor — Production Streamlit Dashboard
============================================================
Reads REAL training results from:
  outputs/training/{run_name}/results.csv
  outputs/training/{run_name}/weights/best.pt
  outputs/training/{run_name}/confusion_matrix.png
  outputs/training/{run_name}/F1_curve.png  etc.

Runs REAL-TIME YOLOv5s inference on webcam / uploaded video/photo.

Run:
  streamlit run dashboard/app.py
"""

import sys, os, time, json, io, threading, subprocess, signal, tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cv2
from PIL import Image

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from config import get_config
    CFG = get_config()
    TRAINING_DIR = CFG.paths.outputs / "training"
    WEIGHTS_DIR  = CFG.paths.weights
    DASHBOARD_DIR= CFG.paths.dashboard_data
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    st.error(f"Config error: {e}")
    st.stop()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classroom Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main-title {
    font-size:2rem; font-weight:700; letter-spacing:-0.5px; text-align:center;
    background:linear-gradient(120deg,#38bdf8,#22c55e); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; padding-bottom:0.8rem;
  }
  .run-card {
    background:rgba(56,189,248,0.05); border:1px solid rgba(56,189,248,0.2);
    border-radius:10px; padding:12px 16px; margin-bottom:8px;
  }
  .run-card.best { border-color:rgba(34,197,94,0.5); background:rgba(34,197,94,0.07); }
  .metric-chip {
    display:inline-block; padding:2px 10px; border-radius:12px;
    font-size:11px; font-weight:600; margin:2px;
  }
  .chip-green { background:rgba(34,197,94,0.15); color:#22c55e; }
  .chip-blue  { background:rgba(56,189,248,0.15); color:#38bdf8; }
  .chip-amber { background:rgba(245,158,11,0.15); color:#f59e0b; }
  .chip-red   { background:rgba(239,68,68,0.15);  color:#ef4444; }
  .live-dot { width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:6px; }
  .dot-on  { background:#22c55e; animation:pulse 1.5s infinite; }
  .dot-off { background:#ef4444; }
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
  .student-card { border-radius:10px;padding:10px 14px;margin:5px 0;border-left:4px solid; }
  .card-high   { border-color:#22c55e;background:rgba(34,197,94,0.07); }
  .card-medium { border-color:#f59e0b;background:rgba(245,158,11,0.07); }
  .card-low    { border-color:#ef4444;background:rgba(239,68,68,0.07); }
  div[data-testid="stMetric"] {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1);
    border-radius:10px; padding:10px 14px;
  }
  code { font-family:'JetBrains Mono',monospace; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
BEHAVIORS = ["blackboard_screen","blackboard_teacher","handrise_read_write",
             "talk_teacher","discussing","standing","talking"]
HIGH_ATTN  = ["blackboard_screen","blackboard_teacher","handrise_read_write","talk_teacher"]
MED_ATTN   = ["discussing"]
LOW_ATTN   = ["standing","talking"]

BEH_COLORS = {
    "blackboard_screen":"#22c55e","blackboard_teacher":"#4ade80",
    "handrise_read_write":"#16a34a","talk_teacher":"#15803d",
    "discussing":"#f59e0b","standing":"#ef4444","talking":"#e11d48",
}
ATTN_COLORS = {"HIGH":"#22c55e","MEDIUM":"#f59e0b","LOW":"#ef4444"}

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8",size=11), margin=dict(t=30,b=10,l=10,r=10),
)

def dark(fig, h=340):
    fig.update_layout(**PLOT_BASE, height=h)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)",zerolinecolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)",zerolinecolor="rgba(255,255,255,0.06)")
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# Training Run Discovery — reads REAL files
# ═══════════════════════════════════════════════════════════════════════════════

def discover_runs() -> Dict[str, dict]:
    """Scan outputs/training/ and return metadata for each run that has results.csv."""
    runs = {}
    if not TRAINING_DIR.exists():
        return runs
    for d in sorted(TRAINING_DIR.iterdir()):
        if not d.is_dir():
            continue
        results_csv = d / "results.csv"
        weights     = d / "weights" / "best.pt"
        if not results_csv.exists():
            continue
        # Parse variant from folder name: actions_yolov5s_20260404_0701 → yolov5s
        name = d.name
        parts = name.split("_")
        variant = next((p for p in parts if p.startswith("yolov5")), None)
        task    = "actions" if name.startswith("actions") else "emotions" if name.startswith("emotions") else "unknown"

        # Read last row of results.csv for final metrics
        try:
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            last = df.iloc[-1]
            # YOLOv5 results.csv column names vary — handle both formats
            def gcol(candidates):
                for c in candidates:
                    if c in df.columns: return float(last[c])
                return None
            map50    = gcol(["metrics/mAP_0.5","metrics/mAP50","mAP_0.5"])
            map5095  = gcol(["metrics/mAP_0.5:0.95","metrics/mAP50-95","mAP_0.5:0.95"])
            prec     = gcol(["metrics/precision","Precision","precision"])
            recall   = gcol(["metrics/recall","Recall","recall"])
            epoch    = int(last.get("epoch", len(df)-1)) if "epoch" in df.columns else len(df)-1
            box_loss = gcol(["train/box_loss","box_loss"])
            obj_loss = gcol(["train/obj_loss","obj_loss"])
            cls_loss = gcol(["train/cls_loss","cls_loss"])
        except Exception as ex:
            continue

        size_mb = round(weights.stat().st_size/(1024*1024),1) if weights.exists() else None
        f1 = round(2*prec*recall/(prec+recall),3) if prec and recall and (prec+recall)>0 else None

        runs[name] = dict(
            path=d, name=name, variant=variant, task=task,
            results_csv=results_csv,
            weights=weights if weights.exists() else None,
            df=df,
            map50=round(map50,4) if map50 else None,
            map5095=round(map5095,4) if map5095 else None,
            precision=round(prec,4) if prec else None,
            recall=round(recall,4) if recall else None,
            f1=f1,
            epochs_trained=epoch+1,
            size_mb=size_mb,
            box_loss=round(box_loss,4) if box_loss else None,
            # curve images
            confusion_matrix = d/"confusion_matrix.png",
            f1_curve         = d/"F1_curve.png",
            p_curve          = d/"P_curve.png",
            pr_curve         = d/"PR_curve.png",
            r_curve          = d/"R_curve.png",
            results_img      = d/"results.png",
            labels_img       = d/"labels.jpg",
            labels_corr      = d/"labels_correlogram.jpg",
            train_batch0     = d/"train_batch0.jpg",
            val_batch_labels = d/"val_batch0_labels.jpg",
            val_batch_pred   = d/"val_batch0_pred.jpg",
        )
    return runs

# ═══════════════════════════════════════════════════════════════════════════════
# Live inference helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_yolov5_model(weights_path: Path):
    """Load YOLOv5 model via torch.hub or direct path."""
    try:
        import torch
        model = torch.hub.load(
            str(PROJECT_ROOT/"models"/"yolov5"),
            "custom",
            path=str(weights_path),
            source="local",
            force_reload=False,
            verbose=False,
        )
        model.conf = CFG.yolo.conf_threshold
        model.iou  = CFG.yolo.iou_threshold
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_resource
def get_model(weights_path_str: str):
    return load_yolov5_model(Path(weights_path_str))

def run_inference_on_frame(model, frame_bgr: np.ndarray):
    """Run YOLOv5 inference on a single BGR frame, return annotated frame + detections."""
    import torch
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results   = model(frame_rgb, size=CFG.yolo.img_size)
    det_list  = []
    names     = model.names if hasattr(model,'names') else CFG.actions.classes

    # Draw on frame
    annotated = frame_bgr.copy()
    color_map = {
        "HIGH":   (0,220,80),
        "MEDIUM": (0,190,255),
        "LOW":    (0,0,240),
    }

    if results.xyxy[0] is not None and len(results.xyxy[0]):
        for det in results.xyxy[0].cpu().numpy():
            x1,y1,x2,y2,conf,cls = det
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            cls_id  = int(cls)
            label   = names[cls_id] if cls_id < len(names) else f"cls{cls_id}"
            level   = CFG.actions.get_attention_level(label)
            color   = color_map.get(level,(200,200,200))
            txt     = f"{label.replace('_',' ')} {conf:.2f}"

            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
            tw,th = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)[0]
            cv2.rectangle(annotated,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
            cv2.putText(annotated,txt,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

            det_list.append(dict(
                class_name=label, confidence=round(float(conf),3),
                attention_level=level,
                bbox=[x1,y1,x2,y2],
            ))

    # HUD overlay
    h_f,w_f = annotated.shape[:2]
    overlay = annotated.copy()
    cv2.rectangle(overlay,(0,0),(340,130),(0,0,0),-1)
    cv2.addWeighted(overlay,0.65,annotated,0.35,0,annotated)
    cv2.putText(annotated,"Classroom Monitor — YOLOv5s",(8,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,200),1,cv2.LINE_AA)
    n_det = len(det_list)
    high_n = sum(1 for d in det_list if d["attention_level"]=="HIGH")
    med_n  = sum(1 for d in det_list if d["attention_level"]=="MEDIUM")
    low_n  = sum(1 for d in det_list if d["attention_level"]=="LOW")
    cv2.putText(annotated,f"Detections: {n_det}",(8,42),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)
    cv2.putText(annotated,f"HIGH:{high_n}  MED:{med_n}  LOW:{low_n}",(8,62),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)
    cv2.putText(annotated,f"Model: YOLOv5s | Conf:{CFG.yolo.conf_threshold}",(8,82),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)
    cv2.putText(annotated,f"Classes: {','.join(CFG.actions.classes[:3])}...",(8,102),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)

    return cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB), det_list

# ═══════════════════════════════════════════════════════════════════════════════
# Read live dashboard data (from realtime_classroom.py output)
# ═══════════════════════════════════════════════════════════════════════════════

def read_json_safe(path:Path)->dict:
    try:
        if path.exists():
            with open(path) as f: return json.load(f)
    except: pass
    return {}

def read_live_status():  return read_json_safe(DASHBOARD_DIR/"live_status.json")
def read_live_scores():  return read_json_safe(DASHBOARD_DIR/"live_scores.json")
def read_live_det():     return read_json_safe(DASHBOARD_DIR/"live_detections.json")
def read_live_beh():     return read_json_safe(DASHBOARD_DIR/"live_behaviors.json")
def read_live_att():     return read_json_safe(DASHBOARD_DIR/"live_attendance.json")

def load_attention_ts() -> pd.DataFrame:
    p = DASHBOARD_DIR/"attention_timeseries.csv"
    if p.exists():
        try:
            df=pd.read_csv(p)
            if "timestamp" in df.columns and len(df)>0:
                df["time_min"]=(df["timestamp"]-df["timestamp"].min())/60
            return df
        except: pass
    return pd.DataFrame()

def load_detections_log() -> pd.DataFrame:
    p = CFG.paths.detections_csv
    if p.exists():
        try: return pd.read_csv(p)
        except: pass
    return pd.DataFrame()

def load_attention_scores() -> pd.DataFrame:
    p = CFG.paths.attention_csv
    if p.exists():
        try: return pd.read_csv(p)
        except: pass
    return pd.DataFrame()

def load_attendance_log() -> pd.DataFrame:
    p = CFG.paths.attendance_csv
    if p.exists():
        try: return pd.read_csv(p)
        except: pass
    return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar(runs):
    with st.sidebar:
        st.markdown("## 🎓 EduVision")
        st.caption(f"Project: `{PROJECT_ROOT.name}`")
        st.divider()

        page = st.radio("Navigation", [
            "📹 Live Inference",
            "📊 Session Analytics",
            "🤖 Model Metrics",
            "📈 YOLOv5 Comparison",
            "🖼️ Training Visuals",
            "📋 Attendance",
        ])

        st.divider()
        st.markdown("**📂 Upload for Inference**")
        uploaded = st.file_uploader(
            "Photo or video",
            type=["jpg","jpeg","png","mp4","avi","mov"],
            label_visibility="collapsed",
        )

        st.divider()
        # Live monitor status
        status = read_live_status()
        is_live = status.get("is_running", False)
        if is_live:
            st.markdown(f'<span class="live-dot dot-on"></span><b>LIVE</b> — {status.get("fps",0):.1f} FPS', unsafe_allow_html=True)
            st.caption(f"Frame: {status.get('frame_count',0)} | Students: {status.get('num_students',0)}")
        else:
            st.markdown('<span class="live-dot dot-off"></span>Monitor offline', unsafe_allow_html=True)
            st.caption("Start `realtime_classroom.py --headless` for live feed")

        st.divider()
        st.markdown("**Training runs found:**")
        if runs:
            for rname, r in runs.items():
                tag = r.get("map50","?")
                task_icon = "🎯" if r["task"]=="actions" else "😊"
                st.caption(f"{task_icon} `{rname[:30]}` — mAP50: `{tag}`")
        else:
            st.warning(f"No runs in `{TRAINING_DIR.relative_to(PROJECT_ROOT)}`")

        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    return page, uploaded

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Inference
# ═══════════════════════════════════════════════════════════════════════════════
def page_live_inference(uploaded, runs):
    st.markdown('<div class="main-title">📹 Live Inference — YOLOv5s</div>', unsafe_allow_html=True)

    # ── Control bar ───────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2,2,1,1])

    # Pick best actions weights
    actions_runs = {k:v for k,v in runs.items() if v["task"]=="actions" and v.get("weights")}
    if actions_runs:
        with col_ctrl1:
            sel_run = st.selectbox(
                "Action model weights",
                list(actions_runs.keys()),
                format_func=lambda k: f"{k} (mAP50={actions_runs[k].get('map50','?')})"
            )
        weights_path = actions_runs[sel_run]["weights"]
    else:
        with col_ctrl1:
            st.error("No trained weights found. Train models first.")
        weights_path = None

    with col_ctrl2:
        conf = st.slider("Confidence threshold", 0.1, 0.95, CFG.yolo.conf_threshold, 0.05)
    with col_ctrl3:
        use_webcam = st.checkbox("Use webcam", value=False)
    with col_ctrl4:
        run_live = st.button("▶ Start monitor", type="primary")

    st.divider()

    # ── Start full pipeline ───────────────────────────────────────────────────
    if run_live and not weights_path:
        st.error("No weights available. Train models first.")

    # Check if background realtime_classroom.py is running
    status = read_live_status()
    is_pipeline_live = status.get("is_running", False)

    if is_pipeline_live:
        st.success(f"🟢 Pipeline LIVE — FPS: {status.get('fps',0):.1f} | Students: {status.get('num_students',0)} | Frame: {status.get('frame_count',0)}")

    # ── Inference on uploaded media ───────────────────────────────────────────
    vid_col, side_col = st.columns([3,2])

    # Attention scorer state (session-level)
    if "attention_scores" not in st.session_state:
        st.session_state.attention_scores = {}
    if "behavior_history" not in st.session_state:
        st.session_state.behavior_history = []
    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0

    with vid_col:
        st.subheader("📹 Inference Output")
        img_placeholder = st.empty()
        fps_placeholder = st.empty()

        if uploaded is not None and weights_path:
            CFG.yolo.conf_threshold = conf
            model = get_model(str(weights_path))
            if model:
                model.conf = conf

                if uploaded.type.startswith("image"):
                    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                    frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    t0 = time.time()
                    annotated_rgb, det_list = run_inference_on_frame(model, frame_bgr)
                    inf_ms = (time.time()-t0)*1000

                    img_placeholder.image(annotated_rgb, use_container_width=True,
                                          caption=f"Inference: {inf_ms:.1f}ms | {len(det_list)} detections")
                    fps_placeholder.caption(f"File: {uploaded.name} | Model: {sel_run if actions_runs else 'N/A'}")

                    # Save detections to session state
                    for d in det_list:
                        st.session_state.behavior_history.append(d)
                    st.session_state.frame_count += 1

                    # Update attention scores
                    for i,d in enumerate(det_list):
                        tid = i+1
                        level = d["attention_level"]
                        if tid not in st.session_state.attention_scores:
                            st.session_state.attention_scores[tid] = float(CFG.attention.initial_score)
                        if   level=="HIGH":   st.session_state.attention_scores[tid] = min(100, st.session_state.attention_scores[tid]+CFG.attention.high_attention_increment)
                        elif level=="MEDIUM": st.session_state.attention_scores[tid] = min(100, st.session_state.attention_scores[tid]+CFG.attention.medium_attention_increment)
                        elif level=="LOW":    st.session_state.attention_scores[tid] = max(0,   st.session_state.attention_scores[tid]-CFG.attention.low_attention_decrement)

                elif uploaded.type.startswith("video"):
                    # Save to temp file and process
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded.read())
                    tfile.close()

                    cap = cv2.VideoCapture(tfile.name)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    vid_fps      = cap.get(cv2.CAP_PROP_FPS) or 30

                    st.info(f"Video: {total_frames} frames @ {vid_fps:.0f}fps — processing every 5th frame")
                    progress = st.progress(0)

                    frame_idx = 0
                    processed = 0
                    all_dets  = []
                    t_start   = time.time()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame_idx += 1

                        if frame_idx % 5 != 0:   # Process every 5th frame
                            continue

                        annotated_rgb, det_list = run_inference_on_frame(model, frame)
                        all_dets.extend(det_list)

                        img_placeholder.image(annotated_rgb, use_container_width=True,
                                              caption=f"Frame {frame_idx}/{total_frames}")
                        fps_live = processed / max(time.time()-t_start, 0.001)
                        fps_placeholder.caption(f"Processing @ {fps_live:.1f} eff-fps | Detections so far: {len(all_dets)}")
                        progress.progress(min(frame_idx/max(total_frames,1), 1.0))
                        processed += 1

                    cap.release()
                    os.unlink(tfile.name)
                    st.success(f"✅ Processed {processed} frames | Total detections: {len(all_dets)}")
                    st.session_state.behavior_history.extend(all_dets)
                    st.session_state.frame_count += processed

        elif uploaded is None and not is_pipeline_live:
            # Show live frame from pipeline if available
            live_frame_path = DASHBOARD_DIR/"live_frame.jpg"
            if live_frame_path.exists() and (time.time()-live_frame_path.stat().st_mtime) < 10:
                img = cv2.imread(str(live_frame_path))
                if img is not None:
                    img_placeholder.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                img_placeholder.info(
                    "**Upload a photo/video** to run inference, or start the live pipeline:\n\n"
                    "```bash\npython src/inference/realtime_classroom.py --headless --action-weights "
                    + (str(list(actions_runs.values())[0]["weights"]) if actions_runs else "models/weights/yolov5s_actions.pt")
                    + "\n```"
                )

    # ── Student attention panel ───────────────────────────────────────────────
    with side_col:
        st.subheader("👥 Detected Persons")

        # Try live pipeline data first
        live_scores = read_live_scores()
        scores_data = live_scores.get("scores", {})

        if scores_data:
            for tid, info in sorted(scores_data.items(), key=lambda x:-x[1].get("score",0)):
                sc    = info.get("score",50)
                lvl   = info.get("level","MEDIUM")
                name  = info.get("student_name",f"Track_{tid}")
                css   = "card-high" if lvl=="HIGH" else "card-medium" if lvl=="MEDIUM" else "card-low"
                icon  = "🟢" if lvl=="HIGH" else "🟡" if lvl=="MEDIUM" else "🔴"
                st.markdown(
                    f'<div class="student-card {css}"><strong>{icon} {name}</strong><br>'
                    f'Attention: <strong>{sc:.0f}%</strong> — {lvl}</div>',
                    unsafe_allow_html=True
                )
        elif st.session_state.attention_scores:
            for tid, sc in sorted(st.session_state.attention_scores.items(), key=lambda x:-x[1]):
                lvl = "HIGH" if sc>=60 else "MEDIUM" if sc>=40 else "LOW"
                css = "card-high" if lvl=="HIGH" else "card-medium" if lvl=="MEDIUM" else "card-low"
                icon= "🟢" if lvl=="HIGH" else "🟡" if lvl=="MEDIUM" else "🔴"
                st.markdown(
                    f'<div class="student-card {css}"><strong>{icon} Person {tid}</strong><br>'
                    f'Attention: <strong>{sc:.0f}%</strong> — {lvl}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Upload media or start pipeline to see detections")

        if st.button("🗑 Reset session scores"):
            st.session_state.attention_scores.clear()
            st.session_state.behavior_history.clear()
            st.session_state.frame_count = 0
            st.rerun()

    # ── Live detections bar charts ────────────────────────────────────────────
    if st.session_state.behavior_history or scores_data:
        st.divider()
        st.subheader("📊 Real-time Detection Analysis")
        c1,c2 = st.columns(2)

        beh_data = {}
        attn_data= {"HIGH":0,"MEDIUM":0,"LOW":0}

        # From pipeline
        live_beh = read_live_beh()
        if live_beh.get("behaviors"):
            beh_data  = live_beh["behaviors"]
            for tid,info in scores_data.items():
                attn_data[info.get("level","LOW")] = attn_data.get(info.get("level","LOW"),0)+1
        elif st.session_state.behavior_history:
            for d in st.session_state.behavior_history:
                beh_data[d["class_name"]] = beh_data.get(d["class_name"],0)+1
                attn_data[d["attention_level"]] += 1

        with c1:
            if beh_data:
                bdf = pd.DataFrame({"Behavior":list(beh_data),"Count":list(beh_data.values())})
                fig = px.bar(bdf,x="Behavior",y="Count",color="Behavior",color_discrete_map=BEH_COLORS,
                             title="Detected Behaviors")
                fig.update_layout(xaxis_tickangle=-30,showlegend=False)
                st.plotly_chart(dark(fig, 300), use_container_width=True, key="chart_1")

        with c2:
            if any(v>0 for v in attn_data.values()):
                adf = pd.DataFrame({"Level":list(attn_data),"Count":list(attn_data.values())})
                fig = px.pie(adf,values="Count",names="Level",color="Level",
                             color_discrete_map=ATTN_COLORS,title="Attention Distribution")
                st.plotly_chart(dark(fig, 300), use_container_width=True, key="chart_2")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Session Analytics
# ═══════════════════════════════════════════════════════════════════════════════
def page_session_analytics():
    st.markdown('<div class="main-title">📊 Session Analytics</div>', unsafe_allow_html=True)

    ts_df   = load_attention_ts()
    det_df  = load_detections_log()
    summ_df = load_attention_scores()
    att_df  = load_attendance_log()
    status  = read_live_status()

    # Metrics row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Status", "LIVE 🟢" if status.get("is_running") else "Offline 🔴")
    c2.metric("Frames Processed", f"{status.get('frame_count',0):,}")
    c3.metric("FPS", f"{status.get('fps',0):.1f}")
    c4.metric("Avg Attention", f"{status.get('avg_attention',0):.0f}%" if status else "—")
    c5.metric("Students", status.get("num_students",0))
    elapsed = status.get("elapsed_sec",0)
    c6.metric("Duration", f"{int(elapsed//60)}m {int(elapsed%60)}s")

    st.divider()

    if ts_df.empty and det_df.empty and summ_df.empty:
        st.info(
            "No session data yet. Run the monitoring pipeline to collect data:\n\n"
            "```bash\npython src/inference/realtime_classroom.py --headless\n```\n\n"
            "Data will appear here automatically."
        )
        return

    # Attention time series
    if not ts_df.empty:
        st.subheader("📈 Attention Over Time — Per Student")
        x_col = "time_min" if "time_min" in ts_df.columns else "timestamp"
        fig = px.line(ts_df, x=x_col, y="attention_score", color="student_name",
                      labels={x_col:"Time (min)","attention_score":"Attention %"},
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.add_hrect(y0=60,y1=100,fillcolor="#22c55e",opacity=0.05,line_width=0,annotation_text="HIGH")
        fig.add_hrect(y0=40,y1=60, fillcolor="#f59e0b",opacity=0.05,line_width=0,annotation_text="MEDIUM")
        fig.add_hrect(y0=0, y1=40, fillcolor="#ef4444",opacity=0.05,line_width=0,annotation_text="LOW")
        fig.add_hline(y=60,line_dash="dash",line_color="#22c55e",line_width=1)
        fig.add_hline(y=40,line_dash="dash",line_color="#ef4444",line_width=1)
        fig.update_layout(yaxis_range=[0,105],hovermode="x unified")
        st.plotly_chart(dark(fig, 380), use_container_width=True, key="chart_3")

    # Scores summary table
    if not summ_df.empty:
        st.subheader("🏆 Student Attention Summary")
        ranked = summ_df.sort_values("current_score",ascending=False)
        fig = px.bar(ranked,x="student_name",y="current_score",color="current_score",
                     color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],range_color=[0,100],
                     text="current_score",labels={"student_name":"Student","current_score":"Score"})
        fig.update_traces(texttemplate="%{text:.0f}%",textposition="outside")
        st.plotly_chart(dark(fig, 300), use_container_width=True, key="chart_4")
        st.dataframe(ranked,use_container_width=True)

    # Detection log analysis
    if not det_df.empty:
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("🎯 Behavior Distribution")
            if "behavior" in det_df.columns:
                bc = det_df["behavior"].value_counts().reset_index()
                bc.columns=["Behavior","Count"]
                fig=px.bar(bc,x="Behavior",y="Count",color="Behavior",color_discrete_map=BEH_COLORS)
                fig.update_layout(xaxis_tickangle=-30,showlegend=False)
                st.plotly_chart(dark(fig, 300), use_container_width=True, key="chart_5")
        with c2:
            st.subheader("📊 Attention Level Mix")
            if "attention_level" in det_df.columns:
                ac=det_df["attention_level"].value_counts().reset_index()
                ac.columns=["Level","Count"]
                fig=px.pie(ac,values="Count",names="Level",color="Level",color_discrete_map=ATTN_COLORS)
                st.plotly_chart(dark(fig, 300), use_container_width=True, key="chart_6")

        # Heatmap
        if "student_name" in det_df.columns and "timestamp" in det_df.columns and "raw_score" in det_df.columns:
            st.subheader("🌡 Attention Heatmap (Student × Time)")
            try:
                det_df["time_min"] = (det_df["timestamp"]-det_df["timestamp"].min())/60
                pivot = det_df.pivot_table(index="student_name",
                                           columns=pd.cut(det_df["time_min"],bins=min(12,len(det_df)//5+1)),
                                           values="raw_score",aggfunc="mean")
                pivot.columns=[f"{i.mid:.1f}m" for i in pivot.columns]
                fig=px.imshow(pivot,color_continuous_scale="RdYlGn",aspect="auto",zmin=0,zmax=100)
                st.plotly_chart(dark(fig, 300), use_container_width=True, key="chart_7")
            except: pass

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Metrics — reads REAL results.csv
# ═══════════════════════════════════════════════════════════════════════════════
def page_model_metrics(runs):
    st.markdown('<div class="main-title">🤖 Model Metrics</div>', unsafe_allow_html=True)

    if not runs:
        st.error(f"No training runs found in `{TRAINING_DIR}`. Train models first.")
        return

    action_runs  = {k:v for k,v in runs.items() if v["task"]=="actions"}
    emotion_runs = {k:v for k,v in runs.items() if v["task"]=="emotions"}

    tabs = []
    if action_runs:  tabs.append("🎯 Action Models")
    if emotion_runs: tabs.append("😊 Emotion Models")
    if not tabs:
        st.warning("No runs found"); return

    tab_objects = st.tabs(tabs)

    def render_run_metrics(run_name, r, tab):
        with tab:
            st.subheader(f"`{run_name}`")

            # Top metrics
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Variant", r.get("variant","?"))
            c2.metric("mAP@0.5",     f"{r.get('map50','?')}")
            c3.metric("mAP@0.5:0.95",f"{r.get('map5095','?')}")
            c4.metric("Precision",   f"{r.get('precision','?')}")
            c5.metric("Recall",      f"{r.get('recall','?')}")
            c6.metric("F1",          f"{r.get('f1','?')}")

            st.caption(f"Epochs trained: {r.get('epochs_trained','?')} | "
                       f"Weights size: {r.get('size_mb','?')} MB | "
                       f"Weights: `{r.get('weights','N/A')}`")
            st.divider()

            df = r["df"]
            df.columns = [c.strip() for c in df.columns]

            # Loss + mAP curves from real results.csv
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Training & Validation Loss**")
                fig = go.Figure()
                epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]

                def try_add(col_candidates, name, color, dash="solid", yaxis="y"):
                    for c in col_candidates:
                        if c in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df[epoch_col], y=pd.to_numeric(df[c],errors="coerce"),
                                name=name, line=dict(color=color,width=2,dash=dash), yaxis=yaxis
                            ))
                            return True
                    return False

                try_add(["train/box_loss","box_loss"],   "Train box",   "#4f8ef7")
                try_add(["train/obj_loss","obj_loss"],   "Train obj",   "#38bdf8")
                try_add(["train/cls_loss","cls_loss"],   "Train cls",   "#818cf8")
                try_add(["val/box_loss"],                "Val box",     "#f59e0b","dash")
                try_add(["val/obj_loss"],                "Val obj",     "#fb923c","dash")
                try_add(["val/cls_loss"],                "Val cls",     "#fbbf24","dash")
                fig.update_layout(xaxis_title="Epoch",yaxis_title="Loss")
                st.plotly_chart(dark(fig,320), use_container_width=True, key=f"{run_name}_chart_8")

            with c2:
                st.markdown("**mAP & P/R Curves**")
                fig2 = go.Figure()
                try_add(["metrics/mAP_0.5","metrics/mAP50","mAP_0.5"],       "mAP@0.5",     "#22c55e")
                try_add(["metrics/mAP_0.5:0.95","metrics/mAP50-95"],         "mAP@0.5:0.95","#16a34a","dash")
                try_add(["metrics/precision","Precision"],                    "Precision",   "#38bdf8")
                try_add(["metrics/recall","Recall"],                          "Recall",      "#f59e0b")
                fig2.update_layout(xaxis_title="Epoch",yaxis_range=[0,1])
                st.plotly_chart(dark(fig2,320), use_container_width=True, key=f"{run_name}_chart_9")

            # Curve images from training run folder
            st.markdown("**Saved Training Curves**")
            img_cols = st.columns(4)
            imgs_to_show = [
                ("P-R Curve", r["pr_curve"]),
                ("F1 Curve",  r["f1_curve"]),
                ("P Curve",   r["p_curve"]),
                ("R Curve",   r["r_curve"]),
            ]
            for i,(label,path) in enumerate(imgs_to_show):
                if path and path.exists():
                    with img_cols[i%4]:
                        st.image(str(path),caption=label,use_container_width=True)
                else:
                    with img_cols[i%4]:
                        st.caption(f"_{label}: not found_")

            # Confusion matrix
            c1,c2 = st.columns(2)
            with c1:
                if r["confusion_matrix"].exists():
                    st.image(str(r["confusion_matrix"]),caption="Confusion Matrix",use_container_width=True)
            with c2:
                if r["results_img"].exists():
                    st.image(str(r["results_img"]),caption="Training Results Summary",use_container_width=True)

            # Batch visualizations
            batch_imgs = [
                ("Labels", r["labels_img"]),
                ("Label Correlogram", r["labels_corr"]),
                ("Train Batch 0", r["train_batch0"]),
                ("Val Batch Labels", r["val_batch_labels"]),
                ("Val Batch Preds", r["val_batch_pred"]),
            ]
            avail_batch = [(l,p) for l,p in batch_imgs if p and p.exists()]
            if avail_batch:
                st.markdown("**Batch Visualization Samples**")
                bc = st.columns(min(len(avail_batch),3))
                for i,(l,p) in enumerate(avail_batch):
                    with bc[i%3]:
                        st.image(str(p),caption=l,use_container_width=True)

            # Raw results.csv
            with st.expander("📋 Raw results.csv"):
                st.dataframe(df,use_container_width=True)

    # Action runs — one subtab per run
    if action_runs and "🎯 Action Models" in tabs:
        with tab_objects[tabs.index("🎯 Action Models")]:
            run_names = list(action_runs.keys())
            if len(run_names)==1:
                render_run_metrics(run_names[0], action_runs[run_names[0]], st.container())
            else:
                subtabs = st.tabs([f"`{n[:25]}`" for n in run_names])
                for i,(rn,subtab) in enumerate(zip(run_names,subtabs)):
                    render_run_metrics(rn, action_runs[rn], subtab)

    if emotion_runs and "😊 Emotion Models" in tabs:
        with tab_objects[tabs.index("😊 Emotion Models")]:
            run_names = list(emotion_runs.keys())
            if len(run_names)==1:
                render_run_metrics(run_names[0], emotion_runs[run_names[0]], st.container())
            else:
                subtabs = st.tabs([f"`{n[:25]}`" for n in run_names])
                for i,(rn,subtab) in enumerate(zip(run_names,subtabs)):
                    render_run_metrics(rn, emotion_runs[rn], subtab)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: YOLOv5 Comparison — reads REAL results from ALL action runs
# ═══════════════════════════════════════════════════════════════════════════════
def page_comparison(runs):
    st.markdown('<div class="main-title">📈 YOLOv5 Variant Comparison</div>', unsafe_allow_html=True)

    action_runs = {k:v for k,v in runs.items() if v["task"]=="actions"}
    if not action_runs:
        st.error("No action training runs found."); return

    # Build comparison DataFrame
    rows = []
    for rname, r in action_runs.items():
        rows.append(dict(
            Run=rname,
            Variant=r.get("variant","?"),
            mAP50=r.get("map50"),
            mAP5095=r.get("map5095"),
            Precision=r.get("precision"),
            Recall=r.get("recall"),
            F1=r.get("f1"),
            Epochs=r.get("epochs_trained"),
            Size_MB=r.get("size_mb"),
        ))
    comp_df = pd.DataFrame(rows).sort_values("Variant")

    # Best run highlight
    best_idx = comp_df["mAP50"].idxmax() if comp_df["mAP50"].notna().any() else None

    st.subheader("📋 Full Comparison Table")
    def highlight_best(row):
        if best_idx is not None and row.name==best_idx:
            return ["background-color:rgba(34,197,94,0.15)"]*len(row)
        return [""]*len(row)
    st.dataframe(comp_df.style.apply(highlight_best,axis=1), use_container_width=True)

    st.divider()
    YCOLORS = ["#4f8ef7","#22c55e","#a78bfa","#f59e0b","#ef4444","#38bdf8","#fb923c","#f472b6"]

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("📊 mAP@0.5 — All Runs")
        fig=px.bar(comp_df,x="Variant",y="mAP50",color="Run",text="mAP50",
                   color_discrete_sequence=YCOLORS,barmode="group")
        fig.update_traces(texttemplate="%{text:.3f}",textposition="outside")
        fig.update_layout(yaxis_range=[0,1],showlegend=True)
        st.plotly_chart(dark(fig, 320), use_container_width=True, key="chart_10")

    with c2:
        st.subheader("⚡ Model Size vs mAP Trade-off")
        valid=comp_df.dropna(subset=["Size_MB","mAP50"])
        if not valid.empty:
            fig=px.scatter(valid,x="Size_MB",y="mAP50",text="Variant",color="Run",
                           color_discrete_sequence=YCOLORS,size=[20]*len(valid))
            fig.update_traces(textposition="top center")
            fig.update_layout(showlegend=False,xaxis_title="Weights Size (MB)",yaxis_title="mAP@0.5")
            st.plotly_chart(dark(fig, 320), use_container_width=True, key="chart_11")
        else:
            st.info("Size data unavailable (weights missing)")

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("🎯 Precision vs Recall")
        valid=comp_df.dropna(subset=["Precision","Recall"])
        if not valid.empty:
            fig=px.scatter(valid,x="Recall",y="Precision",text="Variant",color="Run",
                           color_discrete_sequence=YCOLORS,size=[20]*len(valid))
            fig.update_traces(textposition="top center")
            fig.update_layout(showlegend=False)
            st.plotly_chart(dark(fig, 300), use_container_width=True, key="chart_12")

    with c2:
        st.subheader("📊 All Metrics Grouped")
        metrics=["mAP50","mAP5095","Precision","Recall","F1"]
        melted=comp_df.melt(id_vars=["Variant","Run"],value_vars=metrics,var_name="Metric",value_name="Value").dropna()
        if not melted.empty:
            fig=px.bar(melted,x="Metric",y="Value",color="Run",barmode="group",
                       color_discrete_sequence=YCOLORS)
            fig.update_layout(yaxis_range=[0,1])
            st.plotly_chart(dark(fig, 300), use_container_width=True, key="chart_13")

    # Radar chart
    

    # Training loss overlay — compare across runs
    st.subheader("📉 Training Loss Overlay — All Runs")
    fig=go.Figure()
    for i,(rname,r) in enumerate(action_runs.items()):
        df=r["df"]; df.columns=[c.strip() for c in df.columns]
        epoch_col="epoch" if "epoch" in df.columns else df.columns[0]
        for mcol,mname,dash in [
            (["metrics/mAP_0.5","metrics/mAP50"],"mAP@0.5","solid"),
            (["val/box_loss","box_loss"],"Val Box Loss","dash"),
        ]:
            for c in mcol:
                if c in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df[epoch_col],y=pd.to_numeric(df[c],errors="coerce"),
                        name=f"{rname[:15]}|{mname}",
                        line=dict(color=YCOLORS[i%len(YCOLORS)],dash=dash,width=1.5),
                    ))
                    break
    fig.update_layout(hovermode="x unified",xaxis_title="Epoch")
    st.plotly_chart(dark(fig, 380), use_container_width=True, key="chart_15")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Training Visuals — show all PNGs/JPGs from all run dirs
# ═══════════════════════════════════════════════════════════════════════════════
def page_training_visuals(runs):
    st.markdown('<div class="main-title">🖼️ Training Visuals</div>', unsafe_allow_html=True)

    if not runs:
        st.error("No training runs found."); return

    sel_run = st.selectbox("Select run", list(runs.keys()))
    r = runs[sel_run]
    st.caption(f"Run: `{r['path']}`")
    st.divider()

    # Collect all image files in that run directory
    run_dir = r["path"]
    img_files = sorted(list(run_dir.glob("*.png")) + list(run_dir.glob("*.jpg")))

    if not img_files:
        st.warning("No image files found in this run directory.")
        return

    # Group them
    groups = {
        "Curves": [f for f in img_files if any(k in f.name for k in ["curve","Curve","results.png"])],
        "Confusion": [f for f in img_files if "confusion" in f.name.lower()],
        "Labels": [f for f in img_files if "label" in f.name.lower()],
        "Batches": [f for f in img_files if "batch" in f.name.lower()],
        "Other": [f for f in img_files if not any(k in f.name.lower() for k in ["curve","confusion","label","batch"])],
    }

    for group_name, files in groups.items():
        if not files: continue
        st.subheader(group_name)
        cols = st.columns(min(len(files),3))
        for i,f in enumerate(files):
            with cols[i%3]:
                st.image(str(f),caption=f.name,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Attendance
# ═══════════════════════════════════════════════════════════════════════════════
def page_attendance():
    st.markdown('<div class="main-title">📋 Attendance</div>', unsafe_allow_html=True)

    att_df = load_attendance_log()
    live_att = read_live_att()

    if not att_df.empty:
        present=(att_df["status"]=="Present").sum() if "status" in att_df.columns else 0
        total=len(att_df)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total",total); c2.metric("✅ Present",present)
        c3.metric("❌ Absent",total-present); c4.metric("Rate",f"{present/max(total,1)*100:.0f}%")
        def color_row(row):
            c="rgba(34,197,94,0.08)" if row.get("status")=="Present" else "rgba(239,68,68,0.08)"
            return [f"background-color:{c}"]*len(row)
        st.dataframe(att_df.style.apply(color_row,axis=1),use_container_width=True,height=400)
    elif live_att:
        present_list = live_att.get("present",[])
        total = live_att.get("total_enrolled",0)
        st.metric("Present",f"{len(present_list)}/{total}")
        for name in present_list:
            st.markdown(f"✅ {name}")
    else:
        st.info(
            "No attendance data yet.\n\n"
            "Enroll students and run the pipeline with attendance enabled:\n"
            "```bash\npython src/attendance/enroll_student.py\n"
            "python src/inference/realtime_classroom.py\n```"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    runs = discover_runs()
    page, uploaded = render_sidebar(runs)

    if   page == "📹 Live Inference":    page_live_inference(uploaded, runs)
    elif page == "📊 Session Analytics": page_session_analytics()
    elif page == "🤖 Model Metrics":     page_model_metrics(runs)
    elif page == "📈 YOLOv5 Comparison": page_comparison(runs)
    elif page == "🖼️ Training Visuals":  page_training_visuals(runs)
    elif page == "📋 Attendance":        page_attendance()

if __name__ == "__main__":
    main()