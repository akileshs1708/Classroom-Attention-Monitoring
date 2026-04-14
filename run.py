"""
Master runner script for the Classroom Monitoring System.
Provides a simple CLI to run any part of the pipeline.
"""

import sys
import argparse
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Classroom Monitoring System",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "command",
        choices=[
            "init",
            "download",
            "prepare-actions",
            "prepare-emotions",
            "train-actions",
            "train-emotions",
            "evaluate",
            "monitor",
            "monitor-headless",
            "enroll",
            "dashboard",
            "live",
            "demo",
        ],
        help="""Command to run:
  init             - Initialize project structure
  download         - Download datasets from Kaggle
  prepare-actions  - Prepare SCB-05 behavior dataset
  prepare-emotions - Prepare emotion dataset
  train-actions    - Train YOLOv5 action model
  train-emotions   - Train YOLOv5 emotion model
  evaluate         - Evaluate and compare models
  monitor          - Start real-time monitoring (with display)
  monitor-headless - Start monitoring without display (for dashboard)
  enroll           - Enroll a student for attendance
  dashboard        - Launch Streamlit dashboard only
  live             - Launch monitor (headless) + dashboard together
  demo             - Quick demo with pretrained COCO model
""",
    )

    parser.add_argument("--source", default=None, help="Video source for monitor")
    parser.add_argument("--name", default=None, help="Student name for enrollment")
    parser.add_argument("--variant", default="yolov5s", help="YOLOv5 variant")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--no-display", action="store_true", help="Headless mode")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    if args.command == "init":
        from config import init_project
        init_project()

    elif args.command == "download":
        from dataset_utils.download_datasets import main as download_main
        download_main()

    elif args.command == "prepare-actions":
        from dataset_utils.prepare_actions_dataset import main as prep_actions
        prep_actions()

    elif args.command == "prepare-emotions":
        from dataset_utils.prepare_emotions_dataset import main as prep_emotions
        prep_emotions()

    elif args.command == "train-actions":
        from train.train_actions import train_single_variant, train_all_variants
        if args.variant == "all":
            train_all_variants()
        else:
            train_single_variant(args.variant, epochs=args.epochs)

    elif args.command == "train-emotions":
        from train.train_emotions import main as train_emo
        train_emo()

    elif args.command == "evaluate":
        from evaluate.compare_models import evaluate_all_action_models
        evaluate_all_action_models()

    elif args.command == "monitor":
        from inference.realtime_classroom import ClassroomMonitor

        source = args.source
        if source is not None:
            try:
                source = int(source)
            except ValueError:
                pass

        monitor = ClassroomMonitor(
            video_source=source if source is not None else 0,
            enable_display=not args.no_display,
        )
        monitor.run()

    elif args.command == "monitor-headless":
        from inference.realtime_classroom import ClassroomMonitor
        source = args.source
        monitor = ClassroomMonitor(
            video_source=source,
            enable_display=False,
        )
        monitor.run_headless(max_frames=args.max_frames)

    elif args.command == "enroll":
        if not args.name:
            print("Error: --name required for enrollment")
            sys.exit(1)
        from attendance.attendance_system import AttendanceSystem
        system = AttendanceSystem()
        system.enroll_student(args.name)
        system.train_recognizer()

    elif args.command == "dashboard":
        import subprocess
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        print(f"Launching dashboard on port {args.port}...")
        subprocess.run([
            "streamlit", "run", str(dashboard_path),
            "--server.port", str(args.port),
            "--server.headless", "true",
        ])

    elif args.command == "live":
        """
        Start both the headless monitor AND the dashboard.
        Monitor runs in background, dashboard in foreground.
        """
        import threading
        from inference.realtime_classroom import ClassroomMonitor, SharedState

        source = args.source

        print("=" * 60)
        print("LAUNCHING LIVE MODE")
        print("  Monitor: headless (background)")
        print(f"  Dashboard: http://localhost:{args.port}")
        print("  Press Ctrl+C to stop")
        print("=" * 60)

        # Start monitor in thread
        shared = SharedState()
        monitor = ClassroomMonitor(
            video_source=source,
            enable_display=False,
            enable_emotions=False,
            enable_attendance=False,
            shared_state=shared,
        )

        monitor_thread = monitor.run_in_thread(max_frames=args.max_frames)

        # Start dashboard in foreground
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        try:
            subprocess.run([
                "streamlit", "run", str(dashboard_path),
                "--server.port", str(args.port),
                "--server.headless", "true",
            ])
        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")
            monitor.stop()
            monitor_thread.join(timeout=5)
            print("[DONE]")

    elif args.command == "demo":
        print("Running demo with pretrained YOLOv5s (COCO)...")
        from inference.realtime_classroom import ClassroomMonitor

        source = args.source
        if source is not None:
            try:
                source = int(source)
            except ValueError:
                pass

        monitor = ClassroomMonitor(
            video_source=source if source is not None else 0,
            enable_emotions=False,
            enable_attendance=False,
            enable_display=not args.no_display,
        )
        monitor.run()


if __name__ == "__main__":
    main()