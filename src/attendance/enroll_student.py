"""
Student enrollment script for the attendance system.
Captures face images from webcam or loads from folder.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config
from attendance.attendance_system import AttendanceSystem

CFG = get_config()


def main():
    parser = argparse.ArgumentParser(description="Enroll a student for attendance")
    parser.add_argument("name", help="Student name/ID")
    parser.add_argument(
        "--source",
        default="camera",
        help="'camera' for webcam or path to folder with face images",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train recognizer after enrollment",
    )

    args = parser.parse_args()

    CFG.paths.create_all()
    system = AttendanceSystem()

    if args.source == "camera":
        system.enroll_student(args.name, camera_source=args.camera)
    else:
        system.enroll_from_folder(args.source, args.name)

    if args.train:
        system.train_recognizer()


if __name__ == "__main__":
    main()