# pyright: reportMissingImports=false

import argparse
from pathlib import Path
import os

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install them with:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc


class AerialTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4

    def predict(self):
        prediction = self.kf.predict()
        return int(prediction[0, 0]), int(prediction[1, 0])

    def update(self, cx, cy):
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        correction = self.kf.correct(measurement)
        return int(correction[0, 0]), int(correction[1, 0])


def build_parser():
    parser = argparse.ArgumentParser(description="Run the aerial tracking pipeline")
    parser.add_argument(
        "--source",
        default=None,
        help="Video file path or camera index. Defaults to demo.mp4 if present, otherwise 0.",
    )
    parser.add_argument("--output", default=None, help="Optional output video path")
    parser.add_argument(
        "--prefer-camera",
        dest="prefer_camera",
        action="store_true",
        default=True,
        help="Try to open the webcam when available (default: True)",
    )
    parser.add_argument(
        "--no-prefer-camera",
        dest="prefer_camera",
        action="store_false",
        help="Do not attempt to open the webcam; use demo video instead",
    )
    parser.add_argument("--show", dest="show", action="store_true", default=True, help="Display the processed video (default: True)")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not display the processed video")
    parser.add_argument("--target-class", type=int, default=None, help="Optional class ID to track")
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=False,
        help="Prompt in terminal to choose 'camera' or 'demo' when no --source is provided",
    )
    parser.add_argument("--camera", dest="camera", action="store_true", help="Use the camera immediately")
    parser.add_argument("--demo", dest="demo", action="store_true", help="Use the demo video immediately")
    parser.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Disable interactive prompt and use automatic source resolution",
    )
    parser.add_argument(
        "--loop",
        dest="loop",
        action="store_true",
        default=False,
        help="Loop input video files until 'q' is pressed",
    )
    parser.add_argument(
        "--no-loop",
        dest="loop",
        action="store_false",
        help="Do not loop input videos (default)",
    )
    parser.add_argument(
        "--open-output",
        dest="open_output",
        action="store_true",
        help="Open the output file in the default media player after creation",
    )
    parser.add_argument(
        "--no-open-output",
        dest="open_output",
        action="store_false",
        help="Do not open the output file after creation (default)",
    )
    return parser


def can_show_video():
    """Check if display is available by attempting to create a window."""
    try:
        cv2.namedWindow("__test_window__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test_window__")
        return True
    except Exception:
        return False


def resolve_source_path(source, prefer_camera=True):
    # If no source provided, prefer camera when requested, otherwise look for demo files
    if source is None:
        if prefer_camera:
            camera_test = cv2.VideoCapture(0)
            if camera_test.isOpened():
                camera_test.release()
                return "0"

        fallback_candidates = [
            Path(__file__).resolve().parent / "videos" / "demo.mp4",
            Path(__file__).resolve().parent / "demo.mp4",
            Path("videos/demo.mp4"),
            Path("demo.mp4"),
        ]
        for candidate in fallback_candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return "0"

    source_path = Path(source)
    if source_path.is_absolute():
        return str(source_path)

    candidates = [Path(__file__).resolve().parent / source_path, Path.cwd() / source_path, source_path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(source_path)


def main():
    parser = build_parser()
    args = parser.parse_args()

    target_class_id = args.target_class
    tracker = None
    
    # Simple menu for camera or demo video
    def show_startup_menu():
        while True:
            print("\n" + "="*50)
            print("AERIAL TRACKER - SELECT VIDEO SOURCE")
            print("="*50)
            print("  c) Camera")
            print("  d) Demo Video")
            print("  q) Quit")
            print("="*50)
            choice = input("Select option (c/d/q): ").strip().lower()
            
            if choice == "q":
                print("Exiting...")
                return None
            elif choice == "c":
                print("Attempting to open camera...")
                try:
                    cap_test = cv2.VideoCapture(0)
                    if cap_test.isOpened():
                        cap_test.release()
                        print("✓ Camera found!")
                        return "0"
                    else:
                        print("✗ Camera not available. Please try again or close other apps using the camera.")
                        cap_test.release()
                except Exception as e:
                    print(f"✗ Error accessing camera: {e}")
                continue
            elif choice == "d":
                demo_candidates = [
                    Path(__file__).resolve().parent / "videos" / "demo.mp4",
                    Path(__file__).resolve().parent / "demo.mp4",
                ]
                for d in demo_candidates:
                    if d.exists():
                        print(f"✓ Demo video found: {d}")
                        return str(d.resolve())
                print("✗ Demo video not found. Please place demo.mp4 in the videos/ folder.")
                continue
            else:
                print("✗ Invalid option. Please enter c, d, or q.")
                continue

    # Show menu and get selection
    selected_source = show_startup_menu()
    if selected_source is None:
        return
        args.source = selection

    source = resolve_source_path(selected_source, prefer_camera=False)
    tracker = AerialTracker("yolov8n.pt")

    def open_capture(src):
        # If src looks like a camera index, open with integer device id
        try:
            if isinstance(src, str) and src.isdigit():
                return cv2.VideoCapture(int(src))
        except Exception:
            pass
        return cv2.VideoCapture(src)

    cap = open_capture(source)
    if not cap.isOpened():
        fallback_video = str((Path(__file__).resolve().parent / "videos" / "demo.mp4").resolve())
        if Path(fallback_video).exists():
            print(f"Camera unavailable. Falling back to demo video: {fallback_video}")
            cap = cv2.VideoCapture(fallback_video)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {source}")
            print("Make sure demo.mp4 exists in the project folder or pass --source with a valid video file path.")
            return

    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Always try to show window if requested, with error handling
    show_window = args.show
    if show_window:
        print("Display window will be shown. The feed should appear in a separate OpenCV window.")

    print("Pipeline started successfully! Press 'q' to quit when showing the video window.")

    # Determine frame display delay from input FPS (ms per frame)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps and fps > 1:
        frame_delay_ms = max(1, int(1000.0 / fps))
    else:
        # fallback for cameras or unknown FPS
        frame_delay_ms = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # If looping is enabled and the source is a file, reopen and continue
            try:
                src_path = Path(source)
            except Exception:
                src_path = None
            is_file_source = src_path is not None and src_path.exists() and not (isinstance(source, str) and source.isdigit())
            if args.loop and is_file_source:
                cap.release()
                cap = open_capture(source)
                continue
            break

        results = tracker.model(frame, stream=False, verbose=False, imgsz=320, conf=0.4)
        detected = False
        cx, cy = 0, 0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if target_class_id is not None and class_id != target_class_id:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detected = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Detected Object",
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                break

        pred_x, pred_y = tracker.predict()

        if detected:
            track_x, track_y = tracker.update(cx, cy)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
            cv2.putText(
                frame,
                "Tracking Active",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            track_x, track_y = pred_x, pred_y
            cv2.putText(
                frame,
                "Target Lost - Predicting Trajectory",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.circle(frame, (track_x, track_y), 8, (255, 0, 0), 2)
        cv2.rectangle(frame, (10, 10), (360, 95), (0, 0, 0), -1)
        cv2.line(frame, (20, 55), (220, 55), (255, 255, 255), 1)
        cv2.putText(
            frame,
            "Aerial Object Detection & Tracking",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Kalman Filter + YOLOv8",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}",
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        if writer is not None:
            writer.write(frame)
        if show_window:
            try:
                cv2.imshow("Tracking Pipeline Output", frame)
                if cv2.waitKey(frame_delay_ms) & 0xFF == ord("q"):
                    break
            except Exception as e:
                print(f"Display error: {e}. Continuing without display.")
                show_window = False
    cap.release()
    if writer is not None:
        writer.release()
    # Optionally open the output file in the default media player (Windows)
    try:
        if args.open_output and args.output and Path(args.output).exists():
            try:
                os.startfile(str(Path(args.output).resolve()))
            except Exception:
                print(f"Could not open output file automatically: {args.output}")
    except NameError:
        # args may not be in scope if main wasn't run normally
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()