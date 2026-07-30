import argparse

import cv2

from .config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_AGE,
    DEFAULT_MIN_HITS,
    DEFAULT_MODEL,
    DEFAULT_SOURCE,
)
from .detector import AerialDetector
from .tracker import KalmanBoxTracker, match_detections_to_trackers


def build_parser():
    parser = argparse.ArgumentParser(description="Run aerial object detection and tracking")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Video path or camera index")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="Tracker association IoU threshold")
    parser.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE, help="Maximum age for a tracker")
    parser.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS, help="Minimum hits before drawing")
    parser.add_argument("--output", default=None, help="Optional output video path")
    return parser


def run(args):
    detector = AerialDetector(model_path=args.model, conf=args.conf, iou=args.iou)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    trackers = []
    next_track_id = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)

        for tracker in trackers:
            tracker.predict()

        matched_pairs, unmatched_trackers, unmatched_detections = match_detections_to_trackers(
            trackers,
            detections,
            iou_threshold=args.iou,
        )

        for tracker_idx, detection_idx in matched_pairs:
            trackers[tracker_idx].update(detections[detection_idx]["bbox"])

        for tracker_idx in unmatched_trackers:
            trackers[tracker_idx].time_since_update += 1
            trackers[tracker_idx].age += 1

        for detection_idx in unmatched_detections:
            tracker = KalmanBoxTracker(detections[detection_idx]["bbox"], next_track_id)
            next_track_id += 1
            trackers.append(tracker)

        active_trackers = []
        for tracker in trackers:
            if tracker.time_since_update <= args.max_age:
                active_trackers.append(tracker)
        trackers = active_trackers

        for tracker in trackers:
            if tracker.hits < args.min_hits:
                continue

            x1, y1, x2, y2 = tracker.get_bbox()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {tracker.id}",
                (int(x1), max(10, int(y1 - 5))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Aerial Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
