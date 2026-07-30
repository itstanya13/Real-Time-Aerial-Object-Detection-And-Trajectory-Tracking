import numpy as np


class KalmanBoxTracker:
    def __init__(self, bbox, track_id):
        self.id = track_id
        self.time_since_update = 0
        self.hits = 1
        self.age = 1

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        self.kf = np.array(
            [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

        self.P = np.diag([100.0, 100.0, 100.0, 100.0, 10.0, 10.0, 10.0, 10.0])
        self.F = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]).astype(np.float32)
        self.R = np.diag([10.0, 10.0, 10.0, 10.0]).astype(np.float32)

    def predict(self):
        self.kf = self.F @ self.kf
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_bbox()

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        z = np.array([cx, cy, w, h], dtype=np.float32)

        y = z - self.H @ self.kf
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.kf = self.kf + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.hits += 1
        self.time_since_update = 0
        return self.get_bbox()

    def get_bbox(self):
        cx, cy, w, h = self.kf[:4]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return (x1, y1, x2, y2)


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def match_detections_to_trackers(trackers, detections, iou_threshold):
    if not trackers:
        return [], [], list(range(len(detections)))

    matched_pairs = []
    unmatched_trackers = list(range(len(trackers)))
    unmatched_detections = list(range(len(detections)))
    cost_matrix = np.full((len(trackers), len(detections)), fill_value=1.0)

    for tracker_idx, tracker in enumerate(trackers):
        predicted_box = tracker.predict()
        for det_idx, detection in enumerate(detections):
            cost_matrix[tracker_idx, det_idx] = 1.0 - compute_iou(predicted_box, detection["bbox"])

    while True:
        if len(unmatched_trackers) == 0 or len(unmatched_detections) == 0:
            break

        min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
        tracker_idx, det_idx = min_idx
        if cost_matrix[tracker_idx, det_idx] > 1.0 - iou_threshold:
            break

        matched_pairs.append((tracker_idx, det_idx))
        unmatched_trackers.remove(tracker_idx)
        unmatched_detections.remove(det_idx)
        cost_matrix[tracker_idx, :] = 1.0
        cost_matrix[:, det_idx] = 1.0

    return matched_pairs, unmatched_trackers, unmatched_detections
