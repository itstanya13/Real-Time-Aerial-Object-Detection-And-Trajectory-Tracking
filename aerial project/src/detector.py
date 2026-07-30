from ultralytics import YOLO


class AerialDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.25, iou=0.7):
        self.model = YOLO(model_path)
        self.confidence_threshold = conf
        self.iou_threshold = iou

    def detect(self, frame):
        results = self.model(
            frame,
            stream=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=640,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf.item())
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                detections.append({"bbox": (x1, y1, x2, y2), "confidence": confidence})

        return detections
