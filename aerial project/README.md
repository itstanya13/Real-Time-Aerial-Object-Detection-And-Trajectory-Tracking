# Real-Time Aerial Object Detection & Trajectory Tracking

This project is a starter implementation for a real-time aerial object detection and tracking pipeline using Python, OpenCV, and YOLOv8.

## Goals
- Detect small aerial objects from video or webcam input.
- Track them across frames using a Kalman filter-style tracker.
- Provide a foundation for fine-tuning YOLOv8 on custom aerial datasets.

## Project structure
- `src/` contains the detection, tracking, and inference code.
- `data/` is intended for videos, annotations, and dataset files.
- `models/` can store trained YOLOv8 weights.

## Setup
1. Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the demo:
   ```bash
   python -m src.inference --source 0
   ```

## Next steps
- Replace the default pretrained model with a fine-tuned YOLOv8 checkpoint.
- Add dataset annotation pipelines and training scripts.
- Improve the tracker with better motion models and ID management.
