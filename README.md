# 🚀 Real-Time Aerial Object Detection & Trajectory Tracking

A real-time computer vision system for detecting, tracking, and predicting the trajectory of small aerial objects using **YOLOv8**, **OpenCV**, and a **Kalman Filter**. The project is designed for applications such as drone surveillance, airspace monitoring, and defense-inspired tracking systems.

## 📌 Overview

Detecting small, fast-moving aerial objects is challenging due to factors such as low object visibility, scale variations, motion blur, and dynamic backgrounds. This project addresses these challenges by combining deep learning-based object detection with state estimation techniques to build an efficient real-time tracking pipeline.

The system first detects aerial objects using a fine-tuned **YOLOv8** model and then predicts and smooths their movement using a **Kalman Filter**, enabling continuous tracking even during brief detection failures or occlusions. The architecture is conceptually similar to Electro-Optical Tracking Systems (EOTS) used in aerospace and defense applications for flight-vehicle tracking. :contentReference[oaicite:0]{index=0}

---

## 🎯 Objectives

- Detect small aerial objects with high accuracy.
- Track detected objects across video frames.
- Predict future object positions using a Kalman Filter.
- Achieve real-time performance with low latency.
- Provide a scalable framework for surveillance and aerospace monitoring applications. :contentReference[oaicite:1]{index=1}

---

## 🏗️ System Architecture

```
Video Input
      │
      ▼
Frame Preprocessing (OpenCV)
      │
      ▼
YOLOv8 Object Detection
      │
      ▼
Track Association
      │
      ▼
Kalman Filter Prediction
      │
      ▼
Trajectory Visualization
      │
      ▼
Output Video
```

The modular architecture allows each stage of the pipeline to be independently improved or replaced without affecting the rest of the system. :contentReference[oaicite:2]{index=2}

---

## ⚙️ Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Kalman Filter
- SciPy

---

## 📂 Dataset

A custom dataset containing **4,000+ manually annotated frames** of small aerial objects was created.

The dataset includes:
- Drones
- Small aerial targets
- Various flight conditions

Data augmentation techniques used:
- Rotation
- Scaling
- Brightness adjustment
- Motion blur simulation

These augmentations improve the model's robustness under real-world conditions. :contentReference[oaicite:3]{index=3}

---

## 🔍 Methodology

### 1. Dataset Preparation
- Collect aerial object images.
- Annotate bounding boxes.
- Split into training, validation, and testing datasets.

### 2. Object Detection
- Fine-tune a pretrained YOLOv8 model using transfer learning.
- Optimize hyperparameters for detecting small objects.

### 3. Object Tracking
- Pass detections into a Kalman Filter.
- Predict object position and velocity.
- Smooth noisy detections.
- Maintain tracking during temporary occlusions.

### 4. Visualization
- Draw bounding boxes.
- Display object IDs.
- Overlay predicted trajectory paths in real time. :contentReference[oaicite:4]{index=4}

---

## 📊 Performance

| Metric | Result |
|---------|--------|
| Detection Accuracy (mAP@0.5) | **91%** |
| Tracking Latency | **<35 ms** |
| Processing Speed | **28 FPS** |
| Dataset Size | **4,000+ Images** |

These results demonstrate that the system is capable of accurate and efficient real-time aerial object tracking. :contentReference[oaicite:5]{index=5}

---

## 🌍 Applications

- Drone Detection
- Anti-Drone Surveillance Systems
- Airspace Monitoring
- Airport Security
- Flight Path Monitoring
- Wildlife Monitoring
- Search and Rescue
- Defense & Aerospace Research :contentReference[oaicite:6]{index=6}

---

## 🚀 Future Improvements

Future enhancements may include:

- Multi-object tracking
- Extended Kalman Filter (EKF)
- DeepSORT integration
- NVIDIA Jetson deployment
- Radar and GPS sensor fusion
- Larger datasets with varied weather and lighting conditions :contentReference[oaicite:7]{index=7}

---

## 📁 Project Structure

```
Aerial-Object-Detection/
│
├── dataset/
├── models/
├── training/
├── tracking/
├── utils/
├── videos/
├── results/
├── requirements.txt
├── inference.py
├── train.py
└── README.md
```

---

## ▶️ Installation

```bash
git clone https://github.com/yourusername/Aerial-Object-Detection.git

cd Aerial-Object-Detection

pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python inference.py
```

---

## 📈 Results

The developed pipeline successfully detects and tracks small aerial objects in real time while predicting future trajectories using a Kalman Filter. The combination of deep learning and state estimation provides accurate, smooth, and reliable tracking suitable for surveillance and research applications. :contentReference[oaicite:8]{index=8}

---

## 👩‍💻 Author

**Monika Sengar**
B.Tech CSE(AI&ML)
Khwaja Moinuddin Chisti Language University Lucknow




