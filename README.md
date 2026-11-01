# Real-Time Human Activity Classification

A deep learning-based system for recognizing human activities from video streams using **YOLO, ResNet18, and GRU**. The system combines spatial feature extraction and temporal sequence learning to classify activities and monitor them in real time through a webcam.

## Overview

This project focuses on recognizing human activities using video clips collected across multiple recording sessions, individuals, and camera perspectives.

The system identifies five activity classes:

- **Sleep** – Detecting sleeping or resting activity.
- **Phone** – Detecting smartphone usage.
- **Writing** – Detecting writing activity.
- **Typing** – Detecting keyboard typing.
- **Background** – Other activities and non-target scenarios.

The model uses a combination of CNN-based visual feature extraction and GRU-based temporal learning to understand both individual frames and changes in activity over time.

## System Architecture

The project follows a multi-stage pipeline:

1. **Video Preprocessing:** Videos are divided into short clips and sampled at 10 FPS.
2. **Person Detection:** YOLO-based person detection and cropping help focus on the relevant subject.
3. **Spatial Feature Extraction:** A pretrained ResNet18 extracts 512-dimensional visual features from each frame.
4. **Temporal Learning:** A GRU-based classifier learns sequential patterns across frames, incorporating motion-related features.
5. **Activity Classification:** The trained model predicts one of the five activity classes.
6. **Real-Time Monitoring:** Rolling-window inference, prediction smoothing, and activity-duration tracking are used for webcam-based monitoring.

## Model Architecture

| Component | Purpose |
|---|---|
| YOLO | Person detection and cropping |
| ResNet18 | Spatial feature extraction |
| GRU | Temporal sequence learning |
| Motion Features | Capturing activity-related movement |
| Classification Head | Predicting activity classes |

**Feature extraction:**
- Pretrained ResNet18 backbone.
- Frozen CNN parameters.
- 512-dimensional feature representation per frame.

**Temporal classification:**
- GRU processes sequential frame features.
- Uses 16-frame windows sampled at 10 FPS.
- Predicts the activity based on temporal information.

## Dataset Preparation

The dataset was collected across multiple recording sessions, individuals, and camera perspectives to introduce variations in appearance, posture, and viewing angles.

Preprocessing pipeline:

- Video clips generated with approximately 3.2-second duration and 1.6-second stride.
- Frames sampled at 10 FPS.
- Frames resized to 224 × 224.
- Person detection and cropping applied where appropriate.
- Sequential features extracted and stored for model training.

The dataset includes five classes: Background, Phone, Sleep, Typing, and Writing.

## Training and Evaluation

- Used a pretrained ResNet18 for spatial feature extraction.
- Extracted frame-level features and stored them for efficient training.
- Trained a GRU-based temporal classifier.
- Used recording-session-based data splitting to evaluate generalization across sessions.
- Evaluated performance using accuracy, precision, recall, and F1-score.

### Performance

| Metric | Score |
|---|---:|
| Accuracy | 83.4% |
| Macro F1-score | 82.9% |

*Evaluation performed on held-out recording sessions.*

## Real-Time Activity Monitoring

The trained model supports webcam-based activity recognition through rolling-window inference.

Key features:
- Continuous activity prediction.
- Prediction smoothing to reduce fluctuations.
- Activity transition detection.
- Activity duration tracking.
- CSV-based activity logging.

## Project Structure

```text
Activity-Classifier/
│
├── data/
│   ├── Background/
│   ├── Phone/
│   ├── Sleep/
│   ├── Typing/
│   └── Write/
│
├── extract_features.py
├── train.py
├── realtime.py
├── video_io.py
├── person_crop.py
├── common.py
├── activity_tracker.py
├── eval_by_session.py
├── cut_clips.py
│
├── README.md
└── requirements.txt
```

## Technologies Used

- Python
- PyTorch
- Torchvision
- YOLO
- ResNet18
- GRU
- OpenCV
- NumPy
- Pandas
- Scikit-learn

## Future Improvements

- Improve person detection and cropping reliability.
- Improve recognition of visually similar activities.
- Extend the system to more activity classes.
- Develop a web-based monitoring dashboard.
- Improve robustness across different environments and camera perspectives.

## Key Highlights

- Combines CNN-based spatial feature extraction with GRU-based temporal learning.
- Uses multi-session, multi-person video data.
- Employs session-based evaluation to assess generalization.
- Supports real-time webcam activity recognition and duration tracking.
