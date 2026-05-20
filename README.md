# Industrial AI Vision

A real-time **Industrial AI Vision System** for detecting Personal Protective Equipment (PPE) using **YOLO (Ultralytics) + OpenCV**.

This system is designed to detect safety compliance in real-time from video streams or webcam feeds.

It identifies:
- Person
- Helmet
- Safety Vest
- Boots

It outputs an annotated video with bounding boxes, FPS counter, timestamps, and system overlays.

---

## Features

- Real-time object detection using YOLO
- Video file and webcam support
- Automatic dependency handling (no manual setup required)
- GPU acceleration with CUDA support (optional)
- CPU fallback mode
- PPE compliance visualization layer
- FPS, frame count, and timestamp overlay
- Screenshot capture
- Export processed video output
- Pause and resume controls

---

## Technology Overview

- Computer Vision  
  https://en.wikipedia.org/wiki/Computer_vision

- YOLO (You Only Look Once) Object Detection  
  https://en.wikipedia.org/wiki/You_Only_Look_Once

- OpenCV Library  
  https://en.wikipedia.org/wiki/OpenCV

- Personal Protective Equipment (PPE)  
  https://en.wikipedia.org/wiki/Personal_protective_equipment

---

## Installation

No manual setup required.

Run the script directly:

```bash
python ppe_detection.py
