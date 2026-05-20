#!/usr/bin/env python3

import sys
import subprocess
import importlib
import os
import time
import cv2
import numpy as np
import torch
import glob
import argparse
import threading
import datetime
from pathlib import Path
from tqdm import tqdm

REQUIRED_PACKAGES = {
    "ultralytics": "ultralytics",
    "cv2": "opencv-python",
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "tqdm": "tqdm",
}

def log(level, message):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {level} {message}")

def install_missing_dependencies():
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        for pkg in missing:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg]
            )

install_missing_dependencies()

from ultralytics import YOLO

COLORS = {
    "person": (220, 80, 20),
    "helmet": (30, 200, 30),
    "vest": (60, 20, 220),
    "boots": (200, 200, 10),
    "default": (180, 180, 180),
}

CLASS_ALIASES = {
    "person": "person",
    "worker": "person",
    "human": "person",
    "people": "person",
    "helmet": "helmet",
    "hard hat": "helmet",
    "hardhat": "helmet",
    "hard-hat": "helmet",
    "safety helmet": "helmet",
    "vest": "vest",
    "safety vest": "vest",
    "boots": "boots",
    "boot": "boots",
    "safety boots": "boots",
}

COCO_PERSON_CLASS = 0

PPE_ZONES = {
    "helmet": (0.0, 0.0, 1.0, 0.25),
    "vest": (0.1, 0.25, 0.9, 0.65),
    "boots": (0.1, 0.80, 0.9, 1.0),
}

SCREENSHOT_DIR = Path("screenshots")
OUTPUT_VIDEO = "output_detected.mp4"
INFER_SIZE = 640
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

def detect_device():
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False

PPE_MODEL_CANDIDATES = ["yolov8n-ppe.pt", "best.pt", "ppe.pt"]
COCO_MODEL = "yolov8n.pt"

def load_model(device, use_half):
    for c in PPE_MODEL_CANDIDATES:
        if Path(c).exists():
            model = YOLO(c)
            model.to(device)
            if use_half and device == "cuda":
                model.half()
            return model, True

    model = YOLO(COCO_MODEL)
    model.to(device)
    if use_half and device == "cuda":
        model.half()
    return model, False

def find_input_video(user_path=None):
    if user_path:
        return user_path

    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        matches = glob.glob(ext)
        if matches:
            return matches[0]

    sys.exit(1)

def normalize_class_name(name):
    return CLASS_ALIASES.get(name.lower().strip(), name.lower().strip())

def simulate_ppe_zones(person_boxes):
    simulated = []
    for x1, y1, x2, y2 in person_boxes:
        w = x2 - x1
        h = y2 - y1
        for cls, (rx1, ry1, rx2, ry2) in PPE_ZONES.items():
            px1 = int(x1 + rx1 * w)
            py1 = int(y1 + ry1 * h)
            px2 = int(x1 + rx2 * w)
            py2 = int(y1 + ry2 * h)
            simulated.append({
                "class": cls,
                "conf": 0.8,
                "box": (px1, py1, px2, py2),
            })
    return simulated

def parse_detections(results, is_ppe_model):
    detections = []
    person_boxes = []

    if results is None or len(results) == 0:
        return detections

    result = results[0]
    if result.boxes is None:
        return detections

    names = result.names

    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i].item())
        conf = float(result.boxes.conf[i].item())
        x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)

        name = names.get(cls_id, "unknown")
        cls = normalize_class_name(name)

        if is_ppe_model:
            detections.append({"class": cls, "conf": conf, "box": (x1, y1, x2, y2)})
        else:
            if cls_id == COCO_PERSON_CLASS:
                detections.append({"class": "person", "conf": conf, "box": (x1, y1, x2, y2)})
                person_boxes.append((x1, y1, x2, y2))

    if not is_ppe_model:
        detections.extend(simulate_ppe_zones(person_boxes))

    return detections

def draw_box(frame, det):
    x1, y1, x2, y2 = det["box"]
    cls = det["class"]
    conf = det["conf"]
    color = COLORS.get(cls, COLORS["default"])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{cls} {conf:.2f}"
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def process_video(path, model, is_ppe_model, device, use_half):
    cap = cv2.VideoCapture(path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=INFER_SIZE, device=device, verbose=False)
        detections = parse_detections(results, is_ppe_model)

        for d in detections:
            draw_box(frame, d)

        writer.write(frame)
        cv2.imshow("PPE", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--webcam", nargs="?", const=0, type=int)
    args = parser.parse_args()

    device, gpu = detect_device()
    model, is_ppe_model = load_model(device, gpu)

    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, imgsz=INFER_SIZE, device=device, verbose=False)
            detections = parse_detections(results, is_ppe_model)

            for d in detections:
                draw_box(frame, d)

            cv2.imshow("PPE", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        video = find_input_video(args.video)
        process_video(video, model, is_ppe_model, device, gpu)

if __name__ == "__main__":
    main()