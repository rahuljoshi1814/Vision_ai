from ultralytics import YOLO
import cv2
import os
from typing import List, Tuple

# Load YOLOv8 pretrained model (you can use a custom one too)
model = YOLO("yolov8n.pt")  # You can switch to yolov8m.pt or custom weights

def detect_objects_yolo(frame) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Runs YOLOv8 on a frame and returns detected labels with bounding boxes.

    Args:
        frame (np.array): Input image frame (BGR)

    Returns:
        List of tuples (label, (x1, y1, x2, y2))
    """
    results = model(frame)[0]
    detections = []
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = map(int, box[:6])
        label = model.names[int(cls_id)]
        detections.append((label, (x1, y1, x2, y2)))
    return detections
