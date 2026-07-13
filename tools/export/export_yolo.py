"""Export the baseline YOLO detector to ONNX for YoloOnnxDetector.

Usage (in a venv with `pip install ultralytics`):
    python export_yolo.py [model.pt]

The baseline uses yolo11m; ultralytics downloads the weights automatically.
Output: yolo11m.onnx next to the weights — copy it to pretrained_models/.
"""
import sys
from ultralytics import YOLO

model = YOLO(sys.argv[1] if len(sys.argv) > 1 else "yolo11m.pt")
model.export(format="onnx", imgsz=640, opset=17, simplify=True)
