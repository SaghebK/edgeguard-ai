import argparse
from ultralytics import YOLO
p=argparse.ArgumentParser()
p.add_argument("--weights", default="yolov10n.pt")
p.add_argument("--imgsz", type=int, default=640)
a=p.parse_args()
out=YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, simplify=True)
print("Exported:", out)
