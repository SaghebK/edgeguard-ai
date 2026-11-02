import argparse, csv, json, time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def polygon_pixels(zone, width, height):
    return np.asarray([[round(x*width), round(y*height)] for x,y in zone], dtype=np.int32)

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def log_event(path, track_id, confidence, point):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp_utc", "event", "track_id", "confidence", "x", "y"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "restricted_area_entry", track_id if track_id is not None else "",
                    f"{confidence:.4f}", point[0], point[1]])

def main():
    ap = argparse.ArgumentParser(description="YOLO restricted-area intrusion detection")
    ap.add_argument("--model", default="yolov10n.pt", help="YOLO .pt or .onnx model")
    ap.add_argument("--source", default="0", help="Camera index or video path")
    ap.add_argument("--config", default="edgeguard/config.json")
    ap.add_argument("--output", default="outputs/events.csv")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = YOLO(args.model)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    states, last_alert = {}, {}
    fps_avg, alpha = None, 0.15
    print("EdgeGuard AI running. Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            h, w = frame.shape[:2]
            polygon = polygon_pixels(cfg["restricted_zone"], w, h)
            result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 conf=float(cfg.get("confidence_threshold", .35)),
                                 imgsz=args.imgsz,
                                 classes=[int(cfg.get("person_class_id", 0))],
                                 verbose=False)[0]
            cv2.polylines(frame, [polygon], True, (0, 165, 255), 2)
            p0 = tuple(polygon[0])
            cv2.putText(frame, cfg.get("zone_name", "RESTRICTED AREA"),
                        (p0[0], max(24, p0[1]-8)), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (0,165,255), 2)

            boxes = result.boxes
            if boxes is not None and len(boxes):
                coords = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None]*len(coords)
                for i, (b, score, tid) in enumerate(zip(coords, confs, ids)):
                    x1,y1,x2,y2 = map(int, b)
                    foot = (int((x1+x2)/2), y2)
                    inside = point_in_polygon(foot, polygon)
                    key = tid if tid is not None else f"det-{i}"
                    color = (0,0,255) if inside else (0,190,0)
                    label = f"person {score:.2f}" + (" | INTRUSION" if inside else "")
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.circle(frame, foot, 4, color, -1)
                    cv2.putText(frame, label, (x1,max(20,y1-7)),
                                cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
                    now = time.monotonic()
                    if inside and not states.get(key, False) and now-last_alert.get(key,0) >= float(cfg.get("event_cooldown_seconds",3)):
                        log_event(Path(args.output), tid, float(score), foot)
                        last_alert[key] = now
                        print(f"[ALERT] Entry detected (track={tid}, conf={score:.2f})")
                    states[key] = inside

            dt = time.perf_counter()-t0
            fps = 1/dt if dt else 0
            fps_avg = fps if fps_avg is None else alpha*fps+(1-alpha)*fps_avg
            cv2.putText(frame, f"FPS: {fps_avg:.1f}", (12,28),
                        cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,0),2)
            if not args.no_display:
                cv2.imshow("EdgeGuard AI", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
