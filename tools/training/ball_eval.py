"""Ball recall per hard case on a YOLO-format split (step 2.3).

Each labelled ball is put in one case from its box shape alone, since the labels carry no tags:
  fast     - motion blur stretches the box: long side / short side >= --blur-ratio
  partial  - a figure or hand hides part of it: box area < --partial-area x median ball area
  free     - everything else
A ball counts as found when a ball prediction above --conf overlaps it with IoU >= --iou.
Misses and false balls are drawn into --out for review by eye.

  ~/.venvs/pitchwise-export/bin/python tools/training/ball_eval.py \
      --model best-v4.pt --images <split>/images --out /tmp/ball_eval
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BALL = 0


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_balls(label_file, w, h):
    boxes = []
    if label_file.exists():
        for line in label_file.read_text().split("\n"):
            p = line.split()
            if len(p) == 5 and int(p[0]) == BALL:
                cx, cy, bw, bh = (float(v) for v in p[1:])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--blur-ratio", type=float, default=1.3)
    ap.add_argument("--partial-area", type=float, default=0.6)
    args = ap.parse_args()

    model = YOLO(args.model)
    images = sorted(args.images.glob(args.glob))
    labels_dir = args.images.parent / "labels"
    args.out.mkdir(parents=True, exist_ok=True)

    rows, frames = [], {}
    for path in images:
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gt = load_balls(labels_dir / (path.stem + ".txt"), w, h)
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, classes=[BALL], verbose=False)[0]
        preds = res.boxes.xyxy.cpu().numpy().tolist()
        scores = res.boxes.conf.cpu().numpy().tolist()
        used = set()
        for g in gt:
            best, best_j = 0.0, None
            for j, p in enumerate(preds):
                if j not in used and (v := iou(g, p)) > best:
                    best, best_j = v, j
            found = best >= args.iou
            if found:
                used.add(best_j)
            gw, gh = g[2] - g[0], g[3] - g[1]
            rows.append(dict(image=path.name, kind="gt", found=found, iou=round(best, 3),
                             ratio=max(gw, gh) / min(gw, gh), area=gw * gh, box=g))
        for j, p in enumerate(preds):
            if j not in used:
                rows.append(dict(image=path.name, kind="false", score=round(scores[j], 3), box=p))
        frames[path.name] = (path, preds, gt)

    gts = [r for r in rows if r["kind"] == "gt"]
    median_area = float(np.median([r["area"] for r in gts])) if gts else 0.0
    for r in gts:
        r["case"] = ("fast" if r["ratio"] >= args.blur_ratio
                     else "partial" if r["area"] < args.partial_area * median_area else "free")

    print(f"{len(images)} images, conf {args.conf}, IoU {args.iou}")
    print(f"{'case':<8}{'balls':>7}{'found':>7}{'recall':>8}")
    for case in ("free", "fast", "partial", "all"):
        sel = [r for r in gts if case == "all" or r["case"] == case]
        n, k = len(sel), sum(r["found"] for r in sel)
        print(f"{case:<8}{n:>7}{k:>7}{(k / n if n else float('nan')):>8.3f}")
    print(f"false balls: {sum(r['kind'] == 'false' for r in rows)}")

    # Draw every frame with a miss or a false ball: GT green, missed GT red, false prediction magenta.
    bad = {r["image"] for r in rows if (r["kind"] == "gt" and not r["found"]) or r["kind"] == "false"}
    for name in sorted(bad):
        path, preds, gt = frames[name]
        img = cv2.imread(str(path))
        for r in rows:
            if r["image"] != name:
                continue
            x1, y1, x2, y2 = (int(v) for v in r["box"])
            if r["kind"] == "gt":
                color, text = ((0, 255, 0), r["case"]) if r["found"] else ((0, 0, 255), f"MISS {r['case']}")
            else:
                color, text = (255, 0, 255), f"FALSE {r['score']}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(args.out / name), img)

    with open(args.out / "balls.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "kind", "case", "found", "iou", "score", "ratio", "area", "x1", "y1", "x2", "y2"])
        for r in rows:
            wr.writerow([r["image"], r["kind"], r.get("case", ""), r.get("found", ""), r.get("iou", ""),
                         r.get("score", ""), round(r.get("ratio", 0), 2), round(r.get("area", 0))] + [round(v) for v in r["box"]])
    print(f"review: {len(bad)} frames in {args.out}")


if __name__ == "__main__":
    main()
