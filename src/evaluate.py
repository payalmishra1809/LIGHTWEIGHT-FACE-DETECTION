"""Accuracy evaluation: precision, recall, F1 at a fixed IoU threshold.

Annotation CSV columns: image,x,y,w,h,condition
(one row per ground-truth face; 'condition' e.g. normal, low_light, small, multi, occluded)
Images are read from --images (default data/raw).

Usage:
  python src/evaluate.py --detectors res10 yunet --sides 320 416 640
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
RES10_PROTO = ROOT / "models/baseline/deploy.prototxt"
RES10_WEIGHTS = ROOT / "models/baseline/res10_300x300_ssd_iter_140000.caffemodel"
YUNET_MODEL = ROOT / "models/optimized/face_detection_yunet_2023mar.onnx"


def build_detector(name, conf=0.5):
    if name == "res10":
        from baseline_detector import Res10Detector
        return Res10Detector(RES10_PROTO, RES10_WEIGHTS, conf)
    if name == "yunet":
        from optimized_detector import YuNetDetector
        return YuNetDetector(YUNET_MODEL, conf)
    raise ValueError(f"Unknown detector: {name}")


def model_path(name):
    return RES10_WEIGHTS if name == "res10" else YUNET_MODEL


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def match(preds, gts, thr=0.5):
    """Greedy matching, highest-score predictions first. Returns (tp, fp, fn)."""
    used = set()
    tp = 0
    for p in sorted(preds, key=lambda d: -d[4]):
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(p, g)
            if v > best:
                best, best_j = v, j
        if best >= thr:
            used.add(best_j)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def detect_resized(det, img, side=None):
    """Resize so the longest side == side (aspect kept), detect, map boxes back.

    For res10 this is only a pre-resample: the detector itself squashes to 300x300.
    """
    if side is None:
        return det.detect(img)
    h, w = img.shape[:2]
    s = side / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)
    return [(x / s, y / s, bw / s, bh / s, sc) for x, y, bw, bh, sc in det.detect(small)]


def load_annotations(path):
    boxes, cond = defaultdict(list), {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            boxes[r["image"]].append((float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])))
            cond[r["image"]] = r.get("condition") or "all"
    return boxes, cond


def evaluate(det, boxes, cond, img_dir, side=None, iou_thr=0.5):
    tot = defaultdict(lambda: [0, 0, 0])  # condition -> tp, fp, fn
    for name, gts in boxes.items():
        img = cv2.imread(str(Path(img_dir) / name))
        if img is None:
            print(f"[warn] cannot read {name}")
            continue
        tp, fp, fn = match(detect_resized(det, img, side), gts, iou_thr)
        for key in (cond[name], "overall"):
            tot[key][0] += tp
            tot[key][1] += fp
            tot[key][2] += fn
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "data/raw"))
    ap.add_argument("--ann", default=str(ROOT / "data/annotations/annotations.csv"))
    ap.add_argument("--detectors", nargs="+", default=["res10", "yunet"])
    ap.add_argument("--sides", nargs="+", type=int, default=[320, 416, 640])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "results/metrics.csv"))
    a = ap.parse_args()

    boxes, cond = load_annotations(a.ann)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "side", "condition", "tp", "fp", "fn",
                    "precision", "recall", "f1", "iou_thr", "conf"])
        for dn in a.detectors:
            det = build_detector(dn, a.conf)
            for side in a.sides:
                for c, (tp, fp, fn) in sorted(evaluate(det, boxes, cond, a.images, side, a.iou).items()):
                    p, r, f1 = prf(tp, fp, fn)
                    w.writerow([dn, side, c, tp, fp, fn, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", a.iou, a.conf])
                    if c == "overall":
                        print(f"{dn:6s} side={side:4d}  P={p:.3f} R={r:.3f} F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")
    print(f"Saved -> {a.out}")


if __name__ == "__main__":
    main()
