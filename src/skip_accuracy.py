"""skip_accuracy.py - how much agreement does frame skipping cost?

Method: run each detector on EVERY frame once (reference). For skip k, the
detector is 'used' only on frames where i % k == 0; on the other frames the
last detections are carried forward. On those skipped frames we compare the
carried boxes with the detector's own per-frame boxes (IoU >= 0.5 match).

NOTE: the reference is the same detector on every frame, not human ground
truth. This measures staleness caused by skipping, not detection correctness.

Usage (works from any folder):
    python src/skip_accuracy.py --video "C:\\Users\\USER\\Downloads\\clip.mp4" --side 640 --skips 2 3 5
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parent.parent
RES10_PROTO = str(ROOT / "models/baseline/deploy.prototxt")
RES10_MODEL = str(ROOT / "models/baseline/res10_300x300_ssd_iter_140000.caffemodel")
YUNET_MODEL = str(ROOT / "models/optimized/face_detection_yunet_2023mar.onnx")


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


class Res10:
    name = "res10"

    def __init__(self, conf):
        self.net = cv2.dnn.readNetFromCaffe(RES10_PROTO, RES10_MODEL)
        self.conf = conf

    def detect(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        out = self.net.forward()
        boxes = []
        for i in range(out.shape[2]):
            if out[0, 0, i, 2] >= self.conf:
                x1, y1, x2, y2 = out[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes


class YuNet:
    name = "yunet"

    def __init__(self, conf):
        self.det = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320),
                                             conf, 0.3, 5000)

    def detect(self, img):
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        if faces is None:
            return []
        return [tuple(f[:4]) for f in faces]


def read_frames(path, side, max_frames):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = side / max(h, w)
        frames.append(cv2.resize(f, (int(w * s), int(h * s))))
    cap.release()
    if not frames:
        raise SystemExit(f"No frames read from {path!r}. Check path/codec (use H.264).")
    return frames


def agreement(ref, carried):
    """Greedy match carried boxes to reference boxes. Returns (n_ref, n_matched, ious)."""
    used, ious = set(), []
    for r in ref:
        best, bj = 0.0, None
        for j, c in enumerate(carried):
            if j in used:
                continue
            v = iou(r, c)
            if v > best:
                best, bj = v, j
        if bj is not None and best >= 0.5:
            used.add(bj)
            ious.append(best)
    return len(ref), len(ious), ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--side", type=int, default=640)
    ap.add_argument("--skips", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--max-frames", type=int, default=2000)
    ap.add_argument("--res10-conf", type=float, default=0.5)
    ap.add_argument("--yunet-conf", type=float, default=0.5)  # same as res10 and as evaluate/benchmark
    ap.add_argument("--out", default=str(ROOT / "results/skip_accuracy.csv"))
    a = ap.parse_args()

    frames = read_frames(a.video, a.side, a.max_frames)
    print(f"{len(frames)} frames loaded at max side {a.side}")

    rows = []
    if a.res10_conf != a.yunet_conf:
        print(f"[warn] thresholds differ (res10 {a.res10_conf}, yunet {a.yunet_conf}); "
              "agreement rates are not comparable across detectors")
    for det, conf in ((Res10(a.res10_conf), a.res10_conf), (YuNet(a.yunet_conf), a.yunet_conf)):
        det.detect(frames[0])  # warm-up
        per_frame = [det.detect(f) for f in frames]
        print(f"{det.name}: reference done")
        for k in a.skips:
            n_ref = n_match = 0
            all_ious = []
            last = []
            for i, ref in enumerate(per_frame):
                if i % k == 0:
                    last = ref  # detector actually runs here
                    continue
                nr, nm, ious = agreement(ref, last)  # skipped frame
                n_ref += nr
                n_match += nm
                all_ious += ious
            rate = n_match / n_ref if n_ref else float("nan")
            miou = float(np.mean(all_ious)) if all_ious else float("nan")
            rows.append({"detector": det.name, "side": a.side, "skip": k, "conf": conf,
                         "skipped_frames_faces": n_ref,
                         "agreement_rate": round(rate, 4),
                         "mean_iou_of_matches": round(miou, 4)})
            print(f"  skip={k}: agreement {rate:.3f}  mean IoU {miou:.3f}  (faces={n_ref})")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {a.out}")


if __name__ == "__main__":
    main()
