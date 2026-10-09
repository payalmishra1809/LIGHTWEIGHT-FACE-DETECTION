"""pr_curve.py - threshold-free accuracy: precision-recall curves, AP@IoU and best-F1.

Detectors are run with a very low score threshold (0.01); every detection is
kept with its score. Detections are pooled over the whole test set, sorted by
score, and greedily matched to unmatched ground-truth boxes of the same image
(IoU >= thr). AP is the all-point-interpolated area under the PR curve.

Usage:
  python src/pr_curve.py --sides 320 416 640
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from evaluate import ROOT, build_detector, detect_resized, iou, load_annotations


def pr_curve(all_dets, n_gt, thr):
    """all_dets: list of (score, is_tp). Returns precision, recall arrays (score-descending)."""
    all_dets.sort(key=lambda d: -d[0])
    tp = np.cumsum([d[1] for d in all_dets], dtype=float)
    fp = np.cumsum([1 - d[1] for d in all_dets], dtype=float)
    return tp / np.maximum(tp + fp, 1e-9), tp / n_gt, np.array([d[0] for d in all_dets])


def ap_all_point(prec, rec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def collect(det, boxes, img_dir, side, thr):
    dets, n_gt = [], 0
    for name, gts in boxes.items():
        img = cv2.imread(str(Path(img_dir) / name))
        if img is None:
            continue
        n_gt += len(gts)
        used = set()
        for p in sorted(detect_resized(det, img, side), key=lambda d: -d[4]):
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j in used:
                    continue
                v = iou(p, g)
                if v > best:
                    best, bj = v, j
            ok = best >= thr
            if ok:
                used.add(bj)
            dets.append((p[4], int(ok)))
    return dets, n_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "data/raw"))
    ap.add_argument("--ann", default=str(ROOT / "data/annotations/annotations.csv"))
    ap.add_argument("--sides", nargs="+", type=int, default=[320, 416, 640])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "results/pr_ap.csv"))
    a = ap.parse_args()
    boxes, _ = load_annotations(a.ann)
    rows, curves = [], {}
    for dn in ("res10", "yunet"):
        det = build_detector(dn, conf=0.01)
        for side in a.sides:
            dets, n_gt = collect(det, boxes, a.images, side, a.iou)
            prec, rec, sc = pr_curve(dets, n_gt, a.iou)
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
            k = int(np.argmax(f1))
            rows.append({"detector": dn, "side": side, "ap": round(ap_all_point(prec, rec), 4),
                         "max_recall": round(float(rec[-1]), 4), "best_f1": round(float(f1[k]), 4),
                         "best_f1_score_thr": round(float(sc[k]), 4), "n_dets": len(dets)})
            print(rows[-1])
            curves[f"{dn}_{side}_prec"], curves[f"{dn}_{side}_rec"] = prec, rec
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(Path(a.out).with_suffix(".npz"), **curves)
    print(f"Saved -> {a.out}")


if __name__ == "__main__":
    main()
