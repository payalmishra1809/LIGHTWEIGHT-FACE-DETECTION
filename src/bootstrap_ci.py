"""bootstrap_ci.py - 95% image-level bootstrap confidence intervals for overall P/R/F1.

Per-image (tp, fp, fn) are computed once per detector/side at conf 0.5, IoU 0.5;
images are then resampled with replacement (seed 0) and micro-averaged P/R/F1
are recomputed for each resample.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from evaluate import ROOT, build_detector, detect_resized, load_annotations, match


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results/bootstrap_ci.csv"))
    a = ap.parse_args()
    boxes, _ = load_annotations(ROOT / "data/annotations/annotations.csv")
    names = list(boxes)
    imgs = [cv2.imread(str(ROOT / "data/raw" / n)) for n in names]
    rng = np.random.default_rng(0)
    rows = []
    for dn in ("res10", "yunet"):
        det = build_detector(dn, 0.5)
        for side in (320, 416, 640):
            c = np.array([match(detect_resized(det, im, side), boxes[n], 0.5) for n, im in zip(names, imgs)], float)
            stats = []
            for _ in range(a.n_boot):
                s = c[rng.integers(0, len(c), len(c))].sum(0)
                stats.append(prf(*s))
            stats = np.array(stats)
            pt = prf(*c.sum(0))
            row = {"detector": dn, "side": side}
            for i, k in enumerate(("precision", "recall", "f1")):
                lo, hi = np.percentile(stats[:, i], [2.5, 97.5])
                row.update({k: round(pt[i], 4), k + "_lo": round(lo, 4), k + "_hi": round(hi, 4)})
            rows.append(row); print(row)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("Saved ->", a.out)


if __name__ == "__main__":
    main()
