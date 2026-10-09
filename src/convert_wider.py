"""Build a balanced test set + annotations.csv from the WIDER FACE validation set.

Usage (Windows example):
  python src/convert_wider.py --wider "C:\\path\\WIDER_val" ^
      --split "C:\\path\\wider_face_split\\wider_face_val_bbx_gt.txt" --per-condition 50

--wider : folder that contains the 'images' directory (e.g. WIDER_val)
--split : wider_face_val_bbx_gt.txt

Selection rules (state these in your paper):
  - faces flagged 'invalid' are dropped
  - images with any face < 20 px (w or h) or > 30 faces are skipped
  - each image gets ONE condition label, by priority:
      extreme_light > occluded > small (median height < 40 px) > multi (>= 5 faces) > normal
  - random sample of N images per condition (seed 42)
WIDER attributes per face: x y w h blur expression illumination invalid occlusion pose
"""
import argparse
import csv
import random
import shutil
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse(split_file):
    lines = Path(split_file).read_text().splitlines()
    i, out = 0, []
    while i < len(lines):
        name = lines[i].strip()
        n = int(lines[i + 1])
        rows = lines[i + 2: i + 2 + max(n, 1)]   # n==0 still has one dummy row
        i += 2 + max(n, 1)
        faces = []
        if n > 0:
            for r in rows:
                v = list(map(int, r.split()))
                faces.append(v)
        out.append((name, faces))
    return out


def label(faces):
    n = len(faces)
    if sum(f[6] == 1 for f in faces) > n / 2:
        return "extreme_light"
    if any(f[8] > 0 for f in faces):
        return "occluded"
    if st.median(f[3] for f in faces) < 40:
        return "small"
    if n >= 5:
        return "multi"
    return "normal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wider", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--per-condition", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    pools = {}
    for name, faces in parse(a.split):
        faces = [f for f in faces if f[7] == 0 and f[2] > 0 and f[3] > 0]   # drop invalid
        if not faces or len(faces) > 30 or any(f[2] < 20 or f[3] < 20 for f in faces):
            continue
        pools.setdefault(label(faces), []).append((name, faces))

    random.seed(a.seed)
    raw = ROOT / "data/raw"
    ann = ROOT / "data/annotations"
    raw.mkdir(parents=True, exist_ok=True)
    ann.mkdir(parents=True, exist_ok=True)

    rows, copied = [], 0
    for cond, items in sorted(pools.items()):
        pick = random.sample(items, min(a.per_condition, len(items)))
        print(f"{cond:14s} available={len(items):5d} selected={len(pick)}")
        for name, faces in pick:
            src = Path(a.wider) / "images" / name
            if not src.exists():
                print(f"[warn] missing {src}")
                continue
            flat = name.replace("/", "_").replace("\\", "_")
            shutil.copy(src, raw / flat)
            copied += 1
            for f in faces:
                rows.append([flat, f[0], f[1], f[2], f[3], cond])

    with open(ann / "annotations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "x", "y", "w", "h", "condition"])
        w.writerows(rows)
    print(f"Copied {copied} images, wrote {len(rows)} boxes -> {ann / 'annotations.csv'}")


if __name__ == "__main__":
    main()
