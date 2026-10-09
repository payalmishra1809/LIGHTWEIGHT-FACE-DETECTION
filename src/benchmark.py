"""Efficiency benchmark: latency, FPS, model size, startup, peak RAM, CPU.

Images are preloaded so disk I/O is excluded from timing.
CPU% = process CPU time / wall time (can exceed 100% on multi-core).

Usage:
  python src/benchmark.py --sides 320 416 640 --repeats 3
  python src/benchmark.py --video clip.mp4 --skips 1 2 3 --sides 640   # frame skipping
"""
import argparse
import csv
import statistics as st
import time
from pathlib import Path

import cv2
import psutil

from evaluate import ROOT, build_detector, detect_resized, load_annotations, model_path

WARMUP = 10
PROC = psutil.Process()


def cpu_time():
    t = PROC.cpu_times()
    return t.user + t.system


def rss_mb():
    return PROC.memory_info().rss / 1024 ** 2


def bench_images(name, images, side, repeats, conf):
    rows = []
    for run in range(1, repeats + 1):
        t0 = time.perf_counter()
        det = build_detector(name, conf)
        startup_ms = (time.perf_counter() - t0) * 1000
        for i in range(WARMUP):                       # warm-up, discarded
            detect_resized(det, images[i % len(images)], side)
        times, peak = [], rss_mb()
        c0, w0 = cpu_time(), time.perf_counter()
        for img in images:
            t = time.perf_counter()
            detect_resized(det, img, side)
            times.append(time.perf_counter() - t)
            peak = max(peak, rss_mb())
        wall = time.perf_counter() - w0
        mean = st.mean(times)
        rows.append({
            "detector": name, "side": side, "run": run,
            "startup_ms": round(startup_ms, 2),
            "model_mb": round(model_path(name).stat().st_size / 1024 ** 2, 3),
            "mean_ms": round(mean * 1000, 3), "fps": round(1 / mean, 2),
            "peak_rss_mb": round(peak, 1),
            "cpu_percent": round((cpu_time() - c0) / wall * 100, 1),
            "cpu_ms_per_image": round((cpu_time() - c0) / len(images) * 1000, 2),
        })
    return rows


def bench_skip(name, frames, side, k, conf):
    det = build_detector(name, conf)
    for i in range(WARMUP):
        detect_resized(det, frames[i % len(frames)], side)
    last, faces_total = [], 0
    c0, w0 = cpu_time(), time.perf_counter()
    for n, f in enumerate(frames):
        if n % k == 0:
            last = detect_resized(det, f, side)       # other frames reuse last boxes
        faces_total += len(last)
    wall = time.perf_counter() - w0
    return {"detector": name, "side": side, "skip": k,
            "effective_fps": round(len(frames) / wall, 2),
            "cpu_percent": round((cpu_time() - c0) / wall * 100, 1),
            "avg_faces_per_frame": round(faces_total / len(frames), 3)}


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "data/raw"))
    ap.add_argument("--ann", default=str(ROOT / "data/annotations/annotations.csv"))
    ap.add_argument("--detectors", nargs="+", default=["res10", "yunet"])
    ap.add_argument("--sides", nargs="+", type=int, default=[320, 416, 640])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--video", help="video file for frame-skipping test")
    ap.add_argument("--skips", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0, help="limit OpenCV threads (1 = low-resource simulation)")
    ap.add_argument("--out", default=str(ROOT / "results/benchmark_results.csv"))
    a = ap.parse_args()
    if a.threads:
        cv2.setNumThreads(a.threads)
        print(f"OpenCV threads limited to {a.threads}")

    if a.video:
        cap, frames = cv2.VideoCapture(a.video), []
        while len(frames) < a.max_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        rows = [bench_skip(d, frames, s, k, a.conf)
                for d in a.detectors for s in a.sides for k in a.skips for _ in range(a.repeats)]
        for r in rows:
            print(r)
        write_csv(str(Path(a.out).with_name("frame_skip_results.csv")), rows)
        return

    boxes, _ = load_annotations(a.ann)
    images = [im for im in (cv2.imread(str(Path(a.images) / n)) for n in boxes) if im is not None]
    print(f"{len(images)} images loaded")
    rows = []
    for d in a.detectors:
        for s in a.sides:
            r = bench_images(d, images, s, a.repeats, a.conf)
            rows += r
            ms, fps, cpu = ([x[k] for x in r] for k in ("mean_ms", "fps", "cpu_percent"))
            cms = [x["cpu_ms_per_image"] for x in r]
            sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
            print(f"{d:6s} side={s:4d}  {st.mean(ms):7.2f}±{sd(ms):.2f} ms  "
                  f"{st.mean(fps):6.1f} FPS  CPU {st.mean(cpu):5.1f}% ({st.mean(cms):.1f} cpu-ms/img)  RSS {r[-1]['peak_rss_mb']} MB")
    write_csv(a.out, rows)


if __name__ == "__main__":
    main()
