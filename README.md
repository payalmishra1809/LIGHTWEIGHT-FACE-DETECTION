# Lightweight Face Detection for Resource-Constrained Devices

Baseline (OpenCV Res10 SSD) vs lightweight (YuNet) comparison.

## Setup
pip install -r requirements.txt

Place models:
- models/baseline/deploy.prototxt
- models/baseline/res10_300x300_ssd_iter_140000.caffemodel
- models/optimized/face_detection_yunet_2023mar.onnx

Place test images in data/raw/ and annotations in data/annotations/annotations.csv
(columns: image,x,y,w,h,condition).

## Run
python src/evaluate.py --detectors res10 yunet --sides 320 416 640
python src/benchmark.py --sides 320 416 640 --repeats 3
python src/benchmark.py --video clip.mp4 --skips 1 2 3 --sides 640

## Fixed settings
Confidence 0.5, IoU 0.5, same hardware and test set for all runs.

## Notes (read before reporting results)
- res10 has a fixed 300x300 input. `--sides` only changes a pre-resample for it, so its
  speed is flat across sides; treat it as one speed figure.
- Tables/plots: `python src/plot_results.py --drop-first 3` (uses the pooled "overall" rows of metrics.csv).
- Extra analysis: `python src/pr_curve.py` (AP, PR curves), `python src/bootstrap_ci.py` (95% CIs).
- Frame-skip agreement: `python src/skip_accuracy.py --video clip.mp4 --side 640 --skips 2 3 5`
  now uses confidence 0.5 for BOTH detectors by default and writes a `conf` column.
