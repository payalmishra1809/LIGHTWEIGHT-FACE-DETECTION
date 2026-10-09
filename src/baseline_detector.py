"""Baseline: OpenCV Res10 SSD (Caffe) face detector. Fixed 300x300 input.

NOTE: the network's prior boxes are tied to a 300x300 input, so the blob size
cannot be changed. Whatever resolution the caller passes in, the image is
squashed (aspect ratio NOT preserved) to 300x300 inside detect(). In
evaluate.py / benchmark.py the --sides option therefore only changes an
intermediate resampling step for res10; its latency is expected to be flat
across sides. Report it that way (one speed figure, not three).
"""
import cv2


class Res10Detector:
    name = "res10"

    def __init__(self, proto, weights, conf=0.5):
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        self.conf = conf

    def detect(self, img):
        """Return list of (x, y, w, h, score) in pixel coordinates of img."""
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        out = self.net.forward()[0, 0]
        faces = []
        for d in out:
            score = float(d[2])
            if score < self.conf:
                continue
            x1, y1 = max(0.0, d[3] * w), max(0.0, d[4] * h)
            x2, y2 = min(float(w), d[5] * w), min(float(h), d[6] * h)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1, score))
        return faces
