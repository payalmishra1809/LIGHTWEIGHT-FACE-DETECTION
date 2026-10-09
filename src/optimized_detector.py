"""Lightweight: OpenCV YuNet (FaceDetectorYN, ONNX)."""
import cv2


class YuNetDetector:
    name = "yunet"

    def __init__(self, model, conf=0.5, nms=0.3, top_k=5000):
        self.det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), conf, nms, top_k)
        self._size = (320, 320)

    def detect(self, img):
        """Return list of (x, y, w, h, score) in pixel coordinates of img."""
        h, w = img.shape[:2]
        if (w, h) != self._size:
            self.det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.det.detect(img)
        if faces is None:
            return []
        return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])) for f in faces]
