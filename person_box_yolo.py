"""Draw a rectangle around the person with a pretrained YOLO person detector.

Better than person_box.py (background difference): it knows what a person is, so it needs no empty-room
picture, ignores moved chairs / laptops / phones, and also works if the camera or lighting changes.
Same interface as PersonBox, so realtime.py can use either one (--box-model yolo | motion).

Install (Colab or PC):
    pip install ultralytics
The weights file (yolov8n.pt, ~6 MB) is downloaded automatically the first time.

Standalone demo:
    python person_box_yolo.py --camera 0
    python person_box_yolo.py --video clip.mp4 --save-video boxed.mp4 --no-window   # e.g. on Colab
    python person_box_yolo.py --image photo.jpg --save-video boxed.jpg --no-window

Speed: yolov8n is small. On a GPU (Colab) it takes a few ms per frame. On a CPU it takes roughly 50-150 ms,
so `every=3` runs the detector on every 3rd frame and keeps the last box in between (about 10 boxes/s at 30 fps).
Bigger weights (yolov8s.pt, yolov8m.pt) are more accurate and need a GPU.

Assumes zero or one person in the frame: if several are found, the largest box is used.
Camera angle matters: if only head and shoulders are visible (close webcam), lower `conf` (e.g. 0.25).
"""
import argparse

import cv2
import numpy as np

from person_box import PersonBox

PERSON_CLASS = 0     # COCO class id of "person"


class YoloPersonBox:
    def __init__(self, weights="yolov8n.pt", conf=0.35, imgsz=640, device=None,
                 every=3, smooth=0.5, hold_frames=15):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("person_box_yolo needs ultralytics:  pip install ultralytics") from e
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device     # device: None = auto, "cpu", 0 = first GPU
        self.every = max(1, int(every))                              # detector runs on every Nth frame
        self.smooth = smooth                                         # 0 = no smoothing, closer to 1 = smoother
        self.hold_frames = hold_frames                               # keep the last box this many frames
        self._box = None                                             # smoothed (x1, y1, x2, y2)
        self._target = None                                          # latest raw detection
        self._missing = 0
        self._n = 0

    # --- same interface as PersonBox (no empty-room picture needed here) ---------------------------
    ready = True

    def calibrate(self, frame):
        pass

    def set_background(self, frame):
        pass

    def reset(self):
        self._box, self._target, self._missing, self._n = None, None, 0, 0

    # -----------------------------------------------------------------------------------------------
    def _detect(self, frame):
        res = self.model.predict(frame, classes=[PERSON_CLASS], conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        xyxy = res.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return xyxy[int(np.argmax(areas))].astype(np.float32)        # largest person

    def update(self, frame):
        """Return (x, y, w, h) in frame coordinates, or None when nobody is there."""
        if self._n % self.every == 0:
            det = self._detect(frame)
            if det is not None:
                self._target = det
                self._missing = 0
            else:
                self._missing += self.every
                if self._missing > self.hold_frames:
                    self._target = None
        self._n += 1

        if self._target is None:
            self._box = None
            return None
        self._box = self._target.copy() if self._box is None else (
            self.smooth * self._box + (1 - self.smooth) * self._target)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        return x1, y1, x2 - x1, y2 - y1

    draw = staticmethod(PersonBox.draw)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0)
    src.add_argument("--video", default=None)
    src.add_argument("--image", default=None)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--every", type=int, default=1, help="run the detector on every Nth frame")
    ap.add_argument("--device", default=None, help="cpu, or 0 for the first GPU (default: automatic)")
    ap.add_argument("--save-video", default="", help="write the result to this video / image file")
    ap.add_argument("--no-window", action="store_true", help="no display (use this on Colab)")
    a = ap.parse_args()

    dev = int(a.device) if a.device and a.device.isdigit() else a.device
    box = YoloPersonBox(a.weights, a.conf, device=dev, every=a.every)

    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            raise SystemExit(f"Cannot read image {a.image}")
        rect = box.update(frame)
        print("box (x, y, w, h):", rect)
        PersonBox.draw(frame, rect)
        if a.save_video:
            cv2.imwrite(a.save_video, frame)
            print("saved", a.save_video)
        return

    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open video source.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, n, detected = None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rect = box.update(frame)
        n += 1
        detected += rect is not None
        PersonBox.draw(frame, rect)
        if a.save_video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if not a.no_window:
            cv2.imshow("person box (yolo)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
        print("saved", a.save_video)
    cv2.destroyAllWindows()
    print(f"frames: {n}, frames with a box: {detected}")


if __name__ == "__main__":
    main()
