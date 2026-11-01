"""Draw a rectangle around the person using background difference.

Assumes: fixed camera, fixed background, zero or one person in frame.
No model download, runs fine on CPU. Needs only OpenCV + numpy.

Standalone demo:
    python person_box.py --camera 0                     # live webcam
    python person_box.py --video clip.mp4               # test on a video file
    python person_box.py --camera 0 --save-bg bg.png    # also save the empty-room picture
Keys: b = recapture the empty-room background, q = quit.

The first 30 frames are used as the "empty room" picture, so start with nobody in front of the camera
(or press b when the frame is empty). For a company installation it is more robust to save one empty-room
picture once (--save-bg) and load it every time (realtime.py --bg-image bg.png).

Weak spot: clothes that are almost the same brightness as the background behind them are not detected
(difference below `diff_thresh`, default 30 on a 0-255 scale). Lower diff_thresh (e.g. 20) if that happens;
very small values pick up lighting noise.

Use inside your own code:
    from person_box import PersonBox
    box = PersonBox()
    # empty frames at startup:  box.calibrate(frame)   (~30 times)   or   box.set_background(empty_frame)
    rect = box.update(frame)          # (x, y, w, h) or None
    PersonBox.draw(frame, rect, label="typing")
"""
import argparse

import cv2
import numpy as np


class PersonBox:
    def __init__(self, diff_thresh=30, min_area_frac=0.01, smooth=0.5,
                 hold_frames=15, adapt=0.01, scale_width=320):
        self.diff_thresh = diff_thresh      # pixel difference that counts as "changed"
        self.min_area_frac = min_area_frac  # ignore blobs smaller than this fraction of the frame
        self.smooth = smooth                # 0 = no smoothing, closer to 1 = smoother but laggier
        self.hold_frames = hold_frames      # keep the last box this many frames after losing the person
        self.adapt = adapt                  # background drift rate while nobody is present
        self.scale_width = scale_width      # work on a small copy of the frame for speed
        self.bg = None
        self._calib = []
        self._box = None
        self._missing = 0
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    @property
    def ready(self):
        return self.bg is not None

    def _prep(self, frame):
        h, w = frame.shape[:2]
        s = self.scale_width / float(w)
        small = cv2.resize(frame, (self.scale_width, int(round(h * s))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (7, 7), 0), s

    def calibrate(self, frame):
        """Feed empty-room frames. The background is the average of the last 30."""
        gray, _ = self._prep(frame)
        self._calib.append(gray.astype(np.float32))
        self._calib = self._calib[-30:]
        self.bg = np.mean(self._calib, axis=0)

    def set_background(self, frame):
        """Use one picture of the empty room as the background (e.g. loaded from a file)."""
        gray, _ = self._prep(frame)
        self.bg = gray.astype(np.float32)
        self._calib = []
        self._box, self._missing = None, 0

    def reset(self):
        self.bg, self._calib, self._box, self._missing = None, [], None, 0

    def update(self, frame):
        """Return (x, y, w, h) in original frame coordinates, or None when nobody is there."""
        if self.bg is None:
            self.calibrate(frame)
            return None
        gray, s = self._prep(frame)
        if gray.shape != self.bg.shape:                     # saved background has another aspect ratio
            gray = cv2.resize(gray, (self.bg.shape[1], self.bg.shape[0]))
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.bg))
        _, mask = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=3)   # join head / arms / torso into one blob
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.min_area_frac * mask.shape[0] * mask.shape[1]
        cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]

        if cnts:
            x, y, w, h = cv2.boundingRect(np.vstack(cnts))
            fx = frame.shape[1] / mask.shape[1]             # small mask -> full frame, per axis
            fy = frame.shape[0] / mask.shape[0]
            new = np.array([x * fx, y * fy, (x + w) * fx, (y + h) * fy], dtype=np.float32)
            self._box = new if self._box is None else (
                self.smooth * self._box + (1 - self.smooth) * new)
            self._missing = 0
        else:
            self._missing += 1
            # Nobody there: let the background follow slow lighting changes.
            # (Never done while a person is present, so someone sitting still is not "absorbed".)
            cv2.accumulateWeighted(gray, self.bg, self.adapt)
            if self._missing > self.hold_frames:
                self._box = None

        if self._box is None:
            return None
        x1, y1, x2, y2 = self._box.astype(int)
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    @staticmethod
    def draw(frame, rect, color=(0, 255, 0), label="person", clear_top=130):
        """Rectangle + label. The label goes above the box when there is room (box top below `clear_top`
        pixels, which keeps it out of realtime.py's text bar), otherwise just inside the top edge."""
        if rect is None:
            return frame
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        ty = y - 6 if y >= clear_top else y + 24
        cv2.putText(frame, label, (x + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--calib-frames", type=int, default=30)
    ap.add_argument("--save-bg", default="", help="save the empty-room picture to this file")
    ap.add_argument("--no-window", action="store_true", help="headless test, no display")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open video source.")
    box = PersonBox()
    n = 0
    detected = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n < a.calib_frames:
            box.calibrate(frame)
            rect = None
            if n == a.calib_frames - 1 and a.save_bg:
                cv2.imwrite(a.save_bg, frame)
                print(f"Saved empty-room picture to {a.save_bg}")
        else:
            rect = box.update(frame)
            detected += rect is not None
        n += 1
        if not a.no_window:
            PersonBox.draw(frame, rect)
            if n <= a.calib_frames:
                cv2.putText(frame, "calibrating - keep frame empty", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("person box", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("b"):
                box.reset()
                n = 0
    cap.release()
    cv2.destroyAllWindows()
    print(f"frames: {n}, frames with a box: {detected}")


if __name__ == "__main__":
    main()
