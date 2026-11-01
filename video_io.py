"""Video reading helpers. Only needs OpenCV + numpy (no PyTorch)."""
import sys
import time

import cv2
import numpy as np


def sane_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 240:
        return 30.0
    return float(fps)


def open_source(source, width=None, height=None):
    """source: '0' or '1' = webcam index, anything else = path to a video file.
    Returns (capture, is_file, fps)."""
    source = str(source)
    is_file = not source.isdigit()
    if is_file:
        cap = cv2.VideoCapture(source)
    elif sys.platform.startswith("win"):
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)   # faster/more reliable on Windows
    else:
        cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        what = f"video file '{source}'" if is_file else f"camera {source}"
        raise SystemExit(f"Could not open {what}. For a webcam try --source 1 (or 2). "
                         "For a file, check the path.")
    if not is_file:
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap, is_file, sane_fps(cap)


class Clock:
    """Time of the current frame: video time for files, wall-clock for webcams.
    Call tick() after every successful cap.read(), then now()."""

    def __init__(self, is_file, fps):
        self.is_file, self.fps = is_file, fps
        self.count = 0
        self.start = time.monotonic()

    def tick(self):
        self.count += 1

    def now(self):
        if self.is_file:
            return (self.count - 1) / self.fps
        return time.monotonic() - self.start


def prepare_frame(frame_bgr, size):
    """OpenCV frame (BGR) -> resized RGB uint8 square.
    Used in training AND live, so it must stay identical in both places."""
    small = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


class ClipAug:
    """Random crop + brightness/contrast change. The same values are used for every frame
    of one clip, so the motion stays natural."""

    def __init__(self, rng):
        self.scale = rng.uniform(0.75, 1.0)
        self.cx, self.cy = rng.uniform(0, 1), rng.uniform(0, 1)
        self.alpha = rng.uniform(0.75, 1.25)     # contrast
        self.beta = rng.uniform(-20, 20)         # brightness

    def __call__(self, frame):
        h, w = frame.shape[:2]
        ch, cw = int(h * self.scale), int(w * self.scale)
        y0, x0 = int((h - ch) * self.cy), int((w - cw) * self.cx)
        return cv2.convertScaleAbs(frame[y0:y0 + ch, x0:x0 + cw], alpha=self.alpha, beta=self.beta)


def sample_clip_frames(path, feature_fps, size, aug=None):
    """Read a clip and keep `feature_fps` frames per second (evenly spaced in time).
    Returns a list of prepared RGB frames (size x size)."""
    cap = cv2.VideoCapture(str(path))
    fps = sane_fps(cap)
    frames, k, i = [], 0, 0                      # k = how many frames kept, i = frame number in file
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        prepared = None
        while i >= int(round(k * fps / feature_fps)):
            if prepared is None:
                prepared = prepare_frame(aug(frame) if aug else frame, size)
            frames.append(prepared)
            k += 1
        i += 1
    cap.release()
    return frames


def read_feature_frames(path, feature_fps):
    """Same frame sampling as sample_clip_frames, but returns the RAW BGR frames (no resize, no augmentation).
    Used when frames must be cropped to the person first."""
    cap = cv2.VideoCapture(str(path))
    fps = sane_fps(cap)
    frames, k, i = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while i >= int(round(k * fps / feature_fps)):
            frames.append(frame)
            k += 1
        i += 1
    cap.release()
    return frames


def prepare_clip(frames_bgr, size, aug=None, rects=None):
    """Crop each frame to rects[j] = (x0, y0, x1, y1) (if given), augment (if given), resize to size x size RGB."""
    out = []
    for j, f in enumerate(frames_bgr):
        if rects is not None:
            x0, y0, x1, y1 = rects[j]
            f = f[y0:y1, x0:x1]
        out.append(prepare_frame(aug(f) if aug else f, size))
    return out
