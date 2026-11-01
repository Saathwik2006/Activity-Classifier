"""Crop every frame to the person (YOLO), so the image network sees the PERSON, not the whole room.

The SAME class is used when making features (extract_features.py --crop-person) and live (realtime.py),
so training and live use see identical crops. Settings are saved in features/meta.json and in model.pt;
realtime.py reads them from model.pt, so you never have to repeat them.

Rules
  * YOLO runs on every `every`-th feature frame (default 2); in between the last box is kept.
  * The largest person box is used (hanging clothes, posters etc. are usually smaller).
  * The box is smoothed (EMA) so the crop does not jitter.
  * If nobody is found for a while (`hold` detections), the WHOLE frame is used. Background clips with
    nobody in view therefore look the same in training and live.
  * top_frac < 1 keeps only the top part of a TALL person box (head, torso, arms, hands at the desk).
  * objects=True: a LAPTOP or PHONE that overlaps / touches the person is added to the crop, so the laptop
    does not get cut out of Typing and the phone stays in Phone. Objects far from the person are ignored.
  * The crop is the region plus `margin` on each side, as square as the frame allows, then resized to the
    network's square input like before.

Preview before the long extraction (needs: pip install ultralytics):
    python person_crop.py --video clips/Typing/room_Gowtham_P1__Typing_01__0003.mp4 --margin 0.1 --top-frac 0.6 --objects
The picture shows the original frames with the boxes (top) and what the network will see (bottom).
"""
import argparse

import cv2
import numpy as np

PERSON, LAPTOP, PHONE = 0, 63, 67           # COCO class ids
NAMES = {PERSON: "person", LAPTOP: "laptop", PHONE: "phone"}


def crop_rect(box, W, H, margin=0.2, top_frac=1.0):
    """box = (x1, y1, x2, y2) or None -> integer crop (x0, y0, x1, y1) inside the W x H frame.
    No box -> the whole frame.
    top_frac < 1 keeps only the top part of a TALL box (head, torso, arms, hands at the desk), because a seated
    person's box usually runs down to the legs under the desk and then the crop cannot zoom in at all."""
    if box is None:
        return 0, 0, W, H
    x1, y1, x2, y2 = box
    if top_frac < 1.0 and (y2 - y1) > (x2 - x1):
        y2 = y1 + (y2 - y1) * top_frac
    w, h = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
    side = max(w, h) * (1 + 2 * margin)             # square-ish crop around the person
    cw, ch = min(side, W), min(side, H)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0 = min(max(cx - cw / 2, 0), W - cw)
    y0 = min(max(cy - ch / 2, 0), H - ch)
    x0, y0 = int(round(x0)), int(round(y0))
    xe, ye = min(W, max(x0 + 2, int(round(x0 + cw)))), min(H, max(y0 + 2, int(round(y0 + ch))))
    return x0, y0, xe, ye


def near_person(person, obj, grow=0.2):
    """True if obj overlaps the person box grown by `grow` x its width on every side."""
    px1, py1, px2, py2 = person
    d = grow * (px2 - px1)
    return not (obj[2] < px1 - d or obj[0] > px2 + d or obj[3] < py1 - d or obj[1] > py2 + d)


class PersonCropper:
    def __init__(self, weights="yolov8n.pt", conf=0.35, margin=0.2, every=2, smooth=0.6,
                 hold=4, imgsz=640, device=None, top_frac=1.0, objects=False, obj_conf=0.25):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("person crop needs ultralytics:  pip install ultralytics") from e
        self.model = YOLO(weights)
        self.weights, self.conf, self.margin, self.every = weights, conf, margin, max(1, int(every))
        self.smooth, self.hold, self.imgsz, self.device = smooth, hold, imgsz, device
        self.top_frac, self.objects, self.obj_conf = top_frac, bool(objects), obj_conf
        self.reset()

    def settings(self):
        """What is saved with the features / model so live use can rebuild the same cropper."""
        return {"weights": self.weights, "conf": self.conf, "margin": self.margin,
                "every": self.every, "smooth": self.smooth, "hold": self.hold, "imgsz": self.imgsz,
                "top_frac": self.top_frac, "objects": self.objects, "obj_conf": self.obj_conf}

    def reset(self):
        self._box, self._ub, self._n, self._miss = None, None, 0, 0
        self.last_objects = []          # [(name, conf, (x1, y1, x2, y2))] near the person at the last detection

    def _detect(self, frame):
        """-> (person box or None, list of (class_id, conf, box) for objects near the person)."""
        classes = [PERSON, LAPTOP, PHONE] if self.objects else [PERSON]
        res = self.model.predict(frame, classes=classes, conf=min(self.conf, self.obj_conf), imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int) if self.objects else np.zeros(len(xyxy), int)
        people = [(b, c) for b, c, k in zip(xyxy, conf, cls) if k == PERSON and c >= self.conf]
        if not people:
            return None, []
        person = max(people, key=lambda bc: (bc[0][2] - bc[0][0]) * (bc[0][3] - bc[0][1]))[0].astype(np.float32)
        extras = [(int(k), float(c), b.astype(np.float32)) for b, c, k in zip(xyxy, conf, cls)
                  if k != PERSON and c >= self.obj_conf and near_person(person, b)]
        return person, extras

    def _union_target(self, person, extras):
        x1, y1, x2, y2 = person
        if self.top_frac < 1.0 and (y2 - y1) > (x2 - x1):
            y2 = y1 + (y2 - y1) * self.top_frac
        ub = [x1, y1, x2, y2]
        for _, _, b in extras:
            ub = [min(ub[0], b[0]), min(ub[1], b[1]), max(ub[2], b[2]), max(ub[3], b[3])]
        return np.array(ub, np.float32)

    def update(self, frame):
        """Call once per feature frame, in order. Returns the crop rect (x0, y0, x1, y1)."""
        if self._n % self.every == 0:
            det, extras = self._detect(frame)
            if det is not None:
                self._box = det if self._box is None else self.smooth * self._box + (1 - self.smooth) * det
                self._miss = 0
                self.last_objects = [(NAMES.get(k, str(k)), c, tuple(map(float, b))) for k, c, b in extras]
                if self.objects:
                    tgt = self._union_target(det, extras)
                    self._ub = tgt if self._ub is None else self.smooth * self._ub + (1 - self.smooth) * tgt
            else:
                self._miss += 1
                if self._miss > self.hold:
                    self._box, self._ub, self.last_objects = None, None, []
        self._n += 1
        H, W = frame.shape[:2]
        if self.objects and self._ub is not None:
            return crop_rect(self._ub, W, H, self.margin, 1.0)      # top_frac is already inside the union
        return crop_rect(self._box, W, H, self.margin, self.top_frac)

    def crop(self, frame):
        x0, y0, x1, y1 = self.update(frame)
        return frame[y0:y1, x0:x1]

    def rects(self, frames):
        """Crop rects for all frames of one clip (starts fresh)."""
        self.reset()
        return [self.update(f) for f in frames]

    def box_xywh(self):
        """Current smoothed person box as (x, y, w, h), or None when nobody is tracked (for drawing)."""
        if self._box is None:
            return None
        x1, y1, x2, y2 = self._box
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="preview.png")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--top-frac", type=float, default=1.0, help="e.g. 0.6 = crop only the upper 60%% of a tall person box")
    ap.add_argument("--objects", action="store_true", help="also keep a laptop / phone that touches the person in the crop")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    from video_io import read_feature_frames
    frames = read_feature_frames(a.video, 10.0)
    if not frames:
        raise SystemExit(f"Cannot read {a.video}")
    dev = int(a.device) if a.device and a.device.isdigit() else a.device
    cropper = PersonCropper(a.weights, a.conf, a.margin, a.every, device=dev, top_frac=a.top_frac, objects=a.objects)
    cropper.reset()
    rects, boxes, objs = [], [], []
    for f in frames:
        rects.append(cropper.update(f))
        boxes.append(cropper.box_xywh())
        objs.append(list(cropper.last_objects))
    idx = sorted({int(round(i)) for i in np.linspace(0, len(frames) - 1, 6)})
    top, bottom = [], []
    for i in idx:
        f = frames[i].copy()
        x0, y0, x1, y1 = rects[i]
        if boxes[i]:
            bx, by, bw, bh = boxes[i]
            cv2.rectangle(f, (bx, by), (bx + bw, by + bh), (0, 220, 0), 3)
        for name, c, b in objs[i]:
            cv2.rectangle(f, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 200, 0), 3)
        cv2.rectangle(f, (x0, y0), (x1 - 1, y1 - 1), (0, 230, 255), 3)
        t = cv2.resize(f, (320, int(round(320 * f.shape[0] / f.shape[1]))))          # keep the true aspect ratio
        top.append(cv2.copyMakeBorder(t, 0, 240 - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) if t.shape[0] < 240 else t[:240])
        tile = np.zeros((240, 320, 3), np.uint8)                       # what the network sees: 224 x 224
        tile[8:232, 48:272] = cv2.resize(frames[i][y0:y1, x0:x1], (224, 224))
        bottom.append(tile)
    cv2.imwrite(a.out, np.vstack([np.hstack(top), np.hstack(bottom)]))
    full = sum(r == (0, 0, frames[0].shape[1], frames[0].shape[0]) for r in rects)
    print(f"{len(frames)} frames; {full} used the whole frame (no person found). Saved {a.out}")
    print("green = person, blue = laptop/phone kept in the crop, yellow = the crop")


if __name__ == "__main__":
    main()
