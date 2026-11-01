"""Step 4: run the trained model live.

    python realtime.py                                  # webcam 0
    python realtime.py --source 1                       # another camera
    python realtime.py --source test.mp4                # a video file
    python realtime.py --log activities.csv             # one CSV row per finished activity
    python realtime.py --source test.mp4 --headless     # no window, prints to the console
    python realtime.py --source test.mp4 --headless --save-video out.mp4   # no window (Colab): write a video
    python realtime.py --box-model motion               # simple background-difference box instead of YOLO
    python realtime.py --no-box                         # no rectangle

Press q (or Esc) in the window to quit.

The rectangle follows the person (0 or 1 person) and is labelled with the stable activity. Default is a YOLO
person detector (--box-model yolo, needs `pip install ultralytics`, falls back to the simple box if missing).
The simple box (--box-model motion) needs a fixed camera and background, and an empty-room picture: the first
~30 frames, or --bg-image bg.png; press b in the window to recapture it.

The raw model output jumps around (a short pause while typing looks like "background").
activity_tracker.py turns it into stable activities with a grace period:
  --smooth 10        average the last 10 predictions (about 1 s) first
  --switch-after 3   a new activity must hold for 3 s before it replaces the current one
  --end-after 10     an activity ends only after 10 s without seeing it (short pauses stay inside it)
"""
import argparse
import collections
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch

from activity_tracker import ActivityTracker
from common import build_backbone, build_head, compute_features
from person_box import PersonBox
from person_crop import PersonCropper
from video_io import Clock, open_source, prepare_frame

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    backbone = build_backbone(ckpt["backbone"], pretrained=False)     # weights come from model.pt
    backbone.load_state_dict(ckpt["backbone_state"])
    h = ckpt["head"]
    head = build_head(h["kind"], ckpt["feature_dim"], len(ckpt["labels"]), h["hidden"], h["layers"], h["dropout"],
                      h.get("motion", False))
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    return ckpt, backbone, head


def make_formatter(is_file, wall0):
    """Time as text: mm:ss.s inside a video file, clock time (HH:MM:SS) for a webcam."""
    def fmt(t):
        if is_file:
            return f"{int(t) // 60:02d}:{t % 60:04.1f}"
        return datetime.fromtimestamp(wall0 + t).strftime("%H:%M:%S")
    return fmt


def draw_overlay(frame, activity, elapsed, idle, lines, feat_ms, head_ms):
    shaded = frame.copy()
    cv2.rectangle(shaded, (0, 0), (frame.shape[1], 112), (0, 0, 0), -1)
    cv2.addWeighted(shaded, 0.6, frame, 0.4, 0, frame)
    colour = (190, 190, 190) if idle else (0, 220, 0)
    title = activity if idle else f"{activity}  {int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
    cv2.putText(frame, title, (10, 38), FONT, 1.0, colour, 2)
    for j, text in enumerate(lines[:3]):
        cv2.putText(frame, text, (10, 62 + 18 * j), FONT, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"CNN {feat_ms:.0f} ms | head {head_ms:.1f} ms | q = quit",
                (10, frame.shape[0] - 10), FONT, 0.5, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--source", default="0", help="webcam index (0, 1, ...) or a video file")
    ap.add_argument("--threshold", type=float, default=0.6, help="below this confidence the model says 'unsure'")
    ap.add_argument("--smooth", type=int, default=10, help="average the last N predictions (less flicker)")
    ap.add_argument("--switch-after", type=float, default=3.0, help="seconds a new activity must hold")
    ap.add_argument("--end-after", type=float, default=10.0, help="seconds without the activity before it ends")
    ap.add_argument("--background", default="background", help="name of the 'nothing happening' class")
    ap.add_argument("--log", default="", help="CSV file: one row per finished activity")
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--no-box", action="store_true", help="do not draw the rectangle around the person")
    ap.add_argument("--box-model", default="yolo", choices=["yolo", "motion"],
                    help="yolo = pretrained person detector (default), motion = background difference")
    ap.add_argument("--box-weights", default="yolov8n.pt", help="YOLO weights (yolov8s.pt / yolov8m.pt = more accurate, GPU)")
    ap.add_argument("--box-conf", type=float, default=0.35,
                    help="YOLO: minimum confidence for a person box (raise to 0.5 if clothes/posters get boxed)")
    ap.add_argument("--box-every", type=int, default=3, help="YOLO runs on every Nth camera frame (3 = about 10 per second)")
    ap.add_argument("--save-video", default="", help="also write the annotated video to this file (works with --headless)")
    ap.add_argument("--bg-image", default="", help="[motion box] picture of the empty room "
                                                   "(otherwise the first frames are used, so start with nobody in view)")
    ap.add_argument("--box-thresh", type=int, default=30,
                    help="[motion box] sensitivity: lower = detects clothes closer to the background colour")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    ckpt, backbone, head = load_model(args.model)
    labels, n, size = ckpt["labels"], ckpt["num_frames"], ckpt["input_size"]
    period = 1.0 / ckpt["feature_fps"]
    print(f"Loaded {args.model}: {len(labels)} classes {labels}")
    print(f"Looking at {ckpt['feature_fps']:g} frames/s, window = {n / ckpt['feature_fps']:.1f} s")
    if args.background not in labels:
        print(f"WARNING: the model has no class called '{args.background}'. "
              f"Use --background <name> or every activity will end only by timeout.")

    cap, is_file, src_fps = open_source(args.source, args.width, args.height)
    clock = Clock(is_file, src_fps)
    fmt = make_formatter(is_file, time.time())
    tracker = ActivityTracker(args.background, args.switch_after, args.end_after)

    show = (not args.headless) or bool(args.save_video)     # draw frames (window and/or output video)
    cropper = None
    if ckpt.get("crop"):                                     # model was trained on person crops: crop the same way
        cropper = PersonCropper(**ckpt["crop"])
        print(f"Person crop: YOLO ({ckpt['crop']['weights']}) on every {ckpt['crop']['every']} feature frame(s); "
              f"the features, and the green box, come from it")
    box, calib_left, writer = None, 0, None
    if not args.no_box and show and cropper is None:
        if args.box_model == "yolo":
            try:
                from person_box_yolo import YoloPersonBox
                box = YoloPersonBox(args.box_weights, conf=args.box_conf, every=args.box_every)
                print(f"Person box: YOLO ({args.box_weights}), every {args.box_every} frame(s)")
            except ImportError as e:
                print(f"{e}\nFalling back to the simple background-difference box (--box-model motion).")
        if box is None:
            box = PersonBox(diff_thresh=args.box_thresh)
            if args.bg_image:
                bg = cv2.imread(args.bg_image)
                if bg is None:
                    raise SystemExit(f"Could not read --bg-image '{args.bg_image}'.")
                box.set_background(bg)
            else:
                calib_left = 30

    feats = collections.deque(maxlen=n)
    recent = collections.deque(maxlen=max(1, args.smooth))
    counts = collections.Counter()
    next_time = 0.0
    raw_label, raw_conf, top3 = "warming up", 0.0, []
    feat_ms, head_ms = 0.0, 0.0
    feat_times = []

    log_file, log = None, None
    if args.log:
        log_file = open(args.log, "w", newline="")
        log = csv.writer(log_file)
        log.writerow(["activity", "start", "end", "seconds"])

    def handle(events):
        for kind, label, start, end in events:
            if kind == "start":
                print(f"{fmt(start)}  {label} started")
            else:
                print(f"{fmt(end)}  {label} ended  ({fmt(start)} to {fmt(end)}, {end - start:.0f} s)")
                if log:
                    log.writerow([label, fmt(start), fmt(end), f"{end - start:.1f}"])
                    log_file.flush()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        clock.tick()
        now = clock.now()

        if now >= next_time:                     # time for the next feature vector
            t0 = time.perf_counter()
            src = frame if cropper is None else cropper.crop(frame)
            feats.append(compute_features(backbone, [prepare_frame(src, size)])[0])
            feat_ms = (time.perf_counter() - t0) * 1000
            feat_times.append(feat_ms)
            next_time = max(next_time + period, now)

            if len(feats) == n:                  # window full -> classify
                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = head(torch.from_numpy(np.stack(feats))[None])
                probs = torch.softmax(logits, dim=1)[0].numpy()
                head_ms = (time.perf_counter() - t0) * 1000
                recent.append(probs)
                avg = np.mean(recent, axis=0)
                order = np.argsort(avg)[::-1]
                raw_conf = float(avg[order[0]])
                raw_label = labels[order[0]] if raw_conf >= args.threshold else "unsure"
                top3 = [(labels[j], float(avg[j])) for j in order[:3]]
                counts[raw_label] += 1
                handle(tracker.update(now, raw_label))

        if show:
            activity, elapsed = tracker.status(now)
            if top3:
                lines = [f"model now: {raw_label} ({raw_conf * 100:.0f}%)"] + \
                        [f"{name}: {p * 100:.0f}%" for name, p in top3[:2]]
            else:
                lines = ["warming up..."]
            idle = activity == args.background
            rect, calibrating = None, False
            if cropper is not None:
                if not args.no_box:
                    rect = cropper.box_xywh()
            elif box is not None:                          # find the person on the clean frame, draw later
                if calib_left > 0:                       # learning the empty room
                    box.calibrate(frame)
                    calib_left -= 1
                    calibrating = True
                else:
                    rect = box.update(frame)
            draw_overlay(frame, activity, elapsed, idle, lines, feat_ms, head_ms)
            if calibrating:
                cv2.putText(frame, "person box: keep the frame empty...", (10, frame.shape[0] - 32),
                            FONT, 0.5, (0, 200, 255), 1)
            elif rect is not None:                       # drawn after the text bar so it is not dimmed
                PersonBox.draw(frame, rect, (190, 190, 190) if idle else (0, 220, 0), f"Person | {activity}")
            if args.save_video:
                if writer is None:
                    fh, fw = frame.shape[:2]
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (fw, fh))
                writer.write(frame)
            if not args.headless:
                cv2.imshow("Activity tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("b") and box is not None:  # recapture the empty room / reset the box
                    box.reset()
                    if isinstance(box, PersonBox):
                        calib_left = 30

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to {args.save_video}")
    if not args.headless:
        cv2.destroyAllWindows()
    handle(tracker.finish())
    if log_file:
        log_file.close()

    print("\nSummary (time spent per activity):")
    totals = tracker.totals()
    if totals:
        for label, (secs, count) in sorted(totals.items(), key=lambda kv: -kv[1][0]):
            print(f"  {label:<16} {secs / 60:6.1f} min  ({secs:5.0f} s, {count} segment{'s' if count != 1 else ''})")
    else:
        print("  no activity found")
    if feat_times:
        print(f"\nAverage CNN time per frame: {np.mean(feat_times):.1f} ms "
              f"(must stay below {period * 1000:.0f} ms to keep up)")
    if counts:
        print("Raw predictions per label:", dict(counts))


if __name__ == "__main__":
    main()
