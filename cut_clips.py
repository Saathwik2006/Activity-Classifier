"""Cut long recordings into short, fixed-length clips.

Expected layout (one folder per session, class name at the start of each filename):

    raw/
      lab_A_p1/
        typing_01.mp4
        background_01.mp4
        <other_chore>_01.mp4
      office_B_p2/
        ...

Output layout (session name is kept in every clip name, so you can split by session later):

    clips/
      typing/
        lab_A_p1__typing_01__0000.mp4
        ...
      background/
        ...

Usage:
    pip install opencv-python
    python cut_clips.py --raw raw --out clips --classes typing background chore3 chore4 chore5
    python cut_clips.py --raw raw --out clips --classes typing background --clip-sec 3.2 --stride-sec 1.6

Defaults: 3.2 s clips, a new clip every 1.6 s (50% overlap), resampled to 10 fps.
"""
import argparse
import sys
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def find_class(stem, classes):
    # Longest class name first, so "desk_work" wins over "desk".
    for c in sorted(classes, key=len, reverse=True):
        if stem.lower().startswith(c.lower()):
            return c
    return None


def cut_video(path, out_dir, session, cls, fps_out, clip_sec, stride_sec, size):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  ! cannot open {path}")
        return 0
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Read every frame, keep only those that land on the 10 fps grid.
    frames, next_t, i = [], 0.0, 0
    step = 1.0 / fps_out
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps_in
        if t >= next_t:
            if size:
                frame = cv2.resize(frame, size)
            frames.append(frame)
            next_t += step
        i += 1
    cap.release()

    n_clip = int(round(clip_sec * fps_out))
    n_stride = max(1, int(round(stride_sec * fps_out)))
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for start in range(0, len(frames) - n_clip + 1, n_stride):
        h, w = frames[start].shape[:2]
        name = f"{session}__{path.stem}__{count:04d}.mp4"
        writer = cv2.VideoWriter(str(out_dir / name),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))
        for f in frames[start:start + n_clip]:
            writer.write(f)
        writer.release()
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--clip-sec", type=float, default=3.2)
    ap.add_argument("--stride-sec", type=float, default=1.6)
    ap.add_argument("--width", type=int, default=0, help="resize width (0 = keep)")
    ap.add_argument("--height", type=int, default=0, help="resize height (0 = keep)")
    a = ap.parse_args()

    size = (a.width, a.height) if a.width and a.height else None
    raw, out = Path(a.raw), Path(a.out)
    totals = {}

    for session_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        print(session_dir.name)
        for v in sorted(session_dir.iterdir()):
            if v.suffix.lower() not in VIDEO_EXTS:
                continue
            cls = find_class(v.stem, a.classes)
            if cls is None:
                print(f"  ! skipped {v.name}: filename does not start with a known class")
                continue
            n = cut_video(v, out / cls, session_dir.name, cls,
                          a.fps, a.clip_sec, a.stride_sec, size)
            totals[cls] = totals.get(cls, 0) + n
            print(f"  {v.name}: {n} clips -> {cls}")

    print("\nClips per class:")
    for c in a.classes:
        print(f"  {c}: {totals.get(c, 0)}")
    if not totals:
        sys.exit("No clips written. Check folder layout and --classes.")


if __name__ == "__main__":
    main()
