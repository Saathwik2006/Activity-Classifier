import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import config
from common import FEATURE_DIMS, VIDEO_EXTS, build_backbone, compute_features
from video_io import ClipAug, prepare_clip, read_feature_frames, sample_clip_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--out", default=str(config.FEATURE_DIR))
    ap.add_argument("--backbone", default=config.BACKBONE, choices=list(FEATURE_DIMS))
    ap.add_argument("--size", type=int, default=config.INPUT_SIZE)
    ap.add_argument("--fps", type=float, default=config.FEATURE_FPS)
    ap.add_argument("--aug-copies", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (only for testing the code)")
    ap.add_argument("--force", action="store_true", help="redo clips that are already done")
    ap.add_argument("--crop-person", action="store_true",
                    help="crop every frame to the person (YOLO) first; use a NEW --out folder for these features")
    ap.add_argument("--crop-weights", default="yolov8n.pt")
    ap.add_argument("--crop-conf", type=float, default=0.35)
    ap.add_argument("--crop-margin", type=float, default=0.2)
    ap.add_argument("--crop-every", type=int, default=2, help="YOLO runs on every Nth feature frame")
    ap.add_argument("--crop-objects", action="store_true",
                    help="also keep a laptop / phone that touches the person inside the crop")
    ap.add_argument("--crop-top-frac", type=float, default=1.0,
                    help="keep only the top part of a tall person box (0.6 = head, torso, hands at the desk)")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    labels = sorted(p.name for p in data.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))) \
        if data.is_dir() else []
    if not labels:
        raise SystemExit(f"No class folders found in {data}. Expected data/<label>/<clip>.mp4")

    cropper = None
    if args.crop_person:
        from person_crop import PersonCropper
        cropper = PersonCropper(args.crop_weights, args.crop_conf, args.crop_margin, args.crop_every,
                                top_frac=args.crop_top_frac, objects=args.crop_objects)
    meta = {"backbone": args.backbone, "size": args.size, "fps": args.fps,
            "pretrained": not args.no_pretrained, "feature_dim": FEATURE_DIMS[args.backbone],
            "crop": cropper.settings() if cropper else None}
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists() and not args.force:
        old = json.loads(meta_path.read_text())
        if any(old.get(k) != v for k, v in meta.items()):
            raise SystemExit(f"{out} was made with different settings ({old}).\n"
                             "Delete that folder, or pass --force, to start again.")
    meta_path.write_text(json.dumps(meta, indent=2))

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"Loading backbone {args.backbone} (pretrained={not args.no_pretrained}) ...")
    backbone = build_backbone(args.backbone, pretrained=not args.no_pretrained)
    torch.save(backbone.state_dict(), out / "backbone.pt")     # train.py bundles this into model.pt (saved on CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone.to(device)                                         # GPU makes this step minutes instead of an hour
    print(f"Feature extraction runs on: {device}")

    n_short, n_done, n_skipped, t_start = 0, 0, 0, time.time()
    for label in labels:
        (out / label).mkdir(exist_ok=True)
        paths = sorted(p for p in (data / label).iterdir() if p.suffix.lower() in VIDEO_EXTS)
        print(f"[{label}] {len(paths)} clips")
        for number, path in enumerate(paths, 1):
            raw = rects = None                       # person crops are found once per clip and shared by its copies
            for copy in range(args.aug_copies + 1):
                name = path.stem if copy == 0 else f"{path.stem}~aug{copy}"
                target = out / label / f"{name}.npy"
                if target.exists() and not args.force:
                    n_skipped += 1
                    continue
                aug = None if copy == 0 else ClipAug(random.Random(f"{path}-{copy}"))
                if cropper is None:
                    frames = sample_clip_frames(path, args.fps, args.size, aug)
                else:
                    if raw is None:
                        raw = read_feature_frames(path, args.fps)
                        rects = cropper.rects(raw) if raw else []
                    frames = prepare_clip(raw, args.size, aug, rects) if raw else []
                if not frames:
                    print(f"  WARNING: could not read {path}, skipped")
                    break
                np.save(target, compute_features(backbone, frames))
                n_done += 1
                if copy == 0 and len(frames) < config.NUM_FRAMES:
                    n_short += 1
            if number % 20 == 0:
                print(f"  {number}/{len(paths)}  ({time.time() - t_start:.0f}s)")

    print(f"\nDone: {n_done} feature files written, {n_skipped} skipped (already existed), "
          f"{time.time() - t_start:.0f}s.")
    if n_short:
        print(f"NOTE: {n_short} clips are shorter than {config.NUM_FRAMES / args.fps:.1f}s "
              f"(= NUM_FRAMES / FPS). They get padded; consider longer clips.")
    print("Next: python train.py")


if __name__ == "__main__":
    main()
