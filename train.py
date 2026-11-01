import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import config
from common import build_head


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------
def load_features(feature_dir):
    meta = json.loads((feature_dir / "meta.json").read_text())
    labels = sorted(p.name for p in feature_dir.iterdir() if p.is_dir() and any(p.glob("*.npy")))
    records = []
    for label_index, label in enumerate(labels):
        for f in sorted((feature_dir / label).glob("*.npy")):
            base = f.stem.split("~aug")[0]               # clip name without the augmentation suffix
            records.append({"feat": np.load(f), "label": label_index, "base": base,
                            "group": base.split("__")[0],   # session = text before "__"
                            "aug": "~aug" in f.stem})
    return meta, labels, records


def split_records(records, labels, val_frac, seed, val_groups):
    """Returns (train, val). Augmented copies are only ever used for training."""
    rng = random.Random(seed)
    train, val = [], []
    for r in records:                                      # explicit hold-out sessions
        if val_groups and r["group"] in val_groups:
            if not r["aug"]:
                val.append(r)
        elif val_groups:
            train.append(r)
    if val_groups:
        return train, val

    for li, label in enumerate(labels):
        recs = [r for r in records if r["label"] == li]
        groups = sorted({r["group"] for r in recs})
        if len(groups) >= 2:                               # proper split by session
            rng.shuffle(groups)
            n_val = min(max(1, round(len(groups) * val_frac)), len(groups) - 1)
            val_g = set(groups[:n_val])
            for r in recs:
                if r["group"] in val_g:
                    if not r["aug"]:
                        val.append(r)
                else:
                    train.append(r)
        else:                                              # only one session: fall back to clip split
            print(f"WARNING: '{label}' has only one session, so validation clips come from the SAME "
                  f"session as training clips. Accuracy for this class will look better than it is. "
                  f"Film it again with a different person/day/camera and a new --session name.")
            bases = sorted({r["base"] for r in recs})
            rng.shuffle(bases)
            n_val = max(1, round(len(bases) * val_frac)) if len(bases) >= 2 else 0
            val_b = set(bases[:n_val])
            for r in recs:
                if r["base"] in val_b:
                    if not r["aug"]:
                        val.append(r)
                else:
                    train.append(r)
    return train, val


def take_window(feat, n, start):
    """n consecutive feature vectors from `start`. Too-short clips are padded with the last one."""
    if len(feat) < n:
        feat = np.concatenate([feat, np.repeat(feat[-1:], n - len(feat), axis=0)])
        start = 0
    return feat[start:start + n]


class TrainSet(Dataset):
    """Random window of NUM_FRAMES features from each clip, different every epoch."""

    def __init__(self, records, n):
        self.records, self.n = records, n

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        T = len(r["feat"])
        start = random.randint(0, T - self.n) if T > self.n else 0
        return torch.from_numpy(np.ascontiguousarray(take_window(r["feat"], self.n, start))), r["label"]


class EvalSet(Dataset):
    """Up to 3 fixed windows per clip (start, middle, end) so the score is steadier."""

    def __init__(self, records, n):
        self.items = []
        for r in records:
            T = len(r["feat"])
            for s in sorted({0, max(0, (T - n) // 2), max(0, T - n)}):
                self.items.append((np.ascontiguousarray(take_window(r["feat"], n, s)), r["label"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y = self.items[i]
        return torch.from_numpy(x), y


# ----------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------
def evaluate(model, loader, n_classes):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss, cm = 0.0, np.zeros((n_classes, n_classes), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            total_loss += ce(logits, y).item()
            for t, p in zip(y.tolist(), logits.argmax(1).tolist()):
                cm[t, p] += 1
    support = cm.sum(1)
    recalls = [cm[i, i] / support[i] for i in range(n_classes) if support[i] > 0]
    return total_loss / cm.sum(), np.trace(cm) / cm.sum(), float(np.mean(recalls)), cm


def print_report(labels, cm):
    width = max(len(l) for l in labels)
    print("\nPer-class results on validation  (precision = of the windows called this class, how many were right;")
    print("recall = of this class's windows, how many were found; F1 = balance of the two):")
    print(f"     {'':<{width}}  windows  precision   recall       F1")
    f1s = []
    for i, label in enumerate(labels):
        n, predicted = cm[i].sum(), cm[:, i].sum()
        prec = cm[i, i] / predicted if predicted else None
        rec = cm[i, i] / n if n else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if n else None)
        if f1 is not None:
            f1s.append(f1)
        pct = lambda v: f"{v * 100:6.1f}%" if v is not None else "   n/a "
        print(f"  {i:2d} {label:<{width}}  {n:7d}  {pct(prec)}  {pct(rec)}  {pct(f1)}")
    print(f"\n  accuracy {np.trace(cm) / cm.sum() * 100:.1f}%    macro F1 {np.mean(f1s) * 100:.1f}%")
    print("\nConfusion matrix (rows = true class, columns = predicted class, numbers = class ids above):")
    print("     " + " ".join(f"{j:4d}" for j in range(len(labels))))
    for i in range(len(labels)):
        print(f"  {i:2d} " + " ".join(f"{v:4d}" for v in cm[i]))


def save_confusion(labels, cm, folder):
    with open(folder / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + labels)
        for i, label in enumerate(labels):
            w.writerow([label] + cm[i].tolist())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1 + 0.8 * len(labels), 1 + 0.8 * len(labels)))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(folder / "confusion_matrix.png", dpi=120)
    except ImportError:
        pass


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(config.FEATURE_DIR))
    ap.add_argument("--out", default=str(config.MODEL_PATH))
    ap.add_argument("--head", default=config.HEAD, choices=["gru", "tcn"])
    ap.add_argument("--hidden", type=int, default=config.HIDDEN)
    ap.add_argument("--layers", type=int, default=config.LAYERS)
    ap.add_argument("--dropout", type=float, default=config.DROPOUT)
    ap.add_argument("--frames", type=int, default=config.NUM_FRAMES, help="window length in frames")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.25, help="share of sessions used for validation")
    ap.add_argument("--val-groups", default="", help="comma-separated session names to hold out")
    ap.add_argument("--patience", type=int, default=25, help="stop if no improvement for this many epochs")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--train-on-all", action="store_true", help="use every clip, no validation, keep last epoch")
    ap.add_argument("--motion", action="store_true",
                    help="also give the head the frame-to-frame change of the features (no new extraction needed)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    feature_dir = Path(args.features)
    if not (feature_dir / "meta.json").exists():
        raise SystemExit(f"No features found in {feature_dir}. Run extract_features.py first.")
    meta, labels, records = load_features(feature_dir)
    n_classes, n = len(labels), args.frames
    if n_classes < 2:
        raise SystemExit("Need at least 2 classes (include a 'background' class too).")

    val_groups = {g.strip() for g in args.val_groups.split(",") if g.strip()}
    train_recs, val_recs = split_records(records, labels, args.val_frac, args.seed, val_groups)
    if args.train_on_all:
        train_recs, val_recs = records, []
    elif not val_recs:
        raise SystemExit("No validation clips. Add more clips/sessions, or use --train-on-all.")

    print("\nClips per class (originals only):  total / train / validation")
    for i, label in enumerate(labels):
        tot = sum(1 for r in records if r["label"] == i and not r["aug"])
        tr = sum(1 for r in train_recs if r["label"] == i and not r["aug"])
        va = sum(1 for r in val_recs if r["label"] == i)
        sessions = len({r["group"] for r in records if r["label"] == i})
        print(f"  {label:<20} {tot:4d} / {tr:4d} / {va:4d}   ({sessions} sessions)")

    # Feature mean/std from the training clips, stored inside the model.
    all_feats = np.concatenate([r["feat"] for r in train_recs])
    head = build_head(args.head, meta["feature_dim"], n_classes, args.hidden, args.layers, args.dropout, args.motion)
    head.mean.copy_(torch.from_numpy(all_feats.mean(0)))
    head.std.copy_(torch.from_numpy(all_feats.std(0) + 1e-5))
    if args.motion:                                   # scale of the change between neighbouring frames, per feature
        diffs = np.concatenate([(r["feat"][1:] - r["feat"][:-1]) / head.std.numpy() for r in train_recs])
        head.dstd.copy_(torch.from_numpy(diffs.std(0) + 1e-5))
    n_params = sum(p.numel() for p in head.parameters())
    print(f"\nHead: {args.head}{' + motion' if args.motion else ''}, hidden={args.hidden}, layers={args.layers}, dropout={args.dropout}"
          f"  ->  {n_params:,} trainable parameters")

    counts = np.bincount([r["label"] for r in train_recs], minlength=n_classes).astype(np.float32)
    weights = None
    if not args.no_class_weights:
        weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    train_loader = DataLoader(TrainSet(train_recs, n), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EvalSet(val_recs, n), batch_size=64) if val_recs else None
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = {"score": (-1.0, -float("inf")), "epoch": 0, "state": None, "cm": None}
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        head.train()
        loss_sum, seen = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(head(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
        scheduler.step()

        if val_loader is None:
            print(f"epoch {epoch:3d}/{args.epochs}  train_loss {loss_sum / seen:.3f}")
            continue
        val_loss, val_acc, macro_recall, cm = evaluate(head, val_loader, n_classes)
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss {loss_sum / seen:.3f}  val_loss {val_loss:.3f}  "
              f"val_acc {val_acc * 100:5.1f}%  mean_class_recall {macro_recall * 100:5.1f}%")
        score = (macro_recall, -val_loss)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "cm": cm,
                    "state": {k: v.clone() for k, v in head.state_dict().items()}}
        elif epoch - best["epoch"] >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping.")
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if val_loader is not None:
        head.load_state_dict(best["state"])
        print(f"\nBest epoch: {best['epoch']}  (mean class recall {best['score'][0] * 100:.1f}%)")
        print_report(labels, best["cm"])
        save_confusion(labels, best["cm"], out_path.parent)
    print(f"\nTraining took {time.time() - t_start:.0f}s")

    torch.save({
        "labels": labels,
        "backbone": meta["backbone"],
        "input_size": int(meta["size"]),
        "feature_dim": int(meta["feature_dim"]),
        "feature_fps": float(meta["fps"]),
        "crop": meta.get("crop"),                 # person-crop settings (None = whole frame), used by realtime.py
        "num_frames": int(n),
        "head": {"kind": args.head, "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
                 "motion": args.motion},
        "head_state": head.state_dict(),
        "backbone_state": torch.load(feature_dir / "backbone.pt", map_location="cpu", weights_only=True),
    }, out_path)
    print(f"Saved {out_path}   (backbone + head + class names in one file)")
    print("Next: python realtime.py")


if __name__ == "__main__":
    main()
