"""Results of a trained model PER SESSION (use it on the sessions you held out).

    python eval_by_session.py --model /content/drive/MyDrive/model_test.pt \
        --features /content/drive/MyDrive/features \
        --sessions room_Gowtham_P1,room_Gowtham_P2,lib_Vivek_P1

For every session it prints the accuracy, and for every class in that session: how many windows, the recall,
and what the model most often said instead. This shows whether one person / angle / room causes most of the errors.
Run it from the folder that has train.py and common.py. Without --sessions it uses every session (training ones
too, which will look much better than they really are).
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from common import build_head
from train import load_features, take_window


def summarize(rows, labels):
    """rows: list of (session, true_class_index, predicted_class_index).
    Returns {session: (n_windows, accuracy, [(class, n, recall, (wrong_class, share) or None)])}."""
    out = {}
    for s in sorted({r[0] for r in rows}):
        rs = [(t, p) for ss, t, p in rows if ss == s]
        acc = sum(t == p for t, p in rs) / len(rs)
        per = []
        for c in range(len(labels)):
            preds = [p for t, p in rs if t == c]
            if not preds:
                continue
            wrong = [p for p in preds if p != c]
            top = None
            if wrong:
                k, cnt = Counter(wrong).most_common(1)[0]
                top = (labels[k], cnt / len(preds))
            per.append((labels[c], len(preds), sum(p == c for p in preds) / len(preds), top))
        out[s] = (len(rs), acc, per)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--sessions", default="", help="comma-separated session names (default: all)")
    a = ap.parse_args()

    ckpt = torch.load(a.model, map_location="cpu", weights_only=True)
    labels, n = ckpt["labels"], ckpt["num_frames"]
    h = ckpt["head"]
    head = build_head(h["kind"], ckpt["feature_dim"], len(labels), h["hidden"], h["layers"], h["dropout"],
                      h.get("motion", False))
    head.load_state_dict(ckpt["head_state"])
    head.eval()

    _, feat_labels, records = load_features(Path(a.features))
    if feat_labels != labels:
        raise SystemExit(f"Class order differs: model {labels} vs features {feat_labels}")
    wanted = {s.strip() for s in a.sessions.split(",") if s.strip()}

    X, meta = [], []
    for r in records:
        if r["aug"] or (wanted and r["group"] not in wanted):
            continue
        T = len(r["feat"])
        for s in sorted({0, max(0, (T - n) // 2), max(0, T - n)}):   # same 3 windows per clip as train.py
            X.append(np.ascontiguousarray(take_window(r["feat"], n, s)))
            meta.append((r["group"], r["label"]))
    if not X:
        raise SystemExit("No clips matched. Check --sessions against the names in your features folder.")

    with torch.no_grad():
        preds = head(torch.from_numpy(np.stack(X))).argmax(1).tolist()
    rows = [(g, t, p) for (g, t), p in zip(meta, preds)]

    for session, (count, acc, per) in summarize(rows, labels).items():
        print(f"\n== {session}   {count} windows   accuracy {acc * 100:.1f}%")
        for cls, k, rec, top in per:
            wrong = f"   mostly called {top[0]} ({top[1] * 100:.0f}%)" if top else ""
            print(f"   {cls:<12} windows {k:4d}   recall {rec * 100:5.1f}%{wrong}")


if __name__ == "__main__":
    main()
