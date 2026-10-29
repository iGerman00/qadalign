"""Train the sung / not-sung head on the frozen aligner encoder.

    python tools/train_sung_head.py --init runs/full/best.pt --data data --epochs 6

Frames inside a lyric segment are positives, frames inside an instrumental
window negatives. The aligner is frozen; only the 8.3K-parameter head trains.
Expect AUC near 0.88.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as K
from qadalign.aurora import AuroraASRFeatureExtractor, AuroraASRForCTC, AuroraASRConfig
from qadalign.sung import SungHead

WIN = 30.0
FPS = 25.0          # 40 ms frames


def build_model(ckpt):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    vocab = sd["head.weight"].shape[0]
    cfg = AuroraASRConfig(vocab_size=vocab, blank_token_id=0)
    model = AuroraASRForCTC(cfg)
    model.load_state_dict(sd, strict=False)
    return model


def pool(model, fe, rows, wav_dir, dev, cap):
    """Encoder features plus per-frame sung labels, one window at a time."""
    hidden = {}
    model.final_ln.register_forward_hook(lambda m, i, o: hidden.__setitem__("h", o.detach()))
    X, Y = [], []
    for s in rows:
        a0 = max(0.0, s["start"] - WIN / 2)
        a1 = min(s["end"] + WIN / 2, s["end"] + WIN)
        wav, offset = K.read_audio({"track_id": s["track_id"], "start": a0,
                                    "end": a1}, wav_dir, pad=0.0)
        if len(wav) < 8000:
            continue
        enc = fe([wav], sampling_rate=K.SR, return_tensors="pt")
        with torch.no_grad():
            model(enc["input_features"].to(dev), attention_mask=enc["attention_mask"].to(dev))
        h = hidden["h"][0].float()
        n = h.shape[0]
        y = np.zeros(n, np.float32)
        if s.get("targets") or s.get("tier") not in (None, "instr"):
            i0 = max(0, int((s["start"] - offset) / 0.04))
            i1 = min(n, int((s["end"] - offset) / 0.04))
            if i1 > i0:
                y[i0:i1] = 1.0
        X.append(h.half().cpu())
        Y.append(torch.from_numpy(y))
        if len(X) >= cap:
            break
    if not X:
        raise SystemExit(
            "no windows pooled from the rows given: check that the data directory "
            "has segments for the requested split (see prepare_dataset.py)")
    return torch.cat(X), torch.cat(Y)


@torch.no_grad()
def metrics(head, X, Y, dev, thresh=0.5):
    head.eval()
    s, l = [], []
    for i in range(0, len(X), 8192):
        s.append(torch.sigmoid(head(X[i:i + 8192].float().to(dev))).cpu())
        l.append(Y[i:i + 8192])
    s, l = torch.cat(s).numpy(), torch.cat(l).numpy()
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(s))
    n1, n0 = l.sum(), len(l) - l.sum()
    auc = (ranks[l == 1].sum() - n1 * (n1 - 1) / 2) / max(1e-9, n1 * n0)
    p = s > thresh
    return (float(auc), float((p & (l == 1)).sum() / max(1, p.sum())),
            float((p & (l == 1)).sum() / max(1, (l == 1).sum())),
            float((~p & (l == 0)).sum() / max(1, (l == 0).sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="trained aligner checkpoint")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/sung_head.pt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train", type=int, default=4000, help="windows to pool")
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    data = Path(a.data)
    dev = a.device
    model = build_model(a.init).to(dev).eval()
    fe = AuroraASRFeatureExtractor()
    wav_dir = data / "wav"

    sung = K.load_segments(data / "segments.jsonl", ("train",))
    if not sung:   # small corpora can hash entirely out of train
        sung = K.load_segments(data / "segments.jsonl", ("train", "val", "test"))
        print(f"WARNING: no train segments, using all {len(sung)}")
    voc = []
    for line in (data / "instrumental.jsonl").open():
        s = json.loads(line)
        if s["split"] == "train":
            s["targets"] = []
            voc.append(s)
    val_sung = K.load_segments(data / "segments.jsonl", ("val",))
    val_voc = []
    for line in (data / "instrumental.jsonl").open():
        s = json.loads(line)
        if s["split"] == "val":
            s["targets"] = []
            val_voc.append(s)

    if not sung:
        raise SystemExit("no segments found; run tools/prepare_dataset.py first")
    if not (val_sung or val_voc):
        cut = max(1, len(sung) // 5)
        val_sung, val_voc = sung[-cut:], voc[-cut:] if voc else []
        print(f"WARNING: no val segments, holding out {len(val_sung)} training windows")
        sung, voc = sung[:-cut], voc[:-cut] if voc else []

    Xtr, Ytr = pool(model, fe, sung + voc, wav_dir, dev, a.train)
    Xva, Yva = pool(model, fe, val_sung + val_voc, wav_dir, dev, a.val)
    print(f"{len(Xtr)} train frames ({Ytr.mean()*100:.0f}% sung), {len(Xva)} val")

    dim = Xtr.shape[1]
    head = SungHead(dim).to(dev)
    print(f"head {sum(p.numel() for p in head.parameters())} params on {dim}-dim encoder")
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    pw = torch.tensor([(1 - float(Ytr.mean())) / max(1e-6, float(Ytr.mean()))], device=dev)

    for ep in range(1, a.epochs + 1):
        head.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), 8192):
            b = perm[i:i + 8192]
            loss = F.binary_cross_entropy_with_logits(
                head(Xtr[b].float().to(dev)), Ytr[b].to(dev), pos_weight=pw)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += float(loss.detach()) * len(b)
        auc, prec, rec, spec = metrics(head, Xva, Yva, dev)
        print(f"ep {ep} loss {tot/len(Xtr):.4f} val AUC {auc:.3f} "
              f"prec {prec:.3f} rec {rec:.3f} spec {spec:.3f}", flush=True)

    auc = metrics(head, Xva, Yva, dev)[0]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "dim": dim, "auc": auc}, a.out)
    print(f"val AUC {auc:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
