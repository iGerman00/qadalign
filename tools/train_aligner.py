"""Train the aligner from the Aurora ASR checkpoint.

Three stages, run in order, each warm-starting from the previous checkpoint:

    python tools/train_aligner.py --stage phone --base ./aurora --data data --epochs 20
    python tools/train_aligner.py --stage gaps  --init runs/phone/best.pt --data data --epochs 20
    python tools/train_aligner.py --stage full  --init runs/gaps/best.pt  --data data --epochs 5

phone  fresh 39-phone CTC head, word-level segments only
gaps   adds windows with no lyric, phased in from epoch 3
full   all segment types at their configured weights

Checkpoints are ranked here by phone error. Select on alignment instead with
tools/evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import common as K
from qadalign.aurora import AuroraASRFeatureExtractor, AuroraASRForCTC, AuroraASRConfig


def build_base(base: str | None):
    """Encoder from the public Aurora checkpoint, with a fresh phone head."""
    if base and Path(base, "config.json").exists():
        cfg = AuroraASRConfig(**{k: v for k, v in
                                 json.loads(Path(base, "config.json").read_text()).items()
                                 if k not in ("architectures", "auto_map", "model_type",
                                              "transformers_version")})
        cfg.vocab_size = K.NUM_PHONES
        cfg.blank_token_id = 0
        model = AuroraASRForCTC(cfg)
        sd = Path(base, "model.safetensors")
        if sd.exists():
            from safetensors.torch import load_file
            state = load_file(str(sd))
            # the base has a 257-piece BPE head; keep everything that still fits
            own = model.state_dict()
            keep = {k: v for k, v in state.items()
                    if k in own and own[k].shape == v.shape}
            model.load_state_dict(keep, strict=False)
            print(f"encoder loaded from {base}: {len(keep)}/{len(own)} tensors, "
                  f"head reinitialised")
        return K.swap_head(model)
    from transformers import AutoModelForCTC
    model = AutoModelForCTC.from_pretrained(base or "AuroraAI-Research/AuroraASR-1M",
                                            trust_remote_code=True)
    return K.swap_head(model)


def load_rows(data: Path, splits, tiers, limit=None):
    rows = [s for s in K.load_segments(data / "segments.jsonl", splits)
            if ("instr" if s.get("tier") == "instr" else s.get("tier")) in tiers]
    return rows[:limit] if limit else rows


def load_instrumentals(data: Path, splits, limit=None):
    import json as _j
    rows = []
    for line in (data / "instrumental.jsonl").open():
        s = _j.loads(line)
        if s["split"] in splits:
            s["targets"] = []
            rows.append(s)
    return rows[:limit] if limit else rows


def evaluate(model, fe, rows, dev, instr_rows=None):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for b in K.batcher(rows, 240.0, 24):
            f, m, l, lab = K.collate(fe, [(K.read_audio(s, s["_wav"])[0], s["targets"])
                                          for s in b])
            out = model(f.to(dev), attention_mask=m.to(dev))
            for p in K.greedy(out.logits, l):
                preds.append(p)
            refs += [s["targets"] for s in b]
    spur = 0
    if instr_rows:
        with torch.no_grad():
            for b in K.batcher(instr_rows[:200], 240.0, 24):
                f, m, l, lab = K.collate(fe, [(K.read_audio(s, s.get("_wav", rows[0]["_wav"]))[0], [])
                                              for s in b])
                out = model(f.to(dev), attention_mask=m.to(dev))
                spur += sum(len(p) for p in K.greedy(out.logits, l))
    model.train()
    return K.phone_error_rate(preds, refs), spur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["phone", "gaps", "full"], required=True)
    ap.add_argument("--base", default=None, help="Aurora checkpoint dir (stage phone)")
    ap.add_argument("--init", default=None, help="warm start checkpoint")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--body-lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--secs", type=float, default=240.0)
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    data = Path(a.data)
    out = Path(a.out) / a.stage
    out.mkdir(parents=True, exist_ok=True)

    model = build_base(a.base)
    if a.init:
        state = torch.load(a.init, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"warm start {a.init}")
    model.to(a.device)
    fe = AuroraASRFeatureExtractor()

    wav_dir = data / "wav"
    if a.stage == "phone":
        rows = load_rows(data, ("train",), ("word",))
        val = load_rows(data, ("val",), ("word",), limit=500)
        epochs = a.epochs or 20
        body_lr, head_lr = a.body_lr or 3e-4, a.head_lr or 1e-3
        instr, neg_start, nw, lw, cw = [], 0, 1.0, 0.0, 0.0
    elif a.stage == "gaps":
        rows = load_rows(data, ("train",), ("word",))
        val = load_rows(data, ("val",), ("word",), limit=500)
        epochs = a.epochs or 20
        body_lr, head_lr = a.body_lr or 5e-5, a.head_lr or 1.5e-4
        instr, neg_start, nw, lw, cw = load_instrumentals(data, ("train",)), 3, 1.0, 0.0, 0.0
    else:
        rows = load_rows(data, ("train",), ("word",))
        val = load_rows(data, ("val",), ("word",), limit=800)
        epochs = a.epochs or 5
        body_lr, head_lr = a.body_lr or 5e-5, a.head_lr or 1.5e-4
        instr = load_instrumentals(data, ("train",))
        neg_start, nw, lw, cw = 0, 1.0, 0.55, 0.12

    if not val:
        # a small corpus can hash entirely into train; never report a score on
        # zero rows, which reads as a perfect one
        val = rows[:max(1, len(rows) // 5)]
        print(f"WARNING: no val segments, scoring on {len(val)} training segments instead")
    for r in rows + val:
        r["_wav"] = str(wav_dir)
    if not rows:
        raise SystemExit("no training segments found; run tools/prepare_dataset.py first")
    print(f"{a.stage}: {len(rows)} segments, {len(instr)} instrumental windows, "
          f"{epochs} epochs")

    params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW([
        {"params": params, "lr": body_lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ], weight_decay=0.01)
    base_lrs = [g["lr"] for g in opt.param_groups]
    total = max(1, epochs * max(1, sum(1 for _ in K.batcher(rows, a.secs, a.items))))
    warm = min(500, max(100, total // 25))
    step, best, log = 0, (9e9, ""), []

    for ep in range(1, epochs + 1):
        model.train()
        items = K.expand(rows, nw)
        line_rows = load_rows(data, ("train",), ("line",)) if lw else []
        chunk_rows = load_rows(data, ("train",), ("chunk",)) if cw else []
        for r in line_rows + chunk_rows:
            r["_wav"] = str(wav_dir)
        items += K.expand(line_rows, lw) + K.expand(chunk_rows, cw)
        items += K.expand(instr, 2.0 if ep >= neg_start else 0.0, seed=ep)
        if ep == neg_start:
            print(f"  instrumental negatives phased in at epoch {ep}")

        run, nb, t0 = 0.0, 0, time.time()
        for batch in K.batcher(items, a.secs, a.items, seed=ep):
            f, m, l, lab = K.collate(fe, [(K.read_audio(s, wav_dir)[0], s["targets"])
                                          for s in batch])
            o = model(f.to(a.device), attention_mask=m.to(a.device), labels=lab.to(a.device))
            o.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            step += 1
            sc = min(1.0, step / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = max(b * 0.05, b * sc)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += float(o.loss.detach())
            nb += 1

        per, spur = evaluate(model, fe, val, a.device, instr)
        print(f"ep {ep:02d} loss {run/max(1,nb):.4f} val_PER {per*100:5.2f}% "
              f"spurious {spur:5d} {time.time()-t0:.0f}s", flush=True)
        log.append({"epoch": ep, "loss": run / max(1, nb), "val_per": per})
        if per < best[0]:
            best = (per, str(out / "best.pt"))
            torch.save({"model": model.state_dict(), "epoch": ep, "val_per": per},
                       best[1])
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"best PER {best[0]*100:.2f}% -> {best[1]}")


if __name__ == "__main__":
    main()
