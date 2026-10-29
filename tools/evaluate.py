"""Measure alignment error against word-level references.

    python tools/evaluate.py --weights weights/qadalign-0.1.pt --pairs data/pairs \
        --split test --limit 50

Reports line-start and word-onset error and the fraction of line-final words
under half their reference duration.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
from qadalign.align import Config, align
from qadalign.model import load

AUDIO_EXT = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".opus", ".aiff", ".aif"}


def key(text):
    n = re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()
    return " ".join(n)[:40]


def reference_lines(meta):
    """Reference lines, built from words when the pair has them."""
    if meta.get("words"):
        lines, cur = [], []
        by_line = {}
        for w in meta["words"]:
            by_line.setdefault(w.get("line", 0), []).append(w)
        for k in sorted(by_line):
            ws = by_line[k]
            lines.append({"text": " ".join((w.get("text") or w.get("w")) for w in ws),
                          "start": ws[0]["start"], "end": ws[-1]["end"], "words": ws})
        return lines
    return [ln for ln in (meta.get("lines") or []) if ln.get("start") is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()

    pairs = Path(a.pairs)
    meta_all = []
    for src in sorted(pairs.iterdir()):
        if src.suffix.lower() not in AUDIO_EXT:
            continue
        mp = src.with_suffix(".json")
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        if a.split and meta.get("split", "test") != a.split:
            continue
        meta_all.append((src, meta))
    if a.limit:
        meta_all = meta_all[:a.limit]
    if not meta_all:
        raise SystemExit("no pairs matched")

    model = load(a.weights, device=a.device, int8=a.int8)
    cfg = Config(device=a.device)
    lerr, werr, short = [], [], []

    for src, meta in meta_all:
        ref = reference_lines(meta)
        if not ref:
            continue
        text = "\n".join(ln["text"] for ln in ref)
        try:
            out = align(src, text, model, cfg=cfg)["lines"]
        except Exception as e:
            print(f"  {src.name}: {e}")
            continue
        sm = difflib.SequenceMatcher(a=[key(x["text"]) for x in out],
                                    b=[key(x["text"]) for x in ref], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                continue
            for m in range(i2 - i1):
                p, r = out[i1 + m], ref[j1 + m]
                lerr.append(abs(p["start"] - r["start"]))
                if r.get("words") and p.get("words"):
                    for pw, rw in zip(p["words"], r["words"]):
                        werr.append(abs(pw["start"] - rw["start"]))
                    rd = r["words"][-1]["end"] - r["words"][-1]["start"]
                    if rd > 0.15:
                        pd = p["end"] - p["words"][-1]["start"]
                        short.append(pd / rd < 0.5)

    le, we = np.array(lerr), np.array(werr) if werr else np.array([np.nan])
    print(f"{len(meta_all)} files, {len(le)} matched lines, {len(we)} words")
    print(f"line start: median {np.median(le)*1000:5.0f}ms  p90 {np.percentile(le,90)*1000:5.0f}ms  "
          f"<=250ms {np.mean(le<=0.25)*100:4.1f}%  <=1s {np.mean(le<=1.0)*100:4.1f}%  "
          f">2s {np.mean(le>2.0)*100:4.1f}%")
    if werr:
        print(f"word onset: median {np.median(we)*1000:5.0f}ms  p90 {np.percentile(we,90)*1000:5.0f}ms  "
              f"<=250ms {np.mean(we<=0.25)*100:4.1f}%")
        print(f"line-final words cut under half: {np.mean(short)*100:.0f}%")


if __name__ == "__main__":
    main()
