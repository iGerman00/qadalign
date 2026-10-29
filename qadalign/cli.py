"""Command line interface.

    python -m qadalign.cli audio.mp3 lyrics.txt -o out.json
    python -m qadalign.cli audio.mp3 lyrics.txt --flat words
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .align import Config, align
from .model import DEFAULT_WEIGHTS, load


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="qadalign",
        description="Align plain-text lyrics to an audio file, line and word level.")
    p.add_argument("audio", help="audio file (anything torchaudio/libsndfile reads)")
    p.add_argument("lyrics", help="lyrics as plain text, one line per lyric line")
    p.add_argument("-o", "--out", default=None, help="output JSON (default: stdout)")
    p.add_argument("--flat", choices=["lines", "words"], default=None,
                   help="emit a flat list instead of the nested structure")
    p.add_argument("--weights", default=None, help=f"checkpoint (default {DEFAULT_WEIGHTS.name})")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--int8", action="store_true", help="quantise weights ~4x smaller")
    p.add_argument("--gate", type=float, default=0.3, help="sung threshold for the first lyric")
    p.add_argument("--skip", type=float, default=1.8, help="price of audio with no lyric")
    p.add_argument("--snap", type=float, default=1.0, help="max seconds to extend a line end")
    a = p.parse_args(argv)

    for f in (a.audio, a.lyrics):
        if not Path(f).is_file():
            p.error(f"no such file: {f}")

    text = Path(a.lyrics).read_text(encoding="utf-8", errors="replace")
    model = load(a.weights, device=a.device, int8=a.int8)
    cfg = Config(device=a.device, gate=a.gate, skip_cost=a.skip, snap_max=a.snap)
    result = align(a.audio, text, model, cfg=cfg)

    if a.flat == "lines":
        payload = [{"text": ln["text"], "start": ln["start"], "end": ln["end"]}
                   for ln in result["lines"]]
    elif a.flat == "words":
        payload = [{"text": w["text"], "start": w["start"], "end": w["end"]}
                   for ln in result["lines"] for w in ln["words"]]
    else:
        payload = result

    blob = json.dumps(payload, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(blob, encoding="utf-8")
        n = len(result["lines"])
        print(f"{n} lines, {sum(len(x['words']) for x in result['lines'])} words -> {a.out}",
              file=sys.stderr)
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
