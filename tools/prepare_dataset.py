"""Build training data from audio and timed lyric pairs.

    python tools/prepare_dataset.py --pairs data/pairs --out data

Produces:
    <out>/wav/<id>.wav            16 kHz mono
    <out>/segments.jsonl          one row per singable segment, with phone sequence
    <out>/instrumental.jsonl      windows with no lyric, used as negative examples

Each data identifier lands in exactly one split when the pair JSON carries an
"identifier" field; otherwise the split is a stable hash of the file name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import phones_of

AUDIO_EXT = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".opus", ".aiff", ".aif"}

MAX_SEG = 12.0          # longest segment fed to the model
MIN_SEG = 1.0
INSTR_WIN = 3.0
INSTR_MIN_GAP = 6.0
CHUNK_MAX = 24.0         # longest multi-line context segment


def decode(src: Path) -> np.ndarray:
    from qadalign.audio import load_audio
    return load_audio(src)


def split_for(meta: dict, tid: str) -> str:
    key = str(meta.get("identifier") or tid)
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def chunk_words(words, cap=MAX_SEG):
    """Split a long line into singable pieces on word boundaries."""
    out, cur, t0 = [], [], None
    for w in words:
        if t0 is None:
            t0 = w["start"]
        if cur and w["end"] - t0 > cap:
            out.append((cur, t0, cur[-1]["end"]))
            cur, t0 = [], w["start"]
        cur.append(w)
    if cur:
        out.append((cur, t0, cur[-1]["end"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="directory of audio + .json pairs")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--chunks", action="store_true",
                    help="also emit multi-line context segments")
    a = ap.parse_args()

    pairs = Path(a.pairs)
    out = Path(a.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)

    segs, instr = [], []
    n_files = 0
    for src in sorted(pairs.iterdir()):
        if src.suffix.lower() not in AUDIO_EXT:
            continue
        meta_path = src.with_suffix(".json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        tid = src.stem
        wav = decode(src)
        import soundfile as sf
        sf.write(str(out / "wav" / f"{tid}.wav"), wav, 16000)
        dur = len(wav) / 16000
        split = split_for(meta, tid)
        n_files += 1

        words = [w for w in (meta.get("words") or [])
                 if w.get("start") is not None and w.get("end") is not None]
        sung = []
        if words:
            chorus = " ".join((w.get("text") or w.get("w")) for w in words)
            for grp, t0, t1 in chunk_words(words):
                text = " ".join(w["text"] for w in grp)
                ph = phones_of(text)
                if ph and MIN_SEG <= t1 - t0 <= MAX_SEG:
                    segs.append({"track_id": tid, "split": split, "tier": "word",
                                 "start": round(t0, 3), "end": round(t1, 3),
                                 "text": text, "phones": ph})
                    sung.append((t0, t1))
            del chorus
        for ln in (meta.get("lines") or []):
            if not (ln.get("start") is not None and ln.get("end") is not None):
                continue
            if words:
                continue                      # word-level supervision is strictly better
            text = ln.get("text") or ""
            ph = phones_of(text)
            if ph and MIN_SEG <= ln["end"] - ln["start"] <= MAX_SEG:
                segs.append({"track_id": tid, "split": split, "tier": "line",
                             "start": round(ln["start"], 3), "end": round(ln["end"], 3),
                             "text": text, "phones": ph})
                sung.append((ln["start"], ln["end"]))

        # multi-line context chunks: longer windows teach the model context, and
        # the shipped run mixed them in at a low weight
        if a.chunks and words:
            t0 = words[0]["start"]
            grp, start = [], t0
            for w in words:
                if w["end"] - start > CHUNK_MAX and grp:
                    text = " ".join((x.get("text") or x.get("w")) for x in grp)
                    ph = phones_of(text)
                    if ph:
                        segs.append({"track_id": tid, "split": split, "tier": "chunk",
                                     "start": round(start, 3), "end": round(grp[-1]["end"], 3),
                                     "text": text, "phones": ph})
                    grp, start = [], w["start"]
                grp.append(w)
            if grp:
                text = " ".join((x.get("text") or x.get("w")) for x in grp)
                ph = phones_of(text)
                if ph and grp[-1]["end"] - start >= 6.0:
                    segs.append({"track_id": tid, "split": split, "tier": "chunk",
                                 "start": round(start, 3), "end": round(grp[-1]["end"], 3),
                                 "text": text, "phones": ph})

        # instrumental negatives: gaps with no lyric at all
        sung.sort()
        gaps = []
        prev = 0.0
        for s0, s1 in sung:
            if s0 - prev >= INSTR_MIN_GAP:
                gaps.append((prev, s0))
            prev = max(prev, s1)
        if dur - prev >= INSTR_MIN_GAP:
            gaps.append((prev, dur))
        for g0, g1 in gaps:
            t = g0 + 0.5
            while t + INSTR_WIN <= g1:
                instr.append({"track_id": tid, "split": split, "tier": "instr",
                              "start": round(t, 3), "end": round(t + INSTR_WIN, 3)})
                t += INSTR_WIN

    with (out / "segments.jsonl").open("w") as f:
        for s in segs:
            f.write(json.dumps(s) + "\n")
    with (out / "instrumental.jsonl").open("w") as f:
        for s in instr:
            f.write(json.dumps(s) + "\n")

    w = sum(1 for s in segs if s["tier"] == "word")
    print(f"{n_files} files -> {len(segs)} segments ({w} word-level), "
          f"{len(instr)} instrumental windows")


if __name__ == "__main__":
    main()
