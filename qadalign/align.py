"""Alignment pipeline: lyric text and audio to timed lines and words.

1. Decode to 16 kHz mono.
2. Score overlapping 30 s windows for phone posteriors and per-frame sung
   probability.
3. Gate where the lyric may start.
4. Run constrained CTC Viterbi over the whole lyric in one pass.
5. Group phone spans into words and lines, then set line ends.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

from . import ctc as A
from . import g2p
from .audio import SR, load_audio

FRAME = 0.04          # 40 ms per model frame


@dataclass
class Config:
    skip_cost: float = 1.8      # price of audio with no lyric (ad-libs, extra syllables)
    del_cost: float = 4.0       # price of lyric with no audio
    release_cost: float = 0.0   # price of leaving a phone; 0 measured best
    gate: float = 0.3           # sung probability needed to allow the first lyric
    gate_rel: float = 0.5       # ...and at least this fraction of the track's loud-sung level
    tail: float = 0.30          # seconds added after a line's last phone
    snap_max: float = 1.0       # cap when extending a line to where singing stops
    window: float = 30.0
    hop: float = 15.0
    smooth: int = 25            # frames (1 s) of smoothing on the sung signal
    gate_run: float = 1.0       # seconds the sung signal must stay above the gate
    device: str = "cpu"
    vocab: int = 40


_TAG = re.compile(r"\[[^\]]*\]")


def parse_lyrics(text: str):
    """Text -> [{"text", "words"}], one entry per non-empty line.

    The line text is kept exactly as written. Section tags carry no phones, so a
    tag-only line survives as an untimed entry rather than disappearing.
    """
    lines, oov = [], {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        words = []
        for tok in _TAG.sub(" ", raw).split():
            if not any(c.isalnum() for c in tok) or set(tok) == {"*"}:
                continue
            phones = g2p.phones_for(tok, oov)
            if phones:
                words.append({"w": tok, "phones": phones})
        lines.append({"text": line, "words": words})
    return lines


def sustained_onset(sm, gate, min_run):
    """First frame where the sung signal stays above `gate` for `min_run` frames."""
    above = sm > gate
    if min_run <= 1:
        idx = np.nonzero(above)[0]
        return int(idx[0]) if len(idx) else None
    run = 0
    for i, v in enumerate(above):
        run = run + 1 if v else 0
        if run >= min_run:
            return i - min_run + 1
    return None


def _windows(total, cfg):
    t = 0.0
    while t < total - 1.0:
        yield t, min(total, t + cfg.window)
        if t + cfg.window >= total:
            break
        t += cfg.hop


@torch.no_grad()
def frame_posteriors(model, fe, wav, cfg: Config):
    """Per-frame phone log-probs and sung probability, cross-faded over windows."""
    total = len(wav) / SR
    n = int(total / FRAME) + 2
    acc = np.zeros((n, cfg.vocab), np.float32)
    sacc = np.zeros(n, np.float32)
    wsum = np.zeros(n, np.float32)
    for a, b in _windows(total, cfg):
        chunk = wav[int(a * SR):int(b * SR)]
        if len(chunk) < 4000:
            break
        enc = fe([chunk], sampling_rate=SR, return_tensors="pt")
        logits, sung = model(enc["input_features"].to(cfg.device),
                             attention_mask=enc["attention_mask"].to(cfg.device))
        lp = logits.float().log_softmax(-1)[0].cpu().numpy()
        sp = sung.cpu().numpy()
        idx = int(round(a / FRAME)) + np.arange(lp.shape[0])
        w = np.hanning(lp.shape[0] + 2)[1:-1].astype(np.float32)
        ok = idx < n
        acc[idx[ok]] += lp[ok] * w[ok, None]
        m = min(len(sp), len(idx))
        sacc[idx[:m][ok[:m]]] += sp[:m][ok[:m]] * w[:m][ok[:m]]
        wsum[idx[ok]] += w[ok]
    # frames past the final window carry no weight; keep them finite
    den = np.maximum(wsum, 1e-6)
    return acc / den[:, None], sacc / den


def align(audio_path, lyrics_text: str, model, fe=None, cfg: Config | None = None):
    """Align lyrics to audio. Returns {"lines": [...]} with nested word timings."""
    from .model import feature_extractor
    cfg = cfg or Config()
    fe = fe or feature_extractor()
    wav = load_audio(audio_path)
    lines = parse_lyrics(lyrics_text)
    flat = [w for ln in lines for w in ln["words"]]
    if not flat:
        raise ValueError("no pronounceable lyric words found")
    targets = A.target_ids(flat)

    lp, sung = frame_posteriors(model, fe, wav, cfg)
    if lp.shape[0] <= len(targets):
        raise ValueError("lyrics longer than the audio can hold")

    # nothing is sung before the vocals start: gate the first lyric
    if cfg.gate > 0:
        sm = np.convolve(sung, np.ones(cfg.smooth) / cfg.smooth, mode="same")
        # require a fraction of the level reached when the file is plainly sung
        level = max(cfg.gate, cfg.gate_rel * float(np.percentile(sm, 90)))
        onset = sustained_onset(sm, level, int(cfg.gate_run / FRAME))
        if onset:
            lp[:onset, 1:] = -1e9

    path, ext, skipped = A.viterbi_robust(lp, targets, skip_cost=cfg.skip_cost,
                                          del_cost=cfg.del_cost,
                                          release_cost=cfg.release_cost)
    words = A.word_spans(A.phone_spans(path, ext, skipped), flat)

    out, k, prev_end = [], 0, 0.0
    for ln in lines:
        n = len(ln["words"])
        chunk = [w for w in words[k:k + n] if w["start"] is not None]
        k += n
        if not chunk:
            # nothing singable on this line, such as a section tag: keep it
            out.append({"text": ln["text"], "start": prev_end, "end": prev_end,
                        "words": []})
            continue
        entry = {"text": ln["text"], "start": float(chunk[0]["start"]),
                 "end": float(chunk[-1]["end"]),
                 "words": [{"text": w["w"], "start": float(w["start"]),
                            "end": float(w["end"])} for w in chunk]}
        out.append(entry)
        prev_end = entry["end"]

    # words stay highlighted until the next word starts
    for ln in out:
        for j in range(len(ln["words"]) - 1):
            ln["words"][j]["end"] = max(ln["words"][j]["end"], ln["words"][j + 1]["start"])
        if ln["words"]:
            ln["end"] = max(ln["end"], ln["words"][-1]["end"])

    # a line ends shortly after its last phone, not at the next line's start
    timed = [i for i, ln in enumerate(out) if ln["words"]]
    for pos, i in enumerate(timed):
        ln = out[i]
        nxt = out[timed[pos + 1]]["start"] if pos + 1 < len(timed) else None
        limit = (nxt - 0.05) if nxt is not None else 1e9
        end = ln["end"]
        if limit > end:
            a = int(end / FRAME)
            b = min(int(min(limit, end + cfg.snap_max) / FRAME), len(sung))
            if b > a:
                on = np.nonzero(sung[a:b] > cfg.gate)[0]
                if len(on):
                    end = max(end, (a + int(on[-1])) * FRAME)
        ln["end"] = min(end + 0.10, limit)

    for ln in out:
        ln["start"] = round(ln["start"], 2)
        ln["end"] = round(max(ln["end"], ln["start"]), 2)
        for w in ln["words"]:
            w["start"] = round(w["start"], 2)
            w["end"] = round(max(w["end"], w["start"] + 0.02), 2)

    return {"lines": out, "skipped_fraction": float(skipped.mean())}
