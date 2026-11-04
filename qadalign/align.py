"""Alignment pipeline: lyric text and audio to timed lines and words.

1. Decode to 16 kHz mono.
2. Score overlapping 30 s windows for phone posteriors and per-frame sung
   probability.
3. Gate where the lyric may start.
4. Run constrained CTC Viterbi over the whole lyric in one pass.
5. Group phone spans into words and lines, place the words the path gave no
   audio, then set line ends.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

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
_SYL = re.compile(r"[aeiouy]+")
SYL_DUR = 0.30        # seconds of singing assumed per syllable when filling gaps


def parse_lyrics(text: str):
    """Text -> [{"text", "words"}], one entry per non-empty line.

    The line text is kept exactly as written. Section tags carry no phones, so a
    tag-only line survives as an untimed entry rather than disappearing. A word
    no pronunciation was found for gets placeholder vowels per syllable,
    so it still takes audio and keeps its place. Any word the path leaves empty
    is placed from its neighbours.
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
            words.append({"w": tok, "phones": phones or g2p.skeleton(tok),
                          "skel": not phones})
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


def frame_posteriors(model, fe, wav, cfg: Config):
    """Per-frame phone log-probs and sung probability, cross-faded over windows."""
    import torch

    with torch.no_grad():
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


def _weight(text: str) -> int:
    """Syllables a word is assumed to hold, from its spelling."""
    return max(1, len(_SYL.findall(re.sub(r"[^a-z]", "", text.lower()))))


def _fill(words, lo, hi):
    """Give each run of untimed words a share of the gap, by syllable count."""
    n, i = len(words), 0
    while i < n:
        if words[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j + 1 < n and words[j + 1]["start"] is None:
            j += 1
        a = words[i - 1]["end"] if i > 0 else lo
        b = words[j + 1]["start"] if j + 1 < n else hi
        if a is None:
            a = 0.0
        if b is None or b < a:
            b = a
        total = sum(_weight(words[k]["w"]) for k in range(i, j + 1)) or 1
        t, span = a, b - a
        for k in range(i, j + 1):
            d = span * _weight(words[k]["w"]) / total
            words[k]["start"], words[k]["end"] = t, t + d
            t += d
        i = j + 1


def place_words(lines):
    """Time every word the path left untimed, and each line's own bounds.

    A skipped word takes a share of the silence between its neighbours,
    weighted by syllable count. A line nothing was timed on is spread over the
    gap between the lines around it, one syllable roughly every SYL_DUR seconds.
    """
    anchors = [next((w["start"] for w in ln["words"] if w["start"] is not None), None)
               for ln in lines]
    prev_end = 0.0
    for i, ln in enumerate(lines):
        words = ln["words"]
        if not words:
            ln["start"] = ln["end"] = prev_end
            continue
        nxt = next((anchors[j] for j in range(i + 1, len(lines))
                    if anchors[j] is not None), None)
        here = anchors[i]
        if here is None:
            lo = prev_end + 0.02
            hi = nxt - 0.05 if nxt is not None else lo + SYL_DUR * sum(
                _weight(w["w"]) for w in words)
            _fill(words, lo, max(hi, lo))
        else:
            lead = next(k for k, w in enumerate(words) if w["start"] is not None)
            last = len(words) - 1 - next(k for k, w in enumerate(reversed(words))
                                         if w["start"] is not None)
            lo = here
            if lead:
                want = SYL_DUR * sum(_weight(w["w"]) for w in words[:lead])
                lo = max(prev_end + 0.02, here - want)
            hi = words[last]["end"]
            if last < len(words) - 1:
                hi += SYL_DUR * sum(_weight(w["w"]) for w in words[last + 1:])
                if nxt is not None:
                    hi = min(hi, nxt - 0.05)
                hi = max(hi, words[last]["end"])
            _fill(words, lo, hi)
        ln["start"] = float(words[0]["start"])
        ln["end"] = float(words[-1]["end"])
        prev_end = ln["end"]


def decode(lp, sung, lines, cfg: Config):
    """Constrained CTC path over phone and sung posteriors -> timed lines."""
    flat = [w for ln in lines for w in ln["words"]]
    if not flat:
        raise ValueError("no lyric words found")
    targets = A.target_ids(flat)
    if lp.shape[0] <= len(targets):
        raise ValueError("lyrics longer than the audio can hold")

    # nothing is sung before the vocals start: gate the first lyric
    if cfg.gate > 0:
        sm = np.convolve(sung, np.ones(cfg.smooth) / cfg.smooth, mode="same")
        # require a fraction of the level reached when the file is plainly sung
        level = max(cfg.gate, cfg.gate_rel * float(np.percentile(sm, 90)))
        onset = sustained_onset(sm, level, int(cfg.gate_run / FRAME))
        if onset:
            lp = lp.copy()
            lp[:onset, 1:] = -1e9

    path, ext, skipped = A.viterbi_robust(lp, targets, skip_cost=cfg.skip_cost,
                                          del_cost=cfg.del_cost,
                                          release_cost=cfg.release_cost)
    spans = A.word_spans(A.phone_spans(path, ext, skipped), flat)

    out, k = [], 0
    for ln in lines:
        n = len(ln["words"])
        out.append({"text": ln["text"], "words": spans[k:k + n]})
        k += n
    place_words(out)
    for ln in out:
        ln["words"] = [{"text": w["w"], "start": w["start"], "end": w["end"]}
                       for w in ln["words"]]

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
        ln["end"] = min(end + cfg.tail, limit)

    for ln in out:
        ln["start"] = round(ln["start"], 2)
        ln["end"] = round(max(ln["end"], ln["start"]), 2)
        for w in ln["words"]:
            w["start"] = round(w["start"], 2)
            w["end"] = round(max(w["end"], w["start"] + 0.02), 2)

    return {"lines": out, "skipped_fraction": float(skipped.mean())}


def align(audio_path, lyrics_text: str, model, fe=None, cfg: Config | None = None):
    """Align lyrics to audio. Returns {"lines": [...]} with nested word timings."""
    from .model import feature_extractor
    cfg = cfg or Config()
    fe = fe or feature_extractor()
    wav = load_audio(audio_path)
    lp, sung = frame_posteriors(model, fe, wav, cfg)
    return decode(lp, sung, parse_lyrics(lyrics_text), cfg)
