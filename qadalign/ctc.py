"""Constrained CTC Viterbi over a known phone sequence.

Standard CTC topology: blank at 0, states alternate blank/label. Blank states
absorb pauses and melismas for free, which is why phone CTC suits singing.

Two extra moves let the path survive lyrics and audio disagreeing.
"""
from __future__ import annotations

import numpy as np

from .phones import PHONE2ID, BLANK

NEG = -1e30


def viterbi_robust(log_probs, targets, skip_cost=1.8, del_cost=4.0,
                   release_cost=0.0):
    """Align `targets` (phone ids) to frames.

    skip_cost    price of calling a frame unmatched audio.
    del_cost     price of advancing past one target without consuming audio.
    release_cost optional price for leaving a phone state.

    Returns (path, ext, skipped): path[t] is the state at frame t, ext[s] the
    label state s emits (0 = blank), skipped[t] True for unassigned frames.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T = log_probs.shape[0]
    L = len(targets)
    S = 2 * L + 1
    ext = np.zeros(S, dtype=np.int64)
    ext[1::2] = targets

    alpha = np.full(S, NEG)
    alpha[0] = log_probs[0, BLANK]
    if L:
        alpha[1] = log_probs[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)

    repeat_ok = np.zeros(S, dtype=bool)
    repeat_ok[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])

    for t in range(1, T):
        a1 = np.full(S, NEG)
        if release_cost:
            rel = (ext[:-1] != 0).astype(np.float64) * release_cost
            a1[1:] = alpha[:-1] - rel
        else:
            a1[1:] = alpha[:-1]
        a2 = np.full(S, NEG)
        a2[2:] = np.where(repeat_ok[2:], alpha[:-2], NEG)
        a3 = np.full(S, NEG)
        a3[2:] = alpha[:-2] - del_cost if L > 1 else NEG
        a4 = alpha - skip_cost
        src = np.stack([alpha, a1, a2, a3])
        idx = src.argmax(0)
        emit = src[idx, np.arange(S)] + log_probs[t, ext]
        take_skip = a4 > emit
        alpha = np.where(take_skip, a4, emit)
        back[t] = np.where(take_skip, 4, idx).astype(np.int8)

    s = S - 1
    if L and alpha[S - 2] > alpha[s]:
        s = S - 2
    path = np.zeros(T, dtype=np.int64)
    skipped = np.zeros(T, dtype=bool)
    path[-1] = s
    for t in range(T - 1, 0, -1):
        code = int(back[t, s])
        if code == 4:
            skipped[t] = True
        else:
            s -= code
        path[t - 1] = s
    return path, ext, skipped


def phone_spans(path, ext, skipped):
    """[(label, first_frame, last_frame_exclusive)] in target order."""
    spans = []
    for s in range(1, len(ext), 2):
        idx = np.nonzero((path == s) & ~skipped)[0]
        spans.append((int(ext[s]),
                      int(idx[0]) if len(idx) else None,
                      int(idx[-1]) + 1 if len(idx) else None))
    return spans


def word_spans(spans, words, hop=0.04, half=0.02):
    """Group phone spans into word spans.

    `words` is [{w, phones}] with phones in the order they were targeted. A word
    with no frames of its own inherits its neighbour's boundary.
    """
    out, pi = [], 0
    for w in words:
        n = len(w.get("phones") or [])
        chunk = spans[pi:pi + n]
        pi += n
        frames = [(a, b) for _, a, b in chunk if a is not None]
        if not frames:
            out.append({"w": w.get("w"), "start": None, "end": None})
        else:
            a = min(x[0] for x in frames)
            b = max(x[1] for x in frames)
            out.append({"w": w.get("w"), "start": a * hop + half,
                        "end": b * hop + half})
    for i, w in enumerate(out):
        if w["start"] is None:
            nxt = next((x["start"] for x in out[i + 1:] if x["start"] is not None), None)
            prv = next((x["end"] for x in out[i - 1::-1] if x["end"] is not None), None)
            w["start"] = nxt if nxt is not None else prv
            w["end"] = prv if prv is not None else nxt
    return out


def target_ids(words):
    return [PHONE2ID[p] for w in words for p in w["phones"]]
