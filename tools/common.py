"""Shared plumbing for dataset building and training.

Input format (all the trainer knows about): a directory of pairs, one audio file
and one JSON per audio file:

    pairs/<id>.mp3   pairs/<id>.json

The JSON holds either or both of:

    {"words": [{"text": "hello", "start": 1.23, "end": 1.45}, ...],
     "lines": [{"text": "hello world", "start": 1.23, "end": 2.60}, ...]}

Times are seconds. Word entries are the strong signal; line entries alone are
weaker, since word positions inside a line are not marked.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root

BOOTSTRAPPED = True

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from qadalign import g2p
from qadalign.phones import PHONE2ID, BLANK, NUM_PHONES

SR = 16000
PAD_S = 0.25          # padding each side of a segment; CTC absorbs it as blank
FRAME = 0.04


def phones_of(text: str):
    """Phone names for a text span, or None if any word is unknown."""
    out = []
    for tok in g2p.tokenize(text):
        ph = g2p.phones_for(tok)
        if not ph:
            return None
        out += ph
    return out


def load_segments(path, splits=("train",)):
    rows = []
    for line in Path(path).open():
        s = json.loads(line)
        if s["split"] not in splits:
            continue
        ids = [PHONE2ID[p] for p in s["phones"] if p in PHONE2ID]
        if ids:
            s["targets"] = ids
            rows.append(s)
    return rows


def read_audio(seg, wav_dir, pad=PAD_S):
    """Segment audio with padding; short reads are zero-padded, not dropped."""
    t0 = max(0.0, seg["start"] - pad)
    t1 = seg["end"] + pad
    want = max(1, int((t1 - t0) * SR))
    path = Path(wav_dir) / f"{seg['track_id']}.wav"
    with sf.SoundFile(str(path)) as f:
        total = f.frames
        a = min(max(0, int(t0 * SR)), max(0, total - 1))
        b = min(total, int(t1 * SR))
        if b <= a:
            wav = np.zeros(want, dtype=np.float32)
        else:
            f.seek(a)
            wav = f.read(b - a, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if len(wav) < want:
        wav = np.pad(wav, (0, want - len(wav)))
    return wav.astype(np.float32), t0


def collate(fe, batch):
    enc = fe([b[0] for b in batch], sampling_rate=SR, return_tensors="pt")
    lens = enc["attention_mask"].sum(-1)
    maxl = max(1, max(len(b[1]) for b in batch))
    labels = torch.full((len(batch), maxl), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        if b[1]:
            labels[i, :len(b[1])] = torch.tensor(b[1])
    return enc["input_features"], enc["attention_mask"], lens, labels


def batcher(rows, max_secs=240.0, max_items=24, seed=0):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda i: len(rows[i]["targets"]))
    batch, secs = [], 0.0
    for i in order:
        d = rows[i]["end"] - rows[i]["start"] + 2 * PAD_S
        if batch and (secs + d > max_secs or len(batch) >= max_items):
            yield [rows[j] for j in batch]
            batch, secs = [], 0.0
        batch.append(i)
        secs += d
    if batch:
        yield [rows[j] for j in batch]


def expand(rows, weight, boost=0.0, seed=0):
    """Fractional oversampling. boost>0 also repeats short, dense segments, which
    is how slow sustained lines get their share of the batch."""
    if weight <= 0:
        return []
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        n = int(weight)
        frac = weight - n
        out += [r] * n
        if rng.random() < frac:
            out.append(r)
        if boost > 0:
            dur = max(0.2, r["end"] - r["start"])
            rate = len(r["targets"]) / dur
            extra = int(boost * max(0.0, 3.0 - rate) * 4)
            out += [r] * extra
    return out


def swap_head(model, vocab: int = NUM_PHONES):
    d = model.config.d_model
    model.head = nn.Linear(d, vocab)
    nn.init.normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    model.config.vocab_size = vocab
    model.config.blank_token_id = BLANK
    return model


def greedy(logits, lens):
    pred = logits.argmax(-1).cpu().numpy()
    out = []
    for i in range(pred.shape[0]):
        seq, prev = [], -1
        for t in pred[i, : int(lens[i])]:
            t = int(t)
            if t != prev and t != BLANK:
                seq.append(t)
            prev = t
        out.append(seq)
    return out


def edit_distance(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def phone_error_rate(preds, refs):
    e = sum(edit_distance(p, r) for p, r in zip(preds, refs))
    n = sum(len(r) for r in refs)
    return e / max(1, n)
