# Reproducing qadalign 0.1

Rebuilds `weights/qadalign-0.1.pt` from the public Aurora CTC checkpoint and a
set of timed lyric pairs.

One RTX 3090 takes a couple of compute hours, about 50 minutes of it for stage three.
A GPU is required for training but inference runs on CPU.

## 0. Requirements

- An Aurora ASR CTC checkpoint (Apache-2.0), local or fetched:

  ```
  python tools/fetch_base.py --out aurora
  ```

- Some 61 hours of audio material, 36 hours of it with word-level lyric timing.
- Python 3.10+, `pip install -e .`

## 1. Data

One JSON per audio file, beside it, same stem:

```
pairs/<id>.mp3
pairs/<id>.json
```

```json
{
  "identifier": "group-a",
  "words": [{"text": "hello", "start": 1.23, "end": 1.45},
            {"text": "world", "start": 1.45, "end": 1.90}],
  "lines": [{"text": "hello world", "start": 1.23, "end": 1.90}]
}
```

- `words` is the primary label.
- `lines` alone also works; those segments are weighted 0.55 instead of 1.0.
- `identifier` groups files that belong together, so one data identifier never
  lands in both splits. Without it the split is a hash of the file name and the
  reported accuracy will be optimistic.

```
python tools/prepare_dataset.py --pairs pairs --out data --chunks
```

Writes `data/wav/` (16 kHz mono), `data/segments.jsonl` and
`data/instrumental.jsonl`. The instrumental windows are gaps of 6 s or more
containing no lyric. Segments are capped at 12 s and split on word boundaries;
`--chunks` adds multi-line segments of up to 24 s.

## 2. Stage one: phone head

A fresh 39-phone CTC head on the Aurora encoder.

```
python tools/train_aligner.py --stage phone --base ./aurora --data data \
    --epochs 20 --body-lr 3e-4 --head-lr 1e-3
```

Word-level segments only, float32, no augmentation. Expect about 64% phone error
after 20 epochs.

## 3. Stage two: instrumental windows

```
python tools/train_aligner.py --stage gaps --init runs/phone/best.pt --data data \
    --epochs 20 --body-lr 5e-5 --head-lr 1.5e-4
```

Adds instrumental windows at weight 2.0 from epoch 3. Expect about 61% phone
error with a few hundred emissions on instrumental audio.

## 4. Stage three: all segments

```
python tools/train_aligner.py --stage full --init runs/gaps/best.pt --data data \
    --epochs 5 --body-lr 5e-5 --head-lr 1.5e-4
```

Word-level at 1.0 with duration oversampling (boost 0.6), line-level at 0.55,
context chunks at 0.12, instrumental at 2.0. Cosine schedule, 500-step warmup,
gradient clipping at 5.0, AdamW with weight decay 0.01.

Phone error reaches about 75%. Phone error is not alignment error: constrained
decoding needs a preference between paths, not a transcription, which is why
step 5 selects on alignment.

## 5. Select the checkpoint

```
python tools/evaluate.py --weights runs/full/best.pt --pairs pairs --split test
```

Rank every saved epoch and keep the best. On held-out data split by identifier
the target is a line-start median near 284 ms and a word onset near 220 ms.

## 6. Sung head

```
python tools/train_sung_head.py --init runs/full/best.pt --data data --epochs 6 \
    --out runs/sung_head.pt
```

The aligner is frozen. Frames inside lyric segments are positives, frames inside
instrumental windows are negatives. Expect AUC near 0.88.

## 7. Package the weights

```
python - <<'PY'
import torch
ck = torch.load("runs/full/best.pt", map_location="cpu", weights_only=False)
aux = torch.load("runs/sung_head.pt", map_location="cpu", weights_only=False)
torch.save({"model": ck["model"], "sung": aux["head"],
            "meta": {"name": "qadalign", "version": "0.1", "vocab": 40, "d_model": 128,
                     "sung_head_auc": round(float(aux["auc"]), 3)}},
           "weights/qadalign-0.1.pt")
PY
```

## Parameters

| stage | setting | value |
|---|---|---|
| phone | epochs / body lr / head lr | 20 / 3e-4 / 1e-3 |
| phone | precision, augmentation | fp32, none |
| gaps | epochs / body lr / head lr | 20 / 5e-5 / 1.5e-4 |
| gaps | instrumental weight, start epoch | 2.0, 3 |
| full | epochs / body lr / head lr | 5 / 5e-5 / 1.5e-4 |
| full | word / line / chunk / instrumental weight | 1.0 / 0.55 / 0.12 / 2.0 |
| full | duration boost | 0.6 |
| full | batch | 240 s of audio, max 24 segments |
| sung head | epochs / lr / params | 6 / 3e-3 / 8,321 |
| decode | skip / delete / release cost | 1.8 / 4.0 / 0.0 |
| decode | gate, absolute / fraction of p90 | 0.3 / 0.5 |
| decode | tail, snap cap | 0.30 s, 1.0 s |
| decode | window / hop / frame | 30 s / 15 s / 40 ms |

## Deviations

Three places where a reproduction will not match the weights exactly:

1. Line-level pairs. The published weights used approximate word positions
   inferred inside line-level pairs at weight 0.55; `prepare_dataset.py` emits
   line segments directly at the same weight.
2. Checkpoint selection. Epoch ranking is manual here rather than scripted.
3. Data. A different set of audio material gives different absolute numbers.
   The stage-wise phone error figures and the alignment/phone-error divergence
   should reproduce.
