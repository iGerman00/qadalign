# qadalign

> [!Note]
> Check out [Vizzy](https://vizzy.io) - our free web-based music video editor that can use the output of this tool to create a karaoke-style video!

Qadalign is a "quick and dirty" tool to align plain-text lyrics to an audio file. It returns when each line and each word is sung. It uses a small, bespoke ML model based on [AuroraASR-1M](https://huggingface.co/AuroraAI-Research/AuroraASR-1M). It has 929K parameters, weighs ~3.8 MB at float32 and about 1 MB at int8. It runs offline on CPU, taking around 5 seconds for a four-minute file.

The quality is passable for a first draft and will save you time compared to doing it manually, but this is a *tiny* experimental model and it will make mistakes. I'm pretty sure it can actually run on a tiny microcontroller if you're keen enough to make a battery out of a couple potatoes and spend a few evenings in debugging pain.

```
python -m qadalign.cli audio.mp3 lyrics.txt -o out.json
```

```json
{
  "lines": [
    {"text": "And I say it's wasted but I'm not okay",
     "start": 17.62, "end": 23.06,
     "words": [
       {"text": "And", "start": 17.62, "end": 17.78},
       {"text": "I", "start": 17.90, "end": 18.10}
     ]}
  ],
  "skipped_fraction": 0.03
}
```

`skipped_fraction` is the fraction of audio the aligner
assigned to no lyric.

## Install

```
pip install -e .
```

CMUdict handles ordinary English vocabulary. `espeak-ng` is optional and only
consulted for words CMUdict lacks; without it those words are skipped.

## Use

```
python -m qadalign.cli AUDIO LYRICS [-o OUT] [--flat lines|words] [--int8] [--device cuda]
```

Flat output returns the plain `[{"text", "start", "end"}, ...]` shape:

```
python -m qadalign.cli audio.mp3 lyrics.txt --flat lines
python -m qadalign.cli audio.mp3 lyrics.txt --flat words > words.json
```

From Python:

```python
from qadalign.align import align
from qadalign.model import load

model = load()
with open("lyrics.txt") as f:
    result = align("audio.mp3", f.read(), model)
for line in result["lines"]:
    print(line["start"], line["text"])
```

Lyrics are one line per sung line. Text is used exactly as written: section
tags, parentheses, capitalisation and punctuation all survive into the output.
Tags carry no phones, so a tag-only line comes out with zero length instead of
being dropped or sung. Pass lyrics only: a credit or header line will be treated
as a lyric and take up audio time.

## Browser

`web/` runs the model client-side with onnxruntime-web.

```
python3 -m http.server 8000        # from the repository root
```

Open `http://localhost:8000/web/`

`web/data` is a symlink into the package so the dictionary resolves either way.

Payload is ~2.1 MB:
- `web/qadalign.onnx` 1.23 MB and
- `qadalign/data/cmudict.txt.gz` 0.86 MB.

The weights are int8 with a per-row scale, dequantised in the graph, a third of the float32 size. onnxruntime-web (MIT) loads from a CDN on first use.

You can rebuild this yourself with `python web/export_onnx.py` and `python tools/build_dict.py`.

A four-minute file takes about 9 seconds on my machine.

## How it works

Phone-level CTC against a known sequence. The lyrics are converted to a
39-phone ARPAbet sequence and a constrained Viterbi pass finds the most likely
path through the audio. Frames carrying no lyric and lyrics that were never
sung are both permitted at a cost, so a local mismatch stays local.

The audio is scored in overlapping 30 s windows and decoded in one pass over the whole lyric. Line-by-line decoding accumulates drift.

An 8.3K-parameter head on the encoder output predicts, per frame, whether anything is being sung. It gates where the lyric may start and ends each line where singing stops.

Words run until the next word starts; lines end shortly after their last phone rather than at the next line's start.

## Accuracy

On 52 held-out files, 2.8K lines and 17.3K words with word-level references:

| metric | value |
|---|---|
| line start, median | 284 ms |
| line start within 250 ms | 47% |
| line start within 1 s | 69% |
| line start beyond 2 s | 25% |
| word onset, median | 219 ms |
| word onset within 250 ms | 53% |
| line-final word duration, median | 1.02x of reference |

## Failures

- Sustained notes a big weak point. Lines sung below 3.5 phones per second
  have a median error of 1089 ms and 35% beyond 2 s, against 117 ms and 3% for
  dense lines.
- About 16% of line-final words are under half their ground-truth duration.
- Heavy reverb and near-inaudible vocals lose sync.
- A loud passage before the first vocal could still pull the first line early.

## Files

```
qadalign/            package: aligner, CTC, g2p, model loading, CLI
qadalign/aurora/     vendored Aurora model code, unmodified
weights/             model, sung head and metadata in one file
qadalign/data/       bundled CMUdict pronunciations and their licence
tools/               dataset preparation, training, evaluation, dictionary build
web/                 browser runner, ONNX export, notices
docs/REPRODUCE.md    how the weights were produced
```

## AI disclosure

This project is mostly LLM-written as a proof of concept. The LLM was guided by me for research loops and prototyping and all files have been reviewed manually for slop (because I hate it).

## Licence

Apache-2.0. See LICENSE and NOTICE. The encoder is the Aurora ASR model
(Apache-2.0) by AuroraAI-Research; the phone head, sung head and alignment
pipeline are this project's.

Pronunciation data in `qadalign/data/cmudict.txt.gz` is derived from the Carnegie
Mellon Pronouncing Dictionary, BSD-2-Clause, and is bundled so the package needs
no GPL-licensed dependency at runtime. The full notice is in
`qadalign/data/CMUDICT-LICENSE`, `web/LICENSES.md`.
