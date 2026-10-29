"""Fetch the Aurora ASR base checkpoint used as the encoder.

    python tools/fetch_base.py --out aurora

Downloads the public Aurora CTC model (Apache-2.0) and leaves a directory that
tools/train_aligner.py --base accepts. Only the encoder is reused: its BPE text
head is replaced by a 39-phone head in stage one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = "AuroraAI-Research/AuroraASR-1M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="aurora")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(a.repo, local_dir=str(out),
                                 allow_patterns=["*.json", "*.safetensors",
                                                 "*.py", "*.model", "*.txt"])
        print(f"downloaded {a.repo} -> {path}")
    except ImportError:
        from transformers import AutoModelForCTC
        model = AutoModelForCTC.from_pretrained(a.repo, trust_remote_code=True)
        model.save_pretrained(str(out))
        print(f"saved {a.repo} via transformers -> {out}")

    need = ["config.json"]
    missing = [n for n in need if not (out / n).exists()]
    if missing:
        raise SystemExit(f"incomplete download, missing: {missing}")
    print(f"ready: {out}")


if __name__ == "__main__":
    main()
