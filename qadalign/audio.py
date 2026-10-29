"""Audio loading: anything libsndfile or torchaudio can decode -> 16 kHz mono."""
from __future__ import annotations

import numpy as np
import torch

SR = 16000


def load_audio(path, sr: int = SR) -> np.ndarray:
    """Mono float32 at `sr`, regardless of the container or source rate."""
    wav, got = _read(path)
    if got != sr:
        wav = torch.from_numpy(wav)
        wav = torchaudio_resample(wav, got, sr)
        wav = wav.numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def _read(path):
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(0).numpy()
        return wav.astype(np.float32), int(sr)
    except Exception:
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(1)
        return wav, int(sr)


def torchaudio_resample(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    import torchaudio
    return torchaudio.functional.resample(wav, sr_in, sr_out)


def duration(path) -> float:
    import soundfile as sf
    return sf.info(str(path)).frames / sf.info(str(path)).samplerate
