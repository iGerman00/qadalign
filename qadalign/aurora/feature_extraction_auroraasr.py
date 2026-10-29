"""Log-mel feature extractor that reproduces the training-time features exactly."""
from functools import lru_cache

import numpy as np
import torch
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature


@lru_cache(maxsize=4)
def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)), np.float32)
    for i in range(n_mels):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - lo) / (ce - lo)
        down = (hi - freqs) / (hi - ce)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
        if fb[i].sum() == 0:  # filter narrower than one FFT bin: snap to nearest bin
            fb[i, int(np.argmin(np.abs(freqs - ce)))] = 1.0
    return fb


class AuroraASRFeatureExtractor(SequenceFeatureExtractor):
    """80-dim log-mel, 25 ms window / 10 ms hop, per-utterance mean/variance normalisation."""

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        padding_value=0.0,
        n_fft=512,
        win_length=400,
        hop_length=160,
        fmin=20.0,
        fmax=8000.0,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate,
                         padding_value=padding_value, **kwargs)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def _logmel(self, wav):
        w = torch.from_numpy(wav)
        if w.numel() < self.n_fft // 2 + 1:
            raise ValueError(f"audio too short ({w.numel()} samples); need at least {self.n_fft // 2 + 1}")
        fb = torch.from_numpy(_mel_filterbank(self.sampling_rate, self.n_fft, self.feature_size,
                                              float(self.fmin), float(self.fmax)))
        window = torch.hann_window(self.win_length)
        spec = torch.stft(w, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=window, center=True, pad_mode="reflect", return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2
        mel = torch.log(fb @ power + 1e-10).T
        mel = (mel - mel.mean(0, keepdim=True)) / (mel.std(0, keepdim=True) + 1e-5)
        return mel.numpy()

    def __call__(self, raw_speech, sampling_rate=None, return_tensors=None, **kwargs):
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Model expects {self.sampling_rate} Hz audio but got {sampling_rate} Hz. "
                "Please resample first.")
        if isinstance(raw_speech, torch.Tensor):
            raw_speech = raw_speech.detach().cpu().numpy()
        batched = (isinstance(raw_speech, (list, tuple)) and len(raw_speech) > 0
                   and isinstance(raw_speech[0], (np.ndarray, list, tuple, torch.Tensor)))
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 2:
            raw_speech, batched = list(raw_speech), True
        if not batched:
            raw_speech = [raw_speech]
        wavs = []
        for w in raw_speech:
            if isinstance(w, torch.Tensor):
                w = w.detach().cpu().numpy()
            w = np.asarray(w, dtype=np.float32)
            if w.ndim != 1:
                raise ValueError("each waveform must be 1-D (mono)")
            wavs.append(w)
        feats = [self._logmel(w) for w in wavs]
        T = max(f.shape[0] for f in feats)
        out = np.full((len(feats), T, self.feature_size), self.padding_value, dtype=np.float32)
        mask = np.zeros((len(feats), T), dtype=np.int64)
        for i, f in enumerate(feats):
            out[i, :f.shape[0]] = f
            mask[i, :f.shape[0]] = 1
        return BatchFeature({"input_features": out, "attention_mask": mask}, tensor_type=return_tensors)
