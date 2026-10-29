"""Model loading: one file, no network, no `trust_remote_code`.

The checkpoint holds the CTC model, the sung head, and the metadata needed to
rebuild the architecture. Everything comes from the vendored Aurora code.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .aurora import (AuroraASRConfig, AuroraASRFeatureExtractor, AuroraASRForCTC,
                     quantize_model_int8)
from .phones import NUM_PHONES
from .sung import SungHead

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "weights" / "qadalign-0.1.pt"


def feature_extractor() -> AuroraASRFeatureExtractor:
    return AuroraASRFeatureExtractor()


def build_ctc(vocab: int = NUM_PHONES) -> AuroraASRForCTC:
    cfg = AuroraASRConfig(vocab_size=vocab, blank_token_id=0,
                          weight_quantization=None)
    return AuroraASRForCTC(cfg)


class Aligner(torch.nn.Module):
    """CTC aligner plus the sung head, sharing one encoder pass."""

    def __init__(self, ctc: AuroraASRForCTC, sung: SungHead):
        super().__init__()
        self.ctc = ctc
        self.sung = sung
        self._hidden = None
        ctc.final_ln.register_forward_hook(self._capture)

    def _capture(self, _mod, _inp, out):
        self._hidden = out

    def forward(self, input_features, attention_mask=None):
        logits = self.ctc(input_features, attention_mask=attention_mask).logits
        sung = torch.sigmoid(self.sung(self._hidden[0].float()))
        return logits, sung


def load(weights=None, device="cpu", int8: bool = False) -> Aligner:
    """Load a checkpoint. `int8=True` shrinks weights ~4x at a small accuracy cost."""
    path = Path(weights) if weights else DEFAULT_WEIGHTS
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    meta = blob.get("meta", {})
    ctc = build_ctc(meta.get("vocab", NUM_PHONES))
    ctc.load_state_dict(blob["model"])
    sung = SungHead(meta.get("d_model", 128))
    sung.load_state_dict(blob["sung"])
    if int8:
        quantize_model_int8(ctc)
    model = Aligner(ctc, sung).eval().to(device)
    return model
