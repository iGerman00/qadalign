"""Export qadalign weights to ONNX for the browser runner.

    python web/export_onnx.py

Writes web/qadalign.onnx with two outputs: phone logits and per-frame sung
probability. Both come from one pass over the shared encoder.

Weights are stored as int8 plus a per-row scale, dequantised inside the graph.
Constant folding is disabled so the exporter keeps them int8 instead of expanding
them back to float32, which takes the file from 3.86 MB to 1.23 MB.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qadalign.aurora import quantize_model_int8
from qadalign.aurora.modeling_auroraasr import rope_cache
from qadalign.model import load

OUT = Path(__file__).resolve().parent / "qadalign.onnx"


class Export(nn.Module):
    def __init__(self, aligner):
        super().__init__()
        self.ctc = aligner.ctc
        self.sung = aligner.sung

    def forward(self, feats, mask):
        c = self.ctc
        b, t0, _ = feats.shape
        lens = mask.sum(-1).long()
        valid = torch.arange(t0, device=feats.device)[None, :] < lens[:, None]
        x = feats * valid[..., None].to(feats.dtype)
        x = F.relu(c.conv1(x.unsqueeze(1)))
        x = c.conv_dw(x)
        x = F.relu(c.conv_pw(x))
        b, ch, t, fq = x.shape
        x = c.stem_proj(x.permute(0, 2, 1, 3).reshape(b, t, ch * fq))
        olens = c._out_lengths(lens).clamp(max=t)
        mask2 = torch.arange(t, device=x.device)[None, :] < olens[:, None]
        attn = mask2[:, None, None, :]
        rope = rope_cache(t, c.head_dim, x.device)
        for blk in c.blocks:
            x = blk(x, mask2, attn, rope)
        h = c.final_ln(x)
        return c.head(h), torch.sigmoid(self.sung(h))


def main():
    aligner = load().eval()
    quantize_model_int8(aligner.ctc)
    model = Export(aligner).eval()
    feats = torch.zeros(1, 300, 80)
    mask = torch.ones(1, 300, dtype=torch.long)
    torch.onnx.export(
        model, (feats, mask), str(OUT),
        input_names=["features", "mask"],
        output_names=["logits", "sung"],
        dynamic_axes={"features": {1: "time"}, "mask": {1: "time"},
                      "logits": {1: "frames"}, "sung": {1: "frames"}},
        opset_version=17, dynamo=False, do_constant_folding=False,
    )
    print(f"{OUT} {OUT.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
