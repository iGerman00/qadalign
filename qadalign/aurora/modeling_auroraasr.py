"""Tiny weight-shared Conformer-CTC ASR model for Hugging Face `transformers`.

Layer names are identical to `ASRModel` in train_asr.py, so a training checkpoint's state dict
loads without any key remapping. Use with `trust_remote_code=True`.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_auroraasr import AuroraASRConfig
from .feature_extraction_auroraasr import AuroraASRFeatureExtractor


# ----------------------------------------------------------------------------- INT8 support
class Int8Linear(nn.Module):
    """Weight-only INT8 linear layer (symmetric, per-output-channel scales).

    Weights are stored as int8 and dequantised on the fly, so it runs on any device/dtype and
    the checkpoint is ~4x smaller than FP32.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, lin: nn.Linear):
        q = cls(lin.in_features, lin.out_features, bias=lin.bias is not None)
        w = lin.weight.detach().float()
        scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        q.qweight.copy_(torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8))
        q.scale.copy_(scale)
        if lin.bias is not None:
            q.bias.data.copy_(lin.bias.detach())
        return q

    def forward(self, x):
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype).unsqueeze(1)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


def _swap_linears(module, quantize_weights):
    """Replace every nn.Linear by Int8Linear (empty, or quantised from the existing weights)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new = (Int8Linear.from_linear(child) if quantize_weights
                   else Int8Linear(child.in_features, child.out_features, child.bias is not None))
            setattr(module, name, new)
        else:
            _swap_linears(child, quantize_weights)


def quantize_model_int8(model):
    """In-place: convert all Linear layers to INT8 and mark the config so it reloads correctly."""
    _swap_linears(model, quantize_weights=True)
    model.config.weight_quantization = "int8"
    return model


# ----------------------------------------------------------------------------- building blocks
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cache(T, hd, device):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    emb = torch.cat([f, f], dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]


class MHSA(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, rope, attn_mask):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = rope[0].to(q.dtype), rope[1].to(q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class SharedBlock(nn.Module):
    """One set of weights applied `repeats` times, with separate LayerNorms per repeat."""

    def __init__(self, d, heads, ff, k, repeats, drop=0.0):
        super().__init__()
        self.attn = MHSA(d, heads)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.pw2 = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)
        self.norms = nn.ModuleList(
            [nn.ModuleList([nn.LayerNorm(d) for _ in range(4)]) for _ in range(repeats)])
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask, attn_mask, rope):
        for n_att, n_conv, n_cin, n_ff in self.norms:
            x = x + self.drop(self.attn(n_att(x), rope, attn_mask))
            y = F.glu(self.pw1(n_conv(x)), dim=-1)
            y = y.masked_fill(~mask[..., None], 0.0)
            y = self.dw(y.transpose(1, 2)).transpose(1, 2)
            y = self.pw2(F.silu(n_cin(y)))
            x = x + self.drop(y)
            x = x + self.drop(self.fc2(F.silu(self.fc1(n_ff(x)))))
        return x


# ----------------------------------------------------------------------------- HF model
@dataclass
class AuroraASRCTCOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    output_lengths: Optional[torch.LongTensor] = None


class AuroraASRPreTrainedModel(PreTrainedModel):
    config_class = AuroraASRConfig
    main_input_name = "input_features"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class AuroraASRForCTC(AuroraASRPreTrainedModel):
    def __init__(self, config: AuroraASRConfig):
        super().__init__(config)
        d = config.d_model
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1)
        self.conv_dw = nn.Conv2d(16, 16, 3, 2, 1, groups=16)
        self.conv_pw = nn.Conv2d(16, 16, 1)
        self.stem_proj = nn.Linear(16 * (config.n_mels // 4), d)
        self.blocks = nn.ModuleList([
            SharedBlock(d, config.num_heads, config.ff_dim, config.conv_kernel,
                        config.num_repeats, config.dropout)
            for _ in range(config.num_unique_blocks)])
        self.final_ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.vocab_size)
        self.head_dim = d // config.num_heads
        if config.weight_quantization == "int8":
            _swap_linears(self, quantize_weights=False)  # empty INT8 layers; weights come from the checkpoint
        self.post_init()

    @staticmethod
    def _out_lengths(lens):
        l = (lens - 1) // 2 + 1
        return (l - 1) // 2 + 1

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, AuroraASRCTCOutput]:
        """input_features: (B, T, 80) log-mel from AuroraASRFeatureExtractor.

        labels (optional): (B, L) token ids in model vocabulary (0 = blank, sentencepiece id + 1),
        padded with -100. Returns CTC loss when given.
        """
        B, T0, _ = input_features.shape
        if input_lengths is None:
            if attention_mask is not None:
                input_lengths = attention_mask.sum(-1).long()
            else:
                input_lengths = torch.full((B,), T0, dtype=torch.long, device=input_features.device)
        valid = torch.arange(T0, device=input_features.device)[None, :] < input_lengths[:, None]
        x = input_features.to(self.conv1.weight.dtype) * valid[..., None]

        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = self.conv_dw(x)
        x = F.relu(self.conv_pw(x))
        B, C, T, Fq = x.shape
        x = self.stem_proj(x.permute(0, 2, 1, 3).reshape(B, T, C * Fq))
        olens = self._out_lengths(input_lengths).clamp(max=T)
        mask = torch.arange(T, device=x.device)[None, :] < olens[:, None]
        attn_mask = mask[:, None, None, :]
        rope = rope_cache(T, self.head_dim, x.device)
        for blk in self.blocks:
            x = blk(x, mask, attn_mask, rope)
        logits = self.head(self.final_ln(x))

        loss = None
        if labels is not None:
            lab_mask = labels >= 0
            tgt_len = lab_mask.sum(-1)
            flat = labels.masked_select(lab_mask)
            lp = F.log_softmax(logits.float(), -1).transpose(0, 1)
            loss = F.ctc_loss(lp, flat, olens, tgt_len, blank=self.config.blank_token_id,
                              reduction="mean", zero_infinity=True)
        if not return_dict:
            return tuple(v for v in (loss, logits, olens) if v is not None)
        return AuroraASRCTCOutput(loss=loss, logits=logits, output_lengths=olens)

    # ------------------------------------------------------------------ decoding helpers
    def _find_spm(self):
        name = self.config.tokenizer_file
        path = getattr(self.config, "_name_or_path", "") or ""
        if os.path.isdir(path):
            for cand in (os.path.join(path, name), os.path.join(os.path.dirname(os.path.abspath(path)), name)):
                if os.path.isfile(cand):
                    return cand
        from huggingface_hub import hf_hub_download
        return hf_hub_download(path, name)

    def _get_sp(self, spm_path=None):
        if getattr(self, "_sp", None) is None:
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor(model_file=spm_path or self._find_spm())
        return self._sp

    def greedy_decode(self, logits, output_lengths=None, spm_path=None) -> List[str]:
        """Greedy CTC decoding: argmax, collapse repeats, drop blanks, detokenise."""
        sp = self._get_sp(spm_path)
        pred = logits.argmax(-1).cpu().numpy()
        texts = []
        for i in range(pred.shape[0]):
            p = pred[i] if output_lengths is None else pred[i, : int(output_lengths[i])]
            keep = [True] + [bool(a != b) for a, b in zip(p[1:], p[:-1])] if len(p) else []
            ids = [int(t) - 1 for t, k in zip(p, keep) if k and t != self.config.blank_token_id]
            texts.append(sp.decode(ids))
        return texts

    @torch.no_grad()
    def transcribe(self, audio, sampling_rate=16000, spm_path=None) -> List[str]:
        """Transcribe 16 kHz mono float waveform(s) (numpy / torch / list of those)."""
        fe = AuroraASRFeatureExtractor(
            feature_size=self.config.n_mels, sampling_rate=self.config.sampling_rate,
            n_fft=self.config.n_fft, win_length=self.config.win_length,
            hop_length=self.config.hop_length)
        feats = fe(audio, sampling_rate=sampling_rate, return_tensors="pt")
        dev = next(self.parameters()).device
        out = self(feats["input_features"].to(dev), attention_mask=feats["attention_mask"].to(dev))
        return self.greedy_decode(out.logits.float(), out.output_lengths, spm_path)
