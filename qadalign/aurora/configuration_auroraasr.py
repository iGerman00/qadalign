"""Configuration for the tiny weight-shared Conformer-CTC ASR model."""
from transformers import PretrainedConfig


class AuroraASRConfig(PretrainedConfig):
    model_type = "aurora_asr"

    def __init__(
        self,
        vocab_size=257,            # 256 BPE pieces + CTC blank (id 0)
        n_mels=80,
        d_model=128,
        num_heads=4,
        ff_dim=384,
        conv_kernel=15,
        num_unique_blocks=4,
        num_repeats=2,             # each unique block is applied this many times
        dropout=0.0,
        blank_token_id=0,
        sampling_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        tokenizer_file="bpe256.model",
        weight_quantization=None,  # None or "int8" (weight-only, per-output-channel)
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_mels = n_mels
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.conv_kernel = conv_kernel
        self.num_unique_blocks = num_unique_blocks
        self.num_repeats = num_repeats
        self.dropout = dropout
        self.blank_token_id = blank_token_id
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.tokenizer_file = tokenizer_file
        self.weight_quantization = weight_quantization
        kwargs.setdefault("pad_token_id", blank_token_id)
        super().__init__(**kwargs)
