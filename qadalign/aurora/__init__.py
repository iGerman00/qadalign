"""Vendored Aurora ASR model code (Apache-2.0).

Kept in-tree so loading requires no `trust_remote_code` and no network access.
"""
from .configuration_auroraasr import AuroraASRConfig
from .feature_extraction_auroraasr import AuroraASRFeatureExtractor
from .modeling_auroraasr import AuroraASRForCTC, quantize_model_int8

__all__ = ["AuroraASRConfig", "AuroraASRFeatureExtractor", "AuroraASRForCTC",
           "quantize_model_int8"]
