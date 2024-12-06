from dataclasses import dataclass, field
from typing import Optional, Type
from pathlib import Path

import pyaudio
import inspect


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params: dict):
        valid_params = {
            k: v
            for k, v in params.items()
            if k in inspect.signature(cls).parameters
        }
        return cls(**valid_params)
    

@dataclass
class InferenceArgs(BaseModelArgs):
    format = pyaudio.paFloat32
    channels = 1
    rate = 16000  # 16kHz
    record_seconds = 0.25  # 250ms
    chunk = rate // 4 # 4096


@dataclass
class AudioEncoderModelArgs(BaseModelArgs):
    hidden_size: int = 1028
    hidden_layers: int = 12
    num_heads: int = 16
    head_dim: int = hidden_size // num_heads

    channels: int = 512
    kernel_size: int = 3
    num_conv_layers: int = 3
    dilation_growth: int = 2

    codebook_size: int = 2048
    num_acoustic_quantizers: int = 8
    num_semantic_quantizers: int = 1
    downsampling_ratio: int = 8

    rms_norm_eps: float = 1e-5
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 64


@dataclass
class AudioDecoderModelArgs(BaseModelArgs):
    hidden_size: int = 512
    hidden_layers: int = 12
    num_heads: int = 12
    head_dim: int = hidden_size // num_heads

    channels: int = 512
    kernel_size: int = 3
    num_conv_layers: int = 3
    dilation_growth: int = 2

    rms_norm_eps: float = 1e-5
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 64


@dataclass
class VisionEncoderModelArgs(BaseModelArgs):
    hidden_size: int = 512
    hidden_layers: int = 8
    num_heads: int = 16
    num_kv_heads: Optional[int] = 8
    head_dim: int = hidden_size // num_heads
    codebook_size: int = 2048
    num_quantizers: int = 128
    rms_norm_eps: float = 1e-5
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 256
    rope_theta: float = 500000
    max_frames: int = 12


@dataclass
class TemporialTransformer(BaseModelArgs):
    hidden_size: int = 1028
    hidden_layers: int = 12
    num_heads: int = 12
    head_dim: int = hidden_size // num_heads
    attention_dropout: float = 0.0

    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 4096


@dataclass
class DepthTransformer(BaseModelArgs):
    hidden_size: int = 512
    hidden_layers: int = 6
    num_heads: int = 4
    head_dim: int = hidden_size // num_heads
    attention_dropout: float = 0.0

    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 4096


@dataclass
class ModelArgs(BaseModelArgs):
    audio_encoder_args: Type[AudioEncoderModelArgs] = AudioEncoderModelArgs
    audio_decoder_args: Type[AudioDecoderModelArgs] = AudioDecoderModelArgs
    vision_encoder_args: Type[VisionEncoderModelArgs] = VisionEncoderModelArgs

    temporal_transformer_args: Type[TemporialTransformer] = TemporialTransformer
    depth_transformer_args: Type[DepthTransformer] = DepthTransformer

    inference_args: Type[InferenceArgs] = InferenceArgs

    stfu_token_id: int = 0

    vocab_size: int = 128256

    tokenizer_path: Path = field(default=Path('/Users/gokdenizgulmez/Desktop/J.O.S.I.E./tokenizer.model'))
