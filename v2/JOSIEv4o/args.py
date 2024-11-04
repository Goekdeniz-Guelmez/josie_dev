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
class StreamingArgs(BaseModelArgs):
    chunk_size: int = 512
    context_size: int = 2048
    overlap_size: int = 128
    stream_types: tuple = ('temporal', 'depth')


@dataclass
class AudioEncoderModelArgs(BaseModelArgs):
    hidden_size: int = 256
    hidden_layers: int = 22
    num_heads: int = 16
    head_dim: int = hidden_size // num_heads

    codebook_size: int = 2048
    num_quantizers: int = 8
    downsampling_ratio: int = 8

    rms_norm_eps: float = 1e-5
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 64

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_heads


@dataclass
class AudioDecoderModelArgs(BaseModelArgs):
    hidden_size: int = 256
    hidden_layers: int = 4
    num_heads: int = 16
    num_kv_heads: Optional[int] = 8
    head_dim: int = field(init=False)
    codebook_size: int = 2048
    num_quantizers: int = 8
    rms_norm_eps: float = 1e-5
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 64
    sample_rate: int = 16000

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_heads


@dataclass
class VisionEncoderModelArgs(BaseModelArgs):
    hidden_size: int = 512
    hidden_layers: int = 8
    num_heads: int = 16
    num_kv_heads: Optional[int] = 8
    head_dim: int = field(init=False)
    codebook_size: int = 2048
    num_quantizers: int = 8
    rms_norm_eps: float = 1e-5
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 256
    rope_theta: float = 500000
    max_frames: int = 12

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_heads


@dataclass
class ModelArgs(BaseModelArgs):
    audio_encoder_args: Type[AudioEncoderModelArgs] = AudioEncoderModelArgs
    audio_decoder_args: Type[AudioDecoderModelArgs] = AudioDecoderModelArgs
    vision_encoder_args: Type[VisionEncoderModelArgs] = VisionEncoderModelArgs

    inference_args: Type[InferenceArgs] = InferenceArgs
    streaming_args: Type[StreamingArgs] = StreamingArgs

    reasoner_architecture: str = 'LlamaForCausalLM'
    reasoner_hidden_size: int = 896
    reasoner_hidden_layers: int = 24
    reasoner_num_heads: int = 14
    reasoner_num_kv_heads: Optional[int] = 2
    reasoner_head_dim: int = field(init=False)
    reasoner_rms_norm_eps: float = 1e-06
    reasoner_attention_dropout: float = 0.0
    reasoner_max_position_embeddings: int = 1028
    reasoner_rope_theta: float = 1000000.0
    reasoner_vocab_size: int = 128256
    reasoner_multiple_of: int = 256
    reasoner_ffn_dim_multiplier: Optional[float] = None

    tokenizer_path: Path = field(default=Path('/Users/gokdenizgulmez/Desktop/J.O.S.I.E./tokenizer.model'))
    batch_size: int = 2

    def __post_init__(self):
        self.reasoner_head_dim = self.reasoner_hidden_size // self.reasoner_num_heads