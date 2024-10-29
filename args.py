from typing import List, Any, Optional
from dataclasses import dataclass
import inspect


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelArgs(BaseModelArgs):
    encoder_audio_hidden_dim: int = 256
    encoder_audio_hidden_layers: int = 4
    encoder_audio_num_heads: int = 16
    encoder_audio_num_kv_heads: Optional[int] = 8
    encoder_audio_head_dim: int = encoder_audio_hidden_dim // encoder_audio_num_heads
    encoder_audio_codebook_size: int = 2048
    encoder_audio_num_quantizers: int = 8
    encoder_audio_rms_norm_eps: float = 1e-5
    encoder_audio_max_batch_size: int = 1
    encoder_audio_mlp_dropout: float = 0.1
    encoder_audio_attention_dropout: float = 0.1
    encoder_audio_max_position_embeddings: int = 32
    encoder_audio_sample_rate: int = 16000

    encoder_audio_rate: int = 16000
    encoder_audio_channels: int = 1
    encoder_audio_chunk: int = encoder_audio_rate // 4 # 250 ms for 16 kHz audio

    decoder_audio_hidden_dim: int = 256
    decoder_audio_hidden_layers: int = 4
    decoder_audio_num_heads: int = 16
    decoder_audio_num_kv_heads: Optional[int] = 8
    decoder_audio_head_dim: int = decoder_audio_hidden_dim // decoder_audio_num_heads
    decoder_audio_codebook_size: int = 2048
    decoder_audio_num_quantizers: int = 8
    decoder_audio_rms_norm_eps: float = 1e-5
    decoder_audio_max_batch_size: int = 1
    decoder_audio_mlp_dropout: float = 0.1
    decoder_audio_attention_dropout: float = 0.1
    decoder_audio_max_position_embeddings: int = 32
    decoder_audio_sample_rate: int = 16000

    encoder_vision_hidden_dim: int = 512
    encoder_vision_hidden_layers: int = 8
    encoder_vision_num_heads: int = 16
    encoder_vision_num_kv_heads: Optional[int] = 8
    encoder_vision_head_dim: int = encoder_vision_hidden_dim // encoder_vision_num_heads
    encoder_vision_codebook_size: int = 2048
    encoder_vision_num_quantizers: int = 8
    encoder_vision_rms_norm_eps: float = 1e-5
    encoder_vision_max_batch_size: int = 1
    encoder_vision_max_position_embeddings: int = 256
    encoder_vision_rope_theta: float = 500000

    reasoner_hidden_dim: int = 512
    reasoner_hidden_layers: int = 4
    reasoner_num_heads: int = 16
    reasoner_num_kv_heads: Optional[int] = 8
    reasoner_head_dim: int = reasoner_hidden_dim // reasoner_num_heads
    reasoner_rms_norm_eps: float = 1e-5
    reasoner_max_batch_size: int = 1
    reasoner_max_position_embeddings: int = 128
    reasoner_rope_theta: float = 500000
    reasoner_vocab_size: int = 131000
    reasoner_multiple_of: int = 256
    reasoner_ffn_dim_multiplier: Optional[float] = None