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
class AudioRQTransformerArgs(BaseModelArgs):
    encoder_hidden_dim: int = 512
    encoder_hidden_depth_layers: int = 6
    encoder_hidden_temporial_layers: int = 6
    encoder_hidden_spectral_layers: int = 6
    encoder_num_heads: int = 16
    encoder_num_kv_heads: Optional[int] = 8
    encoder_head_dim: int = encoder_hidden_dim // encoder_num_heads
    encoder_codebook_size: int = 2048
    encoder_num_quantizers: int = 8
    encoder_rms_norm_eps: float = 1e-5
    encoder_max_batch_size: int = 1
    encoder_max_position_embeddings: int = 32
    encoder_rope_theta: float = 10000.0
    encoder_use_scaled_rope: bool = True
    encoder_sample_rate: int = 16000
    enable_denoising: bool = True