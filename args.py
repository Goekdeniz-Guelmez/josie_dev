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
    encoder_hidden_dim: int = 256
    encoder_hidden_layers: int = 4
    encoder_num_heads: int = 16
    encoder_num_kv_heads: Optional[int] = 8
    encoder_head_dim: int = encoder_hidden_dim // encoder_num_heads
    encoder_codebook_size: int = 2048
    encoder_num_quantizers: int = 8
    encoder_rms_norm_eps: float = 1e-5
    encoder_max_batch_size: int = 1
    encoder_max_position_embeddings: int = 256
    encoder_use_scaled_rope: bool = True
    encoder_sample_rate: int = 16000
    enable_denoising: bool = True

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