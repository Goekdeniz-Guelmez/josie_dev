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
    hidden_dim: int = 512
    encoder_hidden_depth_layers: int = 12
    encoder_hidden_temporial_layers: int = 12
    encoder_num_heads: int = 16
    encoder_num_kv_heads: Optional[int] = 8
    encoder_head_dim: int = hidden_dim // encoder_num_heads
    codebook_size: int = 1024
    encoder_num_quantizers: int = 8