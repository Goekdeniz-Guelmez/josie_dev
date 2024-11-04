from typing import Tuple

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from JOSIEv4o.args import ModelArgs
from JOSIEv4o.utils import RMSNorm

class Quantizer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if hasattr(args, 'audio_encoder_args'):
            self.args = args.audio_encoder_args
        elif hasattr(args, 'vision_encoder_args'):
            self.args = args.vision_encoder_args
        else:
            self.args = args
            
        self.num_quantizers = self.args.num_quantizers
        self.codebook_size = self.args.codebook_size
        self.hidden_size = self.args.hidden_size
        self.chunk_size = self.hidden_size // self.num_quantizers
        self.rms_eps = self.args.rms_norm_eps
        
        self.temporal_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, (self.hidden_size // self.num_quantizers))
                for _ in range(self.num_quantizers)
            ]
        )
        self.temporal_output_norm = RMSNorm(self.hidden_size, eps=self.rms_eps)

        self.depth_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, (self.hidden_size // self.num_quantizers))
                for _ in range(self.num_quantizers)
            ]
        )
        self.depth_output_norm = RMSNorm(self.hidden_size, eps=self.rms_eps)

    def _clean_input(self, x: torch.Tensor):
        if len(x.shape) == 3:
            b, t, qd = x.shape
            d = qd // self.num_quantizers
            return x.view(b, t, self.num_quantizers, d)
        else:
            t, qd = x.shape
            d = qd // self.num_quantizers
            return x.view(t, self.num_quantizers, d)

    def quantize_temporial(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._clean_input(x)

        descrete_tokens = []
        quantized = []

        for i, codebook in enumerate(self.temporal_codebooks):
            distances = torch.cdist(x[..., i, :], codebook.weight)
            idx = distances.argmin(dim=-1)
            descrete_tokens.append(idx)
            quantized.append(codebook(idx))

        descrete_tokens = torch.stack(descrete_tokens, dim=-1)
        quantized = torch.cat(quantized, dim=-1)

        return quantized, descrete_tokens
    
    def quantize_depth(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._clean_input(x)
        
        descrete_tokens = []
        quantized = []

        for i, codebook in enumerate(self.depth_codebooks):
            distances = torch.cdist(x[..., i, :], codebook.weight)
            idx = distances.argmin(dim=-1)
            descrete_tokens.append(idx)
            quantized.append(codebook(idx))

        descrete_tokens = torch.stack(descrete_tokens, dim=-1)
        quantized = torch.cat(quantized, dim=-1)

        return quantized, descrete_tokens
    
    def forward(self, x: torch.Tensor, stream_type: str = 'temporal') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input based on stream type
        stream_type: One of 'temporal', 'depth', or 'spectral'
        """
        if stream_type == 'temporal':
            quantized, descrete_tokens = self.quantize_temporial(x)
            return self.temporal_output_norm(quantized), descrete_tokens
        
        elif stream_type == 'depth':
            quantized, descrete_tokens = self.quantize_depth(x)
            return self.depth_output_norm(quantized), descrete_tokens

        else:
            raise ValueError(f"Unknown stream type: {stream_type}")