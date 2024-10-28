from typing import Tuple

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import AudioRQTransformerArgs as ModelArgs
from utils import RMSNorm


class AudioQuantizer(nn.Module):
    """Quantizer with temporal, depth, and spectral codebooks"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_quantizers = args.encoder_num_quantizers
        self.codebook_size = args.encoder_codebook_size
        self.hidden_dim = args.encoder_hidden_dim

        self.input_norm = RMSNorm(self.hidden_dim, eps=self.args.encoder_rms_norm_eps)

        # Temporal codebooks - for sequence-level patterns
        self.temporal_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, (self.hidden_dim // self.num_quantizers))
                for _ in range(self.num_quantizers)
            ]
        )
        self.temporal_output_norm = RMSNorm(self.hidden_dim, eps=self.args.encoder_rms_norm_eps)

        # Depth codebooks - for feature-level patterns
        self.depth_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, (self.hidden_dim // self.num_quantizers))
                for _ in range(self.num_quantizers)
            ]
        )
        self.depth_output_norm = RMSNorm(self.hidden_dim, eps=self.args.encoder_rms_norm_eps)

    def quantize_temporial(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        x = rearrange(x, 'b t (q d) -> b t q d', q=self.num_quantizers)

        indices = []
        quantized = []

        for i, codebook in enumerate(self.temporal_codebooks):
            distances = torch.cdist(x[..., i, :], codebook.weight)
            idx = distances.argmin(dim=-1)
            indices.append(idx)
            quantized.append(codebook(idx))

        indices = torch.stack(indices, dim=-1)
        quantized = torch.cat(quantized, dim=-1)

        return quantized, indices
    
    def quantize_depth(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        x = rearrange(x, 'b t (q d) -> b t q d', q=self.num_quantizers)
        
        indices = []
        quantized = []
        
        for i, codebook in enumerate(self.depth_codebooks):
            distances = torch.cdist(x[..., i, :], codebook.weight)
            idx = distances.argmin(dim=-1)
            indices.append(idx)
            quantized.append(codebook(idx))
        
        indices = torch.stack(indices, dim=-1)
        quantized = torch.cat(quantized, dim=-1)
        
        return quantized, indices
    
    def forward(self, x: torch.Tensor, stream_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input based on stream type
        stream_type: One of 'temporal', 'depth', or 'spectral'
        """
        x = self.input_norm(x)

        if stream_type == 'temporal':
            quantized, indices = self.quantize_temporial(x)
            return self.temporal_output_norm(quantized), indices
        
        elif stream_type == 'depth':
            quantized, indices = self.quantize_depth(x)
            return self.depth_output_norm(quantized), indices

        else:
            raise ValueError(f"Unknown stream type: {stream_type}")


