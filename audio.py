from typing import List, Any, Optional, Tuple

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import AudioRQTransformerArgs as ModelArgs


class AudioQuantizer(nn.Module):
    """Quantizer with temporal, depth, and spectral codebooks"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_quantizers = args.encoder_num_quantizers
        self.codebook_size = args.encoder_codebook_size
        self.hidden_dim = args.encoder_hidden_dim

        # Temporal codebooks - for sequence-level patterns
        self.temporal_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, self.hidden_dim // self.num_quantizers)
                for _ in range(self.num_quantizers)
            ]
        )

        # Depth codebooks - for feature-level patterns
        self.depth_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, self.hidden_dim // self.num_quantizers)
                for _ in range(self.num_quantizers)
            ]
        )

        # Spectral codebooks - for frequency-domain patterns
        self.spectral_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, self.hidden_dim // self.num_quantizers)
                for _ in range(self.num_quantizers // 2)
            ]
        )

        # Projections for different feature types
        # self.spectral_proj = nn.Linear(args.hidden_size, args.hidden_size)
        # self.depth_proj = nn.Linear(args.hidden_size, args.hidden_size)

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
        # x = self.depth_proj(x)

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











class DualAudioStreamRQTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
    
    def forward(self, input: torch.Tensor):
        return None