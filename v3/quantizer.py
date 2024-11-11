import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int
    ):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)

    def forward(self, x):
        """Encode vectors to tokens"""
        distances = torch.cdist(x, self.codebook.weight)
        indices = distances.argmin(dim=-1) # semantic_tokens
        quantized = self.codebook(indices) # semantic_vectors
        return indices, quantized

    def decode(self, tokens):
        """Convert tokens back to vectors using codebook"""
        # tokens shape: [B, T]
        vectors = self.codebook(tokens) # Look up vectors from codebook
        return vectors


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        num_quantizers: int
    ):
        super().__init__()
        self.quantizers = nn.ModuleList([
            VectorQuantizer(dim, codebook_size)
            for _ in range(num_quantizers)
        ])

    def forward(self, x):
        """Encode vectors to tokens through multiple quantizers"""
        quantized = torch.zeros_like(x)
        indices = [] # Will contain acoustic_tokens
        residual = x

        for quantizer in self.quantizers:
            idx, quant = quantizer(residual)
            indices.append(idx)
            quantized = quantized + quant
            residual = residual - quant

        return indices, quantized

    def decode(self, tokens):
        """Convert multi-level tokens back to vectors"""
        # tokens is a list of [B, T] tensors, one per quantizer
        quantized = torch.zeros_like(self.quantizers[0].codebook(tokens[0]))

        for quantizer, level_tokens in zip(self.quantizers, tokens):
            quant = quantizer.decode(level_tokens)
            quantized = quantized + quant

        return quantized
