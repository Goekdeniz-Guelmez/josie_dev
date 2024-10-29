from typing import Tuple
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import ModelArgs
from utils import RMSNorm
from transformer import TemporalDepthTransformer

class VideoCNN(nn.Module):
    """CNN backbone for processing video frames"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Assuming input frames are 224x224x3
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, args.encoder_hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, frames, channels, height, width]
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames through CNN
        x = x.view(B * T, C, H, W)
        x = self.conv_layers(x)  # Shape: [B*T, hidden_dim, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # Shape: [B*T, hidden_dim]
        
        # Reshape back to sequence form
        x = x.view(B, T, -1)  # Shape: [B, T, hidden_dim]
        return x

class VideoQuantizer(nn.Module):
    """Quantizer with temporal, depth, and spatial codebooks for video"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_quantizers = args.encoder_num_quantizers
        self.codebook_size = args.encoder_codebook_size
        self.hidden_dim = args.encoder_hidden_dim

        self.input_norm = RMSNorm(self.hidden_dim, eps=self.args.encoder_rms_norm_eps)

        # Temporal codebooks - for sequence-level patterns
        self.temporal_codebooks = nn.ModuleList([
            nn.Embedding(self.codebook_size, (self.hidden_dim // self.num_quantizers))
            for _ in range(self.num_quantizers)
        ])
        self.temporal_output_norm = RMSNorm(self.hidden_dim, eps=self.args.encoder_rms_norm_eps)

        # Depth codebooks - for feature-level patterns
        self.depth_codebooks = nn.ModuleList([
            nn.Embedding(self.codebook_size, (self.hidden_dim // self.num_quantizers))
            for _ in range(self.num_quantizers)
        ])
        self.depth_output_norm = RMSNorm(self.hidden_dim, eps=self.args.encoder_rms_norm_eps)

    def quantize_temporal(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def forward(self, x: torch.Tensor, stream_type: str = 'temporal') -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        
        if stream_type == 'temporal':
            quantized, indices = self.quantize_temporal(x)
            return self.temporal_output_norm(quantized), indices
        elif stream_type == 'depth':
            quantized, indices = self.quantize_depth(x)
            return self.depth_output_norm(quantized), indices
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")

class VideoEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        self.cnn = VideoCNN(args)
        self.quantizer = VideoQuantizer(args)
        self.temporal_transformer = TemporalDepthTransformer(args)
        self.depth_transformer = TemporalDepthTransformer(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, frames, channels, height, width]
        x = self.cnn(x)  # Convert frames to embeddings
        
        temporal_quantized, _ = self.quantizer(x)
        depth_quantized, _ = self.quantizer(x, 'depth')

        temporal_features, _ = self.temporal_transformer(temporal_quantized)
        depth_features, _ = self.depth_transformer(depth_quantized)

        discrete_video_tokens = temporal_features + depth_features
        return discrete_video_tokens