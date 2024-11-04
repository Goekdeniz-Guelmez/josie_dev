from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from JOSIEv4o.utils import RMSNorm

from JOSIEv4o.quantizer import Quantizer
from JOSIEv4o.transformer import Transformer

from JOSIEv4o.args import ModelArgs


class JODIOEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.samples_needed = int(self.args.inference_args.rate * self.args.inference_args.record_seconds)
        
        # Encoder projection
        self.input_projection = nn.Linear(
            self.samples_needed,
            self.args.audio_encoder_args.hidden_size,
            bias=False
        )
        
        # Components
        self.quantizer = Quantizer(self.args)
        self.temporal_transformer = Transformer(self.args)
        self.depth_transformer = Transformer(self.args)
    
    def forward(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        B, T, D = audio_tensor.shape
        projected = self.input_projection(audio_tensor)
        
        quantized_temporal, discrete_temporal_tokens = self.quantizer(
            projected, stream_type='temporal'
        )
        quantized_depth, discrete_depth_tokens = self.quantizer(
            projected, stream_type='depth'
        )
        
        temporal_output, _ = self.temporal_transformer(discrete_temporal_tokens)
        depth_output, _ = self.depth_transformer(discrete_depth_tokens)
        
        discrete_audio_tokens = temporal_output + depth_output
        return discrete_audio_tokens


class JODIODecoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.temporal_transformer = Transformer(self.args)
        self.depth_transformer = Transformer(self.args, is_decoder=True)
    
    def forward(self, audio_token: torch.Tensor) -> torch.Tensor:
        temporal_output, _ = self.temporal_transformer(audio_token)
        depth_output, _ = self.depth_transformer(audio_token)
        
        discrete_audio_tokens = temporal_output + depth_output
        return discrete_audio_tokens


class JODIO(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.jodio_encoder = JODIOEncoder(self.args)
        self.jodio_decoder = JODIODecoder(self.args)

    def encode(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        return self.jodio_encoder(audio_tensor)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.jodio_decoder(x)
    
    def forward(self, x: torch.Tensor, style: str = 'encode') -> torch.Tensor:
        if style == 'encode':
            return self.encode(x)
        elif style == 'decode':
            return self.decode(x)
        else:
            raise ValueError(f"Invalid style '{style}'. Must be 'encode' or 'decode'")