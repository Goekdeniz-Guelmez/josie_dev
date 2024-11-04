from typing import Optional
from collections import deque

import torch
import torch.nn as nn

from JOSIEv4o.quantizer import Quantizer
from JOSIEv4o.JODIOv1 import JODIODecoder
from JOSIEv4o.streaming_transformer import StreamingTransformer

from JOSIEv4o.args import ModelArgs

class StreamingJODIOEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.samples_needed = int(self.args.inference_args.rate * self.args.inference_args.record_seconds)
        
        # Streaming configuration
        self.streaming_config = self.args.streaming_args
        
        # Encoder projection
        self.input_projection = nn.Linear(
            self.samples_needed,
            self.args.audio_encoder_args.hidden_size,
            bias=False
        )
        
        # Components
        self.quantizer = Quantizer(self.args)
        self.temporal_transformer = StreamingTransformer(
            args, 
            streaming_config=self.streaming_config
        )
        self.depth_transformer = StreamingTransformer(
            args, 
            streaming_config=self.streaming_config
        )
        
        # Streaming state
        self.streaming_state = {
            'temporal': {'buffer': deque(maxlen=self.streaming_config.context_size)},
            'depth': {'buffer': deque(maxlen=self.streaming_config.context_size)}
        }
    
    def init_streaming(self):
        """Initialize streaming state for both transformers"""
        self.temporal_transformer.init_streaming()
        self.depth_transformer.init_streaming()
        
        for stream_type in self.streaming_config.stream_types:
            self.streaming_state[stream_type] = {
                'buffer': deque(maxlen=self.streaming_config.context_size),
                'position': 0
            }
    
    def forward(
            self, 
            audio_tensor: torch.Tensor,
            streaming: bool = False
        ) -> torch.Tensor:
        
        B, T, D = audio_tensor.shape
        projected = self.input_projection(audio_tensor)
        
        if streaming:
            if not hasattr(self, 'streaming_state'):
                self.init_streaming()
                
            # Process temporal stream
            quantized_temporal, discrete_temporal_tokens = self.quantizer(
                projected, 
                stream_type='temporal'
            )
            temporal_output, _ = self.temporal_transformer(
                discrete_temporal_tokens,
                streaming=True
            )
            
            # Process depth stream
            quantized_depth, discrete_depth_tokens = self.quantizer(
                projected,
                stream_type='depth'
            )
            depth_output, _ = self.depth_transformer(
                discrete_depth_tokens,
                streaming=True
            )
            
            # Update streaming buffers
            self._update_streaming_buffer('temporal', temporal_output)
            self._update_streaming_buffer('depth', depth_output)
            
        else:
            # Regular non-streaming forward pass
            quantized_temporal, discrete_temporal_tokens = self.quantizer(
                projected,
                stream_type='temporal'
            )
            quantized_depth, discrete_depth_tokens = self.quantizer(
                projected,
                stream_type='depth'
            )
            
            temporal_output, _ = self.temporal_transformer(discrete_temporal_tokens)
            depth_output, _ = self.depth_transformer(discrete_depth_tokens)
        
        discrete_audio_tokens = temporal_output + depth_output
        return discrete_audio_tokens
    
    def _update_streaming_buffer(self, stream_type: str, output: torch.Tensor):
        """Update streaming buffer for given stream type"""
        buffer = self.streaming_state[stream_type]['buffer']
        
        # Add new tokens to buffer
        for token in output[0]:
            buffer.append(token)
        
        # Update position
        self.streaming_state[stream_type]['position'] += output.size(1)


class StreamingJODIO(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        self.jodio_encoder = StreamingJODIOEncoder(self.args)
        self.jodio_decoder = JODIODecoder(self.args)
    
    def init_streaming(self):
        """Initialize streaming state for both encoder and decoder"""
        self.jodio_encoder.init_streaming()
        self.jodio_decoder.init_streaming()
    
    def encode(self, audio_tensor: torch.Tensor, streaming: bool = False) -> torch.Tensor:
        return self.jodio_encoder(audio_tensor, streaming=streaming)
    
    def decode(self, x: torch.Tensor, streaming: bool = False) -> torch.Tensor:
        return self.jodio_decoder(x, streaming=streaming)
    
    def forward(
            self,
            x: torch.Tensor,
            style: str = 'encode',
            streaming: bool = False
        ) -> torch.Tensor:
        
        if style == 'encode':
            return self.encode(x, streaming=streaming)
        elif style == 'decode':
            return self.decode(x, streaming=streaming)
        else:
            raise ValueError(f"Invalid style '{style}'. Must be 'encode' or 'decode'")