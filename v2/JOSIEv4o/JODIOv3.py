import torch
import torch.nn as nn

from JOSIEv4o.quantizer import Quantizer
from JOSIEv4o.JODIOv1 import JODIODecoder
from JOSIEv4o.multistream_transformer import MultiStreamTransformer

from JOSIEv4o.args import ModelArgs

class MultiStreamingJODIOEncoder(nn.Module):
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
        self.multistream_transformer = MultiStreamTransformer(self.args, self.args.streaming_args.stream_types)
    
    def forward(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        B, T, D = audio_tensor.shape
        projected = self.input_projection(audio_tensor)
        
        quantized_temporal, discrete_temporal_tokens = self.quantizer(
            projected, stream_type='temporal'
        )
        quantized_depth, discrete_depth_tokens = self.quantizer(
            projected, stream_type='depth'
        )

        discrete_temporal_and_depth_tokens = discrete_temporal_tokens + discrete_depth_tokens
        
        discrete_audio_tokens, _ = self.multistream_transformer(discrete_temporal_and_depth_tokens)
        return discrete_audio_tokens


class MultiStreamingJODIO(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.jodio_encoder = MultiStreamingJODIOEncoder(self.args)
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