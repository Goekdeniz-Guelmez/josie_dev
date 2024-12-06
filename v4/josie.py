import torch
import torch.nn as nn

from jodio.JODIO import JODIO
from temporial_tranformer import TemporalTransformer
from depth_transformer import DepthTransformer

from args import ModelArgs


class JOSIE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.jodio = JODIO(args)

        self.temporial_transformer = TemporalTransformer(args)
        self.depth_transformer = DepthTransformer(args)
    
    def forward(self, text_tokens: torch.Tensor, user_waveform: torch.Tensor):
        semantic_token, acoustic_tokens, _ = self.jodio.encode(user_waveform)

        temporal_context = self.temporial_transformer(
            text_tokens,
            semantic_token,
            acoustic_tokens
        )

        text_token, semantic_token, acoustic_tokens = self.depth_transformer(temporal_context)

        josies_waveform = self.jodio.decode(semantic_token, acoustic_tokens)

        return text_token, josies_waveform
