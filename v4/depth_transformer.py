import torch
import torch.nn as nn

from jodio.layers.transformer import Transformer
from args import ModelArgs
from utils import RMSNorm


class DepthTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.depth_transformer_args = args.depth_transformer  # hidden_size = 512
        self.temporal_transformer_args = args.temporal_transformer  # hidden_size = 1028

        # Project from temporal (1028) to depth (512) dimension
        self.input_projection = nn.Linear(
            self.temporal_transformer_args.hidden_size,  # 1028
            self.depth_transformer_args.hidden_size,     # 512
            bias=False
        )

        # Text token projection from depth dim to vocab
        self.text_projection = nn.Linear(
            self.depth_transformer_args.hidden_size,  # 512
            self.args.vocab_size,  
            bias=False
        )

        self.depth_transformer = Transformer(self.depth_transformer_args)

        # Semantic token projection from depth dim to codebook size
        self.semantic_projection = nn.Linear(
            self.depth_transformer_args.hidden_size,  # 512
            self.args.audio_encoder_args.codebook_size  # Usually 2048
        )

        # Acoustic token projections 
        self.acoustic_projections = nn.ModuleList([
            nn.Linear(self.depth_transformer_args.hidden_size, # 512
                     self.args.audio_encoder_args.codebook_size)
            for _ in range(self.args.audio_encoder_args.num_acoustic_quantizers - 1)
        ])

    def forward(self, temporal_embedding: torch.Tensor):
        B, L, D = temporal_embedding.shape  # [B, L, 1028]

        # Project from 1028 -> 512
        hidden = self.input_projection(temporal_embedding)  # [B, L, 512]
        
        # Transform
        depth_hidden = self.depth_transformer(hidden)  # [B, L, 512]

        # Project to logits
        text_logits = self.text_projection(depth_hidden)
        semantic_logits = self.semantic_projection(depth_hidden)
        acoustic_logits = [proj(depth_hidden) for proj in self.acoustic_projections]

        return text_logits, semantic_logits, acoustic_logits