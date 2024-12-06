import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Transformer
from args import ModelArgs
from utils import turn_to_token


class DepthTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.depth_args = args.depth_transformer_args  # hidden_size = 512
        self.temporal_transformer_args = args.temporal_transformer_args  # hidden_size = 1028
        
        # Project from temporal (1028) to depth (512) dimension
        self.input_projection = nn.Linear(
            self.temporal_transformer_args.hidden_size,  # 1028
            self.depth_args.hidden_size,  # 512
            bias=False
        )
        
        # Text token projection from depth dim to vocab
        self.text_projection = nn.Linear(
            self.depth_args.hidden_size,  # 512
            self.args.vocab_size,
            bias=False
        )
        
        self.transformer = Transformer(self.depth_args)
        
        # Semantic token projection including text token context
        self.semantic_projection = nn.Linear(
            self.depth_args.hidden_size + 1,  # 512 + 1 for text token
            self.args.audio_encoder_args.codebook_size  # Usually 2048
        )
        
        # Acoustic token projections
        self.acoustic_projections = nn.ModuleList([
            nn.Linear(
                self.depth_args.hidden_size,  # 512
                self.args.audio_encoder_args.codebook_size,
                bias=False
            )
            for _ in range(self.args.audio_encoder_args.num_acoustic_quantizers - 1)
        ])

    def forward(self, temporal_embedding: torch.Tensor):
        B, L, D = temporal_embedding.shape  # [B, L, 1028]
        
        # Project from 1028 -> 512
        hidden = self.input_projection(temporal_embedding)  # [B, L, 512]
        
        # Transform
        depth_hidden = self.transformer(hidden)  # [B, L, 512]
        
        # Get single text token
        text_logits = self.text_projection(depth_hidden[:, 0])  # [B, vocab_size]
        text_token = turn_to_token(text_logits).unsqueeze(1)  # [B, 1, 1]
        
        # Get single semantic token
        semantic_input = torch.cat([depth_hidden[:, 0], text_token[:, 0]], dim=-1)  # [B, 513]
        semantic_logits = self.semantic_projection(semantic_input)
        semantic_token = turn_to_token(semantic_logits).unsqueeze(1)  # [B, 1, 1]
        
        # Get single acoustic token for each projection
        acoustic_tokens = [
            turn_to_token(proj(depth_hidden[:, 0])).unsqueeze(1)  # [B, 1, 1]
            for proj in self.acoustic_projections
        ]

        acoustic_tokens = torch.cat(acoustic_tokens, dim=-1)  # [B, L] where L is num_acoustic_quantizers-1
        
        return text_token, semantic_token, acoustic_tokens