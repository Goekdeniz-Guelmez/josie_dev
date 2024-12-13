import torch
import torch.nn as nn

from transformer import Transformer
from args import ModelArgs
from utils import RMSNorm


class TemporalTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.temporal_args = args.temporal_transformer_args

        # Token embeddings 
        self.text_embedding = nn.Embedding(
            args.vocab_size,
            self.temporal_args.hidden_size
        )

        # Single embedding for semantic tokens
        self.semantic_embedding = nn.Embedding(
            args.audio_encoder_args.codebook_size,  # Usually 2048
            self.temporal_args.hidden_size        
        )

        # Single embedding for acoustic tokens
        self.acoustic_embedding = nn.Embedding(
            args.audio_encoder_args.codebook_size,  # Usually 2048
            self.temporal_args.hidden_size
        )

        # Main transformer
        self.transformer = Transformer(self.temporal_args)
        
        # Norms
        self.norm = RMSNorm(self.temporal_args.hidden_size)
        self.final_norm = RMSNorm(self.temporal_args.hidden_size)

    def forward(
            self, 
            text_tokens: torch.Tensor, # [B, L]
            semantic_tokens: torch.Tensor, # [L] 
            acoustic_tokens: torch.Tensor # [L]
    ) -> torch.tensor:

        # Get embeddings
        text_emb = self.text_embedding(text_tokens) # [B, L1, H]
        semantic_emb = self.semantic_embedding(semantic_tokens).unsqueeze(0) # [B, L2, H]
        acoustic_emb = self.acoustic_embedding(acoustic_tokens).unsqueeze(0) # [B, L3, H]

        print(text_emb.shape, semantic_emb.shape, acoustic_emb.shape)
        input()

        # Concatenate along hidden dimension
        hidden = torch.cat([text_emb, semantic_emb, acoustic_emb], dim=1) # [B, L1+L2+L3, H]

        # Apply transformer
        hidden = self.norm(hidden)
        hidden = self.transformer(hidden)
        return self.final_norm(hidden)