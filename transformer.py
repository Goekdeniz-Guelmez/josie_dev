from typing import List, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import AudioRQTransformerArgs as ModelArgs
from utils import RMSNorm, apply_rotary_emb, repeat_kv, precompute_freqs_cis


class TemporalDepthAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        self.num_heads = args.encoder_num_heads
        self.head_dim = args.encoder_head_dim
        
        self.wq = nn.Linear(
            args.encoder_hidden_dim,
            args.encoder_num_heads * args.encoder_head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.encoder_hidden_dim,
            args.encoder_num_heads * args.encoder_head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.encoder_hidden_dim,
            args.encoder_num_heads * args.encoder_head_dim,
            bias=False
        )
        
        self.wo = nn.Linear(
            args.encoder_num_heads * args.encoder_head_dim,
            args.encoder_hidden_dim,
            bias=False
        )
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.wq(x).view(B, L, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.num_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if mask is not None:
            scores = scores + mask
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)


class TemporalDepthTransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = TemporalDepthAttention(args)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(args.encoder_hidden_dim, 4 * args.encoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * args.encoder_hidden_dim, args.encoder_hidden_dim),
            nn.Dropout(0.1)
        )
        
        self.ln1 = RMSNorm(args.encoder_hidden_dim, args.encoder_rms_norm_eps)
        self.ln2 = RMSNorm(args.encoder_hidden_dim, args.encoder_rms_norm_eps)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.feed_forward(self.ln2(x))
        return x


class TemporalDepthTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, args.encoder_max_position_embeddings, args.encoder_hidden_dim)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalDepthTransformerBlock(args) for _ in range(args.encoder_hidden_layers)
        ])
        
        self.ln_out = RMSNorm(args.encoder_hidden_dim, args.encoder_rms_norm_eps)

        self.lm_head = nn.Linear(
            args.encoder_hidden_dim,
            args.encoder_codebook_size * args.encoder_num_quantizers,
            bias=False
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, is_decoder: bool = False):
        B, L, D = x.shape
        
        positions = self.pos_embedding[:, :L, :]
        x = x + positions
        x = self.dropout(x)
        
        mask = None
        if is_decoder:
            if L > 1:
                mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
                mask = mask.unsqueeze(0)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.ln_out(x)

        logits = self.lm_head(x)

        logits = logits.view(B, L, self.args.encoder_num_quantizers, -1)
        
        tokens = torch.argmax(logits, dim=-1)
        
        return tokens, logits.view(B, L, -1)


# model = TemporalDepthTransformer(ModelArgs)

# tokens = torch.randn(1, 2, 256)
# print(tokens.shape)

# # Forward pass
# output = model(tokens, True)
# print(output.shape)