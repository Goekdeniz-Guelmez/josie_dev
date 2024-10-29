from typing import List, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import ModelArgs
from utils import RMSNorm


class TemporalDepthDecoderAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        self.num_heads = args.decoder_audio_num_heads
        self.head_dim = args.decoder_audio_head_dim
        
        self.wq = nn.Linear(
            args.decoder_audio_hidden_dim,
            args.decoder_audio_num_heads * args.decoder_audio_head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.decoder_audio_hidden_dim,
            args.decoder_audio_num_heads * args.decoder_audio_head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.decoder_audio_hidden_dim,
            args.decoder_audio_num_heads * args.decoder_audio_head_dim,
            bias=False
        )
        
        self.wo = nn.Linear(
            args.decoder_audio_num_heads * args.decoder_audio_head_dim,
            args.decoder_audio_hidden_dim,
            bias=False
        )
        
        self.dropout = nn.Dropout(args.decoder_audio_attention_dropout)

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
    

class DecoderMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.linear1 = nn.Linear(args.decoder_audio_hidden_dim, 4 * args.decoder_audio_hidden_dim, bias=False)
        self.linear2 = nn.Linear(4 * args.decoder_audio_hidden_dim, args.decoder_audio_hidden_dim, bias=False)

        self.dropout = nn.Dropout(args.decoder_audio_mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(F.silu(self.linear1(x))))


class TemporalDepthDecoderTransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_index):
        super().__init__()
        self.layer_index = layer_index
        self.attention = TemporalDepthDecoderAttention(args)
        
        self.feed_forward = DecoderMLP(args)
        
        self.ln1 = RMSNorm(args.decoder_audio_hidden_dim, args.decoder_audio_rms_norm_eps)
        self.ln2 = RMSNorm(args.decoder_audio_hidden_dim, args.decoder_audio_rms_norm_eps)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.feed_forward(self.ln2(x))
        return x


class TemporalDepthDecoderTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, args.decoder_audio_max_position_embeddings, args.decoder_audio_hidden_dim)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalDepthDecoderTransformerBlock(args, layer_index=layer_idx) for layer_idx in range(args.decoder_audio_hidden_layers)
        ])
        
        self.ln_out = RMSNorm(args.decoder_audio_hidden_dim, args.decoder_audio_rms_norm_eps)

        self.lm_head = nn.Linear(
            args.decoder_audio_hidden_dim,
            args.decoder_audio_codebook_size * args.decoder_audio_num_quantizers,
            bias=False
        )

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        
        positions = self.pos_embedding[:, :L, :]
        x = x + positions
        
        mask = None
        if L > 1:
            mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
            mask = mask.unsqueeze(0)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.ln_out(x)

        logits = self.lm_head(x)

        logits = logits.view(B, L, self.args.decoder_audio_num_quantizers, -1)
        
        tokens = torch.argmax(logits, dim=-1)
        
        return tokens, logits.view(B, L, -1)