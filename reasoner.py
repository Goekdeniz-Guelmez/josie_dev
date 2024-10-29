import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from args import ModelArgs
from utils import RMSNorm, apply_rotary_emb, repeat_kv, precompute_freqs_cis


class ReasonerAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.reasoner_num_kv_heads = args.reasoner_num_heads if args.reasoner_num_kv_heads is None else args.reasoner_num_kv_heads
        self.reasoner_num_heads = args.reasoner_num_heads
        self.n_rep = self.reasoner_num_heads // self.reasoner_num_kv_heads
        self.reasoner_head_dim = args.reasoner_head_dim

        self.wq = nn.Linear(
            args.reasoner_hidden_dim,
            args.reasoner_num_heads * self.reasoner_head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.reasoner_hidden_dim,
            self.reasoner_num_kv_heads * self.reasoner_head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.reasoner_hidden_dim,
            self.reasoner_num_kv_heads * self.reasoner_head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.reasoner_num_heads * self.reasoner_head_dim,
            args.reasoner_hidden_dim,
            bias=False
        )
        self.cache_k = torch.zeros(
            (
                args.reasoner_max_batch_size,
                args.reasoner_max_position_embeddings,
                self.reasoner_num_kv_heads,
                self.reasoner_head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.reasoner_max_batch_size,
                args.reasoner_max_position_embeddings,
                self.reasoner_num_kv_heads,
                self.reasoner_head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,  # Shape: (batch_size, seq_len, dim)
        start_pos: int,
        freqs_cis: torch.Tensor,  # Shape: (seq_len, dim//2)
        mask: Optional[torch.Tensor],  # Shape: (seq_len, seq_len) or None
    ):
        B, L, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(B, L, self.reasoner_num_heads, self.reasoner_head_dim)
        xk = xk.view(B, L, self.reasoner_num_kv_heads, self.reasoner_head_dim)
        xv = xv.view(B, L, self.reasoner_num_kv_heads, self.reasoner_head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:B, start_pos : start_pos + L] = xk
        self.cache_v[:B, start_pos : start_pos + L] = xv
        keys = self.cache_k[:B, : start_pos + L]
        values = self.cache_v[:B, : start_pos + L]
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        xq = xq.transpose(1, 2) # (batch_size, n_local_heads, L, head_dim)
        keys = keys.transpose(1, 2) # (batch_size, n_local_heads, cache_len + L, head_dim)
        values = values.transpose(1, 2) # (batch_size, n_local_heads, cache_len + L, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.reasoner_head_dim)
        if mask is not None:
            scores = scores + mask # (batch_size, n_local_heads, L, cache_len + L)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(output)


class ReasonerFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ReasonerTransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_index: int):
        super().__init__()
        self.n_heads = args.reasoner_num_heads
        self.dim = args.reasoner_hidden_dim
        self.reasoner_head_dim = args.reasoner_hidden_dim // args.reasoner_num_heads
        self.attention = ReasonerAttention(args)
        self.feed_forward = ReasonerFeedForward(
            dim=args.reasoner_hidden_dim,
            hidden_dim=4 * args.reasoner_hidden_dim,
            multiple_of=args.reasoner_multiple_of,
            ffn_dim_multiplier=args.reasoner_ffn_dim_multiplier,
        )
        self.layer_index = layer_index
        self.attention_norm = RMSNorm(args.reasoner_hidden_dim, eps=args.reasoner_rms_norm_eps)
        self.ffn_norm = RMSNorm(args.reasoner_hidden_dim, eps=args.reasoner_rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class ReasonerTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.reasoner_vocab_size
        self.reasoner_hidden_layers = args.reasoner_hidden_layers

        self.tok_embeddings = nn.Embedding(
            args.reasoner_vocab_size, args.reasoner_hidden_dim
        )

        self.layers = nn.ModuleList([
            ReasonerTransformerBlock(args, layer_index=layer_idx) for layer_idx in range(args.reasoner_hidden_layers)
        ])

        self.norm = RMSNorm(args.reasoner_hidden_dim, eps=args.reasoner_rms_norm_eps)

        self.text_output = nn.Linear(
            args.reasoner_hidden_dim, args.reasoner_vocab_size, bias=False
        )
        self.audio_output = nn.Linear(
            args.reasoner_hidden_dim, args.decoder_audio_hidden_dim, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            args.reasoner_hidden_dim // args.reasoner_num_heads,
            args.reasoner_max_position_embeddings * 2,
            args.reasoner_rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = tokens.shape
        
        h = self.tok_embeddings(tokens)
        
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + L]

        mask = None
        if L > 1:
            mask = torch.full((L, L), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack(
                [torch.zeros((L, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
            
        h = self.norm(h)
        
        text_stream = self.text_output(h).float()
        audio_stream = self.audio_output(h)
        
        return text_stream, audio_stream