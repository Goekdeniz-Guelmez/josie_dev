from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from collections import deque

from JOSIEv4o.args import ModelArgs
from JOSIEv4o.utils import RMSNorm

@dataclass
class StreamingConfig:
    chunk_size: int = 512  # Size of each streaming chunk
    context_size: int = 2048  # Maximum context size to maintain
    overlap_size: int = 128  # Overlap between chunks to maintain continuity


class StreamingAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.hidden_size = self.args.hidden_size
        self.num_heads = self.args.num_heads
        self.head_dim = self.args.head_dim
        self.attention_dropout = self.args.attention_dropout

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False
        )
        
        self.dropout = nn.Dropout(self.attention_dropout)
        
        # Streaming state
        self.kv_cache = None
        self.cache_size = 0

    def _init_streaming_state(self, max_cache_size: int):
        self.kv_cache = None
        self.cache_size = max_cache_size

    def _update_kv_cache(self, keys: torch.Tensor, values: torch.Tensor, current_length: int):
        if self.kv_cache is None:
            self.kv_cache = (keys, values)
        else:
            cached_keys, cached_values = self.kv_cache
            
            # Concatenate new KV with cache
            keys = torch.cat([cached_keys, keys], dim=2)
            values = torch.cat([cached_values, values], dim=2)
            
            # Trim cache if it exceeds maximum size
            if keys.size(2) > self.cache_size:
                start_idx = keys.size(2) - self.cache_size
                keys = keys[:, :, start_idx:]
                values = values[:, :, start_idx:]
            
            self.kv_cache = (keys, values)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            streaming: bool = False
        ) -> torch.Tensor:
        B, L, D = x.shape

        querys = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if streaming and self.kv_cache is not None:
            cached_keys, cached_values = self.kv_cache
            keys = torch.cat([cached_keys, keys], dim=2)
            values = torch.cat([cached_values, values], dim=2)
            
            # Update cache
            self._update_kv_cache(keys[:, :, -self.cache_size:], values[:, :, -self.cache_size:], L)

        attn = F.scaled_dot_product_attention(
            querys, keys, values,
            attn_mask=mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scale
        )
        
        out = attn.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class MultiLayerPerception(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.linear1 = nn.Linear(self.args.hidden_size, 4 * self.args.hidden_size, bias=False)
        self.linear2 = nn.Linear(4 * self.args.hidden_size, self.args.hidden_size, bias=False)

        self.dropout = nn.Dropout(self.args.mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(F.silu(self.linear1(x))))
    

class StreamingTransformerBlock(nn.Module):
    def __init__(self, args, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.args = args

        self.attention = StreamingAttention(args)
        self.feed_forward = MultiLayerPerception(args)
        
        self.attention_norm = RMSNorm(self.args.hidden_size, self.args.rms_norm_eps)
        self.mlp_norm = RMSNorm(self.args.hidden_size, self.args.rms_norm_eps)
        
    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            streaming: bool = False
        ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask, streaming)
        x = x + self.feed_forward(self.mlp_norm(x))
        return x


class StreamingTransformer(nn.Module):
    def __init__(self, args: ModelArgs, streaming_config: StreamingConfig = None):
        super().__init__()
        if hasattr(args, 'audio_encoder_args'):
            self.args = args.audio_encoder_args
        elif hasattr(args, 'vision_encoder_args'):
            self.args = args.vision_encoder_args
        else:
            self.args = args

        self.streaming_config = streaming_config or StreamingConfig()
        
        self.in_embeddings = nn.Embedding(self.args.codebook_size, self.args.hidden_size)
        self.pos_embedding = self._create_rotary_embedding()

        self.layers = nn.ModuleList([
            StreamingTransformerBlock(self.args, layer_index=idx) 
            for idx in range(self.args.hidden_layers)
        ])
        
        self.norm = RMSNorm(self.args.hidden_size, self.args.rms_norm_eps)

        self.lm_head = nn.Linear(
            self.args.hidden_size,
            self.args.codebook_size * self.args.num_quantizers,
            bias=False
        )

        # Streaming state
        self.streaming_state = None
        
    def _create_rotary_embedding(self) -> nn.Parameter:
        max_seq_len = self.args.max_position_embeddings
        hidden_size = self.args.hidden_size
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        
        pos = torch.arange(max_seq_len, dtype=torch.float)
        sincos = torch.einsum('i,j->ij', pos, inv_freq)
        emb = torch.cat((sincos.sin(), sincos.cos()), dim=-1)
        return nn.Parameter(emb.unsqueeze(0), requires_grad=False)

    def init_streaming(self):
        """Initialize streaming state"""
        self.streaming_state = {
            'past_tokens': deque(maxlen=self.streaming_config.context_size),
            'past_embeddings': deque(maxlen=self.streaming_config.context_size),
            'current_position': 0
        }
        
        # Initialize streaming state for each attention layer
        for layer in self.layers:
            layer.attention._init_streaming_state(self.streaming_config.context_size)

    def forward_streaming(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for streaming inference"""
        if self.streaming_state is None:
            self.init_streaming()

        if len(x.shape) == 3:
            x = x.squeeze(0)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)

        B, L = x.shape
        
        # Get embeddings for current chunk
        current_embeddings = self.in_embeddings(x)
        
        # Calculate positions for current chunk
        positions = self.pos_embedding[:, self.streaming_state['current_position']: self.streaming_state['current_position'] + L, :]
        current_embeddings = current_embeddings + positions
        
        # Update streaming state
        for token in x[0]:
            self.streaming_state['past_tokens'].append(token.item())
        for emb in current_embeddings[0]:
            self.streaming_state['past_embeddings'].append(emb)
        
        # Process through transformer layers
        hidden_states = current_embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, streaming=True)
            
        hidden_states = self.norm(hidden_states)
        
        # Generate predictions
        logits = self.lm_head(hidden_states)
        logits = logits.view(B, L, self.args.num_quantizers, -1)
        tokens = torch.argmax(logits, dim=-1)
        
        # Update position counter
        self.streaming_state['current_position'] += L
        
        return tokens, hidden_states

    def forward(self, x: torch.Tensor, streaming: bool = False):
        """Regular forward pass with streaming option"""
        if streaming:
            return self.forward_streaming(x)
            
        if len(x.shape) == 3:
            x = x.squeeze(0)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)

        B, L = x.shape

        x = self.in_embeddings(x)
        positions = self.pos_embedding[:, :L, :]
        x = x + positions
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)

        logits = self.lm_head(x)
        logits = logits.view(B, L, self.args.num_quantizers, -1)
        tokens = torch.argmax(logits, dim=-1)
        
        return tokens, x