from typing import Optional, Dict, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from JOSIEv4o.args import ModelArgs, AudioEncoderModelArgs, VisionEncoderModelArgs
from JOSIEv4o.utils import RMSNorm

from JOSIEv4o.transformer import MultiLayerPerception


class MultiStreamAttention(nn.Module):
    def __init__(
            self,
            stream_args: Union[AudioEncoderModelArgs, VisionEncoderModelArgs],
            stream_name: str
        ):
        super().__init__()
        self.args = stream_args
        self.stream_name = stream_name
        
        self.hidden_size = self.args.hidden_size
        self.num_heads = self.args.num_heads
        self.head_dim = self.args.head_dim
        self.attention_dropout = self.args.attention_dropout
        self.num_kv_heads = getattr(self.args, 'num_kv_heads', self.num_heads)
        
        self.scale = self.head_dim**-0.5
        
        # Handle grouped query attention if num_kv_heads is specified
        self.num_kv_heads = getattr(self.args, 'num_kv_heads', self.num_heads)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False
        )
        
        self.dropout = nn.Dropout(self.attention_dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_queries_per_kv > 1:
            return x.repeat_interleave(self.num_queries_per_kv, dim=2)
        return x

    def forward(
            self,
            x: torch.Tensor,
            temporal_mask: Optional[torch.Tensor] = None,
            depth_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        B, L, D = x.shape

        # Project queries, keys, and values
        queries = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat keys and values if using grouped query attention
        keys = self._repeat_kv(keys)
        values = self._repeat_kv(values)

        # Apply temporal and depth masks if provided
        mask = None
        if temporal_mask is not None and depth_mask is not None:
            mask = temporal_mask * depth_mask
        elif temporal_mask is not None:
            mask = temporal_mask
        elif depth_mask is not None:
            mask = depth_mask

        attn = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scale
        )
        
        out = attn.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class MultiStreamTransformerBlock(nn.Module):
    def __init__(
            self,
            stream_args: Union[AudioEncoderModelArgs, VisionEncoderModelArgs],
            layer_index: int,
            stream_name: str
        ):
        super().__init__()
        self.layer_index = layer_index
        self.args = stream_args
        self.stream_name = stream_name

        self.attention = MultiStreamAttention(stream_args, stream_name)
        self.feed_forward = MultiLayerPerception(stream_args)
        
        self.attention_norm = RMSNorm(self.args.hidden_size, self.args.rms_norm_eps)
        self.mlp_norm = RMSNorm(self.args.hidden_size, self.args.rms_norm_eps)
        
    def forward(
            self,
            x: torch.Tensor,
            temporal_mask: Optional[torch.Tensor] = None,
            depth_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        # Pre-normalization and attention
        normalized = self.attention_norm(x)
        x = x + self.attention(normalized, temporal_mask, depth_mask)
        
        # Pre-normalization and feed-forward
        normalized = self.mlp_norm(x)
        x = x + self.feed_forward(normalized)
        
        return x
    

class MultiStreamTransformer(nn.Module):
    def __init__(
            self,
            args: ModelArgs,
            stream_types: tuple = ('temporal', 'depth')
        ):
        super().__init__()
        self.args = args
        self.stream_types = stream_types
        
        # Setup stream-specific configurations
        self.streams = {}
        if hasattr(args, 'audio_encoder_args'):
            self.streams['audio'] = {
                'args': args.audio_encoder_args,
                'embedding': nn.Embedding(
                    args.audio_encoder_args.codebook_size,
                    args.audio_encoder_args.hidden_size
                )
            }
        if hasattr(args, 'vision_encoder_args'):
            self.streams['vision'] = {
                'args': args.vision_encoder_args,
                'embedding': nn.Embedding(
                    args.vision_encoder_args.codebook_size,
                    args.vision_encoder_args.hidden_size
                )
            }

        # Create positional embeddings for each stream
        self.pos_embeddings = nn.ModuleDict({
            stream_name: self._create_rotary_embedding(stream_config['args'])
            for stream_name, stream_config in self.streams.items()
        })

        # Create transformer layers for each stream
        self.layers = nn.ModuleDict({
            stream_name: nn.ModuleList([
                MultiStreamTransformerBlock(stream_config['args'], idx, stream_name)
                for idx in range(stream_config['args'].hidden_layers)
            ])
            for stream_name, stream_config in self.streams.items()
        })
        
        # Create output normalization for each stream
        self.norms = nn.ModuleDict({
            stream_name: RMSNorm(
                stream_config['args'].hidden_size,
                stream_config['args'].rms_norm_eps
            )
            for stream_name, stream_config in self.streams.items()
        })

        # Create output heads for each stream
        self.heads = nn.ModuleDict({
            stream_name: nn.Linear(
                stream_config['args'].hidden_size,
                stream_config['args'].codebook_size * stream_config['args'].num_quantizers,
                bias=False
            )
            for stream_name, stream_config in self.streams.items()
        })

    def _create_rotary_embedding(
            self,
            stream_args: Union[AudioEncoderModelArgs, VisionEncoderModelArgs]
        ) -> nn.Parameter:
        max_seq_len = stream_args.max_position_embeddings
        hidden_size = stream_args.hidden_size
        
        theta = getattr(stream_args, 'rope_theta', 10000.0)
        inv_freq = 1.0 / (theta ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        
        pos = torch.arange(max_seq_len, dtype=torch.float)
        sincos = torch.einsum('i,j->ij', pos, inv_freq)
        emb = torch.cat((sincos.sin(), sincos.cos()), dim=-1)
        return nn.Parameter(emb.unsqueeze(0), requires_grad=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embeddings'}

    def _create_masks(
            self,
            x: torch.Tensor,
            stream_name: str,
            stream_type: str
        ) -> torch.Tensor:
        B, L = x.shape[:2]
        device = x.device
        
        if stream_type == 'temporal':
            # Causal mask for temporal attention
            mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
            return mask.unsqueeze(0)
        elif stream_type == 'depth':
            # Full attention for depth processing
            return None
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
            stream_type: str = 'temporal'
        ) -> Dict[str, torch.Tensor]:
        if stream_type not in self.stream_types:
            raise ValueError(f"Invalid stream type: {stream_type}")

        outputs = {}
        hidden_states = {}

        # Process each stream
        for stream_name, x in inputs.items():
            if stream_name not in self.streams:
                continue

            # Normalize input shape
            if len(x.shape) == 3:
                x = x.squeeze(0)
            elif len(x.shape) == 1:
                x = x.unsqueeze(0)

            B, L = x.shape

            # Create embeddings
            stream_config = self.streams[stream_name]
            x = stream_config['embedding'](x)

            # Add positional embeddings
            pos_emb = self.pos_embeddings[stream_name][:, :L, :]
            x = x + pos_emb

            # Create appropriate masks
            temporal_mask = self._create_masks(x, stream_name, 'temporal') if 'temporal' in self.stream_types else None
            depth_mask = self._create_masks(x, stream_name, 'depth') if 'depth' in self.stream_types else None

            # Process through transformer layers
            for layer in self.layers[stream_name]:
                x = layer(x, temporal_mask, depth_mask)

            # Final normalization and prediction
            x = self.norms[stream_name](x)
            logits = self.heads[stream_name](x)

            # Reshape logits according to stream configuration
            logits = logits.view(B, L, stream_config['args'].num_quantizers, -1)
            tokens = torch.argmax(logits, dim=-1)

            outputs[stream_name] = tokens
            hidden_states[stream_name] = x

        return outputs, hidden_states