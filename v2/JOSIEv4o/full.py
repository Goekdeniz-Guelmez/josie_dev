from typing import Optional, Tuple

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from JOSIEv4o.args import ModelArgs
from JOSIEv4o.utils import RMSNorm

class Quantizer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if hasattr(args, 'audio_encoder_args'):
            self.args = args.audio_encoder_args
        elif hasattr(args, 'vision_encoder_args'):
            self.args = args.vision_encoder_args
        else:
            self.args = args
            
        self.num_quantizers = self.args.num_quantizers
        self.codebook_size = self.args.codebook_size
        self.hidden_size = self.args.hidden_size
        self.chunk_size = self.hidden_size // self.num_quantizers
        self.rms_eps = self.args.rms_norm_eps
        
        self.temporal_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, (self.hidden_size // self.num_quantizers))
                for _ in range(self.num_quantizers)
            ]
        )
        self.temporal_output_norm = RMSNorm(self.hidden_size, eps=self.rms_eps)

        self.depth_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, (self.hidden_size // self.num_quantizers))
                for _ in range(self.num_quantizers)
            ]
        )
        self.depth_output_norm = RMSNorm(self.hidden_size, eps=self.rms_eps)

    def _clean_input(self, x: torch.Tensor):
        if len(x.shape) == 3:
            b, t, qd = x.shape
            d = qd // self.num_quantizers
            return x.view(b, t, self.num_quantizers, d)
        else:
            t, qd = x.shape
            d = qd // self.num_quantizers
            return x.view(t, self.num_quantizers, d)

    def quantize_temporial(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._clean_input(x)

        descrete_tokens = []
        quantized = []

        for i, codebook in enumerate(self.temporal_codebooks):
            distances = torch.cdist(x[..., i, :], codebook.weight)
            idx = distances.argmin(dim=-1)
            descrete_tokens.append(idx)
            quantized.append(codebook(idx))

        descrete_tokens = torch.stack(descrete_tokens, dim=-1)
        quantized = torch.cat(quantized, dim=-1)

        return quantized, descrete_tokens
    
    def quantize_depth(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._clean_input(x)
        
        descrete_tokens = []
        quantized = []

        for i, codebook in enumerate(self.depth_codebooks):
            distances = torch.cdist(x[..., i, :], codebook.weight)
            idx = distances.argmin(dim=-1)
            descrete_tokens.append(idx)
            quantized.append(codebook(idx))

        descrete_tokens = torch.stack(descrete_tokens, dim=-1)
        quantized = torch.cat(quantized, dim=-1)

        return quantized, descrete_tokens
    
    def forward(self, x: torch.Tensor, stream_type: str = 'temporal') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input based on stream type
        stream_type: One of 'temporal', 'depth', or 'spectral'
        """
        if stream_type == 'temporal':
            quantized, descrete_tokens = self.quantize_temporial(x)
            return self.temporal_output_norm(quantized), descrete_tokens
        
        elif stream_type == 'depth':
            quantized, descrete_tokens = self.quantize_depth(x)
            return self.depth_output_norm(quantized), descrete_tokens

        else:
            raise ValueError(f"Unknown stream type: {stream_type}")


#################

class Attention(nn.Module):
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

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        B, L, D = x.shape

        querys, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        querys = querys.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

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


class TransformerBlock(nn.Module):
    def __init__(self, args, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.args = args

        self.attention = Attention(args)
        
        self.feed_forward = MultiLayerPerception(args)
        
        self.attention_norm = RMSNorm(self.args.hidden_size, self.args.rms_norm_eps)
        self.mlp_norm = RMSNorm(self.args.hidden_size, self.args.rms_norm_eps)
        
    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, is_decoder: bool = False):
        super().__init__()
        if hasattr(args, 'audio_encoder_args'):
            self.args = args.audio_encoder_args
        elif hasattr(args, 'vision_encoder_args'):
            self.args = args.vision_encoder_args
        else:
            self.args = args

        self.is_decoder = is_decoder

        self.in_embeddings = nn.Embedding(self.args.codebook_size, self.args.hidden_size)
        
        self.pos_embedding = self._create_rotary_embedding()

        self.layers = nn.ModuleList([
            TransformerBlock(self.args, layer_index=idx) for idx in range(self.args.hidden_layers)
        ])
        
        self.norm = RMSNorm(self.args.hidden_size, self.args.rms_norm_eps)

        self.lm_head = nn.Linear(
            self.args.hidden_size,
            self.args.codebook_size * self.args.num_quantizers,
            bias=False
        )

    def _create_rotary_embedding(self) -> nn.Parameter:
        max_seq_len = self.args.max_position_embeddings
        hidden_size = self.args.hidden_size
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        
        pos = torch.arange(max_seq_len, dtype=torch.float)
        sincos = torch.einsum('i,j->ij', pos, inv_freq)
        emb = torch.cat((sincos.sin(), sincos.cos()), dim=-1)
        return nn.Parameter(emb.unsqueeze(0), requires_grad=False)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        """Exclude position embeddings from weight decay."""
        return {'pos_embedding'}

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.squeeze(0)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)
        else:
            x = x

        B, L = x.shape

        x = self.in_embeddings(x)
        
        positions = self.pos_embedding[:, :L, :]
        x = x + positions

        mask = None
        if self.is_decoder:
            if L > 1:
                mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
                mask = mask.unsqueeze(0)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)

        logits = self.lm_head(x)
        logits = logits.view(B, L, self.args.num_quantizers, -1)
        tokens = torch.argmax(logits, dim=-1)
        
        return tokens, x

#################

class JODIOEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.samples_needed = int(self.args.inference_args.rate * self.args.inference_args.record_seconds)
        
        # Encoder projection
        self.encoder_projection = nn.Linear(
            self.samples_needed,
            self.args.audio_encoder_args.hidden_size,
            bias=False
        )
        self.encoder_projection_norm = RMSNorm(
            self.args.audio_encoder_args.hidden_size, 
            self.args.audio_encoder_args.rms_norm_eps
        )
        
        # Components
        self.quantizer = Quantizer(self.args)
        self.temporal_transformer = Transformer(self.args)
        self.depth_transformer = Transformer(self.args)
    
    def _ensure_tensor_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has the correct shape [batch_size, sequence_length]"""
        if tensor.dim() == 1:  # [sequence_length]
            return tensor.unsqueeze(0)  # [1, sequence_length]
        elif tensor.dim() == 3:  # [batch_size, channels, sequence_length]
            batch_size, channels, seq_len = tensor.size()
            return tensor.view(batch_size, channels * seq_len)
        return tensor
    
    def forward(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        B, T, D = audio_tensor.shape
        # Project and normalize
        projected = self.encoder_projection(audio_tensor)
        
        # Reshape for quantizer if needed [batch_size, sequence_length, hidden_size]
        if projected.dim() == 2:
            projected = projected.unsqueeze(1)
        
        # Quantize in both streams
        quantized_temporal, discrete_temporal_tokens = self.quantizer(
            projected, stream_type='temporal'
        )
        quantized_depth, discrete_depth_tokens = self.quantizer(
            projected, stream_type='depth'
        )
        
        # Transform both streams
        temporal_output, _ = self.temporal_transformer(discrete_temporal_tokens)
        depth_output, _ = self.depth_transformer(discrete_depth_tokens)
        
        # Combine outputs
        discrete_audio_tokens = temporal_output + depth_output
        return discrete_audio_tokens


class JODIODecoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.temporal_transformer = Transformer(self.args)
        self.depth_transformer = Transformer(self.args, is_decoder=True)
    
    def forward(self, audio_token: torch.Tensor) -> torch.Tensor:
        # Process through both transformer streams
        temporal_output, _ = self.temporal_transformer(audio_token)
        depth_output, _ = self.depth_transformer(audio_token)
        
        # Combine outputs
        discrete_audio_tokens = temporal_output + depth_output
        return discrete_audio_tokens


class JODIO(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.jodio_encoder = JODIOEncoder(self.args)
        self.jodio_decoder = JODIODecoder(self.args)
    
    def encode(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Encode audio tensor to discrete tokens"""
        return self.jodio_encoder(audio_tensor)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens back to audio representation"""
        return self.jodio_decoder(x)
    
    def forward(self, x: torch.Tensor, style: str = 'encode') -> torch.Tensor:
        """Forward pass with specified style (encode/decode)"""
        if style == 'encode':
            return self.encode(x)
        elif style == 'decode':
            return self.decode(x)
        else:
            raise ValueError(f"Invalid style '{style}'. Must be 'encode' or 'decode'")