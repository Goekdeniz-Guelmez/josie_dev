from typing import Tuple, Optional

from einops import rearrange

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import ModelArgs
from utils import RMSNorm, apply_rotary_emb, repeat_kv, precompute_freqs_cis


class TemporalDepthEncoderAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        self.num_heads = args.encoder_audio_num_heads
        self.head_dim = args.encoder_audio_head_dim
        
        self.wq = nn.Linear(
            args.encoder_audio_hidden_dim,
            args.encoder_audio_num_heads * args.encoder_audio_head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.encoder_audio_hidden_dim,
            args.encoder_audio_num_heads * args.encoder_audio_head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.encoder_audio_hidden_dim,
            args.encoder_audio_num_heads * args.encoder_audio_head_dim,
            bias=False
        )
        
        self.wo = nn.Linear(
            args.encoder_audio_num_heads * args.encoder_audio_head_dim,
            args.encoder_audio_hidden_dim,
            bias=False
        )
        
        self.dropout = nn.Dropout(args.encoder_audio_attention_dropout)

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
    

class EncoderMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.linear1 = nn.Linear(args.encoder_audio_hidden_dim, 4 * args.encoder_audio_hidden_dim, bias=False)
        self.linear2 = nn.Linear(4 * args.encoder_audio_hidden_dim, args.encoder_audio_hidden_dim, bias=False)

        self.dropout = nn.Dropout(args.encoder_audio_mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(F.silu(self.linear1(x))))


class TemporalDepthEncoderTransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_index):
        super().__init__()
        self.layer_index = layer_index
        self.attention = TemporalDepthEncoderAttention(args)
        
        self.feed_forward = EncoderMLP(args)
        
        self.ln1 = RMSNorm(args.encoder_audio_hidden_dim, args.encoder_audio_rms_norm_eps)
        self.ln2 = RMSNorm(args.encoder_audio_hidden_dim, args.encoder_audio_rms_norm_eps)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.feed_forward(self.ln2(x))
        return x


class TemporalDepthEncoderTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, args.encoder_audio_max_position_embeddings, args.encoder_audio_hidden_dim)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalDepthEncoderTransformerBlock(args, layer_index=idx) for idx in range(args.encoder_audio_hidden_layers)
        ])
        
        self.ln_out = RMSNorm(args.encoder_audio_hidden_dim, args.encoder_audio_rms_norm_eps)

        self.lm_head = nn.Linear(
            args.encoder_audio_hidden_dim,
            args.encoder_audio_codebook_size * args.encoder_audio_num_quantizers,
            bias=False
        )

    def forward(self, x: torch.Tensor, is_decoder: bool = False):
        B, L, D = x.shape
        
        positions = self.pos_embedding[:, :L, :]
        x = x + positions
        
        mask = None
        if is_decoder:
            if L > 1:
                mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
                mask = mask.unsqueeze(0)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.ln_out(x)

        logits = self.lm_head(x)

        logits = logits.view(B, L, self.args.encoder_audio_num_quantizers, -1)
        
        tokens = torch.argmax(logits, dim=-1)
        
        return tokens, logits.view(B, L, -1)
    

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


class AudioQuantizer(nn.Module):
    """Quantizer with temporal, depth, and spectral codebooks"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_quantizers = args.encoder_audio_num_quantizers
        self.codebook_size = args.encoder_audio_codebook_size
        self.hidden_dim = args.encoder_audio_hidden_dim

        self.input_norm = RMSNorm(self.hidden_dim, eps=self.args.encoder_audio_rms_norm_eps)

        # Temporal codebooks - for sequence-level patterns
        self.temporal_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, (self.hidden_dim // self.num_quantizers))
                for _ in range(self.num_quantizers)
            ]
        )
        self.temporal_output_norm = RMSNorm(self.hidden_dim, eps=self.args.encoder_audio_rms_norm_eps)

        # Depth codebooks - for feature-level patterns
        self.depth_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, (self.hidden_dim // self.num_quantizers))
                for _ in range(self.num_quantizers)
            ]
        )
        self.depth_output_norm = RMSNorm(self.hidden_dim, eps=self.args.encoder_audio_rms_norm_eps)

    def quantize_temporial(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        x = rearrange(x, 'b t (q d) -> b t q d', q=self.num_quantizers)

        indices = []
        quantized = []

        for i, codebook in enumerate(self.temporal_codebooks):
            distances = torch.cdist(x[..., i, :], codebook.weight)
            idx = distances.argmin(dim=-1)
            indices.append(idx)
            quantized.append(codebook(idx))

        indices = torch.stack(indices, dim=-1)
        quantized = torch.cat(quantized, dim=-1)

        return quantized, indices
    
    def quantize_depth(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        x = rearrange(x, 'b t (q d) -> b t q d', q=self.num_quantizers)
        
        indices = []
        quantized = []
        
        for i, codebook in enumerate(self.depth_codebooks):
            distances = torch.cdist(x[..., i, :], codebook.weight)
            idx = distances.argmin(dim=-1)
            indices.append(idx)
            quantized.append(codebook(idx))
        
        indices = torch.stack(indices, dim=-1)
        quantized = torch.cat(quantized, dim=-1)
        
        return quantized, indices
    
    def forward(self, x: torch.Tensor, stream_type: str = 'temporal') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input based on stream type
        stream_type: One of 'temporal', 'depth', or 'spectral'
        """
        x = self.input_norm(x)

        if stream_type == 'temporal':
            quantized, indices = self.quantize_temporial(x)
            return self.temporal_output_norm(quantized), indices
        
        elif stream_type == 'depth':
            quantized, indices = self.quantize_depth(x)
            return self.depth_output_norm(quantized), indices

        else:
            raise ValueError(f"Unknown stream type: {stream_type}")


class AudioEncoder(nn.Module):
    """Quantizer with temporal, depth, and spectral codebooks"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.quantizer = AudioQuantizer(args)

        self.temporial_transformer = TemporalDepthEncoderTransformer(args)
        self.depth_transformer = TemporalDepthEncoderTransformer(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_quantized, _ = self.quantizer(x)
        depth_quantized, _ = self.quantizer(x, 'depth')

        temporialed, _ = self.temporial_transformer(temporal_quantized)
        depthed, _ = self.depth_transformer(depth_quantized)

        discrete_audio_tokens = temporialed + depthed
        return discrete_audio_tokens



class AudioDecoder(nn.Module):
    """Quantizer with temporal, depth, and spectral codebooks"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.hidden_dim = args.encoder_audio_hidden_dim

        self.quantizer = AudioQuantizer(args)

        self.temporial_transformer = TemporalDepthDecoderTransformer(args)
        self.depth_transformer = TemporalDepthDecoderTransformer(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_quantized, _ = self.quantizer(x)
        depth_quantized, _ = self.quantizer(x, 'depth')

        temporialed, _ = self.temporial_transformer(temporal_quantized)
        depthed, _ = self.depth_transformer(depth_quantized)

        discrete_audio_tokens = temporialed + depthed
        return discrete_audio_tokens


class AudioEncoderDecoder(nn.Module):
    """Quantizer with temporal, depth, and spectral codebooks"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.hidden_dim = args.encoder_audio_hidden_dim

        self.encoder = AudioEncoder(args)
        self.decoder = AudioDecoder(args)
    
    def forward(self, x: torch.Tensor, style: str = 'encode') -> torch.Tensor:
        if style == 'encode':
            output = self.encoder(x)
        elif style == 'decode':
            output = self.decoder(x)
        else:
            return f'style wasnt found {style}'
        
        return output
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
    


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
        next_token = torch.argmax(text_stream[:, -1, :], dim=-1)
        
        return next_token, text_stream, audio_stream


class JOSIE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.audio = AudioEncoderDecoder(args)
        self.reasoner = ReasonerTransformer(args)

    def forward(self, audio_input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = audio_input_tensor.shape # -> torch.Size([1, 1, 256])
        discrete_audio_tokens = self.audio.encode(audio_input_tensor) # -> torch.Size([1, 1, 8]
        next_token, _, audio_stream = self.reasoner(discrete_audio_tokens.squeeze(0)) # -> torch.Size([1]), _, torch.Size([1, 8, 256])
        output = self.audio.decode(audio_stream) # -> torch.Size([1, 8, 8])
        audio_output = output.squeeze().detach().numpy() # -> (8, 8)
        return audio_output, next_token