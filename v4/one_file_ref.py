import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_transformer import DepthTransformer

from dataclasses import dataclass, field
from typing import Optional, Type
from pathlib import Path

import pyaudio
import inspect


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params: dict):
        valid_params = {
            k: v
            for k, v in params.items()
            if k in inspect.signature(cls).parameters
        }
        return cls(**valid_params)
    

@dataclass
class InferenceArgs(BaseModelArgs):
    format = pyaudio.paFloat32
    channels = 1
    rate = 16000  # 16kHz
    record_seconds = 0.25  # 250ms
    chunk = rate // 4 # 4096


@dataclass
class AudioEncoderModelArgs(BaseModelArgs):
    hidden_size: int = 512
    hidden_layers: int = 12
    num_heads: int = 16
    head_dim: int = hidden_size // num_heads

    channels: int = 512
    kernel_size: int = 3
    num_conv_layers: int = 3
    dilation_growth: int = 2

    codebook_size: int = 2048
    num_acoustic_quantizers: int = 8
    num_semantic_quantizers: int = 1
    downsampling_ratio: int = 8

    rms_norm_eps: float = 1e-5
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 64


@dataclass
class AudioDecoderModelArgs(BaseModelArgs):
    hidden_size: int = 512
    hidden_layers: int = 12
    num_heads: int = 12
    head_dim: int = hidden_size // num_heads

    channels: int = 512
    kernel_size: int = 3
    num_conv_layers: int = 3
    dilation_growth: int = 2

    rms_norm_eps: float = 1e-5
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 64


@dataclass
class VisionEncoderModelArgs(BaseModelArgs):
    hidden_size: int = 512
    hidden_layers: int = 8
    num_heads: int = 16
    num_kv_heads: Optional[int] = 8
    head_dim: int = hidden_size // num_heads
    codebook_size: int = 2048
    num_quantizers: int = 32
    rms_norm_eps: float = 1e-5
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 256
    rope_theta: float = 500000
    max_frames: int = 12


@dataclass
class TemporialTransformer(BaseModelArgs):
    hidden_size: int = 1028
    hidden_layers: int = 12
    num_heads: int = 12
    head_dim: int = hidden_size // num_heads
    attention_dropout: float = 0.0

    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 4096


@dataclass
class DepthTransformer(BaseModelArgs):
    hidden_size: int = 512
    hidden_layers: int = 6
    num_heads: int = 4
    head_dim: int = hidden_size // num_heads
    attention_dropout: float = 0.0

    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 4096


@dataclass
class ModelArgs(BaseModelArgs):
    audio_encoder_args: Type[AudioEncoderModelArgs] = AudioEncoderModelArgs
    audio_decoder_args: Type[AudioDecoderModelArgs] = AudioDecoderModelArgs
    vision_encoder_args: Type[VisionEncoderModelArgs] = VisionEncoderModelArgs

    temporal_transformer_args: Type[TemporialTransformer] = TemporialTransformer
    depth_transformer_args: Type[DepthTransformer] = DepthTransformer

    inference_args: Type[InferenceArgs] = InferenceArgs

    stfu_token_id: int = 0

    vocab_size: int = 128256

    tokenizer_path: Path = field(default=Path('/Users/gokdenizgulmez/Desktop/J.O.S.I.E./tokenizer.model'))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

def turn_to_token(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Convert logits to tokens with temperature sampling
    Args:
        logits: Shape [B, L, vocab_size]
        temperature: Controls randomness in sampling (higher = more random)
    Returns:
        tokens: Shape [B, L, 1]
    """
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # Apply temperature scaling
    logits = logits / temperature
    
    # Sample from the distribution
    probs = F.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
    return tokens.view(*logits.shape[:-1], 1)


##### Transformer
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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.pos_embedding = self._create_rotary_embedding()

        self.layers = nn.ModuleList([
            TransformerBlock(self.args, layer_index=idx) for idx in range(self.args.hidden_layers)
        ])

        self.norm = RMSNorm(self.args.hidden_size, self.args.rms_norm_eps)

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
        B, L, D = x.shape # self.args.hidden_size

        positions = self.pos_embedding[:, :L, :]

        x = x + positions

        mask = None
        if L > 1:
            mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
            mask = mask.unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x


# Quantizers
class VectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int
    ):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)

    def forward(self, x):
        """Encode vectors to tokens"""
        distances = torch.cdist(x, self.codebook.weight)
        indices = distances.argmin(dim=-1) # tokens
        quantized = self.codebook(indices) # vectors
        return indices, quantized

    def decode(self, tokens):
        """Convert tokens back to vectors using codebook"""
        # tokens shape: [B, T]
        vectors = self.codebook(tokens) # Look up vectors from codebook
        return vectors


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        num_quantizers: int
    ):
        super().__init__()
        self.quantizers = nn.ModuleList([
            VectorQuantizer(dim, codebook_size)
            for _ in range(num_quantizers)
        ])

    def forward(self, x):
        """Encode vectors to tokens through multiple quantizers"""
        B, L, D = x.shape
        quantized = torch.zeros_like(x)
        indices_list = []
        residual = x

        for quantizer in self.quantizers:
            idx, quant = quantizer(residual)
            indices_list.append(idx)
            quantized = quantized + quant
            residual = residual - quant

        # Stack the indices into a single tensor [B, num_quantizers, T]
        indices = torch.stack(indices_list, dim=1)
        return indices, quantized

    def decode(self, tokens):
        """Convert multi-level tokens back to vectors"""
        # tokens is a list of [B, T] tensors, one per quantizer
        quantized = torch.zeros_like(self.quantizers[0].codebook(tokens[0]))

        for quantizer, level_tokens in zip(self.quantizers, tokens):
            quant = quantizer.decode(level_tokens)
            quantized = quantized + quant

        return quantized


# Encoders
class SeaNetEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Add initial projection to convert from 1 channel to dim channels
        self.input_proj = nn.Conv1d(
            in_channels=1,
            out_channels=dim,
            kernel_size=3,
            padding=1
        )
        self.input_proj = nn.utils.parametrizations.weight_norm(self.input_proj)

        self.conv_blocks = nn.ModuleList([
            ConvBlock(stride=4, channels=dim),
            ConvBlock(stride=5, channels=dim),
            ConvBlock(stride=6, channels=dim),
            ConvBlock(stride=8, channels=dim)
        ])

        self.final_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=2
        )
        self.final_conv = nn.utils.parametrizations.weight_norm(self.final_conv)

    def forward(self, x):
        # x shape: [B, 1, T] # Raw audio
        x = self.input_proj(x)  # [B, dim, T]

        # Process through conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Final downsampling
        x = self.final_conv(x)  # [B, dim, T]
        
        # Transpose to [B, T, dim] format for transformer
        x = x.transpose(1, 2)

        return x.contiguous()


class SeaNetDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Initial upsampling layer
        self.initial_upsample = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.initial_upsample = nn.utils.parametrizations.weight_norm(self.initial_upsample)

        # Transposed convolution blocks
        self.conv_blocks = nn.ModuleList([
            TransposedConvBlock(stride=8, channels=dim),
            TransposedConvBlock(stride=6, channels=dim),
            TransposedConvBlock(stride=5, channels=dim),
            TransposedConvBlock(stride=4, channels=dim)
        ])

        # Changed final projection to generate single channel audio
        self.final_proj = nn.Conv1d(
            in_channels=dim,
            out_channels=1,  # Output single channel audio
            kernel_size=3,
            padding=1
        )
        self.final_proj = nn.utils.parametrizations.weight_norm(self.final_proj)

    def forward(self, x):
        # x shape: [B, T, dim]
        x = x.transpose(1, 2)  # [B, dim, T]
        
        # Initial upsampling
        x = self.initial_upsample(x)
        
        # Process through transposed conv blocks
        for block in self.conv_blocks:
            x = block(x)
            
        # Final projection to waveform
        return self.final_proj(x)  # [B, 1, T]


# Update ConvBlock and TransposedConvBlock to use dynamic channel size
class ConvBlock(nn.Module):
    def __init__(
        self,
        stride: int,
        channels: int,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        dilation_growth: int = 2
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dilation = 1

        for _ in range(num_conv_layers):
            # Calculate padding to maintain same output size
            total_padding = (kernel_size - 1) * current_dilation
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            
            # Create padding layer
            pad_layer = nn.ConstantPad1d((left_padding, right_padding), 0)
            
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=current_dilation,
                padding=0  # We'll use explicit padding
            )
            conv = nn.utils.parametrizations.weight_norm(conv)

            layer = nn.Sequential(
                pad_layer,
                conv,
                nn.SiLU(),
            )
            self.layers.append(layer)
            current_dilation *= dilation_growth

        # Downsample with proper padding
        downsample_padding = kernel_size // 2
        self.downsample = nn.Sequential(
            nn.ConstantPad1d((downsample_padding, downsample_padding), 0),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0
            )
        )
        self.downsample[1] = nn.utils.parametrizations.weight_norm(self.downsample[1])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
        x = self.downsample(x) # (B, D, L)
        return x


class TransposedConvBlock(nn.Module):
    def __init__(
        self,
        stride: int,
        channels: int,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        dilation_growth: int = 2
    ):
        super().__init__()
        
        # Upsample layer
        self.upsample = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,  # Changed padding
            output_padding=stride-1
        )
        self.upsample = nn.utils.parametrizations.weight_norm(self.upsample)

        self.layers = nn.ModuleList()
        current_dilation = dilation_growth**(num_conv_layers-1)

        for _ in range(num_conv_layers):
            # Use sequential with padding layer to ensure sizes match
            total_padding = (kernel_size - 1) * current_dilation
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            
            pad_layer = nn.ConstantPad1d((left_padding, right_padding), 0)
            
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=current_dilation,
                padding=0  # We'll use explicit padding
            )
            conv = nn.utils.parametrizations.weight_norm(conv)

            layer = nn.Sequential(
                pad_layer,
                conv,
                nn.SiLU()
            )
            
            self.layers.append(layer)
            current_dilation = current_dilation // dilation_growth

    def forward(self, x):
        x = self.upsample(x)
        
        # Store original upsampled tensor
        orig_x = x
        
        # Apply convolution layers
        for layer in self.layers:
            x = layer(x)
        
        # Add residual connection with size check
        if x.size(2) != orig_x.size(2):
            # Pad or trim to match sizes
            if x.size(2) > orig_x.size(2):
                x = x[..., :orig_x.size(2)]
            else:
                orig_x = orig_x[..., :x.size(2)]
        
        x = x + orig_x
        return x


# Audio Tokenizer
class JODIO(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_args = args.audio_encoder_args()
        self.decoder_args = args.audio_decoder_args()

        # Encoder
        self.encoder = SeaNetEncoder(self.encoder_args.hidden_size) # [B, T, dim]
        self.encoder_transformer = Transformer(self.encoder_args)

        # Vector Quantizers
        self.semantic_rvq = VectorQuantizer(
            dim=self.encoder_args.hidden_size,
            codebook_size=self.encoder_args.codebook_size,
            # num_quantizers=self.encoder_args.num_semantic_quantizers # 2 output Tokens
        )
        self.acoustic_rvq = ResidualVectorQuantizer(
            dim=self.encoder_args.hidden_size,
            codebook_size=self.encoder_args.codebook_size,
            num_quantizers=self.encoder_args.num_acoustic_quantizers # 8 output Tokens
        )

        # Token embeddings for decoder
        self.text_embedding = nn.Embedding(
            args.vocab_size,
            self.decoder_args.hidden_size
        )
        self.semantic_embedding = nn.Embedding(
            self.encoder_args.codebook_size,
            self.decoder_args.hidden_size
        )
        self.acoustic_embedding = nn.Embedding(
            self.encoder_args.codebook_size,
            self.decoder_args.hidden_size
        )

        # Decoder
        self.decoder_transformer = Transformer(self.decoder_args)
        self.decoder = SeaNetDecoder(self.decoder_args.hidden_size)

    def encode(self, waveform: torch.Tensor):
        """
        Convert waveform to semantic and acoustic tokens
        Args:
            waveform: Input audio at 24kHz [B, 1, T]
        Returns:
            semantic_tokens: First VQ codebook indices [B, T]
            acoustic_tokens: List of 8 RVQ codebook indices [B, T]
        """
        # Ensure input has batch dimension
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
            
        # Encode to latent space
        x = self.encoder(waveform)  # [B, T, D]
        x = self.encoder_transformer(x)  # [B, T, D]

        # Vector quantization
        semantic_tokens, _ = self.semantic_rvq(x)
        acoustic_tokens, _ = self.acoustic_rvq(x)
        
        # Remove batch dimension if it was added
        if waveform.size(0) == 1:
            semantic_tokens = semantic_tokens.squeeze(0).flatten()
            acoustic_tokens = acoustic_tokens.squeeze(0).flatten()

        return semantic_tokens, acoustic_tokens

    def decode(self, text_tokens: torch.Tensor, semantic_tokens: torch.Tensor, acoustic_tokens: torch.Tensor):
        """
        Convert tokens back to waveform
        Args:
            text_tokens: [B, 1, 1]
            semantic_tokens: [B, 1, 1]
            acoustic_tokens: [B, 1, num_acoustic_quantizers-1]
        Returns:
            waveform: Reconstructed audio at 24kHz [B, 1, T]
        """
        # Embed each type of token
        text_emb = self.text_embedding(text_tokens.squeeze(-1))  # [B, 1, H]
        semantic_emb = self.semantic_embedding(semantic_tokens.squeeze(-1))  # [B, 1, H]
        
        # Handle acoustic tokens separately since they have multiple quantizers
        acoustic_embs = []
        for i in range(acoustic_tokens.size(-1)):
            acoustic_emb = self.acoustic_embedding(acoustic_tokens[..., i])  # [B, 1, H]
            acoustic_embs.append(acoustic_emb)
        
        # Combine all embeddings
        combined = torch.cat([text_emb, semantic_emb] + acoustic_embs, dim=1)  # [B, L, H]
        
        # Pass through transformer and decoder
        x = self.decoder_transformer(combined)  # [B, L, H]
        waveform = self.decoder(x)  # [B, 1, T]
        
        return waveform



# Vision Tokenizer
class MultimodalRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_spatial_pos: int = 256,  # Maximum spatial dimension
        max_temporal_pos: int = 32,   # Maximum temporal dimension
        theta: float = 10000.0,
        learned: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.max_spatial_pos = max_spatial_pos
        self.max_temporal_pos = max_temporal_pos

        # Create position embeddings for spatial dimensions
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        
        # Spatial positions (height and width)
        pos_h = torch.arange(max_spatial_pos, dtype=torch.float)
        pos_w = torch.arange(max_spatial_pos, dtype=torch.float)
        
        # Temporal positions
        pos_t = torch.arange(max_temporal_pos, dtype=torch.float)
        
        # Calculate embeddings
        sincos_h = torch.einsum('i,j->ij', pos_h, inv_freq)
        sincos_w = torch.einsum('i,j->ij', pos_w, inv_freq)
        sincos_t = torch.einsum('i,j->ij', pos_t, inv_freq)
        
        # Combine sin and cos
        emb_h = torch.cat((sincos_h.sin(), sincos_h.cos()), dim=-1)
        emb_w = torch.cat((sincos_w.sin(), sincos_w.cos()), dim=-1)
        emb_t = torch.cat((sincos_t.sin(), sincos_t.cos()), dim=-1)
        
        # Make learnable if specified
        if learned:
            self.emb_h = nn.Parameter(emb_h)
            self.emb_w = nn.Parameter(emb_w)
            self.emb_t = nn.Parameter(emb_t)
        else:
            self.register_buffer('emb_h', emb_h)
            self.register_buffer('emb_w', emb_w)
            self.register_buffer('emb_t', emb_t)

    def forward(self, h: int, w: int, t: int = 1):
        """
        Apply rotary embeddings to positions
        Args:
            h: Height
            w: Width 
            t: Number of frames (temporal dimension)
        Returns:
            Tuple of embeddings for each dimension
        """
        emb_h = self.emb_h[:h]
        emb_w = self.emb_w[:w]
        emb_t = self.emb_t[:t] if t > 1 else None
        
        return emb_h, emb_w, emb_t


class SpatioTemporalPatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        hidden_size: int = 512,
        temporal_patch_size: int = 2,
        use_conv: bool = True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        
        if use_conv:
            # Use convolution for patch embedding
            self.proj = nn.Conv3d(
                in_channels,
                hidden_size,
                kernel_size=(temporal_patch_size, patch_size, patch_size),
                stride=(temporal_patch_size, patch_size, patch_size)
            )
        else:
            # Use linear projection
            self.proj = nn.Linear(
                in_channels * patch_size * patch_size * temporal_patch_size,
                hidden_size
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Video tensor of shape [B, C, T, H, W]
        Returns:
            Embedded patches of shape [B, L, hidden_size]
            where L = (T/tp) * (H/p) * (W/p), tp = temporal_patch_size, p = patch_size
        """
        B, C, T, H, W = x.shape
        
        # Project patches
        x = self.proj(x)
        
        # Reshape to sequence
        if isinstance(self.proj, nn.Conv3d):
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x.reshape(B, T // self.temporal_patch_size,
                         H // self.patch_size,
                         W // self.patch_size, -1)
            x = x.flatten(1, 3)  # [B, L, hidden_size]
            
        return x


class JOVIO(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vision_args = args.vision_encoder_args()
        
        # Patch embedding
        self.patch_embed = SpatioTemporalPatchEmbedding(
            in_channels=3,  # RGB
            patch_size=16,
            hidden_size=self.vision_args.hidden_size,
            temporal_patch_size=2,
            use_conv=True
        )
        
        # Multimodal rotary embeddings
        self.mrope = MultimodalRotaryEmbedding(
            dim=self.vision_args.hidden_size,
            max_spatial_pos=self.vision_args.max_position_embeddings,
            max_temporal_pos=self.vision_args.max_frames,
            theta=self.vision_args.rope_theta
        )
        
        # Transformer encoder
        self.transformer = Transformer(self.vision_args)
        
        # Quantizer for converting embeddings to tokens
        self.quantizer = ResidualVectorQuantizer(
            dim=self.vision_args.hidden_size,
            codebook_size=self.vision_args.codebook_size,
            num_quantizers=self.vision_args.num_quantizers
        )
        
    def forward(self, x: torch.Tensor):
        """
        Encode video/image to tokens
        Args:
            x: Video tensor [B, C, T, H, W] or image tensor [B, C, H, W]
        Returns:
            tokens: Quantized tokens from RVQ
        """
        # Add temporal dimension for images
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B, C, 1, H, W]
            
        B, C, T, H, W = x.shape
        
        # Embed patches
        x = self.patch_embed(x)  # [B, L, D]
        
        # Get positional embeddings
        h_emb, w_emb, t_emb = self.mrope(
            H // self.patch_embed.patch_size,
            W // self.patch_embed.patch_size,
            T // self.patch_embed.temporal_patch_size
        )
        
        # Apply positional embeddings
        # Reshape to separate dimensions
        L = x.shape[1]
        x = x.view(B, T // self.patch_embed.temporal_patch_size,
                  H // self.patch_embed.patch_size,
                  W // self.patch_embed.patch_size, -1)
        
        # Apply embeddings to each dimension
        x = x + h_emb.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        x = x + w_emb.unsqueeze(1).unsqueeze(0).unsqueeze(0)
        if T > 1:
            x = x + t_emb.unsqueeze(2).unsqueeze(2).unsqueeze(0)
            
        # Reshape back to sequence
        x = x.view(B, L, -1)
        
        # Transform
        x = self.transformer(x)
        
        # Quantize to tokens
        tokens, _ = self.quantizer(x)
        
        return tokens

    def encode(self, x: torch.Tensor):
        """
        Encode video/image to tokens (alias for forward)
        """
        return self.forward(x)


# Main Transformer for JOSIE model
class JOSIEAttention(nn.Module):
    def __init__(self, args: ModelArgs):
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
            scale=self.scale
        )

        out = attn.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class JOSIEMultiLayerPerception(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.linear1 = nn.Linear(self.args.hidden_size, 4 * self.args.hidden_size, bias=False)
        self.linear2 = nn.Linear(4 * self.args.hidden_size, self.args.hidden_size, bias=False)
        self.linear3 = nn.Linear(self.args.hidden_size, 4 * self.args.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(x) * self.linear3(x)))


class JOSIETransformerBlock(nn.Module):
    def __init__(self, args, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.args = args

        self.attention = JOSIEAttention(args)

        self.feed_forward = JOSIEMultiLayerPerception(args)

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


class JOSIETransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.pos_embedding = self._create_rotary_embedding()

        self.layers = nn.ModuleList([
            JOSIETransformerBlock(self.args, layer_index=idx) for idx in range(self.args.hidden_layers)
        ])

        self.norm = RMSNorm(self.args.hidden_size, self.args.rms_norm_eps)

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
        B, L, D = x.shape

        positions = self.pos_embedding[:, :L, :]
        x = x + positions

        mask = None
        if L > 1:
            mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
            mask = mask.unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x
    

# LLM
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

        # Single embedding for vision tokens
        self.vision_embedding = nn.Embedding(
            args.vision_encoder_args.codebook_size,  # Usually 2048
            self.temporal_args.hidden_size
        )

        # Main transformer
        self.transformer = JOSIETransformer(self.temporal_args)
        
        # Norms
        self.norm = RMSNorm(self.temporal_args.hidden_size)
        self.final_norm = RMSNorm(self.temporal_args.hidden_size)

    def forward(
            self,
            text_tokens: Optional[torch.Tensor] = None,  # [B, L]
            vision_tokens: Optional[torch.Tensor] = None,  # [B, L, C, D]
            semantic_tokens: Optional[torch.Tensor] = None,  # [L] 
            acoustic_tokens: Optional[torch.Tensor] = None  # [L]
    ) -> torch.Tensor:
        batch_size = 1  # Default batch size if no tokens provided
        device = next(self.parameters()).device
        hidden_states = []

        # Process text tokens if provided
        if text_tokens is not None:
            batch_size = text_tokens.size(0)
            text_emb = self.text_embedding(text_tokens)  # [B, L1, H]
            hidden_states.append(text_emb)
        
        # Process vision tokens if provided
        if vision_tokens is not None:
            batch_size = vision_tokens.size(0)
            # Handle multiple quantizer levels
            vision_embs = []
            for i in range(vision_tokens.size(1)):  # Iterate over quantizer levels
                emb = self.vision_embedding(vision_tokens[:, i])
                vision_embs.append(emb)
            vision_emb = torch.cat(vision_embs, dim=1)  # Combine all quantizer embeddings
            hidden_states.append(vision_emb)

        # Process semantic tokens if provided
        if semantic_tokens is not None:
            semantic_emb = self.semantic_embedding(semantic_tokens)
            # Expand to match batch size if needed
            if semantic_emb.dim() == 2:  # [L2, H]
                semantic_emb = semantic_emb.unsqueeze(0).expand(batch_size, -1, -1)
            hidden_states.append(semantic_emb)

        # Process acoustic tokens if provided
        if acoustic_tokens is not None:
            acoustic_emb = self.acoustic_embedding(acoustic_tokens)
            # Expand to match batch size if needed
            if acoustic_emb.dim() == 2:  # [L3, H]
                acoustic_emb = acoustic_emb.unsqueeze(0).expand(batch_size, -1, -1)
            hidden_states.append(acoustic_emb)

        # Handle case where no tokens are provided
        if not hidden_states:
            # Return a zero tensor with correct shape
            return torch.zeros(
                batch_size, 
                1, 
                self.temporal_args.hidden_size, 
                device=device
            )

        # Concatenate along sequence length dimension
        hidden = torch.cat(hidden_states, dim=1)  # [B, L1+L2+L3, H]

        # Apply transformer
        hidden = self.norm(hidden)
        hidden = self.transformer(hidden)
        return self.final_norm(hidden)


# For Semantic, Acustic, and text Token prediction
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
        
        self.transformer = JOSIETransformer(self.depth_args)
        
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


# Real-time Audio Full Duplex Dialog Model
class JOSIE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.jodio = JODIO(args)
        self.jovio = JOVIO(args)

        self.temporial_transformer = TemporalTransformer(args)
        self.depth_transformer = DepthTransformer(args)
    
    def forward(
            self,
            user_waveform: torch.Tensor,
            text_tokens: Optional[torch.tensor] = None,
            user_images: Optional[torch.tensor] = None
        ):
        semantic_token, acoustic_tokens = self.jodio.encode(user_waveform)

        # Handle vision encoding
        vision_tokens = None
        if user_images is not None:
            vision_tokens = self.jovio(user_images)

        temporal_context = self.temporial_transformer(
            text_tokens = text_tokens,
            vision_tokens = vision_tokens,
            semantic_tokens = semantic_token,
            acoustic_tokens = acoustic_tokens
        )

        text_token, semantic_token, acoustic_tokens = self.depth_transformer(temporal_context)
        josies_waveform = self.jodio.decode(text_token, semantic_token, acoustic_tokens)

        return text_token, semantic_token, acoustic_tokens, josies_waveform
