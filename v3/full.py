from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import ModelArgs


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


##################################### RVQ
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
        indices = [] # Will contain acoustic_tokens
        residual = x

        for quantizer in self.quantizers:
            idx, quant = quantizer(residual)
            indices.append(idx)
            quantized = quantized + quant
            residual = residual - quant

        return indices, quantized

    def decode(self, tokens):
        """Convert multi-level tokens back to vectors"""
        # tokens is a list of [B, T] tensors, one per quantizer
        quantized = torch.zeros_like(self.quantizers[0].codebook(tokens[0]))

        for quantizer, level_tokens in zip(self.quantizers, tokens):
            quant = quantizer.decode(level_tokens)
            quantized = quantized + quant

        return quantized


##################################### Encoder Transformer
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


##################################### SEANETS
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
        x = self.final_conv(x)

        return x.transpose(1, 2).contiguous()  # Return [B, T, D] for transformer


class SeaNetDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.initial_upsample = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=2,
            output_padding=1
        )
        self.initial_upsample = nn.utils.parametrizations.weight_norm(self.initial_upsample)

        self.conv_blocks = nn.ModuleList([
            TransposedConvBlock(stride=8, channels=dim),
            TransposedConvBlock(stride=6, channels=dim),
            TransposedConvBlock(stride=5, channels=dim),
            TransposedConvBlock(stride=4, channels=dim)
        ])

        self.final_proj = nn.Conv1d(
            in_channels=dim,
            out_channels=1,  # Single channel audio output
            kernel_size=3,
            padding=2
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

        # Project to waveform
        return self.final_proj(x)  # Return [B, L(1), D(T*480)]


##################################### CONVS
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

        self.upsample = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size-1,
            output_padding=stride-1
        )
        self.upsample = nn.utils.parametrizations.weight_norm(self.upsample)

        self.layers = nn.ModuleList()
        current_dilation = dilation_growth**(num_conv_layers-1)

        for _ in range(num_conv_layers):
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * current_dilation,
                dilation=current_dilation
            )
            conv = nn.utils.parametrizations.weight_norm(conv)

            layer = nn.Sequential(
                conv,
                nn.SiLU()
            )
            self.layers.append(layer)
            current_dilation = current_dilation // dilation_growth

    def forward(self, x):
        x = self.upsample(x)
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
        return x


##################################### Main Speech Tokenizer
class JODIO(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.encoder_args = args.audio_encoder_args()
        self.decoder_args = args.audio_decoder_args()

        # Encoder
        self.encoder = SeaNetEncoder(self.encoder_args.hidden_size) # [B, T, dim]
        self.encoder_transformer = Transformer(self.encoder_args)

        # Vector Quantizers
        self.semantic_rvq = ResidualVectorQuantizer(
            dim=self.encoder_args.hidden_size,
            codebook_size=self.encoder_args.codebook_size,
            num_quantizers=self.encoder_args.num_semantic_quantizers # 2 output Tokens
        )
        self.acoustic_rvq = ResidualVectorQuantizer(
            dim=self.encoder_args.hidden_size,
            codebook_size=self.encoder_args.codebook_size,
            num_quantizers=self.encoder_args.num_acoustic_quantizers # 8 output Tokens
        )

        # Decoder
        self.post_vq_proj = nn.Linear(6, self.encoder_args.hidden_size)
        self.decoder_transformer = Transformer(self.decoder_args)
        self.decoder = SeaNetDecoder(self.encoder_args.hidden_size)

    def encode(self, waveform: torch.Tensor):
        """
        Convert waveform to semantic and acoustic tokens
        Args:
            waveform: Input audio at 24kHz [B, 1, T]
        Returns:
            semantic_tokens: First VQ codebook indices [B, T]
            acoustic_tokens: List of 8 RVQ codebook indices [B, T]
        """
        # Encode to latent space
        x = self.encoder(waveform) # [B, D, L]
        x = self.encoder_transformer(x)

        # Vector quantization
        semantic_tokens, _ = self.semantic_rvq(x)
        acoustic_tokens, _ = self.acoustic_rvq(x)

        return semantic_tokens, acoustic_tokens

    def decode(self, semantic_and_acoustic_tokens):
        """
        Convert tokens back to waveform
        Args:
            semantic_and_acoustic_tokens: Indices from semantic and acoustic [B, T], where T has a dimension 10 (8 Acoustic and 2 Semantic).
        Returns:
            waveform: Reconstructed audio at 24kHz [B, 1, T]
        """
        # Convert tokens to float before passing through linear layer
        x = semantic_and_acoustic_tokens.float()
        x = self.post_vq_proj(x)
        # Decode through transformer
        x = self.decoder_transformer(x)
        # Generate waveform
        return self.decoder(x)
    

args = ModelArgs()

num_samples = int(args.inference_args.rate * args.inference_args.record_seconds)


model = JODIO(ModelArgs())
# print(model)

# Shape: [batch=1, channels=1, time]
waveform = torch.randn(1, args.inference_args.channels, num_samples)

while True:
    semantic_tokens, acoustic_tokens = model.encode(waveform)
    print(semantic_tokens)
    print(acoustic_tokens)