import torch
import torch.nn as nn

from args import ModelArgs

from seanets import SeaNetEncoder, SeaNetDecoder
from transformer import Transformer
from quantizer import VectorQuantizer, ResidualVectorQuantizer

class JODIO(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.encoder_args = args.audio_encoder_args
        self.decoder_args = args.audio_decoder_args

        # Input projection to match encoder's expected channels
        self.input_proj = nn.Conv1d(
            in_channels=1,
            out_channels=self.args.hidden_size,
            kernel_size=3,
            padding=1
        )

        # Encoder
        self.encoder = SeaNetEncoder(self.args.hidden_size)
        self.encoder_transformer = Transformer(self.encoder_args)
        self.pre_vq_proj = nn.Linear(self.args.hidden_size, self.encoder_args.hidden_size)

        # Vector Quantizers
        self.semantic_vq = VectorQuantizer(
            dim=self.encoder_args.hidden_size,
            codebook_size=self.encoder_args.codebook_size
        )
        self.acoustic_rvq = ResidualVectorQuantizer(
            dim=self.encoder_args.hidden_size,
            codebook_size=self.encoder_args.codebook_size,
            num_quantizers=self.encoder_args.num_quantizers
        )

        # Decoder
        self.post_vq_proj = nn.Linear(6, self.args.hidden_size)
        self.decoder_transformer = Transformer(self.decoder_args)
        self.decoder = SeaNetDecoder(self.args.hidden_size)

    def encode(self, waveform: torch.Tensor):
        """
        Convert waveform to semantic and acoustic tokens
        Args:
            waveform: Input audio at 24kHz [B, 1, T]
        Returns:
            semantic_tokens: First VQ codebook indices [B, T]
            acoustic_tokens: List of 8 RVQ codebook indices [B, T]
        """
        # Project input to match encoder's channel dimension
        x = self.input_proj(waveform)  # [B, 1, T] -> [B, hidden_size, T]

        # Encode to latent space
        x = self.encoder(x)  # [B, hidden_size, T]

        # Transpose for transformer (expects [B, T, C])
        x = x.transpose(1, 2)

        x = self.encoder_transformer(x)
        x = self.pre_vq_proj(x)

        # Vector quantization
        semantic_tokens, _ = self.semantic_vq(x)
        acoustic_tokens, _ = self.acoustic_rvq(x)

        return semantic_tokens, acoustic_tokens

    def decode(self, semantic_and_acoustic_tokens):
        """
        Convert tokens back to waveform
        Args:
            semantic_and_acoustic_tokens: Indices from semantic and acoustic [B, T], where T has a dimension 6.
        Returns:
            waveform: Reconstructed audio at 24kHz [B, 1, T]
        """
        x = self.post_vq_proj(semantic_and_acoustic_tokens)

        # Decode through transformer
        x = self.decoder_transformer(x)

        # Transpose back to channel-first for decoder (expects [B, C, T])
        x = x.transpose(1, 2)

        # Generate waveform
        waveform = self.decoder(x)
        return waveform

    def forward(self, waveform):
        """
        Full forward pass: encode -> decode
        """
        semantic_tokens, acoustic_tokens = self.encode(waveform)
        reconstructed = self.decode(semantic_tokens, acoustic_tokens)
        return {
            'semantic_tokens': semantic_tokens,
            'acoustic_tokens': acoustic_tokens,
            'reconstructed': reconstructed
        }
