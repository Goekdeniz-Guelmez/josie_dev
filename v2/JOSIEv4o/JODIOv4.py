import torch
import torch.nn as nn
import torch.nn.functional as F

from JOSIEv4o.args import ModelArgs

from JOSIEv4o.seanets import SeaNetEncoder, SeaNetDecoder
from JOSIEv4o.transformer import Transformer
from JOSIEv4o.quantizer import VectorQuantizer, ResidualVectorQuantizer


class JODIO(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.encoder_args = args.audio_encoder_args
        self.decoder_args = args.audio_decoder_args

        # Encoder
        self.seanet_encoder = SeaNetEncoder(self.args.hidden_size)
        self.encoder_transformer = Transformer(self.encoder_args)

        self.pre_vq_proj = nn.Linear(self.args.hidden_size, self.encoder_args.hidden_size)

        self.semantic_vq = VectorQuantizer(
            dim=self.encoder_args.hidden_size,
            codebook_size=self.encoder_args.codebook_size
        )
        self.acoustic_rvq = ResidualVectorQuantizer(
            dim=self.encoder_args.hidden_size,
            codebook_size=self.encoder_args.codebook_size
        )

        # Decoder
        self.post_vq_proj = nn.Linear(self.encoder_args.hidden_size, self.args.hidden_size)

        self.decoder_transformer = Transformer(self.decoder_args)
        self.decoder = SeaNetDecoder(self.args.hidden_size)

    def encode(
        self,
        waveform: torch.Tensor
    ):
        """
        Convert waveform to semantic and acoustic tokens

        Args:
            waveform: Input audio at 24kHz

        Returns:
            semantic_tokens: First VQ codebook indices [B, T]
            acoustic_tokens: List of 7 RVQ codebook indices [B, T]
        """
        # Encode to latent space
        x = self.encoder(waveform)
        x = self.bottleneck_transformer(x)
        x = self.pre_vq_proj(x)

        # Split into semantic and acoustic paths
        semantic_tokens, semantic_vectors = self.semantic_vq(x)
        acoustic_tokens, acoustic_vectors = self.acoustic_rvq(x)

        return semantic_tokens, acoustic_tokens

    def decode(self, semantic_tokens, acoustic_tokens):
        """
        Convert tokens back to waveform

        Args:
            semantic_tokens: Indices from semantic VQ [B, T]
            acoustic_tokens: List of indices from acoustic RVQ [B, T]

        Returns:
            waveform: Reconstructed audio at 24kHz
        """
        # Convert tokens back to vectors
        semantic_vectors = self.semantic_vq.decode(semantic_tokens)
        acoustic_vectors = self.acoustic_rvq.decode(acoustic_tokens)

        # Combine vectors
        x = semantic_vectors + acoustic_vectors
        x = self.post_vq_proj(x)

        # Decode through transformer and decoder
        x = self.decoder_transformer(x)
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
