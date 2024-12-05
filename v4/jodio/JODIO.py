import torch
import torch.nn as nn

from ..args import ModelArgs

from layers.seanet import SeaNetEncoder, SeaNetDecoder
from layers.quantizer import ResidualVectorQuantizer, VectorQuantizer
from layers.transformer import Transformer


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
        # Encode to latent space
        x = self.encoder(waveform) # [B, D, L]
        x = self.encoder_transformer(x)

        # Vector quantization
        semantic_tokens, _ = self.semantic_rvq(x)
        acoustic_tokens, _ = self.acoustic_rvq(x)
        semantic_tokens = semantic_tokens.squeeze(0).flatten()
        acoustic_tokens = acoustic_tokens.squeeze(0).flatten()
        combined_tokens = torch.cat([semantic_tokens, acoustic_tokens], dim=0)

        return semantic_tokens, acoustic_tokens, combined_tokens

    def decode(self, x: torch.tensor):
        """
        Convert tokens back to waveform
        Args:
            semantic_and_acoustic_tokens: Indices from semantic and acoustic [B, T], where T has a dimension 10 (8 Acoustic and 2 Semantic).
        Returns:
            waveform: Reconstructed audio at 24kHz [B, 1, T]
        """
        # Decode through transformer
        x = self.decoder_transformer(x)
        # Generate waveform
        return self.decoder(x).squeeze(0)
    
    def forward(self, waveform: torch.Tensor) -> torch.tensor:
        _, _, combined_tokens = self.encode(waveform)
        return self.decoder_transformer(combined_tokens)