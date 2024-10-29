from typing import Tuple

import torch
import torch.nn as nn

from reasoner import ReasonerTransformer
from audio import AudioEncoderDecoder
from args import ModelArgs

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