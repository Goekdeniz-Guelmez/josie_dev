from typing import List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import AudioRQTransformerArgs as ModelArgs


# TODO add Temporal, Depth codeblocks and Transformers, maybe add Spectral
# TODO add AudioQuantizer class


class AudioQuantizer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()









        
class DualAudioStreamRQTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
    
    def forward(input: torch.Tensor):
        return None