from typing import List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import AudioRQTransformer as ModelArgs


class DualAudioStreamRQTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args