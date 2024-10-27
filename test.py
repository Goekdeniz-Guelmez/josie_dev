import torch

from args import AudioRQTransformerArgs
from audio import AudioQuantizer

model = AudioQuantizer(AudioRQTransformerArgs)

print(model)

inp = torch.randn(1, 1, 512)
quantized, indices = model.quantize_spectral(inp)

print(quantized.shape)