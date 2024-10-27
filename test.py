import torch

from args import AudioRQTransformerArgs
from audio import AudioQuantizer

model = AudioQuantizer(AudioRQTransformerArgs)

print(model)

inp = torch.randn(1, 1, 512)
quantized, indices = model(inp, stream_type='temporal')
print(quantized.shape)
quantized, indices = model(inp, stream_type='depth')
print(quantized.shape)
quantized, indices = model(inp, stream_type='spectral')
print(quantized.shape)