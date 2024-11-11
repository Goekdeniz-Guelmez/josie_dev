import torch

from JODIO import JODIO
from args import ModelArgs

args = ModelArgs()

model = JODIO(args)

# print(model)

model.eval()  # Set to evaluation mode

# Test with dummy input
batch_size = 2
duration_seconds = 1
sample_rate = 24000
audio_length = duration_seconds * sample_rate
dummy_waveform = torch.randn(batch_size, 1, audio_length)

# Forward pass
output = model(dummy_waveform)
print(output)