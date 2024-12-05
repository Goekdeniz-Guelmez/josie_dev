import torch

from jodio.JODIO import JODIO
from args import ModelArgs

args = ModelArgs()
model = JODIO(args)

# print(model)

model.eval()  # Set to evaluation mode to freeze the model layers

num_samples = int(args.inference_args.rate * args.inference_args.record_seconds)

waveform = torch.randn(1, args.inference_args.channels, num_samples) # Shape: [batch=1, channels=1, time]

out = model(waveform)
print(out)