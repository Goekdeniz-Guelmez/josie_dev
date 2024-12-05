import torch

from jodio.JODIO import JODIO
from args import ModelArgs

args = ModelArgs()
model = JODIO(args)

# print(model)

model.eval()  # Set to evaluation mode to freeze the model layers

num_samples = int(args.inference_args.rate * args.inference_args.record_seconds)

test_encode_amount = 4
test_decode_amount = 12

waveform = torch.randn(1, args.inference_args.channels, num_samples) # Shape: [batch=1, channels=1, time]
reasoner_audio_out_linear_output = torch.randn(1, 8, args.audio_decoder_args.hidden_size)

while True:
    for ie in range(test_encode_amount):
        semantic_tokens, acoustic_tokens, combined_tokens = model.encode(waveform)
        print('------------------------------')
        print(combined_tokens)
    
    for id in range(test_decode_amount):
        output_waveform = model.decode(reasoner_audio_out_linear_output)
        print('------------------------------')
        print(output_waveform)
