import torch
import pyaudio
import numpy as np
from args import ModelArgs
from josie import JOSIE

from tokenizer import Tokenizer

tokenizer = Tokenizer('/Users/gokdenizgulmez/Desktop/J.O.S.I.E./tokenizer.model')

# Initialize AudioQuantizer with given arguments
model = JOSIE(ModelArgs)

FORMATIN = pyaudio.paInt16
FORMATOUT = pyaudio.paFloat32

# Initialize PyAudio
p = pyaudio.PyAudio()
streamin = p.open(format=FORMATIN, channels=ModelArgs.encoder_audio_channels, rate=ModelArgs.encoder_audio_rate, input=True, frames_per_buffer=ModelArgs.encoder_audio_chunk)
streamut = p.open(format=FORMATOUT, channels=ModelArgs.encoder_audio_channels, rate=ModelArgs.encoder_audio_rate, output=True, frames_per_buffer=ModelArgs.encoder_audio_chunk)

# Continuous audio processing loop
print("Recording audio...")
try:
    while True:
        audio_data = streamin.read(ModelArgs.encoder_audio_chunk)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalized to [-1, 1]
        audio_tensor = torch.from_numpy(audio_np).view(1, -1)  # Flatten to [1, CHUNK]
        if audio_tensor.size(1) < ModelArgs.encoder_audio_hidden_dim:
            audio_tensor = torch.cat([audio_tensor] * (ModelArgs.encoder_audio_hidden_dim // audio_tensor.size(1) + 1), dim=1)
        audio_tensor = audio_tensor[:, :ModelArgs.encoder_audio_hidden_dim]  # Trim to exactly hidden_dim
        audio_tensor = audio_tensor.view(1, 1, ModelArgs.encoder_audio_hidden_dim)  # Final shape [1, T, hidden_dim]

        audio_output, next_token = model(audio_tensor)

        streamut.write(audio_output.astype(np.float32).tobytes())
        print(f'{tokenizer.decode([next_token.item()])}', end='', flush=True)

except KeyboardInterrupt:
    print("Audio processing stopped.")
finally:
    # Close the stream gracefully
    streamin.stop_stream()
    streamin.close()
    p.terminate()