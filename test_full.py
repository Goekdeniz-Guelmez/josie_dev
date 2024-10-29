import torch
import pyaudio
import numpy as np
from args import ModelArgs
from audio import AudioEncoderDecoder
from reasoner import ReasonerTransformer

# Initialize AudioQuantizer with given arguments
audio = AudioEncoderDecoder(ModelArgs)
model = ReasonerTransformer(ModelArgs)

# Audio stream configuration
CHUNK = 16000 // 4  # 250 ms for 16 kHz audio
FORMATIN = pyaudio.paInt16
FORMATOUT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000  # 16 kHz sample rate

# Initialize PyAudio
p = pyaudio.PyAudio()
streamin = p.open(format=FORMATIN, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
streamut = p.open(format=FORMATOUT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

# Continuous audio processing loop
print("Recording audio...")
try:
    while True:
        audio_data = streamin.read(CHUNK)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalized to [-1, 1]
        audio_tensor = torch.from_numpy(audio_np).view(1, -1)  # Flatten to [1, CHUNK]
        if audio_tensor.size(1) < ModelArgs.encoder_hidden_dim:
            audio_tensor = torch.cat([audio_tensor] * (ModelArgs.encoder_hidden_dim // audio_tensor.size(1) + 1), dim=1)
        audio_tensor = audio_tensor[:, :ModelArgs.encoder_hidden_dim]  # Trim to exactly hidden_dim
        audio_tensor = audio_tensor.view(1, 1, ModelArgs.encoder_hidden_dim)  # Final shape [1, T, hidden_dim]

        discrete_audio_tokens = audio.encode(audio_tensor) # -> torch.Size([1, 1, 8])

        text_stream, audio_stream = model(discrete_audio_tokens.squeeze(0)) # -> torch.Size([1, 8, 131000]), torch.Size([1, 8, 256])

        output = audio.decode(audio_stream)
        audio_output = output.squeeze().detach().numpy()
        streamut.write(audio_output.astype(np.float32).tobytes())

        next_token = torch.argmax(text_stream[:, -1, :], dim=-1)
        print(f'{next_token.item()} ', end='', flush=True)

except KeyboardInterrupt:
    print("Audio processing stopped.")
finally:
    # Close the stream gracefully
    streamin.stop_stream()
    streamin.close()
    p.terminate()