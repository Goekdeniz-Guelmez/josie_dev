import pyaudio
import torch
import numpy as np
from args import AudioRQTransformerArgs
from audio import AudioEncoderDecoder

# Initialize AudioQuantizer with given arguments
model = AudioEncoderDecoder(AudioRQTransformerArgs)
print(model)

# Audio stream configuration
CHUNK = 16000 // 4  # 250 ms for 16 kHz audio
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1
RATE = 16000  # 16 kHz sample rate
INPUT_DIM = AudioRQTransformerArgs.encoder_head_dim  # Model's input dimension

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Continuous audio processing loop
print("Recording audio...")
try:
    while True:
        # Read audio data from stream
        audio_data = stream.read(CHUNK)
        
        # Convert audio data to tensor and normalize it
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalized to [-1, 1]
        audio_tensor = torch.from_numpy(audio_np).view(1, -1)  # Flatten to [1, CHUNK]

        # Adjust dimensions to match [B, T, hidden_dim]
        if audio_tensor.size(1) < model.hidden_dim:
            # If too short, tile or pad to reach hidden_dim
            audio_tensor = torch.cat([audio_tensor] * (model.hidden_dim // audio_tensor.size(1) + 1), dim=1)
        audio_tensor = audio_tensor[:, :model.hidden_dim]  # Trim to exactly hidden_dim
        audio_tensor = audio_tensor.view(1, 1, model.hidden_dim)  # Final shape [1, T, hidden_dim]

        # Temporal quantization
        discrete_audio_tokens = model(audio_tensor)
        print(f"discrete_audio_tokens: {discrete_audio_tokens.shape}")
        print(discrete_audio_tokens)

except KeyboardInterrupt:
    print("Audio processing stopped.")
finally:
    # Close the stream gracefully
    stream.stop_stream()
    stream.close()
    p.terminate()