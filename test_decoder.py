import torch
import pyaudio
import numpy as np
from args import AudioRQTransformerArgs
from audio import AudioEncoderDecoder

# Initialize audio parameters
RATE = 16000  # Sample rate
CHUNK = 4000  # Buffer size
FORMAT = pyaudio.paFloat32
CHANNELS = 1

# Initialize model
model = AudioEncoderDecoder(AudioRQTransformerArgs)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    output=True,
    frames_per_buffer=CHUNK
)

print("Starting continuous audio generation... (Ctrl+C to stop)")

try:
    while True:
        # Create random discrete tokens with shape [1, 1, 256]
        discrete_tokens = torch.randint(
            low=0,
            high=AudioRQTransformerArgs.encoder_codebook_size,
            size=(1, 1, 256)
        )
        
        # Decode
        output = model.decode(discrete_tokens)
        
        # Convert to audio and play
        audio_output = output.squeeze().numpy()
        
        # Play the synthesized audio
        stream.write(audio_output.astype(np.float32).tobytes())

except KeyboardInterrupt:
    print("\nStopping audio generation...")

finally:
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()