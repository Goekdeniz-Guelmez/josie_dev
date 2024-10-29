import torch
import pyaudio
import numpy as np
from args import ModelArgs
from audio import AudioEncoderDecoder

# Initialize audio parameters
RATE = 16000  # Sample rate
CHUNK = 16000 // 4  # Buffer size
FORMAT = pyaudio.paFloat32
CHANNELS = 1

# Initialize model
model = AudioEncoderDecoder(ModelArgs)

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
            high=ModelArgs.encoder_codebook_size,
            size=(1, 1, 256)
        )
        
        # Decode
        output = model.decode(discrete_tokens)
        
        # Convert to audio and play
        audio_output = output.squeeze().numpy()
        print(audio_output.shape)
        
        # Play the synthesized audio
        stream.write(audio_output.astype(np.float32).tobytes())

except KeyboardInterrupt:
    print("\nStopping audio generation...")

finally:
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()