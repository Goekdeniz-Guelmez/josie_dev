import torch
import numpy as np
import pyaudio
from typing import List
import time

from JOSIEv4o.quantizer import Quantizer
from JOSIEv4o.args import ModelArgs

from audio_tokenizer import AudioTokenizer, process_tokens

class LiveAudioTokenizer:
    def __init__(self):
        # Audio parameters
        self.CHUNK = 4096
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000  # 16kHz
        self.RECORD_SECONDS = 0.25  # 250ms
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Initialize model components
        self.args = ModelArgs()
        self.quantizer = Quantizer(self.args)
        
        # Calculate samples needed for 250ms
        self.SAMPLES_NEEDED = int(self.RATE * self.RECORD_SECONDS)
        
    def get_tokens_from_audio(self, audio_data: np.ndarray) -> List[int]:
        """Convert raw audio chunk to discrete tokens"""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Normalize audio
        if torch.abs(audio_tensor).max() > 0:
            audio_tensor = audio_tensor / torch.abs(audio_tensor).max()
        
        # Reshape: [samples] -> [1, 1, samples] (add batch and channel dims)
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        
        # Project to hidden size
        projection = torch.nn.Linear(self.SAMPLES_NEEDED, self.args.audio_encoder_args.hidden_size)
        projected = projection(audio_tensor)
        
        # Get tokens from quantizer
        _, descrete_tokens = self.quantizer(projected, stream_type='temporal')
        
        # Extract tokens (descrete_tokens) from each quantizer
        # descrete_tokens shape: [batch, time, num_quantizers]
        tokens = []
        for q in range(self.args.audio_encoder_args.num_quantizers):
            # Get token from each quantizer and adjust based on position
            token = descrete_tokens[0, 0, q].item()  # [batch=0, time=0, quantizer=q]
            # Offset token by codebook_size * quantizer_position
            adjusted_token = token + (q * self.args.audio_encoder_args.codebook_size)
            tokens.append(adjusted_token)
            
        return tokens
    
    def stream_tokens(self):
        # Buffer to accumulate audio samples
        audio_buffer = []
        
        def callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            audio_buffer.extend(audio_data.tolist())
            
            # Process when we have enough samples
            if len(audio_buffer) >= self.SAMPLES_NEEDED:
                # Take exactly 250ms worth of samples
                audio_chunk = np.array(audio_buffer[:self.SAMPLES_NEEDED])
                audio_buffer.clear()
                
                # Get tokens
                tokens = self.get_tokens_from_audio(audio_chunk)
                print("Tokens:", tokens)
                
            return (in_data, pyaudio.paContinue)
        
        try:
            # Open audio stream
            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=callback
            )
            
            print("* Recording started - tokens will be printed for each 250ms of audio")
            print("* Press Ctrl+C to stop")
            
            stream.start_stream()
            while stream.is_active():
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n* Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            self.p.terminate()

def main():
    tokenizer = LiveAudioTokenizer()
    tokenizer.stream_tokens()

if __name__ == "__main__":
    main()

