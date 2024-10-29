import torch
import torchaudio
import numpy as np
from args import ModelArgs
from audio import AudioEncoderDecoder
from reasoner import ReasonerTransformer
import sounddevice as sd

class AudioProcessor:
    def __init__(self, model_args):
        # Initialize audio encoder/decoder and model
        self.audio = AudioEncoderDecoder(model_args)
        self.model = ReasonerTransformer(model_args)
        
        # Audio configuration
        self.sample_rate = 16000  # 16 kHz
        self.chunk_size = self.sample_rate // 4  # 250ms chunks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = model_args.encoder_hidden_dim
        
    def get_audio_chunk(self):
        """Record a chunk of audio using sounddevice"""
        audio_data = sd.rec(
            frames=self.chunk_size,
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocking=True
        )
        # Convert to torch tensor and ensure float32
        return torch.from_numpy(audio_data).squeeze().float()  # Shape: [chunk_size]
        
    def process_audio_chunk(self, audio_tensor):
        """Process a single chunk of audio"""
        # Ensure float32 dtype
        audio_tensor = audio_tensor.float()
        
        # First ensure the audio tensor is the right length
        target_length = self.hidden_dim
        current_length = audio_tensor.size(0)
        
        if current_length < target_length:
            # Pad if too short
            padding = torch.zeros(target_length - current_length, dtype=torch.float32)
            audio_tensor = torch.cat([audio_tensor, padding])
        elif current_length > target_length:
            # Truncate if too long
            audio_tensor = audio_tensor[:target_length]
        
        # Reshape for model input: [1, 1, hidden_dim]
        audio_tensor = audio_tensor.view(1, 1, self.hidden_dim)
        
        # Process through model pipeline
        discrete_audio_tokens = self.audio.encode(audio_tensor)
        text_stream, audio_stream = self.model(discrete_audio_tokens.squeeze(0))
        
        return text_stream, audio_stream
        
    def play_audio(self, audio_tensor):
        """Play processed audio using sounddevice"""
        # Ensure float32 dtype and convert to numpy
        audio_numpy = audio_tensor.float().squeeze().detach().cpu().numpy()
        
        # Ensure 2D array with shape (samples, channels)
        if len(audio_numpy.shape) == 1:
            audio_numpy = audio_numpy.reshape(-1, 1)
        
        # Ensure float32 dtype for numpy array
        audio_numpy = audio_numpy.astype(np.float32)
        
        # Ensure we don't have any NaN values
        if np.isnan(audio_numpy).any():
            print("Warning: NaN values detected in audio output")
            audio_numpy = np.nan_to_num(audio_numpy, copy=True).astype(np.float32)
        
        # Clip values to [-1, 1] range
        audio_numpy = np.clip(audio_numpy, -1.0, 1.0)
        
        sd.play(audio_numpy, self.sample_rate, blocking=True)
        
    def run(self):
        """Main processing loop"""
        print("Recording audio...")
        try:
            while True:
                # Get audio input
                audio_tensor = self.get_audio_chunk()
                
                # Normalize audio to [-1, 1] if not already normalized
                if torch.max(torch.abs(audio_tensor)) > 1.0:
                    audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
                
                # Process audio
                text_stream, audio_stream = self.process_audio_chunk(audio_tensor)
                
                # Decode and play audio
                output = self.audio.decode(audio_stream)
                self.play_audio(output)
                
                # Process text output
                next_token = torch.argmax(text_stream[:, -1, :], dim=-1)
                print(f'{next_token.item()} ', end='', flush=True)
                
        except KeyboardInterrupt:
            print("\nAudio processing stopped.")
        
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            raise
            
if __name__ == "__main__":
    processor = AudioProcessor(ModelArgs)
    processor.run()