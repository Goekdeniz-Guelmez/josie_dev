import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import threading
import time

from one_file_ref_v5 import ModelArgs, JOSIE

from PIL import Image
import torchvision.transforms as transforms
import os

class RealTimeSpeechSystem:
    def __init__(self, model_path=None, device='cpu', sample_file=None, 
                 image_path='/Users/gokdenizgulmez/Desktop/josie_dev/v4/images/test.png',
                 default_text='Hello world this is a text input test'):
        # Load your JOSIE model if path is provided
        if model_path:
            self.args = ModelArgs()
            self.model = JOSIE(self.args)
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(device)
            self.model.eval()
        else:
            self.args = ModelArgs()
            self.model = JOSIE(self.args)
        
        # Audio parameters
        self.sample_rate = 24000
        self.chunk_size = 6000
        
        # Processing state
        self.is_running = False
        self.device = device
        
        # Load sample audio file if provided
        self.sample_audio = None
        self.sample_position = 0
        
        if sample_file and os.path.exists(sample_file):
            self.load_audio(sample_file)
        
        # Load image if provided
        self.image = None
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
        else:
            print(f"Warning: Image path {image_path} not found")
        
        # Initialize with default text input
        self.set_text_input(default_text)
        
    def load_audio(self, audio_path):
        """Load audio file"""
        try:
            audio, file_sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                
            # Resample if needed
            if file_sr != self.sample_rate:
                from scipy import signal
                num_samples = int(len(audio) * self.sample_rate / file_sr)
                audio = signal.resample(audio, num_samples)
            
            self.sample_audio = audio.astype(np.float32)
            self.sample_position = 0
            print(f"Loaded audio: {audio_path} ({len(audio)/self.sample_rate:.2f} seconds)")
            return True
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False
        
    def load_image(self, image_path):
        """Load and preprocess image for JOVIO model"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Define image transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to model input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
            
            # Apply transformations
            img_tensor = transform(img)
            
            # Add batch and time dimensions [B, C, T, H, W]
            # For JOVIO, we need [B, C, T, H, W] where T is temporal dimension
            self.image = img_tensor.unsqueeze(0).unsqueeze(2).to(self.device)
            
            print(f"Loaded image from {image_path}, shape: {self.image.shape}")
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def set_text_input(self, text):
        """Set text input for the model"""
        if not text:
            print("Empty text input, using default token")
            self.text_tokens = torch.tensor([[1]]).to(self.device)
            return True
            
        # For testing without a real tokenizer
        print(f"Setting text input: '{text}'")
        
        # Create simple token representation (just for testing)
        # In a real implementation, you would use your model's tokenizer
        tokens = []
        for char in text.lower():
            if 'a' <= char <= 'z':
                tokens.append(ord(char) - ord('a') + 2)  # Start at token 2
            elif char == ' ':
                tokens.append(1)  # Space token
            else:
                tokens.append(28)  # Unknown token
        
        if not tokens:
            tokens = [1]  # Default token
            
        self.text_tokens = torch.tensor([tokens]).to(self.device)
        print(f"Text tokens: {self.text_tokens}")
        return True
        
    def get_input_chunk(self):
        """Get next chunk from sample audio or generate synthetic audio"""
        if self.sample_audio is not None:
            # Get chunk from sample audio
            if self.sample_position + self.chunk_size <= len(self.sample_audio):
                chunk = self.sample_audio[self.sample_position:self.sample_position + self.chunk_size]
                self.sample_position += self.chunk_size
            else:
                # Loop back to beginning if we reach the end
                self.sample_position = 0
                chunk = self.sample_audio[self.sample_position:self.sample_position + self.chunk_size]
                self.sample_position += self.chunk_size
                print("Looping sample audio...")
                
            return chunk
        else:
            # Generate synthetic audio
            return self.generate_synthetic_audio()
    
    def generate_synthetic_audio(self):
        """Generate synthetic speech-like audio"""
        # Create time array
        t = np.linspace(0, self.chunk_size / self.sample_rate, self.chunk_size, False)
        
        # Generate a mixture of frequencies common in speech
        audio = 0.3 * np.sin(2 * np.pi * 120 * t)  # Fundamental
        audio += 0.2 * np.sin(2 * np.pi * 240 * t)  # First harmonic
        audio += 0.1 * np.sin(2 * np.pi * 360 * t)  # Second harmonic
        
        # Add some formants (higher frequencies)
        audio += 0.1 * np.sin(2 * np.pi * 1000 * t)
        audio += 0.05 * np.sin(2 * np.pi * 2500 * t)
        
        # Add noise
        audio += 0.1 * np.random.randn(self.chunk_size)
        
        return audio.astype(np.float32)
    
    def process_audio(self):
        buffer = []
        buffer_size = 2  # Process 1 second of audio at a time
        
        while self.is_running:
            # Get input chunk (either from file or synthetic)
            chunk = self.get_input_chunk()
            buffer.append(chunk)
            
            if len(buffer) >= buffer_size:
                # Process audio
                audio_data = np.concatenate(buffer)
                
                # Convert to tensor with correct shape (channels, length)
                waveform = torch.tensor(audio_data, dtype=torch.float32)
                waveform = waveform.to(self.device)
                
                # Always use the model for output, even if we're using dummy input
                print("\nProcessing with JOSIE model...")
                
                with torch.no_grad():
                    # Call model with all available inputs
                    text_token, semantic_token, acoustic_tokens, response_waveform = self.model(
                        audio_array=waveform,
                        text_tokens=self.text_tokens,
                        user_images=self.image
                    )
                    # Print generated text token for debugging
                    if text_token is not None:
                        print(f"Generated text token: {text_token.cpu().numpy()}")
                
                # Get audio response from model
                response = response_waveform.squeeze().cpu().numpy()
                
                print(f"Playing audio output ({len(response)/self.sample_rate:.2f} seconds)")
                # Play the audio directly
                sd.play(response, self.sample_rate)
                sd.wait()  # Wait for playback to finish
                
                # Keep last chunk for context
                buffer = buffer[-1:]
            
            # Sleep to simulate real-time processing
            time.sleep(0.01)
    
    def start(self):
        # Start processing thread
        self.is_running = True
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        print("Real-time speech system started")
    
    def stop(self):
        self.is_running = False
        
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        
        # Stop any playing audio
        sd.stop()
        
        print("Real-time speech system stopped")

# Interactive CLI for the system
def interactive_cli():
    # Create the speech system
    speech_system = RealTimeSpeechSystem(
        model_path=None,  # Set to your model path if available
        image_path='test/images/test.png'  # Default image path
    )
    
    # Start the system
    speech_system.start()
    
    print("\nInteractive JOSIE CLI")
    print("=====================")
    print("Commands:")
    print("  text [your text]  - Set text input")
    print("  image [path]      - Load new image")
    print("  audio [path]      - Load audio file")
    print("  generate          - Generate response immediately")
    print("  quit              - Exit the program")
    
    try:
        while True:
            cmd = input("\nEnter command: ").strip()
            
            if cmd.startswith("text "):
                text = cmd[5:]
                speech_system.set_text_input(text)
            
            elif cmd.startswith("image "):
                img_path = cmd[6:]
                if speech_system.load_image(img_path):
                    print(f"Image loaded: {img_path}")
                else:
                    print(f"Failed to load image: {img_path}")
            
            elif cmd.startswith("audio "):
                audio_path = cmd[6:]
                if speech_system.load_audio(audio_path):
                    print(f"Audio loaded: {audio_path}")
                else:
                    print(f"Failed to load audio: {audio_path}")
            
            elif cmd == "generate":
                # Force immediate processing by filling the buffer
                print("Generating response...")
                # This will be handled in the next processing cycle
            
            elif cmd == "quit" or cmd == "exit":
                break
            
            else:
                print("Unknown command. Available commands: text, image, audio, generate, quit")
    
    except KeyboardInterrupt:
        pass
    finally:
        speech_system.stop()
        print("System stopped.")

# Run the interactive CLI
if __name__ == "__main__":
    interactive_cli()