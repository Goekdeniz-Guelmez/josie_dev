import torch
import sounddevice as sd
import numpy as np
from JODIO import JODIO
from args import ModelArgs
import queue
import threading

class AudioProcessor:
    def __init__(self):
        # Model setup
        self.args = ModelArgs()
        self.model = JODIO(self.args)
        self.model.eval()
        
        # Audio parameters
        self.sample_rate = self.args.inference_args.rate  # Should be 24000
        self.channels = self.args.inference_args.channels  # Should be 1
        self.chunk_duration = 0.1  # Reduced chunk duration for better real-time processing
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Queues for audio processing
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # For decoder testing
        self.reasoner_output = torch.randn(1, 8, self.args.audio_decoder_args.hidden_size)
        
        # Processing flags
        self.running = True
        
        # Buffer for size matching
        self.output_buffer = np.zeros(self.chunk_samples)

    def audio_callback(self, indata, outdata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status)
        
        # Handle input
        if len(indata.shape) > 1:
            audio_data = indata[:, 0]  # Take first channel if stereo
        else:
            audio_data = indata
            
        self.input_queue.put(audio_data)
        
        # Handle output
        try:
            output_data = self.output_queue.get_nowait()
            # Ensure output size matches expected size
            if len(output_data) >= frames:
                outdata[:] = output_data[:frames].reshape(-1, 1)
            else:
                # Pad with zeros if too short
                outdata[:len(output_data)] = output_data.reshape(-1, 1)
                outdata[len(output_data):].fill(0)
        except queue.Empty:
            outdata.fill(0)

    def process_audio(self):
        """Process audio chunks through the model"""
        while self.running:
            try:
                # Get input audio chunk
                audio_data = self.input_queue.get(timeout=1)
                
                # Normalize audio
                audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
                
                # Convert to tensor
                waveform = torch.FloatTensor(audio_data).unsqueeze(0).unsqueeze(0)
                
                # Process through encoder
                with torch.no_grad():
                    semantic_tokens, acoustic_tokens, combined_tokens = self.model.encode(waveform)
                    print("Encoded tokens:", combined_tokens)
                    
                    # Process through decoder (using test output for now)
                    output_waveform = self.model.decode(self.reasoner_output)
                    
                    # Convert to numpy and normalize
                    output_audio = output_waveform.squeeze().cpu().numpy()
                    output_audio = output_audio / np.max(np.abs(output_audio))
                    
                    # Resample to match chunk size if necessary
                    if len(output_audio) != self.chunk_samples:
                        output_audio = np.interp(
                            np.linspace(0, len(output_audio), self.chunk_samples),
                            np.arange(len(output_audio)),
                            output_audio
                        )
                    
                    self.output_queue.put(output_audio)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing: {e}")

    def run(self):
        """Start the audio processing"""
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.start()
        
        try:
            with sd.Stream(channels=self.channels,
                         samplerate=self.sample_rate,
                         blocksize=self.chunk_samples,
                         callback=self.audio_callback):
                print(f"Started audio stream. Using {self.chunk_samples} samples per chunk.")
                print("Press Ctrl+C to stop.")
                while True:
                    sd.sleep(100)
                    
        except KeyboardInterrupt:
            print("\nStopping audio processing...")
            self.running = False
            process_thread.join()
        except Exception as e:
            print(f"Error in audio stream: {e}")
            self.running = False
            process_thread.join()

if __name__ == "__main__":
    processor = AudioProcessor()
    processor.run()