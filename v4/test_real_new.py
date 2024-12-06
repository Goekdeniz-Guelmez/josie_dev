import torch
import sounddevice as sd
import numpy as np
from JOSIE import JOSIE
from args import ModelArgs
import queue
import threading

class AudioProcessor:
    def __init__(self):
        # Model setup
        self.args = ModelArgs()
        self.model = JOSIE(self.args)
        self.model.eval()
        
        # Audio parameters
        self.sample_rate = self.args.inference_args.rate
        self.channels = self.args.inference_args.channels
        self.chunk_duration = self.args.inference_args.record_seconds
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Queues for audio processing
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # For decoder testing
        self.reasoner_output = torch.randn(1, 8, self.args.audio_decoder_args.hidden_size)
        
        # Processing flags
        self.running = True

    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Status: {status}")
        
        # Handle input
        if len(indata.shape) > 1:
            audio_data = indata[:, 0]
        else:
            audio_data = indata
            
        # Check audio data
        if np.any(np.isnan(audio_data)):
            print("Warning: NaN in input audio")
        if np.all(audio_data == 0):
            print("Warning: Silent input")
            
        self.input_queue.put(audio_data)
        
        # Handle output
        try:
            output_data = self.output_queue.get_nowait()
            if len(output_data) >= frames:
                outdata[:] = output_data[:frames].reshape(-1, 1)
            else:
                outdata[:len(output_data)] = output_data.reshape(-1, 1)
                outdata[len(output_data):].fill(0)
        except queue.Empty:
            outdata.fill(0)

    def process_audio(self):
        while self.running:
            try:
                # Get input audio chunk
                audio_data = self.input_queue.get(timeout=1)
                
                # Normalize audio (avoid division by zero)
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data.astype(np.float32) / max_val
                else:
                    audio_data = audio_data.astype(np.float32)
                
                # Convert to tensor with proper shape
                waveform = torch.FloatTensor(audio_data).unsqueeze(0)
                
                # Process through encoder
                with torch.no_grad():
                    output_waveform = self.model(torch.tensor([[1, 2, 3]]), waveform) # torch.Size([1, 1, 8000])
                    
                    # Convert to numpy and normalize
                    output_audio = output_waveform.squeeze().cpu().numpy()
                    max_val = np.max(np.abs(output_audio))
                    if max_val > 0:
                        output_audio = output_audio / max_val
                    
                    # Resample to match chunk size
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
                import traceback
                traceback.print_exc()

    def run(self):
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