import torch
import pyaudio
import numpy as np
import threading
import queue
from pathlib import Path
from typing import Optional

from one_file_ref import JOSIE, ModelArgs

class JOSIETester:
    def __init__(self, model_path: Optional[Path] = None):
        # Initialize model
        self.args = ModelArgs()
        self.inference_args = self.args.inference_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = JOSIE(self.args).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Audio setup
        self.pa = pyaudio.PyAudio()
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Calculate sizes based on inference args
        self.chunk_size = int(self.inference_args.rate * self.inference_args.record_seconds)
        
        # Flags for control
        self.is_running = False
        self.input_stream = None
        self.output_stream = None
        
        # Random token context (batch_size=1, sequence_length=1)
        self.token_context = torch.randint(
            0, self.args.vocab_size, 
            (1, 1), 
            device=self.device
        )

    def audio_input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream"""
        if status:
            print(f"Input stream error: {status}")
        
        # Convert audio data to numpy array and ensure it's writable
        audio_data = np.frombuffer(in_data, dtype=np.float32).copy()
        self.input_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def audio_output_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio output stream"""
        if status:
            print(f"Output stream error: {status}")
        
        try:
            # Get audio data from output queue
            audio_data = self.output_queue.get_nowait()
            return (audio_data.tobytes(), pyaudio.paContinue)
        except queue.Empty:
            # If no data available, return silence
            return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)

    def process_audio(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Get input audio chunk
                audio_data = self.input_queue.get(timeout=1.0)
                
                # Convert to tensor
                waveform = torch.from_numpy(audio_data).to(self.device)
                waveform = waveform.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                
                # Process through model
                with torch.no_grad():
                    # Generate audio from the tokens
                    text_token, semantic_token, acoustic_tokens, output_waveform = self.model(
                        text_tokens=torch.tensor([[1, 2, 3, 4]]),
                        user_waveform=waveform
                    )
                    
                    # Update context with new token
                    self.token_context = text_token
                
                # Convert output waveform to numpy
                output_audio = output_waveform.cpu().numpy()
                
                # Put in output queue
                self.output_queue.put(output_audio)
                
                # Print tokens for debugging
                print(f"Text Token: {text_token.item()}")
                print(f"Semantic Token: {semantic_token.item()}")
                print(f"Acoustic Tokens: {acoustic_tokens.squeeze().tolist()}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing: {e}")
                import traceback
                traceback.print_exc()
                break

    def start(self):
        """Start audio streams and processing"""
        self.is_running = True
        
        # Start input stream
        self.input_stream = self.pa.open(
            format=self.inference_args.format,
            channels=self.inference_args.channels,
            rate=self.inference_args.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_input_callback
        )
        
        # Start output stream
        self.output_stream = self.pa.open(
            format=self.inference_args.format,
            channels=self.inference_args.channels,
            rate=self.inference_args.rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_output_callback
        )
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()
        
        print("JOSIE is listening... Press Enter to stop.")

    def stop(self):
        """Stop audio streams and processing"""
        self.is_running = False
        
        if self.process_thread:
            self.process_thread.join()
            
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        self.pa.terminate()
        print("JOSIE stopped.")

def main():
    # Initialize and start tester
    tester = JOSIETester()
    
    try:
        tester.start()
        input()  # Wait for Enter key
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tester.stop()

if __name__ == "__main__":
    main()