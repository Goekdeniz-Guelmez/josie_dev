from typing import List, Dict
import pyaudio
import time

import numpy as np

import torch
import torch.nn as nn

from JOSIEv4o.args import ModelArgs, InferenceArgs


class LiveAudioTokenizer:
    def __init__(
            self,
            model: nn.Module,
            args: ModelArgs,
            inference_args: InferenceArgs = InferenceArgs,
            history_length: int = 4  # Store 1 second of history (4 x 250ms)
        ):
        # Audio parameters
        if hasattr(args, 'inference_args'):
            inference_args = args.inference_args
        else:
            inference_args = inference_args
            
        self.CHUNK = inference_args.chunk
        self.FORMAT = inference_args.format
        self.CHANNELS = inference_args.channels
        self.RATE = inference_args.rate
        self.RECORD_SECONDS = inference_args.record_seconds
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Initialize model components
        self.args = args
        self.model = model
        
        # Calculate samples needed for 250ms
        self.SAMPLES_NEEDED = int(self.RATE * self.RECORD_SECONDS)
        
        # Token history management
        self.token_history: List[List[int]] = []
        self.history_length = history_length
        
        # Optional: maintain raw audio history if needed
        self.audio_history: List[np.ndarray] = []

    def get_tokens_from_audio(self, audio_data: np.ndarray) -> Dict[str, List]:
        """
        Convert raw audio chunk to discrete tokens and maintain history
        Returns both current tokens and complete history
        """
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Normalize audio
        if torch.abs(audio_tensor).max() > 0:
            audio_tensor = audio_tensor / torch.abs(audio_tensor).max()
            
        # Get tokens from quantizer
        discrete_tokens = self.model(audio_tensor.unsqueeze(0).unsqueeze(0))
        
        # Extract tokens from each quantizer
        current_tokens = []
        for q in range(self.args.audio_encoder_args.num_quantizers):
            # Get token from each quantizer and adjust based on position
            token = discrete_tokens[0, 0, q].item()
            # Offset token by codebook_size * quantizer_position
            adjusted_token = token + (q * self.args.audio_encoder_args.codebook_size)
            current_tokens.append(adjusted_token)
        
        # Update token history
        self.token_history.append(current_tokens)
        if len(self.token_history) > self.history_length:
            self.token_history.pop(0)
            
        # Optional: Update audio history
        self.audio_history.append(audio_data)
        if len(self.audio_history) > self.history_length:
            self.audio_history.pop(0)
            
        return {
            'current_tokens': current_tokens,
            'token_history': self.token_history.copy(),  # Return copy to prevent external modification
            'full_context': [token for chunk in self.token_history for token in chunk]  # Flattened history
        }

    def stream_tokens(self, callback_fn=None):
        """
        Stream audio and process tokens with history
        Optional callback_fn to handle tokens externally
        """
        audio_buffer = []
        
        def audio_callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            audio_buffer.extend(audio_data.tolist())
            
            if len(audio_buffer) >= self.SAMPLES_NEEDED:
                # Take exactly 250ms worth of samples
                audio_chunk = np.array(audio_buffer[:self.SAMPLES_NEEDED])
                audio_buffer.clear()
                
                # Get tokens with history
                token_data = self.get_tokens_from_audio(audio_chunk)
                
                # Either use provided callback or print
                if callback_fn:
                    callback_fn(token_data)
                else:
                    print("\nCurrent Tokens:", token_data['current_tokens'])
                    print("History Length:", len(token_data['token_history']), "chunks")
                    print("Full Context Tokens:", len(token_data['full_context']))
                    
            return (in_data, pyaudio.paContinue)
        
        try:
            # Open audio stream
            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=audio_callback
            )
            
            print("\n* Recording started - Processing audio in 250ms chunks")
            print("* Maintaining history of last", self.history_length, "chunks")
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
    
    def get_full_history(self) -> Dict[str, List]:
        """Get complete token and audio history"""
        return {
            'token_history': self.token_history.copy(),
            'audio_history': self.audio_history.copy(),
            'full_context': [token for chunk in self.token_history for token in chunk]
        }