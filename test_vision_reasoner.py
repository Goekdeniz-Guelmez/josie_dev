import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
from torchvision import transforms
from args import ModelArgs
from tqdm import tqdm
from reasoner import ReasonerTransformer
from vision import VideoEncoder

def convert_to_discrete_tokens(tokens: torch.Tensor, codebook_size: int) -> torch.Tensor:
    """
    Convert continuous embeddings to discrete token indices
    Args:
        tokens: Tensor of shape [batch_size, sequence_length, hidden_dim]
        codebook_size: Size of the codebook for quantization
    Returns:
        Tensor of shape [batch_size, sequence_length] with discrete token indices
    """
    B, L, D = tokens.shape
    
    # Convert to float32 if not already
    tokens = tokens.float()
    
    # Create random codebook vectors
    codebook = torch.randn(codebook_size, D // 8, device=tokens.device)  # D//8 because we have 8 quantizers
    
    # Reshape to group by quantizer dimension
    num_quantizers = 8
    tokens = tokens.view(B, L, num_quantizers, -1)
    
    discrete_tokens = []
    for i in range(num_quantizers):
        # Get tokens for current quantizer
        quantizer_tokens = tokens[:, :, i, :].float()  # Ensure float type
        
        # Calculate distances
        distances = torch.cdist(quantizer_tokens.reshape(-1, quantizer_tokens.size(-1)), codebook)
        indices = distances.argmin(dim=-1).view(B, L)
        
        # Offset indices for each quantizer
        indices = indices + (i * codebook_size)
        discrete_tokens.append(indices)
    
    # Stack and reshape
    discrete_tokens = torch.stack(discrete_tokens, dim=-1)
    discrete_tokens = discrete_tokens.view(B, -1)
    
    return discrete_tokens.long()  # Convert back to long for final indices

def process_video_and_predict(
    video_path: str,
    video_vision: VideoEncoder,
    reasoner: ReasonerTransformer,
    chunk_size: int = 32,
    frame_limit: int = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process video through vision encoder and reasoner for next token prediction
    """
    # Move models to device
    video_vision = video_vision.to(device)
    reasoner = reasoner.to(device)
    video_vision.eval()
    reasoner.eval()
    
    print(f"Loading video: {video_path}")
    frames = load_video_frames(video_path, frame_limit)
    total_frames = len(frames)
    print(f"Loaded {total_frames} frames")
    
    frame_chunks = chunk_frames(frames, chunk_size)
    print(f"Split into {len(frame_chunks)} chunks of {chunk_size} frames")
    
    all_vision_tokens = []
    print("Processing video chunks through vision encoder...")
    for chunk in tqdm(frame_chunks):
        video_tensor = preprocess_frames(chunk)
        video_tensor = video_tensor.to(device)
        
        with torch.no_grad():
            # Get vision tokens and convert to discrete indices
            vision_tokens = video_vision(video_tensor)
            
            # Debug info
            print(f"Vision tokens shape: {vision_tokens.shape}")
            print(f"Vision tokens dtype: {vision_tokens.dtype}")
            
            discrete_tokens = convert_to_discrete_tokens(
                vision_tokens, 
                codebook_size=1024  # Using fixed codebook size
            )
            all_vision_tokens.append(discrete_tokens)
    
    # Combine vision tokens
    vision_tokens = torch.cat(all_vision_tokens, dim=1)
    print(f"Discrete vision tokens shape: {vision_tokens.shape}")
    
    # Process through reasoner for prediction
    print("Processing through reasoner for prediction...")
    with torch.no_grad():
        text_stream, _ = reasoner(vision_tokens)
        
        # Get the last token prediction
        next_token_logits = text_stream[:, -1, :]
        next_token_pred = torch.argmax(next_token_logits, dim=-1)
        
        # Get top 5 predictions with probabilities
        top_k = 5
        probs = torch.softmax(next_token_logits, dim=-1)
        top_probs, top_tokens = torch.topk(probs, k=top_k, dim=-1)
    
    # Print predictions
    print("\nNext Token Predictions:")
    print(f"Top predicted token: {next_token_pred.item()}")
    print("\nTop 5 predictions with probabilities:")
    for token, prob in zip(top_tokens[0], top_probs[0]):
        print(f"Token {token.item()}: {prob.item():.4f}")
    
    return next_token_pred, text_stream

def load_video_frames(video_path: str, frame_limit: int = None) -> List[np.ndarray]:
    """Load frames from a video file"""
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
        if frame_limit and len(frames) >= frame_limit:
            break
            
    cap.release()
    return frames

def preprocess_frames(
    frames: List[np.ndarray], 
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """Preprocess video frames"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    processed_frames = []
    for frame in frames:
        processed = transform(frame)
        processed_frames.append(processed)
    
    video_tensor = torch.stack(processed_frames, dim=0)
    return video_tensor.unsqueeze(0)

def chunk_frames(frames: List[np.ndarray], chunk_size: int = 32) -> List[List[np.ndarray]]:
    """Split frames into chunks"""
    return [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]

# Example usage
if __name__ == "__main__":
    # Initialize models
    video_vision = VideoEncoder(ModelArgs)
    reasoner = ReasonerTransformer(ModelArgs)
    
    # Process video and get prediction
    video_path = "/Users/gokdenizgulmez/Desktop/J.O.S.I.E./2DE83153-6872-4882-8005-B1AC27262A66.mp4"
    next_token, text_stream = process_video_and_predict(
        video_path=video_path,
        video_vision=video_vision,
        reasoner=reasoner,
        chunk_size=ModelArgs.encoder_vision_max_frames
    )
    
    print(f"\nFinal text stream shape: {text_stream.shape}")
    print(f"Predicted next token: {next_token.item()}")
    print("\nProcessing complete!")