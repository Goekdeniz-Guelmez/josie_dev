import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from torchvision import transforms
from args import ModelArgs
from tqdm import tqdm


def load_video_frames(video_path: str, frame_limit: int = None) -> List[np.ndarray]:
    """
    Load frames from a video file
    Args:
        video_path: Path to video file
        frame_limit: Maximum number of frames to load (optional)
    Returns:
        List of frames as numpy arrays
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
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
    """
    Preprocess video frames for the model
    Args:
        frames: List of numpy array frames
        target_size: Target frame size (height, width)
    Returns:
        Tensor of shape [1, num_frames, channels, height, width]
    """
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
    """
    Split frames into chunks of specified size
    Args:
        frames: List of frames
        chunk_size: Number of frames per chunk
    Returns:
        List of frame chunks
    """
    return [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]

def visualize_tokens(tokens: torch.Tensor, save_path: str = None):
    """
    Visualize the discrete tokens as a heatmap
    Args:
        tokens: Tensor of shape [batch_size, sequence_length, hidden_dim]
        save_path: Optional path to save the visualization
    """
    tokens = tokens[0].detach().cpu().numpy()
    
    plt.figure(figsize=(15, 10))
    plt.imshow(tokens, aspect='auto', cmap='viridis')
    plt.colorbar(label='Token Values')
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Frame')
    plt.title(f'Video Discrete Tokens Visualization (Total Frames: {tokens.shape[0]})')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def process_video_chunks(
    video_path: str,
    model: torch.nn.Module,
    chunk_size: int = 32,
    frame_limit: int = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    visualization_path: str = None
) -> torch.Tensor:
    """
    Process a video file through the encoder in chunks
    Args:
        video_path: Path to video file
        model: VideoEncoder model
        chunk_size: Number of frames to process at once
        frame_limit: Maximum number of frames to process
        device: Device to run the model on
        visualization_path: Optional path to save token visualization
    Returns:
        Concatenated encoded tokens tensor
    """
    print(f"Loading video: {video_path}")
    frames = load_video_frames(video_path, frame_limit)
    total_frames = len(frames)
    print(f"Loaded {total_frames} frames")
    
    # Split into chunks
    frame_chunks = chunk_frames(frames, chunk_size)
    print(f"Split into {len(frame_chunks)} chunks of {chunk_size} frames")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Process each chunk
    all_tokens = []
    print("Processing chunks...")
    for i, chunk in enumerate(tqdm(frame_chunks)):
        # Preprocess chunk
        video_tensor = preprocess_frames(chunk)
        video_tensor = video_tensor.to(device)
        
        # Encode chunk
        with torch.no_grad():
            chunk_tokens = model(video_tensor)
            all_tokens.append(chunk_tokens)
    
    # Concatenate all tokens along the sequence dimension
    combined_tokens = torch.cat(all_tokens, dim=1)
    print(f"Final token shape: {combined_tokens.shape}")
    
    # Visualize combined tokens
    if visualization_path:
        print(f"Saving token visualization to: {visualization_path}")
        visualize_tokens(combined_tokens, visualization_path)
    else:
        visualize_tokens(combined_tokens)
        
    return combined_tokens

def analyze_tokens(tokens: torch.Tensor):
    """
    Analyze and print statistics about the encoded tokens
    Args:
        tokens: Tensor of shape [batch_size, sequence_length, hidden_dim]
    """
    tokens = tokens[0].detach().cpu()  # Take first batch
    
    print("\nToken Statistics:")
    print(f"Total frames encoded: {tokens.shape[0]}")
    print(f"Hidden dimension size: {tokens.shape[1]}")
    print(f"Value range: {tokens.min():.3f} to {tokens.max():.3f}")
    print(f"Mean value: {tokens.mean():.3f}")
    print(f"Standard deviation: {tokens.std():.3f}")
    
    # Analyze temporal patterns
    temporal_variation = tokens.std(dim=0).mean()
    print(f"Average temporal variation: {temporal_variation:.3f}")
    
    # Analyze feature patterns
    feature_variation = tokens.std(dim=1).mean()
    print(f"Average feature variation: {feature_variation:.3f}")

# Example usage
if __name__ == "__main__":
    # Initialize model
    from video_vision import VideoEncoder
    model = VideoEncoder(ModelArgs)
    
    # Process video
    video_path = "/Users/gokdenizgulmez/Desktop/J.O.S.I.E./2DE83153-6872-4882-8005-B1AC27262A66.mp4"
    tokens = process_video_chunks(
        video_path=video_path,
        model=model,
        chunk_size=ModelArgs.encoder_max_position_embeddings,
        frame_limit=None,  # Process all frames
        visualization_path="tokens_visualization.png"
    )

    print(tokens.shape)
    
    # Analyze the tokens
    # analyze_tokens(tokens)
    
    print("Processing complete!")