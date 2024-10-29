import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from args import ModelArgs
    
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

def chunk_frames(
    frames: List[np.ndarray], 
    chunk_size: int = 256  # Changed to match position embeddings
) -> List[List[np.ndarray]]:
    """
    Split frames into chunks of specified size, ensuring we don't exceed position embedding limit
    Args:
        frames: List of frames
        chunk_size: Number of frames per chunk (matching position embeddings)
    Returns:
        List of frame chunks
    """
    return [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]

def process_video_chunks(
    video_path: str,
    model: torch.nn.Module,
    chunk_size: int = 256,  # Changed to match position embeddings
    frame_limit: int = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    visualization_path: str = None
) -> torch.Tensor:
    """
    Process a video file through the encoder in chunks, ensuring position embedding compatibility
    """
    print(f"Loading video: {video_path}")
    frames = load_video_frames(video_path, frame_limit)
    total_frames = len(frames)
    print(f"Loaded {total_frames} frames")
    
    # Split into chunks of size matching position embeddings
    frame_chunks = chunk_frames(frames, chunk_size)
    print(f"Split into {len(frame_chunks)} chunks of max {chunk_size} frames")
    
    model = model.to(device)
    model.eval()
    
    all_tokens = []
    print("Processing chunks...")
    for i, chunk in enumerate(tqdm(frame_chunks)):
        # Preprocess chunk
        video_tensor = preprocess_frames(chunk)
        video_tensor = video_tensor.to(device)
        
        # Ensure chunk doesn't exceed position embedding limit
        if video_tensor.shape[1] > ModelArgs.encoder_vision_max_position_embeddings:
            video_tensor = video_tensor[:, :ModelArgs.encoder_vision_max_position_embeddings]
        
        # Encode chunk
        with torch.no_grad():
            chunk_tokens = model(video_tensor)
            all_tokens.append(chunk_tokens)
    
    # Concatenate all tokens along the sequence dimension
    combined_tokens = torch.cat(all_tokens, dim=1)
    print(f"Final token shape: {combined_tokens.shape}")
    
    if visualization_path:
        print(f"Saving token visualization to: {visualization_path}")
        visualize_tokens(combined_tokens, visualization_path)
    
    return combined_tokens

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

if __name__ == "__main__":
    # Initialize model
    from vision import VideoEncoder
    model = VideoEncoder(ModelArgs)
    
    # Process video with correct chunk size
    video_path = "/Users/gokdenizgulmez/Desktop/J.O.S.I.E./2DE83153-6872-4882-8005-B1AC27262A66.mp4"
    tokens = process_video_chunks(
        video_path=video_path,
        model=model,
        chunk_size=ModelArgs.encoder_vision_max_frames,
        frame_limit=None,
        visualization_path="tokens_visualization.png"
    )
    
    print(f"Final output shape: {tokens.shape}")
    print("Processing complete!")