import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

def turn_to_token(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Convert logits to tokens with temperature sampling
    Args:
        logits: Shape [B, L, vocab_size]
        temperature: Controls randomness in sampling (higher = more random)
    Returns:
        tokens: Shape [B, L, 1]
    """
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # Apply temperature scaling
    logits = logits / temperature
    
    # Sample from the distribution
    probs = F.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
    return tokens.view(*logits.shape[:-1], 1)