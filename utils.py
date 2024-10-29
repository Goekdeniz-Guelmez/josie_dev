from typing import Tuple

import math

import torch
from torch import nn

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Args:
        dim (int): Dimension to normalize over
        eps (float): Epsilon for numerical stability. Default: 1e-6
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Shape: (dim,)

    def _norm(self, x):
        """
        Compute RMS normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            torch.Tensor: Normalized tensor of shape (batch_size, seq_len, dim)
        """
        # x shape: (batch_size, seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Apply RMS normalization with learned scale.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            torch.Tensor: Normalized and scaled tensor of shape (batch_size, seq_len, dim)
        """
        # x shape: (batch_size, seq_len, dim)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight  # Shape: (batch_size, seq_len, dim)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute frequencies for rotary positional embeddings.
    
    Args:
        dim (int): Model dimension
        end (int): Maximum sequence length
        theta (float): RoPE theta parameter. Default: 10000.0
        
    Returns:
        torch.Tensor: Complex frequencies tensor of shape (end, dim//2)
    """
    # freqs shape: (dim//2,)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # t shape: (end,)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    # freqs shape: (end, dim//2)
    freqs = torch.outer(t, freqs)
    # freqs_cis shape: (end, dim//2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(
    freqs_cis: torch.Tensor,  # Shape: (seq_len, head_dim/2)
    x: torch.Tensor,  # Shape: (batch_size, seq_len, n_heads, head_dim/2)
) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting against query/key tensors.
    
    Args:
        freqs_cis: Complex frequency tensor
        x: Query or key tensor to broadcast against
        
    Returns:
        torch.Tensor: Reshaped frequencies for broadcasting
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,  # Shape: (batch_size, seq_len, n_heads, head_dim)
    xk: torch.Tensor,  # Shape: (batch_size, seq_len, n_kv_heads, head_dim)
    freqs_cis: torch.Tensor,  # Shape: (seq_len, head_dim/2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Args:
        xq: Query tensor
        xk: Key tensor
        freqs_cis: Precomputed RoPE frequencies in complex form
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed query and key tensors with same shapes as input
    """
    # Reshape last dimension as complex numbers
    # Shape: (batch_size, seq_len, n_heads, head_dim/2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape frequencies for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply rotary embeddings via complex multiplication
    # Shape: (batch_size, seq_len, n_heads, head_dim)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match the number of query heads in multi-query attention.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_kv_heads, head_dim)
        n_rep (int): Number of times to repeat each head
        
    Returns:
        torch.Tensor: Repeated tensor of shape (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape  # Shape: (bs, slen, n_kv_heads, head_dim)
    if n_rep == 1:
        return x
    return (
        # Shape: (bs, slen, n_kv_heads, 1, head_dim)
        x[:, :, :, None, :]
        # Shape: (bs, slen, n_kv_heads, n_rep, head_dim)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        # Shape: (bs, slen, n_kv_heads * n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )