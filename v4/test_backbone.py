import torch

from temporial_tranformer import TemporalTransformer
from depth_transformer import DepthTransformer

from args import ModelArgs

args = ModelArgs()

# Initialize
temporal = TemporalTransformer(args)

print(temporal)

text_tokens = torch.tensor([[5, 28, 103]])
semantic_tokens = torch.tensor([[245]])
acoustic_tokens = torch.tensor([[53, 645, 321, 654, 76, 2, 43, 52]])


# Forward pass
temporal_context = temporal(
    text_tokens,
    semantic_tokens,
    acoustic_tokens
)

print('-----------')
print(temporal_context.shape)

# Pass context to depth transformer
depth = DepthTransformer(args)

print(depth)

text_logits, semantic_logits, acoustic_logits = depth(temporal_context)

print('-----------')
print(text_logits)
print('-----------')
print(semantic_logits)
print('-----------')
print(acoustic_logits)