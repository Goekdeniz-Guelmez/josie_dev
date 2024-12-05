import torch
import torch.nn as nn

class SeaNetEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Add initial projection to convert from 1 channel to dim channels
        self.input_proj = nn.Conv1d(
            in_channels=1,
            out_channels=dim,
            kernel_size=3,
            padding=1
        )
        self.input_proj = nn.utils.parametrizations.weight_norm(self.input_proj)

        self.conv_blocks = nn.ModuleList([
            ConvBlock(stride=4, channels=dim),
            ConvBlock(stride=5, channels=dim),
            ConvBlock(stride=6, channels=dim),
            ConvBlock(stride=8, channels=dim)
        ])

        self.final_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=2
        )
        self.final_conv = nn.utils.parametrizations.weight_norm(self.final_conv)

    def forward(self, x):
        # x shape: [B, 1, T] # Raw audio
        x = self.input_proj(x)  # [B, dim, T]

        # Process through conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Final downsampling
        x = self.final_conv(x)

        return x.transpose(1, 2).contiguous()  # Return [B, T, D] for transformer


class SeaNetDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Modified initial upsample to handle small input sizes
        self.initial_upsample = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=4,  # Increased kernel size
            stride=2,
            padding=1,      # Adjusted padding
            output_padding=0
        )
        self.initial_upsample = nn.utils.parametrizations.weight_norm(self.initial_upsample)

        self.conv_blocks = nn.ModuleList([
            TransposedConvBlock(stride=8, channels=dim),
            TransposedConvBlock(stride=6, channels=dim),
            TransposedConvBlock(stride=5, channels=dim),
            TransposedConvBlock(stride=4, channels=dim)
        ])

        self.final_proj = nn.Conv1d(
            in_channels=dim,
            out_channels=1,  # Single channel audio output
            kernel_size=3,
            padding=1       # Adjusted padding
        )
        self.final_proj = nn.utils.parametrizations.weight_norm(self.final_proj)

    def forward(self, x):
        # x shape: [B, T, dim]
        x = x.transpose(1, 2)  # [B, dim, T]
        
        # Initial upsampling
        x = self.initial_upsample(x)
        
        # Process through transposed conv blocks
        for block in self.conv_blocks:
            x = block(x)
            
        # Project to waveform
        return self.final_proj(x)  # Return [B, 1, T]


# Update ConvBlock and TransposedConvBlock to use dynamic channel size
class ConvBlock(nn.Module):
    def __init__(
        self,
        stride: int,
        channels: int,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        dilation_growth: int = 2
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dilation = 1

        for _ in range(num_conv_layers):
            # Calculate padding to maintain same output size
            total_padding = (kernel_size - 1) * current_dilation
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            
            # Create padding layer
            pad_layer = nn.ConstantPad1d((left_padding, right_padding), 0)
            
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=current_dilation,
                padding=0  # We'll use explicit padding
            )
            conv = nn.utils.parametrizations.weight_norm(conv)

            layer = nn.Sequential(
                pad_layer,
                conv,
                nn.SiLU(),
            )
            self.layers.append(layer)
            current_dilation *= dilation_growth

        # Downsample with proper padding
        downsample_padding = kernel_size // 2
        self.downsample = nn.Sequential(
            nn.ConstantPad1d((downsample_padding, downsample_padding), 0),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0
            )
        )
        self.downsample[1] = nn.utils.parametrizations.weight_norm(self.downsample[1])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
        x = self.downsample(x) # (B, D, L)
        return x


class TransposedConvBlock(nn.Module):
    def __init__(
        self,
        stride: int,
        channels: int,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        dilation_growth: int = 2
    ):
        super().__init__()
        
        # Upsample layer
        self.upsample = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,  # Changed padding
            output_padding=stride-1
        )
        self.upsample = nn.utils.parametrizations.weight_norm(self.upsample)

        self.layers = nn.ModuleList()
        current_dilation = dilation_growth**(num_conv_layers-1)

        for _ in range(num_conv_layers):
            # Use sequential with padding layer to ensure sizes match
            total_padding = (kernel_size - 1) * current_dilation
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            
            pad_layer = nn.ConstantPad1d((left_padding, right_padding), 0)
            
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=current_dilation,
                padding=0  # We'll use explicit padding
            )
            conv = nn.utils.parametrizations.weight_norm(conv)

            layer = nn.Sequential(
                pad_layer,
                conv,
                nn.SiLU()
            )
            
            self.layers.append(layer)
            current_dilation = current_dilation // dilation_growth

    def forward(self, x):
        x = self.upsample(x)
        
        # Store original upsampled tensor
        orig_x = x
        
        # Apply convolution layers
        for layer in self.layers:
            x = layer(x)
        
        # Add residual connection with size check
        if x.size(2) != orig_x.size(2):
            # Pad or trim to match sizes
            if x.size(2) > orig_x.size(2):
                x = x[..., :orig_x.size(2)]
            else:
                orig_x = orig_x[..., :x.size(2)]
        
        x = x + orig_x
        return x
