import torch
import torch.nn as nn

class SeaNetEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Add initial projection to convert from 1 channel to dim channels
        self.initial_proj = nn.Conv1d(
            in_channels=1,
            out_channels=dim,
            kernel_size=3,
            padding=1
        )
        self.initial_proj = nn.utils.weight_norm(self.initial_proj)
        
        self.conv_blocks = nn.ModuleList([
            # Use dim instead of hardcoded 512 channels
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
        self.final_conv = nn.utils.weight_norm(self.final_conv)

    def forward(self, x):
        # x shape: [B, 1, T] # Raw audio
        x = self.initial_proj(x)  # [B, dim, T]
        
        # Process through conv blocks
        for block in self.conv_blocks:
            x = block(x)
            
        # Final downsampling
        x = self.final_conv(x)
        
        return x.transpose(1, 2)  # Return [B, T, dim] for transformer


class SeaNetDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.initial_upsample = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=2,
            output_padding=1
        )
        self.initial_upsample = nn.utils.weight_norm(self.initial_upsample)

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
            padding=2
        )
        self.final_proj = nn.utils.weight_norm(self.final_proj)

    def forward(self, x):
        # x shape: [B, T, dim]
        x = x.transpose(1, 2)  # [B, dim, T]
        
        # Initial upsampling
        x = self.initial_upsample(x)
        
        # Process through transposed conv blocks
        for block in self.conv_blocks:
            x = block(x)
            
        # Project to waveform
        x = self.final_proj(x)
        
        return x  # [B, 1, T*480]


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
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * current_dilation,
                dilation=current_dilation
            )
            conv = nn.utils.parametrizations.weight_norm(conv)
            
            layer = nn.Sequential(
                conv,
                nn.ELU(),
            )
            self.layers.append(layer)
            current_dilation *= dilation_growth

        self.downsample = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size-1
        )
        self.downsample = nn.utils.weight_norm(self.downsample)

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
        x = self.downsample(x)
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
        
        self.upsample = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size-1,
            output_padding=stride-1
        )
        self.upsample = nn.utils.weight_norm(self.upsample)

        self.layers = nn.ModuleList()
        current_dilation = dilation_growth**(num_conv_layers-1)

        for _ in range(num_conv_layers):
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * current_dilation,
                dilation=current_dilation
            )
            conv = nn.utils.parametrizations.weight_norm(conv)
            
            layer = nn.Sequential(
                conv,
                nn.ELU()
            )
            self.layers.append(layer)
            current_dilation = current_dilation // dilation_growth

    def forward(self, x):
        x = self.upsample(x)
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
        return x