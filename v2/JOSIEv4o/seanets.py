import torch
import torch.nn as nn

from JOSIEv4o.args import ModelArgs


class ConvBlock(nn.Module):
    def __init__(
        self,
        stride: int,
        channels: int = 512,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        dilation_growth: int = 2
    ):
        super().__init__()
        self.layer = nn.ModuleList()

        # In encoder: dilation grows: 1 -> 2 -> 4
        current_dilation = 1  # Start with dilation rate of 1

        for _ in range(num_conv_layers):
            # Dilated causal conv layer
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * current_dilation,  # Causal padding
                dilation=current_dilation
            )
            # Apply weight normalization
            conv = nn.utils.weight_norm(conv)

            # Create sequential block
            layer = nn.Sequential(
                conv,
                nn.ELU(),  # ELU activation as mentioned in paper
            )
            self.layers.append(layer)

            # Increase dilation for next layer
            current_dilation *= dilation_growth

        # Final strided convolution for downsampling
        self.downsample = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size-1  # Causal padding
        )
        self.downsample = nn.utils.weight_norm(self.downsample)

    def forward(self, x):
        # Apply dilated convolutions
        for layer in self.layers:
            # Residual connection
            residual = x
            x = layer(x)
            x = x + residual  # Add residual connection

        # Downsample
        x = self.downsample(x)
        return x


class TransposedConvBlock(nn.Module):
    def __init__(self,
                 stride,
                 channels=512,
                 kernel_size=3,
                 num_conv_layers=3,
                 dilation_growth=2):
        super().__init__()

        # Initial upsampling with transposed conv
        self.upsample = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size-1,  # Causal padding
            output_padding=stride-1  # Ensure correct output size
        )
        self.upsample = nn.utils.weight_norm(self.upsample)

        # Dilated conv layers after upsampling
        self.layers = nn.ModuleList()
        current_dilation = dilation_growth**(num_conv_layers-1)  # Start with largest dilation

        for _ in range(num_conv_layers):
            # Dilated causal conv layer
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * current_dilation,
                dilation=current_dilation
            )
            conv = nn.utils.weight_norm(conv)

            layer = nn.Sequential(
                conv,
                nn.ELU()
            )
            self.layers.append(layer)

            # Decrease dilation for next layer
            current_dilation = current_dilation // dilation_growth

    def forward(self, x):
        # Upsample first
        x = self.upsample(x)

        # Apply dilated convolutions
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Add residual connection

        return x


class SeaNetEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            # 4 conv blocks with different strides (4,5,6,8)
            ConvBlock(stride=4),
            ConvBlock(stride=5),
            ConvBlock(stride=6),
            ConvBlock(stride=8)
        ])
        self.final_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=2  # Causal padding
        )
    def forward(self, x):
        # x shape: [B, 1, T]  # Raw audio

        # Initial projection to dim channels
        x = self.initial_proj(x) if hasattr(self, 'initial_proj') else x

        # Process through conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Final downsampling
        x = self.final_conv(x)

        # x shape: [B, dim, T/480]  # 12.5Hz after all strides
        return x.transpose(1, 2)  # Return [B, T, dim] for transformer


class SeaNetDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Initial upsampling from final encoder stride
        self.initial_upsample = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=2,
            output_padding=1
        )
        self.initial_upsample = nn.utils.weight_norm(self.initial_upsample)

        # Transposed conv blocks - reverse order of encoder strides
        self.conv_blocks = nn.ModuleList([
            TransposedConvBlock(stride=8),
            TransposedConvBlock(stride=6),
            TransposedConvBlock(stride=5),
            TransposedConvBlock(stride=4)
        ])

        # Final projection to waveform
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

        # x shape: [B, 1, T*480]  # Back to 24kHz
        return x
