"""
Model definitions for squat form classification (MLP, GRU, TCN).
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Middle-frame baseline classifier."""

    def __init__(self, input_dim: int = 51, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, 51) → logits: (batch, num_classes)"""
        mid = x[:, x.shape[1] // 2]  # Select middle frame
        return self.net(mid)


class GRU(nn.Module):
    """
    Bidirectional GRU sequence classifier.

    Reads the full pose sequence and averages hidden states across
    all time steps before classification.
    """

    def __init__(self, input_dim: int = 51, hidden_dim: int = 96, num_classes: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, 51) → logits: (batch, num_classes)"""
        out, _ = self.rnn(x)          # (batch, T, hidden*2)
        pooled = out.mean(dim=1)      # Mean-pool over time
        return self.head(pooled)


class TCNBlock(nn.Module):
    """Residual block: two dilated Conv1d layers with skip connection."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_path(x) + self.skip(x)


class TCN(nn.Module):
    """
    Temporal Convolutional Network with dilated residual blocks.

    Three blocks with increasing dilation (1 → 2 → 4) give an
    exponentially growing receptive field across the 16-frame sequence.
    """

    def __init__(self, input_dim: int = 51, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(input_dim, 64, dilation=1),
            TCNBlock(64, 96, dilation=2),
            TCNBlock(96, 128, dilation=4),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, 51) → logits: (batch, num_classes)"""
        # Conv1d expects (batch, channels, time)
        return self.net(x.transpose(1, 2))


# Registry for easy lookup by name
MODEL_REGISTRY = {
    "MLP": MLP,
    "GRU": GRU,
    "TCN": TCN,
}
