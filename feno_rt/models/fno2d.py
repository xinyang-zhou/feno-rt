"""Compact Fourier neural operator used by the velocity encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )

    @staticmethod
    def compl_mul2d(value, weights):
        return torch.einsum("bixy,ioxy->boxy", value, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dropout_rate=0.1):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        return self.mlp2(self.dropout(x))


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, dropout_rate=0.1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.padding = 0
        self.dropout_rate = dropout_rate
        self.p = nn.Linear(self.in_channels, self.width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.mlp0 = MLP(width, width, width, dropout_rate)
        self.mlp1 = MLP(width, width, width, dropout_rate)
        self.mlp2 = MLP(width, width, width, dropout_rate)
        self.mlp3 = MLP(width, width, width, dropout_rate)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final_proj = nn.Linear(width, width)

    def forward(self, x):
        x = self.p(x).permute(0, 3, 1, 2)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for conv, mlp, linear in (
            (self.conv0, self.mlp0, self.w0),
            (self.conv1, self.mlp1, self.w1),
            (self.conv2, self.mlp2, self.w2),
        ):
            x = F.relu(mlp(conv(x)) + linear(x))
            x = self.dropout(x)
        x = self.mlp3(self.conv3(x)) + self.w3(x)
        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]
        return self.final_proj(x.permute(0, 2, 3, 1))
