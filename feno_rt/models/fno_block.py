"""Coordinate-conditioned FNO encoder block."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno2d import FNO2d
from .pos_emb import ContinuousSincosEmbed


class FNOBlock(nn.Module):
    def __init__(
        self,
        H,
        W,
        feat_channels,
        out_channels,
        coord_dim,
        pos_embed_dim,
        modes1,
        modes2,
        width,
        dropout_rate,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.coord_dim = coord_dim
        self.pos_emb = ContinuousSincosEmbed(dim=pos_embed_dim, ndim=coord_dim)
        self.fno = FNO2d(
            modes1=modes1,
            modes2=modes2,
            width=width,
            in_channels=feat_channels + pos_embed_dim,
            dropout_rate=dropout_rate,
        )
        self.fno.q = nn.Sequential(
            nn.Conv2d(width, width * 4, 1),
            nn.ReLU(),
            nn.Conv2d(width * 4, out_channels, 1),
        )
        self.fno.padding = 0

    def forward(self, x, query_H=None, query_W=None):
        batch_size = x.shape[0]
        features = x[..., : self.feat_channels].view(
            batch_size, self.H, self.W, self.feat_channels
        )
        coords = x[..., self.feat_channels :].reshape(
            batch_size * self.H * self.W, self.coord_dim
        )
        position = self.pos_emb(coords).view(batch_size, self.H, self.W, -1)
        output = self.fno(torch.cat([features, position], dim=-1))
        if query_H is None and query_W is None:
            return output.view(batch_size, self.H * self.W, self.out_channels)
        output = output.permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(output, (query_H, query_W))
        return pooled.permute(0, 2, 3, 1).contiguous().view(
            batch_size, query_H * query_W, self.out_channels
        )
