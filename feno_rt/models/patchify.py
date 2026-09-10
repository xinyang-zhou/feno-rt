"""Two-dimensional token patching helpers."""

import torch.nn as nn


class Patchify(nn.Module):
    def __init__(self, P, H, W):
        super().__init__()
        self.P = P
        self.H = H
        self.W = W
        self.npatch_H = H // P
        self.npatch_W = W // P

    def forward(self, x):
        if x.shape[1] != self.H * self.W:
            raise ValueError("input token count must equal H * W")
        batch_size = len(x)
        if self.P == 1:
            return x
        x = x.view(batch_size, self.H, self.W, -1)
        x = x.view(
            batch_size,
            self.npatch_H,
            self.P,
            self.npatch_W,
            self.P,
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(batch_size, self.npatch_H * self.npatch_W, -1)


class UnPatchify(nn.Module):
    def __init__(self, P, H, W):
        super().__init__()
        self.P = P
        self.H = H
        self.W = W
        self.npatch_H = H // P
        self.npatch_W = W // P

    def forward(self, x):
        if self.P == 1:
            return x
        batch_size = len(x)
        x = x.view(
            batch_size,
            self.npatch_H,
            self.npatch_W,
            self.P,
            self.P,
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(batch_size, self.H * self.W, -1)
