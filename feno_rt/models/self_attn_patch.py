"""Patchified self-attention block used by encoder and decoder."""

import torch
import torch.nn as nn
from kappamodules.layers import LinearProjection
from kappamodules.transformer import PerceiverBlock

from .patchify import Patchify, UnPatchify
from .pos_emb import ContinuousSincosEmbed


class PatchifiedSelfAttentionBlocks(nn.Module):
    def __init__(self, P, H, W, input_dim, dim, num_heads, enc_depth, init_weights):
        super().__init__()
        if H % P != 0 or W % P != 0:
            raise ValueError("dimensions must be divisible by patch size")
        self.P = P
        self.H = H
        self.W = W
        self.npatch_H = H // P
        self.npatch_W = W // P
        self.input_dim = input_dim
        self.dim = dim
        self.num_heads = num_heads
        self.enc_depth = enc_depth
        self.init_weights = init_weights
        self.block = PerceiverBlock(
            dim=dim * (P**2),
            num_heads=num_heads,
            kv_dim=dim * (P**2),
            init_weights=init_weights,
        )
        self.blocks = nn.ModuleList(self.block for _ in range(enc_depth))
        self.patchify = Patchify(P=P, H=H, W=W)
        self.unpatchify = UnPatchify(P=P, H=H, W=W)
        self.patch_pos = self._get_patch_pos()
        self.pos_emb = ContinuousSincosEmbed(dim=input_dim * (P**2), ndim=2)
        self.embbed_patch_pos = self.pos_emb(self.patch_pos)
        self.patch_linear = LinearProjection(
            2 * input_dim * (P**2),
            dim * (P**2),
            init_weights=init_weights,
            optional=True,
        )

    def forward(self, x):
        x = self.patchify(x)
        position = self.embbed_patch_pos.to(x.device).unsqueeze(0).repeat(
            x.shape[0], 1, 1
        )
        x = self.patch_linear(torch.cat([x, position], dim=-1))
        for block in self.blocks:
            x = block(q=x, kv=x)
        return self.unpatchify(x)

    def _get_patch_pos(self):
        return torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, self.npatch_H, dtype=torch.float32),
                torch.linspace(0, 1, self.npatch_W, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
