"""Continuous sinusoidal coordinate embeddings used by the FENO models."""

import torch
import torch.nn as nn


class ContinuousSincosEmbed(nn.Module):
    def __init__(
        self,
        dim,
        ndim,
        max_wavelength: int = 10000,
        dtype=torch.float32,
        scale: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        if effective_dim_per_wave <= 0:
            raise ValueError("embedding dimension is too small")
        self.register_buffer(
            "omega",
            1.0
            / max_wavelength
            ** (
                torch.arange(0, effective_dim_per_wave, 2, dtype=dtype)
                / effective_dim_per_wave
            ),
        )
        self.scale = scale

    def forward(self, coords):
        out_dtype = coords.dtype
        if coords.shape[-1] != self.ndim:
            raise ValueError(f"expected {self.ndim} coordinate dimensions")
        out = (
            self.scale
            * coords.unsqueeze(-1).to(self.omega.dtype)
            @ self.omega.unsqueeze(0)
        )
        embedding = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        if coords.ndim not in {2, 3}:
            raise NotImplementedError("coordinates must be rank two or three")
        embedding = embedding.flatten(start_dim=-2).to(out_dtype)
        if self.padding > 0:
            padding = torch.zeros(
                *embedding.shape[:-1],
                self.padding,
                device=embedding.device,
                dtype=embedding.dtype,
            )
            embedding = torch.cat([embedding, padding], dim=-1)
        return embedding

    def __repr__(self):
        return f"{type(self).__name__}(dim={self.dim})"
