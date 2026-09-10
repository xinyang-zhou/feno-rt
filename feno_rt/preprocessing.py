"""Input preparation used by the FENO inference adapter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

from .config import FENOModelConfig

ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class NormalizationStats:
    v_mean: float
    v_std: float
    seis_mean: float
    seis_std: float

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NormalizationStats":
        with np.load(path) as values:
            return cls(
                v_mean=float(values["v_mean"]),
                v_std=float(values["v_std"]),
                seis_mean=float(values["seis_mean"]),
                seis_std=float(values["seis_std"]),
            )


def make_2d_grid(
    height: int,
    width: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    z = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    z, x = torch.meshgrid(z, x, indexing="ij")
    return torch.stack((z.reshape(-1), x.reshape(-1)), dim=-1)


def load_velocity_bin(
    path: Union[str, Path],
    original_size: int,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    velocity = np.fromfile(path, dtype=np.float32)
    expected = original_size * original_size
    if velocity.size != expected:
        raise ValueError(f"Expected {expected} float32 values in {path}, found {velocity.size}")
    velocity = velocity.reshape(original_size, original_size, order="F")
    z_idx = np.linspace(0, original_size - 1, target_height, dtype=int)
    x_idx = np.linspace(0, original_size - 1, target_width, dtype=int)
    return velocity[z_idx, :][:, x_idx].astype(np.float32, copy=False)


def build_velocity_input(
    velocity: ArrayLike,
    config: FENOModelConfig,
    normalization: Optional[NormalizationStats],
    *,
    already_normalized: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    velocity_tensor = torch.as_tensor(velocity, dtype=dtype, device=device)
    expected_shape = (config.velocity_height, config.velocity_width)
    if tuple(velocity_tensor.shape) != expected_shape:
        raise ValueError(
            f"velocity must have shape {expected_shape}, got {tuple(velocity_tensor.shape)}"
        )
    if not already_normalized:
        if normalization is None:
            raise ValueError("normalization is required for a physical-scale velocity field")
        if normalization.v_std <= 0:
            raise ValueError("v_std must be positive")
        velocity_tensor = (velocity_tensor - normalization.v_mean) / normalization.v_std

    coordinates = make_2d_grid(
        config.velocity_height,
        config.velocity_width,
        device=velocity_tensor.device,
        dtype=velocity_tensor.dtype,
    )
    values = velocity_tensor.reshape(-1, 1)
    return torch.cat((values, coordinates), dim=-1).unsqueeze(0).contiguous()


def prepare_source_receiver_batch(
    source_positions: ArrayLike,
    config: FENOModelConfig,
    *,
    receiver_positions: Optional[ArrayLike] = None,
    positions_are_normalized: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sources = torch.as_tensor(source_positions, device=device, dtype=dtype)
    if sources.ndim == 1:
        sources = sources.unsqueeze(0)
    if sources.ndim != 2 or sources.shape[-1] != 2:
        raise ValueError(f"source_positions must have shape (batch, 2), got {tuple(sources.shape)}")

    if receiver_positions is None:
        receivers = torch.zeros(config.num_receivers, 2, device=device, dtype=dtype)
        receivers[:, 0] = float(config.receiver_depth)
        receivers[:, 1] = torch.arange(config.num_receivers, device=device, dtype=dtype)
    else:
        receivers = torch.as_tensor(receiver_positions, device=device, dtype=dtype)

    batch_size = sources.shape[0]
    if receivers.ndim == 2:
        if receivers.shape != (config.num_receivers, 2):
            raise ValueError(
                "receiver_positions must have shape "
                f"({config.num_receivers}, 2), got {tuple(receivers.shape)}"
            )
        receivers = receivers.unsqueeze(0).expand(batch_size, -1, -1)
    elif receivers.ndim == 3:
        if receivers.shape != (batch_size, config.num_receivers, 2):
            raise ValueError(
                "batched receiver_positions must have shape "
                f"({batch_size}, {config.num_receivers}, 2), got {tuple(receivers.shape)}"
            )
    else:
        raise ValueError("receiver_positions must have rank 2 or 3")

    sources = sources.unsqueeze(1).expand(-1, config.num_receivers, -1)
    if not positions_are_normalized:
        sources = sources / config.domain_extent
        receivers = receivers / config.domain_extent
    return sources, receivers


def prepare_frequencies(
    frequencies: ArrayLike,
    batch_size: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    values = torch.as_tensor(frequencies, device=device, dtype=dtype)
    if values.ndim == 0:
        values = values.expand(batch_size)
    else:
        values = values.reshape(-1)
    if values.numel() != batch_size:
        raise ValueError(f"Expected {batch_size} frequencies, got {values.numel()}")
    if torch.any(values <= 0):
        raise ValueError("frequencies must be positive")
    return values


def denormalize_seismograms(
    values: torch.Tensor, normalization: NormalizationStats
) -> torch.Tensor:
    return values * normalization.seis_std + normalization.seis_mean
