"""Precision controls for FENO inference."""

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Union

import torch


class PrecisionMode(str, Enum):
    """Supported Stage 4 inference precision modes."""

    FP32 = "fp32"
    TF32 = "tf32"
    BF16 = "bf16"
    FP16 = "fp16"

    @classmethod
    def parse(cls, value: Union[str, "PrecisionMode"]) -> "PrecisionMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"precision must be one of: {choices}") from error


@dataclass(frozen=True)
class PrecisionPolicy:
    """Autocast and TF32 policy without mutating model parameter storage."""

    mode: PrecisionMode
    device: torch.device

    @classmethod
    def create(
        cls,
        mode: Union[str, PrecisionMode],
        device: Union[str, torch.device],
    ) -> "PrecisionPolicy":
        resolved_mode = PrecisionMode.parse(mode)
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda" and resolved_mode in (
            PrecisionMode.TF32,
            PrecisionMode.FP16,
        ):
            raise ValueError(f"{resolved_mode.value} inference requires a CUDA device")
        return cls(mode=resolved_mode, device=resolved_device)

    @property
    def autocast_dtype(self):
        if self.mode is PrecisionMode.BF16:
            return torch.bfloat16
        if self.mode is PrecisionMode.FP16:
            return torch.float16
        return None

    @property
    def signature(self) -> str:
        return self.mode.value

    @property
    def encoder_signature(self) -> str:
        return "tf32" if self.mode is PrecisionMode.TF32 else "fp32"

    @property
    def decoder_static_signature(self) -> str:
        # Compatible parameter sets may produce a medium-only residual stream
        # outside the finite FP16 range. Keep this one-time, cacheable stage in
        # FP32 while retaining FP16 for the online tail.
        return "fp32" if self.mode is PrecisionMode.FP16 else self.signature

    @contextmanager
    def activate_encoder(self) -> Iterator[None]:
        """Keep the FNO FFT path in FP32 while applying the TF32 policy."""
        if self.device.type != "cuda":
            yield
            return
        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        allow_tf32 = self.mode is PrecisionMode.TF32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        try:
            with torch.autocast(device_type="cuda", enabled=False):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            torch.backends.cudnn.allow_tf32 = previous_cudnn

    @contextmanager
    def activate_decoder_static(self) -> Iterator[None]:
        """Build FP16 decoder static state in FP32 to avoid residual overflow."""
        if self.mode is PrecisionMode.FP16:
            with self.activate_encoder():
                yield
            return
        with self.activate():
            yield

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Apply this policy only for the enclosed inference work."""
        autocast_dtype = self.autocast_dtype
        if self.device.type != "cuda":
            if autocast_dtype is None:
                yield
            else:
                with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                    yield
            return

        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        allow_tf32 = self.mode is PrecisionMode.TF32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        try:
            if autocast_dtype is None:
                yield
            else:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            torch.backends.cudnn.allow_tf32 = previous_cudnn
