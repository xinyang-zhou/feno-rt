"""CUDA Graph buckets and persistent I/O buffers for the cached online tail."""

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch

from feno_rt.models.decoder_attention_freq import WaveletContext


@dataclass(frozen=True)
class CUDAGraphKey:
    batch_bucket: int
    prefix_signature: Tuple[Tuple[int, ...], str]
    frequency_signature: Tuple[Tuple[int, ...], str]
    key_signature: Tuple[Tuple[int, ...], str]
    value_signature: Tuple[Tuple[int, ...], str]


@dataclass
class _CapturedTail:
    graph: torch.cuda.CUDAGraph
    prefix: torch.Tensor
    frequency: torch.Tensor
    wavelet_tokens: torch.Tensor
    wavelet_k: torch.Tensor
    wavelet_v: torch.Tensor
    output: torch.Tensor

    @property
    def static_buffer_bytes(self) -> int:
        tensors = (
            self.prefix,
            self.frequency,
            self.wavelet_tokens,
            self.wavelet_k,
            self.wavelet_v,
            self.output,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


class CUDAGraphTailRunner:
    """Capture one graph per batch bucket and tensor signature.

    Inputs and output live in persistent buffers. The returned tensor is cloned so
    a later replay cannot mutate a result still owned by a caller.
    """

    def __init__(
        self,
        buckets: Iterable[int] = (1, 2, 4, 8, 16, 32, 64),
        *,
        warmup_iterations: int = 2,
        fallback_on_error: bool = True,
    ) -> None:
        resolved = tuple(sorted(set(int(value) for value in buckets)))
        if not resolved or resolved[0] <= 0:
            raise ValueError("CUDA Graph buckets must contain positive integers")
        if warmup_iterations < 1:
            raise ValueError("warmup_iterations must be at least one")
        self.buckets = resolved
        self.warmup_iterations = warmup_iterations
        self.fallback_on_error = fallback_on_error
        self._graphs: Dict[CUDAGraphKey, _CapturedTail] = {}
        self._failed_keys = set()
        self._lock = RLock()
        self._requests = 0
        self._replays = 0
        self._captures = 0
        self._capture_failures = 0
        self._fallbacks = 0
        self._padded_requests = 0
        self._padded_slots = 0

    def bucket_for(self, batch_size: int) -> int:
        for bucket in self.buckets:
            if batch_size <= bucket:
                return bucket
        raise ValueError(
            f"batch size {batch_size} exceeds largest CUDA Graph bucket {self.buckets[-1]}"
        )

    @staticmethod
    def _signature(tensor: torch.Tensor) -> Tuple[Tuple[int, ...], str]:
        return tuple(tensor.shape[1:]), str(tensor.dtype)

    def _key(
        self,
        bucket: int,
        prefix: torch.Tensor,
        frequency: torch.Tensor,
        wavelet: WaveletContext,
    ) -> CUDAGraphKey:
        return CUDAGraphKey(
            batch_bucket=bucket,
            prefix_signature=self._signature(prefix),
            frequency_signature=self._signature(frequency),
            key_signature=self._signature(wavelet.k),
            value_signature=self._signature(wavelet.v),
        )

    @staticmethod
    def _allocate_like(value: torch.Tensor, batch_bucket: int) -> torch.Tensor:
        return torch.empty(
            (batch_bucket,) + tuple(value.shape[1:]),
            device=value.device,
            dtype=value.dtype,
        )

    @staticmethod
    def _copy_and_pad(target: torch.Tensor, source: torch.Tensor) -> None:
        batch_size = source.shape[0]
        target[:batch_size].copy_(source)
        if batch_size < target.shape[0]:
            target[batch_size:].copy_(source[-1:].expand_as(target[batch_size:]))

    def _capture(
        self,
        fn: Callable[[torch.Tensor, torch.Tensor, Optional[WaveletContext]], torch.Tensor],
        bucket: int,
        prefix: torch.Tensor,
        frequency: torch.Tensor,
        wavelet: WaveletContext,
    ) -> _CapturedTail:
        static_prefix = self._allocate_like(prefix, bucket)
        static_frequency = self._allocate_like(frequency, bucket)
        static_tokens = self._allocate_like(wavelet.tokens, bucket)
        static_k = self._allocate_like(wavelet.k, bucket)
        static_v = self._allocate_like(wavelet.v, bucket)
        self._copy_and_pad(static_prefix, prefix)
        self._copy_and_pad(static_frequency, frequency)
        self._copy_and_pad(static_tokens, wavelet.tokens)
        self._copy_and_pad(static_k, wavelet.k)
        self._copy_and_pad(static_v, wavelet.v)
        static_wavelet = WaveletContext(tokens=static_tokens, k=static_k, v=static_v)

        capture_stream = torch.cuda.Stream(device=prefix.device)
        capture_stream.wait_stream(torch.cuda.current_stream(prefix.device))
        with torch.cuda.stream(capture_stream):
            for _ in range(self.warmup_iterations):
                output = fn(static_prefix, static_frequency, static_wavelet)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            stream=capture_stream,
            capture_error_mode="thread_local",
        ):
            output = fn(static_prefix, static_frequency, static_wavelet)
        torch.cuda.current_stream(prefix.device).wait_stream(capture_stream)
        return _CapturedTail(
            graph=graph,
            prefix=static_prefix,
            frequency=static_frequency,
            wavelet_tokens=static_tokens,
            wavelet_k=static_k,
            wavelet_v=static_v,
            output=output,
        )

    def run(
        self,
        fn: Callable[[torch.Tensor, torch.Tensor, Optional[WaveletContext]], torch.Tensor],
        prefix: torch.Tensor,
        frequency: torch.Tensor,
        wavelet: WaveletContext,
    ) -> torch.Tensor:
        if prefix.device.type != "cuda":
            raise ValueError("CUDA Graph replay requires CUDA tensors")
        batch_size = prefix.shape[0]
        if batch_size < 1:
            raise ValueError("CUDA Graph replay requires a non-empty batch")
        bucket = self.bucket_for(batch_size)
        key = self._key(bucket, prefix, frequency, wavelet)
        with self._lock:
            self._requests += 1
            if batch_size < bucket:
                self._padded_requests += 1
                self._padded_slots += bucket - batch_size
            captured = self._graphs.get(key)
            if captured is None and key not in self._failed_keys:
                try:
                    captured = self._capture(fn, bucket, prefix, frequency, wavelet)
                except Exception:
                    self._capture_failures += 1
                    self._failed_keys.add(key)
                    if not self.fallback_on_error:
                        raise
                else:
                    self._graphs[key] = captured
                    self._captures += 1

            if captured is None:
                self._fallbacks += 1
                return fn(prefix, frequency, wavelet)

            self._copy_and_pad(captured.prefix, prefix)
            self._copy_and_pad(captured.frequency, frequency)
            self._copy_and_pad(captured.wavelet_tokens, wavelet.tokens)
            self._copy_and_pad(captured.wavelet_k, wavelet.k)
            self._copy_and_pad(captured.wavelet_v, wavelet.v)
            captured.graph.replay()
            self._replays += 1
            return captured.output[:batch_size].clone()

    def clear(self) -> int:
        with self._lock:
            count = len(self._graphs)
            self._graphs.clear()
            self._failed_keys.clear()
            return count

    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "buckets": list(self.buckets),
                "requests": self._requests,
                "captures": self._captures,
                "replays": self._replays,
                "capture_failures": self._capture_failures,
                "fallbacks": self._fallbacks,
                "padded_requests": self._padded_requests,
                "padded_slots": self._padded_slots,
                "resident_graphs": len(self._graphs),
                "static_buffer_bytes": sum(
                    item.static_buffer_bytes for item in self._graphs.values()
                ),
            }
