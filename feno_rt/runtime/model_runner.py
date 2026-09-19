"""Cache-aware FENO model runner."""

from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from feno_rt.config import DEFAULT_INFERENCE_CONFIG, FENOModelConfig
from feno_rt.models.decoder_attention_freq import DecoderStaticContext, WaveletContext
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.preprocessing import (
    ArrayLike,
    NormalizationStats,
    build_velocity_input,
    denormalize_seismograms,
    prepare_frequencies,
    prepare_source_receiver_batch,
)
from feno_rt.runtime.cuda_graph import CUDAGraphTailRunner
from feno_rt.runtime.context_cache import (
    FENOCacheBundle,
    FENOCacheConfig,
    GeometryPrefixCacheKey,
    MediumCacheKey,
    WaveletCacheKey,
)


@dataclass(frozen=True)
class MediumContext:
    """Runner-owned encoder and decoder-static state for one velocity model."""

    medium_id: str
    latent: torch.Tensor
    decoder_context: DecoderStaticContext
    cache_key: MediumCacheKey
    owner_token: object = field(repr=False, compare=False)

    @property
    def device(self) -> torch.device:
        return self.latent.device

    @property
    def dtype(self) -> torch.dtype:
        return self.latent.dtype


class FENOModelRunner:
    """Run FENO with runner-local medium, geometry, and wavelet caches."""

    CACHE_LEVELS = frozenset(("medium", "geometry", "all"))

    def _nvtx_range(self, name: str):
        """Return an NVTX range on CUDA and a no-op context on CPU."""
        if self.device.type == "cuda":
            return torch.cuda.nvtx.range(name)
        return nullcontext()

    def __init__(
        self,
        model: FENOFreq,
        *,
        config: FENOModelConfig = DEFAULT_INFERENCE_CONFIG,
        normalization: Optional[NormalizationStats] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_config: Optional[FENOCacheConfig] = None,
    ) -> None:
        self.config = config
        self.normalization = normalization
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.dtype = next(self.model.parameters()).dtype
        self.caches = FENOCacheBundle(cache_config)
        self._context_owner = object()
        self._cuda_graph_runner: Optional[CUDAGraphTailRunner] = None
        self._metrics_lock = RLock()
        self._runtime_metrics = self._empty_runtime_metrics()

    def _invalidate_cuda_graphs(self) -> None:
        if self._cuda_graph_runner is not None:
            self._cuda_graph_runner.clear()

    def enable_cuda_graphs(
        self,
        *,
        buckets: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        warmup_iterations: int = 2,
        fallback_on_error: bool = True,
    ) -> Dict[str, Any]:
        if self.device.type != "cuda":
            raise ValueError("CUDA Graphs require a CUDA model runner")
        self._invalidate_cuda_graphs()
        self._cuda_graph_runner = CUDAGraphTailRunner(
            buckets=buckets,
            warmup_iterations=warmup_iterations,
            fallback_on_error=fallback_on_error,
        )
        return self._cuda_graph_runner.metrics()

    def disable_cuda_graphs(self) -> int:
        if self._cuda_graph_runner is None:
            return 0
        count = self._cuda_graph_runner.clear()
        self._cuda_graph_runner = None
        return count

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._cuda_graph_runner is not None

    def execution_config(self) -> Dict[str, Any]:
        return {
            "cuda_graph": (
                self._cuda_graph_runner.metrics()
                if self._cuda_graph_runner is not None
                else {"enabled": False}
            ),
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        config: FENOModelConfig = DEFAULT_INFERENCE_CONFIG,
        normalization: Optional[NormalizationStats] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_config: Optional[FENOCacheConfig] = None,
    ) -> "FENOModelRunner":
        resolved_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = FENOFreq(config.encoder_config(), config.decoder_config())
        state_dict = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
        model.load_state_dict(state_dict)
        return cls(
            model,
            config=config,
            normalization=normalization,
            device=resolved_device,
            cache_config=cache_config,
        )

    @staticmethod
    def _tensor_digest(*values: torch.Tensor) -> str:
        digest = sha256()
        for value in values:
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _medium_key(self, velocity: ArrayLike, already_normalized: bool) -> MediumCacheKey:
        canonical = build_velocity_input(
            velocity,
            self.config,
            self.normalization,
            already_normalized=already_normalized,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        return MediumCacheKey(velocity_digest=self._tensor_digest(canonical[..., :1]))

    def medium_cache_key(
        self, velocity: ArrayLike, *, already_normalized: bool = False
    ) -> MediumCacheKey:
        """Return the canonical velocity identity within this runner."""
        return self._medium_key(velocity, already_normalized)

    def _context_key(self, context: MediumContext) -> MediumCacheKey:
        self._validate_context(context)
        return context.cache_key

    @staticmethod
    def _empty_runtime_metrics() -> Dict[str, int]:
        return {
            "geometry_requests": 0,
            "geometry_unique": 0,
            "geometry_in_batch_dedup_hits": 0,
            "wavelet_requests": 0,
            "wavelet_unique": 0,
            "wavelet_in_batch_dedup_hits": 0,
        }

    def _record_dedup(self, kind: str, requests: int, unique: int) -> None:
        with self._metrics_lock:
            self._runtime_metrics[f"{kind}_requests"] += requests
            self._runtime_metrics[f"{kind}_unique"] += unique
            self._runtime_metrics[f"{kind}_in_batch_dedup_hits"] += requests - unique

    def reset_cache_stats(self) -> None:
        self.caches.reset_stats()
        with self._metrics_lock:
            self._runtime_metrics = self._empty_runtime_metrics()

    def cache_metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            runtime_metrics = dict(self._runtime_metrics)
        runtime_metrics["geometry_dedup_rate"] = (
            runtime_metrics["geometry_in_batch_dedup_hits"]
            / runtime_metrics["geometry_requests"]
            if runtime_metrics["geometry_requests"]
            else 0.0
        )
        runtime_metrics["wavelet_dedup_rate"] = (
            runtime_metrics["wavelet_in_batch_dedup_hits"]
            / runtime_metrics["wavelet_requests"]
            if runtime_metrics["wavelet_requests"]
            else 0.0
        )
        return {"caches": self.caches.snapshots(), "runtime": runtime_metrics}

    def clear_caches(self, *, force: bool = False) -> Dict[str, int]:
        return self.caches.clear(force=force)

    @torch.inference_mode()
    def prepare_medium(
        self,
        velocity: ArrayLike,
        *,
        medium_id: Optional[str] = None,
        already_normalized: bool = False,
        use_cache: bool = True,
    ) -> MediumContext:
        key = self._medium_key(velocity, already_normalized)
        if use_cache:
            cached = self.caches.medium.get(key)
            if cached is not None:
                return replace(cached, medium_id=medium_id) if medium_id else cached

        model_input = build_velocity_input(
            velocity,
            self.config,
            self.normalization,
            already_normalized=already_normalized,
            device=self.device,
            dtype=self.dtype,
        )
        latent = self.model.encoder(model_input)
        decoder_context = self.model.decoder.prepare_static_context(latent)
        context = MediumContext(
            medium_id=medium_id or key.velocity_digest,
            latent=latent,
            decoder_context=decoder_context,
            cache_key=key,
            owner_token=self._context_owner,
        )
        if use_cache:
            self.caches.medium.put(key, context)
        return context

    def _validate_context(self, context: MediumContext) -> None:
        if not isinstance(context, MediumContext):
            raise TypeError("context must be a MediumContext created by this runner")
        if context.owner_token is not self._context_owner:
            raise ValueError("MediumContext was created by a different model runner")
        if context.latent.ndim != 3 or context.latent.shape[0] != 1:
            raise ValueError(
                "MediumContext.latent must have shape (1, latent_tokens, dim); "
                f"got {tuple(context.latent.shape)}"
            )
        if context.device != self.device:
            raise ValueError(f"Context is on {context.device}, but the runner is on {self.device}")

    @torch.inference_mode()
    def prepare_decoder_context(self, context: MediumContext) -> DecoderStaticContext:
        self._validate_context(context)
        return context.decoder_context

    def _prepare_request_tensors(
        self,
        source_positions: ArrayLike,
        frequencies: ArrayLike,
        receiver_positions: Optional[ArrayLike],
        positions_are_normalized: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sources_cpu, receivers_cpu = prepare_source_receiver_batch(
            source_positions,
            self.config,
            receiver_positions=receiver_positions,
            positions_are_normalized=positions_are_normalized,
            device=torch.device("cpu"),
            dtype=self.dtype,
        )
        frequencies_cpu = prepare_frequencies(
            frequencies,
            sources_cpu.shape[0],
            device=torch.device("cpu"),
            dtype=self.dtype,
        )
        return (
            sources_cpu.to(self.device),
            receivers_cpu.to(self.device),
            frequencies_cpu.to(self.device),
            sources_cpu,
            receivers_cpu,
            frequencies_cpu,
        )

    def _geometry_prefix_batch(
        self,
        decoder_context: DecoderStaticContext,
        medium_key: MediumCacheKey,
        sources: torch.Tensor,
        receivers: torch.Tensor,
        sources_cpu: torch.Tensor,
        receivers_cpu: torch.Tensor,
    ) -> torch.Tensor:
        keys: List[GeometryPrefixCacheKey] = []
        unique_keys: List[GeometryPrefixCacheKey] = []
        unique_request_indices: List[int] = []
        key_to_index: Dict[GeometryPrefixCacheKey, int] = {}
        inverse: List[int] = []
        for index in range(sources_cpu.shape[0]):
            key = GeometryPrefixCacheKey(
                medium=medium_key,
                geometry_digest=self._tensor_digest(sources_cpu[index], receivers_cpu[index]),
            )
            keys.append(key)
            unique_index = key_to_index.get(key)
            if unique_index is None:
                unique_index = len(unique_keys)
                key_to_index[key] = unique_index
                unique_keys.append(key)
                unique_request_indices.append(index)
            inverse.append(unique_index)
        self._record_dedup("geometry", len(keys), len(unique_keys))

        values: List[Optional[torch.Tensor]] = []
        missing_unique_indices: List[int] = []
        for unique_index, key in enumerate(unique_keys):
            value = self.caches.geometry.get(key)
            values.append(value)
            if value is None:
                missing_unique_indices.append(unique_index)

        if missing_unique_indices:
            request_indices = torch.tensor(
                [unique_request_indices[index] for index in missing_unique_indices],
                device=sources.device,
                dtype=torch.long,
            )
            computed = self.model.decoder.prepare_geometry_prefix(
                decoder_context,
                sources.index_select(0, request_indices),
                receivers.index_select(0, request_indices),
            )
            for offset, unique_index in enumerate(missing_unique_indices):
                value = computed[offset : offset + 1].clone()
                values[unique_index] = value
                self.caches.geometry.put(unique_keys[unique_index], value)

        unique_values = torch.cat([value for value in values if value is not None], dim=0)
        inverse_tensor = torch.tensor(inverse, device=unique_values.device, dtype=torch.long)
        return unique_values.index_select(0, inverse_tensor)

    def _wavelet_key(self, frequency: torch.Tensor) -> WaveletCacheKey:
        effective = frequency.detach().to(device="cpu", dtype=torch.float32).reshape(1).contiguous()
        bits = effective.view(torch.uint8).numpy().tobytes().hex()
        return WaveletCacheKey(frequency_bits=bits)

    def request_cache_keys(
        self,
        context: MediumContext,
        source_position: ArrayLike,
        frequency: ArrayLike,
        *,
        receiver_positions: Optional[ArrayLike] = None,
        positions_are_normalized: bool = False,
    ) -> Dict[str, Any]:
        """Build the exact cache identities used by one queued request."""
        medium_key = self._context_key(context)
        sources_cpu, receivers_cpu = prepare_source_receiver_batch(
            source_position,
            self.config,
            receiver_positions=receiver_positions,
            positions_are_normalized=positions_are_normalized,
            device=torch.device("cpu"),
            dtype=self.dtype,
        )
        frequencies_cpu = prepare_frequencies(
            frequency,
            sources_cpu.shape[0],
            device=torch.device("cpu"),
            dtype=self.dtype,
        )
        if sources_cpu.shape[0] != 1:
            raise ValueError("a queued request must contain exactly one source position")
        geometry_key = GeometryPrefixCacheKey(
            medium=medium_key,
            geometry_digest=self._tensor_digest(sources_cpu[0], receivers_cpu[0]),
        )
        keys: Dict[str, Any] = {
            "medium": medium_key,
            "geometry": geometry_key,
        }
        if self.model.decoder.use_wavelet_attn:
            keys["wavelet"] = self._wavelet_key(frequencies_cpu[0])
        return keys

    def cache_residency(self, keys: Dict[str, Any]) -> Dict[str, bool]:
        """Return residency without affecting LRU ordering or cache metrics."""
        caches = {
            "medium": self.caches.medium,
            "geometry": self.caches.geometry,
            "wavelet": self.caches.wavelet,
        }
        return {
            name: caches[name].peek(key) is not None
            for name, key in keys.items()
            if name in caches
        }

    def _wavelet_context_batch(
        self, frequencies: torch.Tensor, frequencies_cpu: torch.Tensor
    ) -> WaveletContext:
        keys: List[WaveletCacheKey] = []
        unique_keys: List[WaveletCacheKey] = []
        unique_request_indices: List[int] = []
        key_to_index: Dict[WaveletCacheKey, int] = {}
        inverse: List[int] = []
        for index in range(frequencies_cpu.shape[0]):
            key = self._wavelet_key(frequencies_cpu[index])
            keys.append(key)
            unique_index = key_to_index.get(key)
            if unique_index is None:
                unique_index = len(unique_keys)
                key_to_index[key] = unique_index
                unique_keys.append(key)
                unique_request_indices.append(index)
            inverse.append(unique_index)
        self._record_dedup("wavelet", len(keys), len(unique_keys))

        values: List[Optional[WaveletContext]] = []
        missing_unique_indices: List[int] = []
        for unique_index, key in enumerate(unique_keys):
            value = self.caches.wavelet.get(key)
            values.append(value)
            if value is None:
                missing_unique_indices.append(unique_index)

        if missing_unique_indices:
            request_indices = torch.tensor(
                [unique_request_indices[index] for index in missing_unique_indices],
                device=frequencies.device,
                dtype=torch.long,
            )
            computed = self.model.decoder.prepare_wavelet_context(
                frequencies.index_select(0, request_indices)
            )
            for offset, unique_index in enumerate(missing_unique_indices):
                value = WaveletContext(
                    tokens=computed.tokens[offset : offset + 1].clone(),
                    k=computed.k[offset : offset + 1].clone(),
                    v=computed.v[offset : offset + 1].clone(),
                )
                values[unique_index] = value
                self.caches.wavelet.put(unique_keys[unique_index], value)

        resolved = [value for value in values if value is not None]
        inverse_tensor = torch.tensor(inverse, device=frequencies.device, dtype=torch.long)
        return WaveletContext(
            tokens=torch.cat([value.tokens for value in resolved], dim=0).index_select(0, inverse_tensor),
            k=torch.cat([value.k for value in resolved], dim=0).index_select(0, inverse_tensor),
            v=torch.cat([value.v for value in resolved], dim=0).index_select(0, inverse_tensor),
        )

    @torch.inference_mode()
    def forward_batch(
        self,
        context: MediumContext,
        source_positions: ArrayLike,
        frequencies: ArrayLike,
        *,
        receiver_positions: Optional[ArrayLike] = None,
        positions_are_normalized: bool = False,
        denormalize: bool = False,
        cache_level: str = "all",
    ) -> torch.Tensor:
        self._validate_context(context)
        if cache_level not in self.CACHE_LEVELS:
            raise ValueError(f"cache_level must be one of {sorted(self.CACHE_LEVELS)}")

        with self._nvtx_range("feno.input_prepare"):
            sources, receivers, frequency_tensor, sources_cpu, receivers_cpu, frequencies_cpu = (
                self._prepare_request_tensors(
                    source_positions,
                    frequencies,
                    receiver_positions,
                    positions_are_normalized,
                )
            )

        decoder_context = context.decoder_context

        with self._nvtx_range("feno.geometry_resolve"):
            if cache_level == "medium":
                prefix = self.model.decoder.prepare_geometry_prefix(
                    decoder_context, sources, receivers
                )
            else:
                prefix = self._geometry_prefix_batch(
                    decoder_context,
                    context.cache_key,
                    sources,
                    receivers,
                    sources_cpu,
                    receivers_cpu,
                )

        with self._nvtx_range("feno.wavelet_resolve"):
            wavelet_context = (
                self._wavelet_context_batch(frequency_tensor, frequencies_cpu)
                if cache_level == "all" and self.model.decoder.use_wavelet_attn
                else None
            )

        with self._nvtx_range("feno.decoder_tail"):
            if self._cuda_graph_runner is not None and wavelet_context is not None:
                output = self._cuda_graph_runner.run(
                    self.model.decoder.forward_from_geometry_prefix,
                    prefix,
                    frequency_tensor,
                    wavelet_context,
                )
            else:
                output = self.model.decoder.forward_from_geometry_prefix(
                    prefix, frequency_tensor, wavelet_context
                )

        with self._nvtx_range("feno.output"):
            if denormalize:
                if self.normalization is None:
                    raise ValueError("normalization is required to denormalize model output")
                output = denormalize_seismograms(output, self.normalization)

        return output

    @torch.inference_mode()
    def forward_uncached(
        self,
        velocity: ArrayLike,
        source_positions: ArrayLike,
        frequencies: ArrayLike,
        *,
        receiver_positions: Optional[ArrayLike] = None,
        already_normalized: bool = False,
        positions_are_normalized: bool = False,
        denormalize: bool = False,
    ) -> torch.Tensor:
        """Reference path that recomputes encoder and the original decoder."""
        model_input = build_velocity_input(
            velocity,
            self.config,
            self.normalization,
            already_normalized=already_normalized,
            device=self.device,
            dtype=self.dtype,
        )
        sources, receivers = prepare_source_receiver_batch(
            source_positions,
            self.config,
            receiver_positions=receiver_positions,
            positions_are_normalized=positions_are_normalized,
            device=self.device,
            dtype=self.dtype,
        )
        frequency_tensor = prepare_frequencies(
            frequencies,
            sources.shape[0],
            device=self.device,
            dtype=self.dtype,
        )
        latent = self.model.encoder(model_input)
        output = self.model.decoder(
            latent.expand(sources.shape[0], -1, -1),
            sources,
            receivers,
            frequency_tensor,
        )
        if denormalize:
            if self.normalization is None:
                raise ValueError("normalization is required to denormalize model output")
            output = denormalize_seismograms(output, self.normalization)
        return output
