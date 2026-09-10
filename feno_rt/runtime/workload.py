"""Deterministic multi-medium workloads and Stage 7 capacity planning."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_FREQUENCIES = (5.0, 10.0, 20.0, 40.0)


@dataclass(frozen=True)
class MultiMediumWorkloadConfig:
    """Parameters for a locality-heavy, capacity-stress serving trace.

    Velocity variants are deterministic perturbations of one supplied base
    medium. They are suitable for cache and scheduling measurements, not for
    claiming accuracy on independent geological models.
    """

    medium_count: int = 8
    request_count: int = 256
    geometry_cardinality: int = 32
    frequencies: Tuple[float, ...] = DEFAULT_FREQUENCIES
    medium_zipf_alpha: float = 1.1
    geometry_zipf_alpha: float = 1.2
    arrival_rate: float = 0.0
    perturbation_fraction: float = 0.002
    seed: int = 20260911

    def __post_init__(self) -> None:
        frequencies = tuple(float(value) for value in self.frequencies)
        object.__setattr__(self, "frequencies", frequencies)
        if self.medium_count < 2:
            raise ValueError("medium_count must be at least two")
        if self.request_count < self.medium_count:
            raise ValueError("request_count must cover every medium")
        if self.geometry_cardinality < 1:
            raise ValueError("geometry_cardinality must be positive")
        if not frequencies or any(value <= 0.0 for value in frequencies):
            raise ValueError("frequencies must contain positive values")
        if len(set(frequencies)) != len(frequencies):
            raise ValueError("frequencies must be unique")
        if self.medium_zipf_alpha <= 0.0 or self.geometry_zipf_alpha <= 0.0:
            raise ValueError("Zipf exponents must be positive")
        if self.arrival_rate < 0.0:
            raise ValueError("arrival_rate must be non-negative")
        if not 0.0 < self.perturbation_fraction <= 0.05:
            raise ValueError("perturbation_fraction must be in (0, 0.05]")


def array_digest(value: np.ndarray) -> str:
    """Hash a numeric array including its shape and dtype."""

    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def build_velocity_variants(
    base_velocity: np.ndarray,
    config: MultiMediumWorkloadConfig,
) -> Dict[str, np.ndarray]:
    """Build smooth, low-amplitude variants without retaining private data.

    The first medium is the exact base array. Remaining media combine smooth
    sinusoidal perturbations with deterministic phases. The returned arrays are
    independent float32 buffers so callers can safely hand them to async code.
    """

    base = np.asarray(base_velocity, dtype=np.float32)
    if base.ndim != 2 or min(base.shape) < 2:
        raise ValueError("base_velocity must be a two-dimensional grid")
    if not np.isfinite(base).all():
        raise ValueError("base_velocity must contain only finite values")

    z = np.linspace(0.0, 1.0, base.shape[0], dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, base.shape[1], dtype=np.float32)[None, :]
    rng = np.random.default_rng(config.seed)
    variants: Dict[str, np.ndarray] = {"medium-000": base.copy()}
    positive_floor = (
        float(np.min(base[base > 0.0])) * 0.5 if np.any(base > 0.0) else None
    )
    for index in range(1, config.medium_count):
        phase_z, phase_x, phase_cross = rng.uniform(0.0, 2.0 * np.pi, size=3)
        z_order = 1 + index % 3
        x_order = 1 + (index * 2) % 4
        pattern = (
            np.sin(z_order * np.pi * z + phase_z)
            * np.cos(x_order * np.pi * x + phase_x)
            + 0.5
            * np.sin(
                (z_order + 1) * np.pi * z
                + (x_order + 1) * np.pi * x
                + phase_cross
            )
        ) / 1.5
        variant = base * (
            1.0 + np.float32(config.perturbation_fraction) * pattern.astype(np.float32)
        )
        if positive_floor is not None and np.all(base > 0.0):
            np.maximum(variant, positive_floor, out=variant)
        variants[f"medium-{index:03d}"] = np.ascontiguousarray(
            variant, dtype=np.float32
        )
    return variants


def _zipf_probabilities(cardinality: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, cardinality + 1, dtype=np.float64)
    probabilities = ranks ** (-alpha)
    return probabilities / probabilities.sum()


def build_multi_medium_trace(
    config: MultiMediumWorkloadConfig,
    *,
    domain_extent: float,
) -> List[Dict[str, Any]]:
    """Create a deterministic trace that covers every configured medium."""

    if domain_extent <= 2.0:
        raise ValueError("domain_extent must be greater than two")
    rng = np.random.default_rng(config.seed)
    margin = max(1.0, min(domain_extent * 0.03, domain_extent / 3.0))
    geometries = rng.uniform(
        margin,
        domain_extent - margin,
        size=(config.geometry_cardinality, 2),
    ).astype(np.float32)
    medium_indices = rng.choice(
        config.medium_count,
        size=config.request_count,
        p=_zipf_probabilities(config.medium_count, config.medium_zipf_alpha),
    )
    geometry_indices = rng.choice(
        config.geometry_cardinality,
        size=config.request_count,
        p=_zipf_probabilities(
            config.geometry_cardinality, config.geometry_zipf_alpha
        ),
    )
    frequency_indices = rng.integers(
        0, len(config.frequencies), size=config.request_count
    )

    # Compulsory first touches make coverage deterministic even for short,
    # heavily skewed traces.
    medium_indices[: config.medium_count] = np.arange(config.medium_count)
    coverage = min(config.request_count, config.geometry_cardinality)
    geometry_indices[:coverage] = np.arange(coverage)
    frequency_coverage = min(config.request_count, len(config.frequencies))
    frequency_indices[:frequency_coverage] = np.arange(frequency_coverage)

    if config.arrival_rate > 0.0:
        arrivals = np.cumsum(
            rng.exponential(1.0 / config.arrival_rate, size=config.request_count)
        )
        arrivals -= arrivals[0]
    else:
        arrivals = np.zeros(config.request_count, dtype=np.float64)

    return [
        {
            "sequence": index,
            "request_id": f"stage7-0-{index:06d}",
            "context_id": f"medium-{int(medium_indices[index]):03d}",
            "source_position": [
                float(geometries[geometry_indices[index], 0]),
                float(geometries[geometry_indices[index], 1]),
            ],
            "frequency": float(config.frequencies[frequency_indices[index]]),
            "arrival_s": float(arrivals[index]),
        }
        for index in range(config.request_count)
    ]


def trace_digest(trace: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(trace), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def lru_hit_curve(keys: Sequence[Any], *, max_capacity: int = 0) -> List[Dict[str, Any]]:
    """Simulate inclusive LRU hit rates for capacities 1..max_capacity."""

    if not keys:
        raise ValueError("keys must not be empty")
    unique_count = len(set(keys))
    limit = unique_count if max_capacity <= 0 else min(max_capacity, unique_count)
    if limit < 1:
        raise ValueError("max_capacity must be positive when specified")
    rows: List[Dict[str, Any]] = []
    for capacity in range(1, limit + 1):
        cache: "OrderedDict[Any, None]" = OrderedDict()
        hits = 0
        for key in keys:
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                if len(cache) >= capacity:
                    cache.popitem(last=False)
                cache[key] = None
        misses = len(keys) - hits
        rows.append(
            {
                "capacity_entries": capacity,
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / len(keys),
            }
        )
    return rows


def summarize_multi_medium_trace(
    trace: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not trace:
        raise ValueError("trace must not be empty")
    medium_keys = [str(item["context_id"]) for item in trace]
    geometry_keys = [
        (
            str(item["context_id"]),
            tuple(float(value) for value in item["source_position"]),
        )
        for item in trace
    ]
    frequencies = [float(item["frequency"]) for item in trace]
    medium_counts = Counter(medium_keys)
    geometry_counts = Counter(geometry_keys)
    frequency_counts = Counter(frequencies)
    return {
        "requests": len(trace),
        "medium_count": len(medium_counts),
        "geometry_key_count": len(geometry_counts),
        "frequency_count": len(frequency_counts),
        "medium_request_counts": dict(sorted(medium_counts.items())),
        "frequency_request_counts": {
            str(key): value for key, value in sorted(frequency_counts.items())
        },
        "most_popular_medium_fraction": max(medium_counts.values()) / len(trace),
        "most_popular_geometry_fraction": max(geometry_counts.values()) / len(trace),
        "medium_lru_hit_curve": lru_hit_curve(medium_keys),
        "geometry_lru_hit_curve": lru_hit_curve(geometry_keys),
    }


def _round_up(value: float, quantum: int) -> int:
    return int(math.ceil(max(value, 0.0) / quantum) * quantum)


def plan_tier_capacities(
    trace: Sequence[Mapping[str, Any]],
    entry_bytes: Mapping[str, int],
    *,
    velocity_bytes_per_medium: int,
    target_medium_hit_rate: float = 0.80,
    headroom_fraction: float = 0.25,
    allocation_quantum_bytes: int = 64 * 1024 * 1024,
) -> Dict[str, Any]:
    """Convert an observed workload and cache entry sizes into a tier plan."""

    required = {"medium", "decoder_context", "geometry_prefix", "wavelet"}
    missing = required.difference(entry_bytes)
    if missing:
        raise ValueError(f"missing cache entry sizes: {sorted(missing)}")
    sizes = {name: int(entry_bytes[name]) for name in required}
    if any(value <= 0 for value in sizes.values()):
        raise ValueError("cache entry sizes must be positive")
    if velocity_bytes_per_medium <= 0:
        raise ValueError("velocity_bytes_per_medium must be positive")
    if not 0.0 < target_medium_hit_rate < 1.0:
        raise ValueError("target_medium_hit_rate must be in (0, 1)")
    if headroom_fraction < 0.0:
        raise ValueError("headroom_fraction must be non-negative")
    if allocation_quantum_bytes < 1:
        raise ValueError("allocation_quantum_bytes must be positive")

    summary = summarize_multi_medium_trace(trace)
    curve = summary["medium_lru_hit_curve"]
    selected = next(
        (row for row in curve if row["hit_rate"] >= target_medium_hit_rate),
        curve[-1],
    )
    hot_medium_count = int(selected["capacity_entries"])
    medium_counts = Counter(str(item["context_id"]) for item in trace)
    hot_medium_ids = [
        medium_id
        for medium_id, _ in sorted(
            medium_counts.items(), key=lambda item: (-item[1], item[0])
        )[:hot_medium_count]
    ]
    hot_medium_set = set(hot_medium_ids)
    all_geometry_keys = {
        (
            str(item["context_id"]),
            tuple(float(value) for value in item["source_position"]),
        )
        for item in trace
    }
    hot_geometry_keys = {
        key for key in all_geometry_keys if key[0] in hot_medium_set
    }
    frequency_count = len({float(item["frequency"]) for item in trace})
    medium_count = int(summary["medium_count"])
    factor = 1.0 + headroom_fraction

    gpu_raw = {
        "medium": hot_medium_count * sizes["medium"],
        "decoder_context": hot_medium_count * sizes["decoder_context"],
        "geometry_prefix": len(hot_geometry_keys) * sizes["geometry_prefix"],
        "wavelet": frequency_count * sizes["wavelet"],
    }
    gpu_capacities = {
        name: _round_up(value * factor, allocation_quantum_bytes)
        for name, value in gpu_raw.items()
    }
    full_context_payload = (
        medium_count * (sizes["medium"] + sizes["decoder_context"])
        + len(all_geometry_keys) * sizes["geometry_prefix"]
        + frequency_count * sizes["wavelet"]
    )
    pinned_raw = full_context_payload - sum(gpu_raw.values())
    pinned_raw = max(pinned_raw, 0)
    disk_source_raw = medium_count * velocity_bytes_per_medium
    return {
        "policy": {
            "target_medium_lru_hit_rate": target_medium_hit_rate,
            "predicted_medium_lru_hit_rate": selected["hit_rate"],
            "hot_medium_count": hot_medium_count,
            "hot_medium_ids": hot_medium_ids,
            "headroom_fraction": headroom_fraction,
            "allocation_quantum_bytes": allocation_quantum_bytes,
        },
        "entry_bytes": sizes,
        "working_set": {
            "medium_count": medium_count,
            "all_geometry_keys": len(all_geometry_keys),
            "hot_geometry_keys": len(hot_geometry_keys),
            "frequency_count": frequency_count,
            "full_materialized_context_bytes": full_context_payload,
        },
        "gpu": {
            "raw_working_set_bytes": sum(gpu_raw.values()),
            "raw_component_bytes": gpu_raw,
            "recommended_component_capacity_bytes": gpu_capacities,
            "recommended_total_capacity_bytes": sum(gpu_capacities.values()),
        },
        "pinned_cpu": {
            "raw_overflow_context_bytes": pinned_raw,
            "recommended_capacity_bytes": _round_up(
                pinned_raw * factor, allocation_quantum_bytes
            ),
        },
        "disk": {
            "minimum_source_velocity_bytes": disk_source_raw,
            "recommended_source_capacity_bytes": _round_up(
                disk_source_raw * factor, allocation_quantum_bytes
            ),
            "materialized_context_upper_bound_bytes": full_context_payload,
        },
        "assumptions": [
            "medium and decoder entries scale per resident medium",
            "geometry entries are keyed by medium and source/receiver geometry",
            "wavelet entries are shared across media for identical frequencies",
            "pinned and disk tiers are capacity targets, not Stage 7.0 implementations",
        ],
    }
