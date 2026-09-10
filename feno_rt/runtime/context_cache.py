"""Capacity-bounded caches used by the Stage-2 FENO runtime.

The cache implementation deliberately owns policy, accounting, and lifecycle,
while model-specific context objects remain in the model/runner modules.  This
keeps the LRU reusable for medium, decoder, geometry, and wavelet entries.
"""

from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from threading import RLock
from typing import Any, Callable, Dict, Generic, Hashable, List, Mapping, Optional, Tuple, TypeVar

import torch


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(frozen=True)
class MediumCacheKey:
    """Identity of an encoded velocity model within one runtime."""

    model_version: str
    velocity_digest: str
    normalization_version: str
    dtype: str
    device: str


@dataclass(frozen=True)
class DecoderContextCacheKey:
    """Identity of decoder state derived from a medium context."""

    medium: MediumCacheKey


@dataclass(frozen=True)
class GeometryPrefixCacheKey:
    """Identity of a frequency-independent receiver query prefix."""

    decoder_context: DecoderContextCacheKey
    geometry_digest: str


@dataclass(frozen=True)
class WaveletCacheKey:
    """Identity of an exact-frequency wavelet representation."""

    model_version: str
    frequency_bits: str
    output_steps: int
    duration_seconds: float
    dtype: str
    device: str


@dataclass(frozen=True)
class CacheSnapshot:
    name: str
    capacity_bytes: int
    resident_bytes: int
    resident_entries: int
    pinned_entries: int
    hits: int
    misses: int
    insertions: int
    evictions: int
    invalidations: int
    rejections: int

    @property
    def hit_rate(self) -> float:
        accesses = self.hits + self.misses
        return self.hits / accesses if accesses else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "capacity_bytes": self.capacity_bytes,
            "resident_bytes": self.resident_bytes,
            "resident_entries": self.resident_entries,
            "pinned_entries": self.pinned_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "insertions": self.insertions,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "rejections": self.rejections,
        }


@dataclass
class _CacheEntry(Generic[V]):
    value: V
    size_bytes: int
    ref_count: int = 0


def value_nbytes(value: Any) -> int:
    """Return the tensor payload size of a nested cache value.

    Python object overhead is intentionally excluded: GPU tensor payloads are
    the capacity-limiting resource and are stable across Python versions.
    """

    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if is_dataclass(value) and not isinstance(value, type):
        return sum(value_nbytes(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, Mapping):
        return sum(value_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(value_nbytes(item) for item in value)
    return 0


class CacheLease(Generic[K, V]):
    """A context manager that pins an entry against LRU eviction."""

    def __init__(self, cache: "TensorLRUCache[K, V]", key: K, value: V) -> None:
        self._cache = cache
        self.key = key
        self.value = value
        self._released = False

    def __enter__(self) -> V:
        return self.value

    def release(self) -> None:
        if not self._released:
            self._cache.release(self.key)
            self._released = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.release()


class TensorLRUCache(Generic[K, V]):
    """Thread-safe byte-capacity LRU with reference-counted leases."""

    def __init__(
        self,
        capacity_bytes: int,
        *,
        name: str,
        size_fn: Callable[[V], int] = value_nbytes,
    ) -> None:
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        self.capacity_bytes = int(capacity_bytes)
        self.name = name
        self._size_fn = size_fn
        self._entries: "OrderedDict[K, _CacheEntry[V]]" = OrderedDict()
        self._resident_bytes = 0
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._insertions = 0
        self._evictions = 0
        self._invalidations = 0
        self._rejections = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _lookup(self, key: K, *, count_access: bool) -> Optional[_CacheEntry[V]]:
        entry = self._entries.get(key)
        if entry is None:
            if count_access:
                self._misses += 1
            return None
        if count_access:
            self._hits += 1
        self._entries.move_to_end(key)
        return entry

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            entry = self._lookup(key, count_access=True)
            return None if entry is None else entry.value

    def peek(self, key: K) -> Optional[V]:
        """Read without changing hit/miss counters or recency."""
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry.value

    def acquire(self, key: K) -> Optional[CacheLease[K, V]]:
        with self._lock:
            entry = self._lookup(key, count_access=True)
            if entry is None:
                return None
            entry.ref_count += 1
            return CacheLease(self, key, entry.value)

    def release(self, key: K) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise KeyError(f"cannot release missing {self.name} cache entry")
            if entry.ref_count <= 0:
                raise RuntimeError(f"{self.name} cache entry is not pinned")
            entry.ref_count -= 1

    def _evict_until_fits(self, incoming_bytes: int) -> bool:
        while self._resident_bytes + incoming_bytes > self.capacity_bytes:
            victim_key = next(
                (key for key, entry in self._entries.items() if entry.ref_count == 0),
                None,
            )
            if victim_key is None:
                return False
            victim = self._entries.pop(victim_key)
            self._resident_bytes -= victim.size_bytes
            self._evictions += 1
        return True

    def put(self, key: K, value: V, *, size_bytes: Optional[int] = None) -> bool:
        measured_bytes = self._size_fn(value) if size_bytes is None else int(size_bytes)
        if measured_bytes < 0:
            raise ValueError("cache entry size must be non-negative")
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                if existing.ref_count:
                    raise RuntimeError(f"cannot replace pinned {self.name} cache entry")
                self._resident_bytes -= existing.size_bytes
                del self._entries[key]

            if measured_bytes > self.capacity_bytes or not self._evict_until_fits(measured_bytes):
                if existing is not None:
                    self._entries[key] = existing
                    self._resident_bytes += existing.size_bytes
                self._rejections += 1
                return False

            self._entries[key] = _CacheEntry(value=value, size_bytes=measured_bytes)
            self._resident_bytes += measured_bytes
            self._insertions += 1
            return True

    def get_or_create(self, key: K, factory: Callable[[], V]) -> Tuple[V, bool]:
        value = self.get(key)
        if value is not None:
            return value, True
        value = factory()
        self.put(key, value)
        return value, False

    def invalidate(self, key: K, *, force: bool = False) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or (entry.ref_count and not force):
                return False
            del self._entries[key]
            self._resident_bytes -= entry.size_bytes
            self._invalidations += 1
            return True

    def invalidate_where(self, predicate: Callable[[K], bool], *, force: bool = False) -> int:
        with self._lock:
            keys = [key for key in self._entries if predicate(key)]
        return sum(self.invalidate(key, force=force) for key in keys)

    def clear(self, *, force: bool = False) -> int:
        return self.invalidate_where(lambda _key: True, force=force)

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._insertions = 0
            self._evictions = 0
            self._invalidations = 0
            self._rejections = 0

    def snapshot(self) -> CacheSnapshot:
        with self._lock:
            return CacheSnapshot(
                name=self.name,
                capacity_bytes=self.capacity_bytes,
                resident_bytes=self._resident_bytes,
                resident_entries=len(self._entries),
                pinned_entries=sum(entry.ref_count > 0 for entry in self._entries.values()),
                hits=self._hits,
                misses=self._misses,
                insertions=self._insertions,
                evictions=self._evictions,
                invalidations=self._invalidations,
                rejections=self._rejections,
            )

    def keys(self) -> List[K]:
        with self._lock:
            return list(self._entries.keys())


class MediumContextCache(TensorLRUCache[MediumCacheKey, Any]):
    def __init__(self, capacity_bytes: int) -> None:
        super().__init__(capacity_bytes, name="medium")


class DecoderContextCache(TensorLRUCache[DecoderContextCacheKey, Any]):
    def __init__(self, capacity_bytes: int) -> None:
        super().__init__(capacity_bytes, name="decoder_context")


class GeometryPrefixCache(TensorLRUCache[GeometryPrefixCacheKey, torch.Tensor]):
    def __init__(self, capacity_bytes: int) -> None:
        super().__init__(capacity_bytes, name="geometry_prefix")


class WaveletContextCache(TensorLRUCache[WaveletCacheKey, Any]):
    def __init__(self, capacity_bytes: int) -> None:
        super().__init__(capacity_bytes, name="wavelet")


@dataclass(frozen=True)
class FENOCacheConfig:
    medium_capacity_bytes: int = 256 * 1024 * 1024
    decoder_capacity_bytes: int = 2 * 1024 * 1024 * 1024
    geometry_capacity_bytes: int = 512 * 1024 * 1024
    wavelet_capacity_bytes: int = 512 * 1024 * 1024


class FENOCacheBundle:
    """Own all Stage-2 cache levels and dependency-aware invalidation."""

    def __init__(self, config: Optional[FENOCacheConfig] = None) -> None:
        self.config = config or FENOCacheConfig()
        self.medium = MediumContextCache(self.config.medium_capacity_bytes)
        self.decoder = DecoderContextCache(self.config.decoder_capacity_bytes)
        self.geometry = GeometryPrefixCache(self.config.geometry_capacity_bytes)
        self.wavelet = WaveletContextCache(self.config.wavelet_capacity_bytes)

    def snapshots(self) -> Dict[str, Dict[str, Any]]:
        caches = (self.medium, self.decoder, self.geometry, self.wavelet)
        return {cache.name: cache.snapshot().to_dict() for cache in caches}

    def reset_stats(self) -> None:
        for cache in (self.medium, self.decoder, self.geometry, self.wavelet):
            cache.reset_stats()

    def clear(self, *, force: bool = False) -> Dict[str, int]:
        return {
            "geometry_prefix": self.geometry.clear(force=force),
            "decoder_context": self.decoder.clear(force=force),
            "medium": self.medium.clear(force=force),
            "wavelet": self.wavelet.clear(force=force),
        }

    def invalidate_medium(self, key: MediumCacheKey, *, force: bool = False) -> Dict[str, int]:
        decoder_keys = [item for item in self.decoder.keys() if item.medium == key]
        decoder_key_set = set(decoder_keys)
        geometry_count = self.geometry.invalidate_where(
            lambda item: item.decoder_context in decoder_key_set,
            force=force,
        )
        decoder_count = sum(self.decoder.invalidate(item, force=force) for item in decoder_keys)
        medium_count = int(self.medium.invalidate(key, force=force))
        return {
            "geometry_prefix": geometry_count,
            "decoder_context": decoder_count,
            "medium": medium_count,
        }
