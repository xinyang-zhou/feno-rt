"""Policy and lifecycle tests for capacity-bounded caches."""

import unittest

import torch

from feno_rt.runtime.context_cache import (
    FENOCacheBundle,
    FENOCacheConfig,
    GeometryPrefixCacheKey,
    MediumCacheKey,
    TensorLRUCache,
    WaveletCacheKey,
)


def medium_key(name: str) -> MediumCacheKey:
    return MediumCacheKey(velocity_digest=name)


class TensorLRUCacheTest(unittest.TestCase):
    def test_lru_evicts_least_recently_used_entry(self) -> None:
        cache = TensorLRUCache[str, torch.Tensor](8, name="test")
        cache.put("a", torch.tensor([1.0]))
        cache.put("b", torch.tensor([2.0]))
        self.assertIsNotNone(cache.get("a"))

        cache.put("c", torch.tensor([3.0]))

        self.assertIsNone(cache.peek("b"))
        self.assertIsNotNone(cache.peek("a"))
        self.assertIsNotNone(cache.peek("c"))
        self.assertEqual(cache.snapshot().evictions, 1)
        self.assertEqual(cache.snapshot().resident_bytes, 8)

    def test_lease_prevents_eviction_until_release(self) -> None:
        cache = TensorLRUCache[str, torch.Tensor](4, name="test")
        cache.put("a", torch.tensor([1.0]))
        lease = cache.acquire("a")
        self.assertIsNotNone(lease)
        self.assertFalse(cache.put("b", torch.tensor([2.0])))
        self.assertEqual(cache.snapshot().pinned_entries, 1)
        self.assertEqual(cache.snapshot().rejections, 1)

        lease.release()
        self.assertTrue(cache.put("b", torch.tensor([2.0])))
        self.assertIsNone(cache.peek("a"))

    def test_invalidate_refuses_pinned_entry_without_force(self) -> None:
        cache = TensorLRUCache[str, torch.Tensor](4, name="test")
        cache.put("a", torch.tensor([1.0]))
        lease = cache.acquire("a")
        self.assertFalse(cache.invalidate("a"))
        self.assertTrue(cache.invalidate("a", force=True))
        with self.assertRaises(KeyError):
            lease.release()


class CacheBundleTest(unittest.TestCase):
    def test_medium_invalidation_cascades_to_dependent_entries(self) -> None:
        bundle = FENOCacheBundle(
            FENOCacheConfig(
                medium_capacity_bytes=64,
                geometry_capacity_bytes=64,
                wavelet_capacity_bytes=64,
            )
        )
        mkey = medium_key("medium-a")
        gkey = GeometryPrefixCacheKey(
            medium=mkey,
            geometry_digest="geometry-a",
        )
        wkey = WaveletCacheKey(frequency_bits="00002041")
        bundle.medium.put(mkey, torch.tensor([1.0]))
        bundle.geometry.put(gkey, torch.tensor([3.0]))
        bundle.wavelet.put(wkey, torch.tensor([4.0]))

        invalidated = bundle.invalidate_medium(mkey)

        self.assertEqual(invalidated["medium"], 1)
        self.assertEqual(invalidated["geometry_prefix"], 1)
        self.assertIsNotNone(bundle.wavelet.peek(wkey))


if __name__ == "__main__":
    unittest.main()
