"""Release-manifest and benchmark-helper tests without large assets."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from feno_rt import __version__
from feno_rt.config import FENOModelConfig
from feno_rt.preprocessing import NormalizationStats


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release" / "feno_test.manifest.json"
CHECKSUMS = ROOT / "release" / "feno_test.sha256"


def load_benchmark_module():
    path = ROOT / "benchmarks" / "benchmark_feno_deepwave.py"
    spec = importlib.util.spec_from_file_location("feno_efficiency_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load benchmark module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseManifestTest(unittest.TestCase):
    def test_manifest_matches_package_and_uses_release_asset_names(self):
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["package_version"], __version__)
        self.assertEqual(payload["purpose"], "performance-only")
        self.assertEqual(payload["checkpoint"]["filename"], "feno_test.pth")
        self.assertEqual(
            payload["normalization"]["filename"], "norm_params_freq.npz"
        )
        for section in ("checkpoint", "normalization"):
            digest = payload[section]["sha256"]
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(value in "0123456789abcdef" for value in digest))

    def test_checksum_file_matches_manifest(self):
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        checksums = {}
        for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
            digest, filename = line.split(maxsplit=1)
            checksums[filename] = digest
        self.assertEqual(
            checksums,
            {
                payload["checkpoint"]["filename"]: payload["checkpoint"]["sha256"],
                payload["normalization"]["filename"]: payload["normalization"]["sha256"],
            },
        )

    def test_synthetic_velocity_is_deterministic(self):
        module = load_benchmark_module()
        config = FENOModelConfig(
            original_size=8,
            velocity_height=8,
            velocity_width=8,
            latent_height=2,
            latent_width=2,
            output_steps=8,
            receiver_depth=1,
            num_receivers=4,
            encoder_dim=8,
            encoder_depth=1,
            encoder_heads=2,
            decoder_dim=8,
            decoder_depth=1,
            decoder_heads=2,
            fno_modes1=2,
            fno_modes2=2,
            fno_width=8,
            patch_size=2,
            position_embedding_dim=4,
            frequency_condition_dim=4,
            dropout_rate=0.0,
        )
        normalization = NormalizationStats(2000.0, 500.0, 0.0, 1.0)
        left = module.synthetic_velocity(config, normalization)
        right = module.synthetic_velocity(config, normalization)
        np.testing.assert_array_equal(left, right)
        self.assertEqual(left.shape, (8, 8))
        self.assertEqual(left.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
