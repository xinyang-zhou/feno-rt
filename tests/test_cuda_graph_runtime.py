"""CPU-safe tests for CUDA Graph bucket configuration."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feno_rt.runtime import CUDAGraphTailRunner


class CUDAGraphRuntimeTest(unittest.TestCase):
    def test_bucket_selection_and_limits(self) -> None:
        runner = CUDAGraphTailRunner(buckets=(1, 4, 8))
        self.assertEqual(runner.bucket_for(1), 1)
        self.assertEqual(runner.bucket_for(2), 4)
        self.assertEqual(runner.bucket_for(8), 8)
        with self.assertRaisesRegex(ValueError, "exceeds largest"):
            runner.bucket_for(9)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integers"):
            CUDAGraphTailRunner(buckets=())
        with self.assertRaisesRegex(ValueError, "at least one"):
            CUDAGraphTailRunner(warmup_iterations=0)


if __name__ == "__main__":
    unittest.main()
