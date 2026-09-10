"""CPU-only Stage 6 package, demo, and self-containment tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from feno_rt import __version__
from feno_rt.runtime import load_trace


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class Stage6PackagingTest(unittest.TestCase):
    def test_package_metadata_and_python_sources_have_no_private_path_dependency(self):
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(f'version = "{__version__}"', metadata)
        private_home = "/" + "home" + "/" + "xinyang" + "/"
        for path in (ROOT / "feno_rt").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(private_home, source, str(path))
            self.assertNotIn("from TD", source, str(path))
            self.assertNotIn("import TD", source, str(path))

    def test_public_demo_manifest_and_trace_are_valid(self):
        manifest = json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["deterministic"])
        self.assertEqual(manifest["license"], "CC0-1.0")
        for name, description in manifest["files"].items():
            self.assertEqual(sha256(DEMO_DIR / name), description["sha256"])
        records = load_trace(DEMO_DIR / "requests.jsonl")
        requests = [item for item in records if item["kind"] == "request"]
        self.assertEqual(len(requests), 24)
        self.assertTrue(all(item["result"]["status"] == "unverified" for item in requests))

    def test_demo_generator_is_byte_deterministic(self):
        module_path = ROOT / "scripts" / "data" / "generate_demo.py"
        spec = importlib.util.spec_from_file_location("feno_generate_demo", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            module.generate(Path(left))
            module.generate(Path(right))
            for name in ("velocity.npy", "requests.jsonl", "manifest.json"):
                self.assertEqual(
                    (Path(left) / name).read_bytes(),
                    (Path(right) / name).read_bytes(),
                )

if __name__ == "__main__":
    unittest.main()
