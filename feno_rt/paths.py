"""Project-local paths used by inference and observability entry points."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
