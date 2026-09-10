#!/usr/bin/env python3
"""Fail if an inference-only public release contains forbidden assets."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_ROOTS = (
    "checkpoints",
    "notebooks",
    "scripts/train",
    "scripts/evaluate",
    "artifacts/evaluation",
    "artifacts/training",
    "data/raw",
    "data/processed",
)
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".onnx",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
}
ALLOWED_BINARY_FILES = {"data/demo/velocity.npy"}
PRIVATE_TEXT_MARKERS = (
    "/home" + "/xinyang/",
    "scripts/train/",
    "scripts/evaluate/",
    "dataset_freq_seis",
    "train_idx_freq",
    "feno_best_huoqiu_freq.pth",
    "norm_params_freq.npz",
)


def audit() -> list[str]:
    violations: list[str] = []
    for relative in FORBIDDEN_ROOTS:
        path = ROOT / relative
        if path.exists():
            violations.append(f"forbidden path exists: {relative}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden binary suffix: {relative}")
        if path.suffix.lower() == ".npy" and relative not in ALLOWED_BINARY_FILES:
            violations.append(f"non-demo NumPy asset: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            violations.append(f"unexpected file larger than 5 MiB: {relative}")
        if path == Path(__file__).resolve():
            continue
        if path.suffix.lower() in {".py", ".toml", ".yml", ".yaml"}:
            text = path.read_text(encoding="utf-8")
            for marker in PRIVATE_TEXT_MARKERS:
                if marker in text:
                    violations.append(f"private marker {marker!r}: {relative}")
    return violations


def main() -> int:
    violations = audit()
    if violations:
        print("public release audit failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("public release audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
