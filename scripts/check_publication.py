"""Check the Git index for common private metadata and unpublished artifacts.

This is a small publication guard, not a comprehensive secret scanner. It reads
tracked/index content, so ignored local notes and archives are never printed.
"""

import argparse
from pathlib import PurePosixPath
import re
import subprocess


CONTENT_RULES = {
    "personal absolute path": re.compile(rb"/(?:home|Users)/[^\s\"'<>]+"),
    "GPU device identifier": re.compile(rb"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub credential": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
}
PRIVATE_SUFFIXES = (".nsys-rep", ".qdstrm", ".sqlite", ".sqlite-wal", ".sqlite-shm",
                    ".tar.gz", ".zip", ".pth", ".pt", ".ckpt", ".safetensors",
                    ".onnx", ".npz", ".npy")


def check_file(name, content):
    path = PurePosixPath(name)
    issues = []
    if any(part in (".local", ".venv", "venv", "__pycache__") for part in path.parts):
        issues.append("local-only directory")
    if path.name == ".env" or path.name.startswith(".env."):
        issues.append("environment file")
    if name.endswith(PRIVATE_SUFFIXES):
        issues.append("raw profiler, archive, or model artifact")
    if path.parts and path.parts[0].startswith("feno_results_"):
        issues.append("raw result collection")
    issues.extend(label for label, pattern in CONTENT_RULES.items() if pattern.search(content))
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    names = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    failures = []
    count = 0
    for name in filter(None, names):
        # Reading the index prevents an unstaged scrub from hiding staged data.
        content = subprocess.check_output(["git", "show", ":" + name])
        issues = check_file(name, content)
        count += 1
        if issues:
            failures.append((name, issues))
    for name, issues in failures:
        # Report categories and file names, never the detected value.
        print(f"{name}: {', '.join(issues)}")
    if failures:
        print(f"Publication check failed: {len(failures)} files require review.")
        return 1
    if not args.quiet:
        print(f"Publication check passed: {count} indexed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
