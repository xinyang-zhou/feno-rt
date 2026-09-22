#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: bash scripts/validate_server.sh NEW_EXTERNAL_OUTPUT_DIR" >&2
  exit 2
fi
feno_output=$1
feno_python=${FENO_PYTHON:-python}
: "${FENO_CHECKPOINT:?set FENO_CHECKPOINT}"
: "${FENO_NORMALIZATION:?set FENO_NORMALIZATION}"
"$feno_python" - "$feno_output" <<'PY'
from pathlib import Path
import subprocess
import sys
import torch
root = Path.cwd().resolve()
out = Path(sys.argv[1]).resolve()
if out == root or root in out.parents or out.exists():
    raise SystemExit("output must be a new directory outside the checkout")
if subprocess.check_output(["git", "status", "--porcelain"], text=True):
    raise SystemExit("formal validation requires a clean checkout")
if not torch.cuda.is_available():
    raise SystemExit("a CUDA device is required for server validation")
PY
mkdir -p "$feno_output"
git rev-parse HEAD > "$feno_output/commit.txt"
"$feno_python" -m unittest discover -s tests -v 2>&1 | tee "$feno_output/tests.log"
"$feno_python" benchmarks/run_scheduler_ab_matrix.py --output-dir "$feno_output/scheduler_main"
"$feno_python" benchmarks/run_scheduler_ab_matrix.py \
  --config benchmarks/configs/scheduler_batch1_ab.json --output-dir "$feno_output/scheduler_batch1"
echo "Scheduler matrices completed. Profiling commands: docs/BENCHMARK_REPRODUCTION.md"
