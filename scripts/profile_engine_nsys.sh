#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "usage: bash scripts/profile_engine_nsys.sh OUT_DIR [profile_engine.py options]" >&2
  exit 2
fi
feno_output=$1
shift
if [ -e "$feno_output" ]; then
  echo "output must be a new directory: $feno_output" >&2
  exit 2
fi
mkdir -p "$feno_output"
feno_nsys=${FENO_NSYS_BIN:-nsys}
feno_python=${FENO_PYTHON:-python}
"$feno_nsys" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-graph-trace=graph --output="$feno_output/engine" \
  "$feno_python" benchmarks/profile_engine.py --nvtx \
  --output-dir "$feno_output/diagnostic" "$@"
"$feno_python" benchmarks/summarize_engine_profile.py "$feno_output/diagnostic"
"$feno_nsys" export --type=sqlite --output="$feno_output/engine.sqlite" \
  "$feno_output/engine.nsys-rep"
