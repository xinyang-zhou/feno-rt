# Dependency locks

- `runtime.lock` pins the direct inference dependencies validated by the public smoke suite.
- `runtime-cu128-py39.lock` records the complete validated Linux x86_64,
  Python 3.9 and CUDA 12.8 transitive environment.

The project metadata deliberately allows compatible patch/minor releases, while
the lock files provide exact reconstruction points. CUDA PyTorch packages are
platform-specific; select the official PyTorch index that matches the target
driver, then install with the lock as a constraint. Do not copy the CUDA lock to
a CPU or non-Linux deployment.

The repository CI installs a CPU PyTorch wheel and the project package, then
runs all tests. GPU-specific cases self-skip there and require explicit local
devices as documented in the README.
