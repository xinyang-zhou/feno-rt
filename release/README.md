# v0.8.0 release checklist

This release makes the FENO-RT inference and cache benchmark reproducible with
the separately distributed `feno_test` checkpoint. It is a systems-performance
release, not a scientific-accuracy release.

## Required GitHub Release assets

- `feno_test.pth`
- `norm_params_freq.npz`
- `feno_test.manifest.json`
- `feno_test.sha256`
- `verification.json`
- `feno_test_single_gpu.json`
- `feno_test_model_card.md`

Optional profiling assets:

- `feno_test_single_gpu_profile.nsys-rep`
- `feno_test_single_gpu_profile.sqlite`
- `feno_test_single_gpu_profile_stats_cuda_api_sum.csv`
- `feno_test_single_gpu_profile_stats_cuda_gpu_kern_sum.csv`

## Final local checks

Run CPU tests with every GPU hidden:

```bash
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -v
```

Run only the focused GPU suites on physical GPU 0. Stage 4 can hard-mask the
other devices; Stage 7 intentionally checks an unmapped physical device and is
therefore run separately, but its test code accesses only `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unittest tests.test_stage4_gpu -v
env -u CUDA_VISIBLE_DEVICES python -m unittest tests.test_stage7_tier_gpu -v
```

Verify the staged checkpoint on physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/verify_checkpoint.py \
  --manifest release/feno_test.manifest.json \
  --checkpoint /path/to/feno_test.pth \
  --normalization /path/to/norm_params_freq.npz \
  --device cuda:0
```

Then run `python scripts/audit_public_release.py`, build both distributions,
install the wheel in a clean environment, and validate the checksums with
`sha256sum -c feno_test.sha256` from the asset directory.

## Publication boundary

Create tag `v0.8.0` only after reviewing the Git diff. Do not place checkpoint
or normalization binaries in the Git tree. The release description must state
that training stopped after epoch 0, the checkpoint is benchmark-only, and the
speedup applies to repeated hot-cache queries. The recorded uncached FENO path
is slower than Deepwave for every measured batch.
