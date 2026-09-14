# Harness guide

Two routes to a profilable binary on DCU. Pick by how the kernel reaches the GPU.

## Route A: standalone hipcc binary (preferred for pure kernels)

Use when the kernel is self-contained (inputs are synthetic tensors, no framework).

```bash
# inside the DTK container:
hipcc --offload-arch=gfx938 -O3 harness.hip -o harness
```

- Start from `helpers/harness_template.hip` — it has allocation, warmup, correctness print, and HIP-event timing already wired.
- Timing: `hipEventRecord`/`hipEventElapsedTime` around a loop of ≥100 launches after warmup; report the min over ≥3 reps.
- This route compiles in seconds and keeps rocprof runs clean (no framework kernels in the trace).

## Route B: torch cpp_extension (for PyTorch-integrated kernels)

```python
from torch.utils.cpp_extension import load
mod = load(
    name="my_kernel_dcu",
    sources=["kernel.cu", "kernel_binding.cpp"],   # .cu is fine — hipify handles it
    extra_cflags=["-O3", "-std=c++17"],
    build_directory="/tmp/build_dir",               # MUST be writable
)
```

Environment (verified):

```bash
export ROCM_HOME=/opt/dtk
export CUDA_HOME=/opt/dtk
export HIP_PATH=/opt/dtk
export PYTORCH_ROCM_ARCH=gfx938
```

**Traps on this route (all hit in practice):**

1. **hipify writes `kernel.hip` into the source directory.** A read-only source mount (e.g. a task checkout mounted `:ro`) fails with `Failed to save kernel.hip ... Read-only file system`. Fix: `cp -r` the sources to a writable directory and build there.
2. Torch auto-detects ROCm mode and compiles with `hipcc --offload-arch=gfx938 -D__HIP_PLATFORM_AMD__=1` — you don't need to pass `extra_cuda_cflags` with arch flags.
3. Build takes ~40 s for a two-file extension; keep the `build_directory` between runs to avoid rebuilding.

## Route selection

| Situation | Route |
|---|---|
| Kernel from a benchmark task / pure HIP-CUDA source | A |
| Kernel must interoperate with torch tensors and autograd wrappers | B |
| Profiling an existing vLLM/FlashInfer-style operator | B, loading the operator's own sources |

For Route B, run rocprof on a small Python driver script:

```bash
rocprof -i prof_input.txt -o results.csv python3 driver.py
```

The trace will include framework kernels — filter by `KernelName` prefix in analysis.
