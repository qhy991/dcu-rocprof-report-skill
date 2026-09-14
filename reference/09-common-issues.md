# Common issues (all observed on real hardware, DTK 26.04 / DCU-3G / gfx938)

Every item below cost real debugging time. They are ordered by how likely you are to hit them.

## 1. `__shfl_xor_sync` VMFaults — the #1 CUDA-port trap

**Symptoms:**
- With the canonical CUDA 32-bit mask `__shfl_xor_sync(0xFFFFFFFFu, v, offset)`: **compile error** — a `static_assert` in DTK's hip headers demands `sizeof(MaskT) == 8` (a 64-bit mask).
- With the 64-bit mask `0xFFFFFFFFFFFFFFFFull` that satisfies the assert: **compiles, then VMFaults at runtime** (SIGSEGV / `Memory access fault` GPU dump, kernel aborts).

**Root cause:** DTK 26.04's `__shfl_xor_sync` 64-bit-mask code path is broken on gfx938. Proven by bisection: a kernel with only the vector load/store path passes; adding only the 64-bit-mask shuffle reduction faults; the same reduction with the unmasked intrinsic passes.

**Fix:**

```cpp
// CUDA:    v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
v += __shfl_xor(v, offset, 32);   // unmasked intrinsic, explicit width — works on DTK
```

For standalone hipcc compiles of code that still contains masked shuffles, `-DHIP_ENABLE_WARP_SYNC_BUILTINS=1` enables them, **but the 64-bit mask runtime fault remains** — replacing the intrinsic is the only working route found.

## 2. `__float2bfloat16_rn` doesn't exist

```cpp
// CUDA:    out = __float2bfloat16_rn(y);
out = __float2bfloat16(y);    // rounding on DTK's hip/hip_bf16.h
```

Include `<hip/hip_bf16.h>` in standalone builds (torch extension builds get it transitively).

## 3. Inline PTX is not accepted

CUDA inline asm like `st.global.cs.v4.b32` fails to compile under hipify (`invalid input constraint 'l'`, unsupported asm). Replace cache-hint asm with plain C++ stores:

```cpp
// asm volatile("st.global.cs.v4.b32 ...") →
*dst = out_p.v4;    // compiler still emits 128-bit global stores
```

`__ldg()` is emulated fine and can stay. These are performance hints, not semantics — correctness survives the loss.

## 4. rocprof aborts on multi-group metric sets

Requesting more counters than one hardware group holds → rocprof prints `Input metrics out of HW limit. Proposed metrics group set: ...` and then **aborts** (`Error: Context Create failed`, core dumped) instead of replaying per group like upstream ROCm. Keep ≤~6 generic metrics per run; do multiple runs for more.

## 5. torch extension build fails on read-only sources

`Failed to save kernel.hip ... Read-only file system` — hipify writes the translated source next to the input. Copy sources to a writable dir (`cp -r`, `chmod u+w -R`) and build from there. Required env: `ROCM_HOME=HIP_PATH=CUDA_HOME=/opt/dtk`, `PYTORCH_ROCM_ARCH=gfx938`.

## 6. Container torch fails without the hyhal mount

`librocm_smi64.so.2: cannot open shared object` — the GPU user-space runtime lives on the host. Always run the image with:

```bash
docker run --device /dev/dri --device /dev/kfd -v /opt/hyhal:/opt/hyhal:ro ...
```

and `source /opt/dtk/env.sh` inside.

## 7. Reading a VMFault GPU dump

When a kernel faults, the runtime prints an HSA queue analysis. What it gives you and how to read it:

```
VMFault ... DUMP AQL PACKET ... MATCH KERNEL COMMAND
grid: x:1792, y:1, z:1      <- total work-ITEMS (×256-thread CTAs = 7 CTAs here)
workgroup: x:256
group_segment_size: 32       <- LDS bytes/CTA
kernel_object: 0x...
```

- **AQL `grid` is work-items, not CTAs** — divide by the workgroup size before comparing with your launch.
- A fault with a *matching* kernel command means the kernel launched and faulted during execution (bad pointer/mask path — see issue 1), not a launch failure.
- `write index vs read index` in the queue shows how far dispatch got before the fault.

## 8. Device numbering: GPUs are 8–15

`rocprof` CSV `gpu-id`, `rocminfo` nodes, and NUMA topology all number the 8 DCU devices as **8–15** (nodes 0–7 are host CPU NUMA nodes, ~188 GB RAM each — easy to misread as GPU VRAM). `nvidia-smi`-style "GPU 0" intuition fails here.

## 9. rocprof not on PATH

`source /opt/dtk/env.sh` adds `/opt/dtk/bin` but not `/opt/dtk/rocprofiler/bin`. Export it explicitly. On the host, `hy-smi` needs its full path `/usr/local/hyhal/bin/hy-smi` (PATH copy has permission issues).

## 10. Kernel names carry `[clone .kd]`

Filter CSV `KernelName` values with prefix/regex matches (`rmsnorm.*`), never equality — DTK appends a `[clone .kd]` clone suffix to device kernels.

---

## Proven port example (reference for effort estimation)

A fused RMSNorm kernel (bf16, h=4096, warp-shuffle reduction, 128-bit vector I/O) ported from CUDA to DCU needed exactly the three source changes in issues 1–3, plus the environment setup in issues 5–6. Result: all 14 correctness workloads pass at atol=rtol=0.01, geomean 13.8× vs eager PyTorch reference — i.e. **the port is mechanical and the hardware performs**; budget hours, not weeks, per kernel, dominated by issue-1-style bisections.
