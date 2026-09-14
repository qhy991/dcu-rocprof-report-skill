# Diagnosis playbook

Counter pattern → diagnosis → fix. Each entry lists the **confirming** counters and the ones that would **refute** it — check both before committing to a fix.

## A. Memory-bandwidth bound

**Confirm:** `FETCH_SIZE`+`WRITE_SIZE` at ≥70% of a measured peak bandwidth; `MemUnitBusy` high; `L2CacheHit` low for the traffic you expect to be re-read.
**Refute:** hit rates high and bytes already near the algorithmic minimum — then it's latency, not bandwidth.
**Fix:** wider vector loads (uint4/128-bit), fewer passes over data, fp8/bf16 instead of fp32, fuse producer/consumer.

## B. Latency-bound (too few waves in flight)

**Confirm:** `GPUBusy` high but `VALUBusy`/`MemUnitBusy` both low; `SQ_WAVES_LT_*` shows most waves under-filled; occupancy from `arch_vgpr`/`lds` columns far below device limit.
**Refute:** occupancy actually high → look at dependency stalls (E).
**Fix:** more CTAs (smaller tiles), higher occupancy (fewer registers, less LDS), or independent work per thread (ILP).

## C. Wrong pipe: MFMA not engaged

**Confirm:** on a GEMM/attention-class kernel, `SQ_INSTS_MMOP_*` is zero or negligible while `SQ_INSTS_VALU_FMA_F32` is huge.
**Refute:** kernel is elementwise/reduction by design — MMOP zero is correct.
**Fix:** use `hipblasLt` / MFMA path / WMMA intrinsics; check that the framework dispatch didn't silently fall back to the VALU path. This is the most common "it's 5× slower than NVIDIA for no reason" cause on CDNA-class hardware.

## D. LDS bank conflicts

**Confirm:** `SQ_LDS_BANK_CONFLICT` / `LDSBankConflict` non-trivial relative to `LDSInsts`; `ALUStalledByLDS` high.
**Refute:** conflicts near zero → LDS volume itself is the problem (restructure access).
**Fix:** pad shared arrays (e.g. +1 column on 4-byte, +2 on 8-byte), transpose the access pattern, or use per-lane privatization.

## E. Dependency-chain bound

**Confirm:** `SQ_VALU_DEP_STALL` dominant; `GPUBusy` high; long FMA chains in source with no independent work between them.
**Refute:** starvation counters (`SQ_VALU_STARVE`) bigger than dependency stalls → operand delivery, restructure loads.
**Fix:** unroll with multiple accumulators, interleave independent computations.

## F. Tail / quantization effect

**Confirm:** `PartialWaveRate` high; `SQ_WAVES_LT_64` dominant; grid barely over a wave boundary; runtime scales superlinearly with small shape changes.
**Refute:** full waves dominate → not this.
**Fix:** persistent-CTA / grid-stride loops so the last wave isn't mostly empty; align shapes to wave multiples where the API allows.

## G. Launch-config mismatch (harness bug, not kernel bug)

**Confirm:** `grd`/`wgr`/`lds` in the CSV don't match the intended launch; `Wavefronts` inconsistent with `grd/64`.
**Refute:** — (this is always a bug if present)
**Fix:** fix the harness; re-collect. Never analyze a run whose launch config you haven't verified.

## H. Kernel isn't the one you think

**Confirm:** `KernelName` in the trace is the framework's fallback (eager torch op) instead of your kernel.
**Refute:** —
**Fix:** dispatch/import path; on ports, DTK's failure to compile an intrinsic sometimes makes the *load* fall back silently — check `build.log` for the actual code path taken.
