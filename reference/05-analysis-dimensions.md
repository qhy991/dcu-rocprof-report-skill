# Analysis dimensions on gfx938

Six dimensions, in the order worth checking. On any given kernel 1–2 dominate — find them, don't grade all six.

## 1. Wavefronts and grid shape

- `Wavefronts` vs the theoretical minimum (`total_threads / 64`).
- `grd`, `wgr` from the CSV: is the CTA count what you intended?
- A grid that is too small (fewer CTAs than the device can hold) means the GPU is idle by construction — fix the launch before anything else.

## 2. Compute pipe utilization

- `VALUBusy`, `SALUBusy` (derived %): which pipe is saturated.
- `SQ_INSTS_VALU_*` families: the instruction mix — a bf16 kernel should show `SQ_INSTS_VALU_FMA_F32` dominating if upcasting correctly.
- `SQ_INSTS_MMOP_*` (MMOP = matrix-core instructions, the CDNA MFMA path, split by dtype: F16/BF16/F32/I8/FP8): any non-zero count means MFMA is engaged. If a GEMM-class kernel shows **zero** MMOP instructions, it is running on VALU — usually the single biggest fixable problem.
- `SQ_ACTIVE_INST_VALU` vs `SQ_INSTS_VALU`: issue-rate vs instruction-count divergence.

## 3. Memory

- `FETCH_SIZE`, `WRITE_SIZE` (KB, derived): actual video-memory traffic including cache effects.
- Achieved bandwidth = bytes / kernel duration. Compare against a **measured** peak (a memset/traf benchmark on the same device), not a datasheet number.
- `L2CacheHit`, `L1CacheHit`, `DCacheHit`: hit rates tell you whether the traffic you *saved* was actually saved.
- `TCC_*` / `TCP_*` families for deep-dive (L2 channel efficiency, tag conflicts).

## 4. LDS

- `LDSInsts`, `FlatLDSInsts`: LDS instruction volume.
- `SQ_LDS_BANK_CONFLICT` / `LDSBankConflict`: replayed accesses — each conflict costs a cycle per bank.
- `ALUStalledByLDS`: cycles the ALU waited on LDS. High + high bank conflicts = pad your shared arrays or change access pattern.
- `lds` column in the CSV: bytes per CTA, feeds the occupancy calculation.

## 5. Occupancy

- `arch_vgpr`, `sgpr`, `lds` columns → theoretical waves per CU given the limits.
- `SQ_WAVES_LT_16/LT_32/LT_64` distribution: tail/wave quantization.
- `PartialWaveRate`: fraction of waves that are not full — high on small/ragged grids.
- Low occupancy is only a problem if stalls (dimension 6) show latency-bound behavior.

## 6. Stalls

- `SQ_VALU_DEP_STALL` (dependency chains), `SQ_VALU_STARVE` (operand starvation), `SQ_WAIT_INST_*` (pipe backpressure), `SQ_LDS_CMD_FIFO_FULL`, `SQ_VMEM_*_FIFO_FULL` (memory queue full).
- `MemUnitStalled`, `WriteUnitStalled`: memory-side pressure.
- The largest of these usually names the bottleneck directly.

## Cross-cutting: consistency checks

- `Wavefronts == grd / 64` (for full CTAs) — catches launch misreads.
- `GPUBusy` near 100% with low `VALUBusy` and low `MemUnitBusy` = scheduling/latency bound, not throughput bound.
- Sum of `SQ_INSTS_VALU + SQ_INSTS_SALU + SQ_INSTS_VMEM + SQ_INSTS_LDS` ≈ instruction mix total; a VMEM count inconsistent with your access pattern means you're not looking at your kernel (wrong dispatch in the trace).
