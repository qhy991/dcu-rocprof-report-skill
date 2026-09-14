# gfx938 / DCU metric names (verified on DTK 26.04)

Enumerated live via `rocprof --list-basic` / `--list-derived` on a DCU-3G (gfx938) device, DTK 26.04. The full lists are large (≈300 derived metrics); this page curates the ones that matter for analysis. Enumerate your own with:

```bash
rocprof --list-basic
rocprof --list-derived
# definitions: /opt/dtk/rocprofiler/lib/rocprofiler/metrics.xml
```

**Do not assume upstream-ROCm or NVIDIA names work.** On this device `Wavefronts` is the wave count (not NVIDIA "warp"), memory bytes come from `FETCH_SIZE`/`WRITE_SIZE` (not `dram__bytes`), and matrix-core activity is `SQ_INSTS_MMOP_*` (not `mfma`).

## First-pass core (one counter group)

| Metric | Meaning |
|---|---|
| `Wavefronts` | Total wavefronts dispatched |
| `VALUInsts` | Vector ALU instructions |
| `SALUInsts` | Scalar ALU instructions |
| `SFetchInsts` | Scalar fetch (constant/instruction) ops |
| `FETCH_SIZE` | KB fetched from video memory (all cache effects included) |
| `GPUBusy` | % cycles the GPU command processor is busy |

## Utilization / rate (derived)

| Metric | Meaning |
|---|---|
| `VALUBusy` / `SALUBusy` | VALU/SALU pipe active % |
| `MemUnitBusy` / `MemUnitStalled` | Memory unit activity / stalls |
| `VALUUtilization`, `VALUInstrRate`, `SALUInstrRate`, `ThreadRate`, `WaveRate` | Rate ratios |
| `InstrStallsRate`, `ALUStallsRate`, `ALUStalledByLDS` | Stall fractions |
| `WriteUnitStalled`, `VectMenRdRate`, `VectMemWrRate`, `ScalMemRdRate` | Vector/scalar memory rates |

## Memory

| Metric | Meaning |
|---|---|
| `FETCH_SIZE`, `WRITE_SIZE` | KB read/written at video memory (the bandwidth numerators) |
| `L1CacheHit`, `L2CacheHit`, `DCacheHit`, `ICacheHit` | Hit rates (L1 scalar dcache = `DCacheHit`) |
| `L1CacheBW`, `L2CacheBW`, `L2ChanEfficiency` | Throughput ratios |
| `L1ToL2RdLatency`, `L1Wavelatency` | Latency metrics |
| `L2ReadReqs`, `L2WriteReqs`, `MemWrites32B` | Request counts |
| `TCC_*` (e.g. `TCC_EA_RDREQ_sum`, `TCC_WRREQ_STALL_*`) | L2 channel / EA detail — sum over 32 TCC instances |
| `TCP_*` | L1-per-CU detail (tag conflicts, FIFO stalls) |

## Matrix core (MFMA ⇒ MMOP on this part)

`SQ_INSTS_MMOP` total, split by dtype: `SQ_INSTS_MMOP_F64/F32/F16/BF16/I8/FP8`, plus `SQ_ACTIVE_INST_MMOP` (busy cycles). Zero on a GEMM-class kernel = MFMA not engaged (playbook entry C).

## LDS

`LDSInsts`, `FlatLDSInsts`, `LDSBankConflict`, `SQ_LDS_BANK_CONFLICT`, `SQ_LDS_CMD_FIFO_FULL`, `SQ_LDS_DATA_FIFO_FULL`, `SQ_LDS_UNALIGNED_STALL`, `SQ_LDS_MEM_VIOLATIONS`.

## Occupancy / waves

`SQ_WAVES`, `SQ_WAVES_CU`, `SQ_LEVEL_WAVES`, `SQ_WAVES_EQ_64`, `SQ_WAVES_LT_16/LT_32/LT_48/LT_64`, `PartialWaveRate`, `SQ_WAVES_SAVED/RESTORED` (context switches), `SQ_IFETCH` / `SQ_IFETCH_XNACK` (page faults — non-zero XNACK is a memory-layout problem).

## Stalls (SQ block)

`SQ_VALU_DEP_STALL`, `SQ_VALU_STARVE`, `SQ_VALU_SRC_C_CONFLICT`, `SQ_WAIT_INST_VALU/VMEM/LDS/SCA/EXP_GDS/FLAT/MISC`, `SQ_VMEM_RD_SRC_CD_CONFLICT`, `SQ_VMEM_*_FIFO_FULL`, `SQ_TTRACE_STALL`.

## Instruction mix (VALU detail)

`SQ_INSTS_VALU_{ADD,MUL,FMA,TRANS}_{F16,F32,F64}`, `SQ_INSTS_VALU_CVT`, `SQ_INSTS_VALU_INT32/INT64`, `SQ_INSTS_VMEM_RD/WR` (+`_REPLAY`), `SQ_INSTS_SMEM` (+replay), `SQ_INSTS_BRANCH`, `SQ_INSTS_EXP` (exports), `SQ_INSTS_SENDMSG`, `SQ_ACCUM_PRECISION_LO/HI` (if present).

## Instruction-level / emulation caveats

- Many `SQ_INSTS_MMOP_*` and VALU-typed counters are marked **"(per-simd, emulated)"** in `--list-basic` — treat their absolute values with suspicion, their ratios less so.
- Graphics-block counters (`TD_*`, `SPI_*`, `CPC_*`/`CPF_*` command-processor, `GRBM_*` global busy) exist; `GRBM_GUI_ACTIVE`-style totals are useful for whole-GPU busy %.
