# Collection (rocprof on DTK)

All commands run **inside the DTK container** with GPU devices and the hyhal runtime mounted:

```bash
docker run --rm --device /dev/dri --device /dev/kfd \
  -v /opt/hyhal:/opt/hyhal:ro \
  <dtk-image> bash -c 'source /opt/dtk/env.sh; export PATH=/opt/dtk/rocprofiler/bin:$PATH; ...'
```

`source /opt/dtk/env.sh` does **not** put rocprof on PATH — add `/opt/dtk/rocprofiler/bin` yourself.

## rocprof v1: pmc counter collection

Metrics are requested through an input file, one block per collection:

```
pmc: Wavefronts VALUInsts SALUInsts SFetchInsts FETCH_SIZE GPUBusy
```

Run:

```bash
rocprof -i prof_input.txt -o results.csv ./harness
```

Verified behavior (DTK 26.04, gfx938):

- Output CSV goes to `-o` path **and** a `results_<timestamp>.csv` in an auto-created `/tmp/rpl_data_<timestamp>_<pid>/` scratch dir.
- stdout shows the effective input (parsed from `input0.xml`) — check it parsed your metrics correctly.
- Metrics beyond one hardware counter group: rocprof prints `Input metrics out of HW limit. Proposed metrics group set:` followed by a multi-group split — **and then aborts** with `Error: Context Create failed` (core dumped). Upstream ROCm re-runs the app per group automatically; DTK 26.04 does not survive it. **Keep each run to one group** (≈6 generic metrics; groups of specific-block counters vary). If you need more, do multiple runs.

### Suggested first-pass metric set (fits one group)

```
pmc: Wavefronts VALUInsts SALUInsts SFetchInsts FETCH_SIZE GPUBusy
```

Second-pass candidates (pick a group answering the surviving question):

```
pmc: WRITE_SIZE L2CacheHit L1CacheHit
pmc: SQ_WAVES SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_LDS LDSInsts
pmc: VALUBusy SALUBusy MemUnitBusy ALUStalledByLDS
```

### Input file filters (upstream rocprof v1 syntax)

```
kernel: <regex>        # restrict collection to matching kernels
gpu-index: 8           # single device (DCU devices are ids 8-15)
range: <name>          # roctx range
```

The `kernel:` regex must tolerate the `[clone .kd]` suffix DTK appends to kernel names — match a prefix, e.g. `kernel: rmsnorm.*`.

## rocprofv2: traces

`rocprofv2` (same bin dir) collects API/trace data instead of raw counters:

```bash
rocprofv2 --kernel-trace -o out.json ./harness       # kernel dispatch trace
rocprofv2 --hip-trace -o out.json ./harness          # HIP API trace
rocprofv2 -i pmc.txt --plugin file -d out_dir ./harness
```

Use `--kernel-trace` for launch-config and duration timelines (per-dispatch `grd`, `wgr`, duration — the closest thing ncu's timeline on this stack). Counter collection through v2 (`-i` + `--plugin file`) exists but v1's CSV is the verified path here.

## Other tools on the image

- `rocm-smi` (`/opt/dtk/bin`) — device status, clocks, power.
- `hipprof`, `rocsys` (`/opt/dtk/rocprofiler/bin`) — sampling-based profilers; try when pmc counters are too intrusive.
- On the **host**: `/usr/local/hyhal/bin/hy-smi` (use the full path — the bare name hits permission issues), `rocprof`/`xsys` under `/opt/hyhal`.

## Tips

- **Warmup inside the harness.** The first kernel launch includes module load; either warm up before the profiled region or use `kernel:` filters to isolate.
- rocprof serializes and replay-does kernel-level collection; expect the profiled run to be slower than the timed run. Never take *duration* from a profiled run as the performance number — time separately with hipEvents.
- Check `grd`, `wgr`, `lds`, `sgpr`, `arch_vgpr` columns in the CSV against what you intended to launch — they are the fastest way to catch a wrong-launch harness.
