# Collection (rocprof on DTK)

All commands run **inside the DTK container** with GPU devices and the hyhal runtime mounted. First inspect whether the task owns an HCU admission route. When it does, run the profiler as that route's child command rather than starting a second Docker container. For example, in `bw1100-bench` (paths and receipt names are create-only):

```bash
HIP_VISIBLE_DEVICES=1 BWBENCH_TIMEOUT=600 \
  bash scripts/rocprof.sh "$IMAGE" \
  .local/profile/run-001/admission.json \
  .local/profile/run-001/pmc.txt \
  .local/profile/run-001/reports/metrics.csv \
  'Rmsnorm2dFwd' \
  python3 /work/.local/profile/run-001/harness.py
```

Create `.local/profile/run-001/pmc.txt` and the harness first. In this repo,
`scripts/rocprof.sh` invokes `scripts/dtk.sh gpu` and requires a matching
kernel row plus all requested metric columns before writing a create-only
validation JSON. The repository is mounted writable at `/work` and `/tmp`
holds profiler scratch files. Keep the
benchmark's original workload, exact source/image identity, selected idle HCU,
and terminal receipt together. Check that the terminal says `completed` with
exit 0 **and** that `metrics-validation.json` says `passed` for the actual
target kernel. Other repositories can invoke rocprof as a child of their own
admission wrapper with equivalent checks.
An empty CSV, missing terminal, or SSH disconnection is not a profile. Do not
substitute profiled durations for separately measured no-profiler latency.

### JIT-backed library kernels

On the node4 DTK vLLM 0.29/AITER 0.1.5 image, a first AITER RMSNorm call *inside*
rocprof failed before the kernel ran: its JIT flag probe invoked
`aicc --offload-arch=native`, which received profiler counter text as invalid
GPU target IDs. The failed gateway receipt was `not_qualified`; the partial
CSV contained input-generation kernels only and was **not** a baseline profile.
Prewarm the exact source and original workload once through the same admission
gateway **without** rocprof. Set `AITER_JIT_DIR` *before import* to a persistent
writable path under the repository's ignored `.local/`, not the container's
ephemeral `/tmp`. After a completed prewarm receipt and observed HCU release,
run rocprof with the same image, HCU and cache path but a new profiler receipt.
This produced three actual `Rmsnorm2dFwd` counter rows on HCU1 (`gpu-id=9`),
with a completed terminal receipt. Preserve the failed first attempt; do not
relabel its partial CSV as successful. Other JIT libraries need their own
cache/target checks rather than an assumed equivalent fix.

Only when there is **no** project-owned gateway, a standalone command can use
the skill's `helpers/profile_container.sh` or the equivalent raw Docker route:

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
