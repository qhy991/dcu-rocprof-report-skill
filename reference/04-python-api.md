# Parsing rocprof output (Python)

There is no `ncu_report` equivalent on this stack — rocprof v1 writes plain CSV. The schema below is **verified on DTK 26.04 / gfx938** by collecting a vector-add kernel.

## Verified CSV schema

Header (fixed columns first, then your requested metrics in request order):

```
Index,KernelName,gpu-id,queue-id,queue-index,pid,tid,grd,wgr,lds,scr,arch_vgpr,accum_vgpr,sgpr,wave_size,sig,obj,<metric1>,<metric2>,...
```

Example row (vector-add, 2^20 elements, 256-thread CTAs):

```
0,"vecadd(float const*, float const*, float*, int) [clone .kd]",8,0,10,351,351,1048576,256,0,0,8,0,16,64,0x0,0x7f7b8e2ee840,16384.0,14.0,3.0,4.0,8192.75,100.0
```

Column semantics:

| Column | Meaning | Sanity check |
|---|---|---|
| `grd` | grid size in **work-items** (not CTAs) | CTAs × `wgr` = `grd` |
| `wgr` | workgroup (CTA) size in threads | matches your launch |
| `lds` | LDS bytes per CTA | matches your `__shared__`/dynamic LDS |
| `arch_vgpr` / `accum_vgpr` / `sgpr` | register usage | bounds occupancy |
| `wave_size` | 64 on gfx938 | |
| `Wavefronts` | waves dispatched (total) | `grd / wave_size` |
| `KernelName` | mangled + `[clone .kd]` suffix | filter with prefix match |

## Parsing patterns

```python
import csv

def load_rows(path):
    with open(path, newline="") as f:
        # skip any leading non-CSV noise lines; header starts with "Index,"
        lines = [l for l in f if not l.startswith(("RPL:", "ROCProfiler"))]
        return list(csv.DictReader(lines))

rows = load_rows("reports/results.csv")
r = rows[0]
waves   = float(r["Wavefronts"])
valu    = float(r["VALUInsts"])
grd, wgr = int(r["grd"]), int(r["wgr"])
assert grd / wgr * (wgr // 64) == waves  # grid consistency check
```

Aggregate across rows of the same kernel (profiled runs may emit one row per
dispatch). **Sum counters, average percentages, and sum each launch's expected
waves before comparing them with aggregate `Wavefronts`:**

```python
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(float))
percentages = defaultdict(list)
expected_waves = defaultdict(float)
for r in rows:
    k = r["KernelName"].split("(")[0]
    for m in ("Wavefronts", "VALUInsts", "FETCH_SIZE"):
        agg[k][m] += float(r[m])
    percentages[k].append(float(r["GPUBusy"]))
    expected_waves[k] += int(r["grd"]) / int(r["wave_size"])
# mean(percentages[k]) is a per-dispatch average, not cumulative GPU busy.
# Compare agg[k]["Wavefronts"] with expected_waves[k], not one row's grid.
```

## Ratios worth computing

See 05-analysis-dimensions for interpretation; the formulas:

```python
valu_ipc        = VALUInsts / (SQ_BUSY_CYCLES or elapsed_cycles)   # or VALUInstrRate directly
bytes_total     = FETCH_SIZE + WRITE_SIZE  # only when both metrics were collected, KB
achieved_bw_pct = bytes_total / 1024 / kernel_seconds / peak_bytes  # use a separate no-profiler timing and measured peak
waves_per_cu    = SQ_WAVES_CU / num_CUs                             # occupancy indicator
```

Prefer **derived metrics rocprof already computes** (`VALUBusy`, `SALUBusy`, `MemUnitBusy`, `L2CacheHit`) over hand-rolled ratios — they encode the correct per-instance sums/maxes. Hand-roll only what's not in the list (see 08-gfx938-metric-names).

`helpers/analyze_csv.py` implements the counter and percentage aggregation;
run it on the CSV before writing any analysis. Its optional
`--peak-bw-gbps` requires an independently measured peak and it prints a
traffic floor only when **both** read and write metrics were collected. With
only `FETCH_SIZE`, total traffic is unknown rather than zero writes. Profiled
durations themselves are not speed scores.
