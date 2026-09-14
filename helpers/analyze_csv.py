#!/usr/bin/env python3
"""Parse a rocprof v1 CSV (DTK 26.04 / gfx938 schema) and print a summary.

Usage:
    python3 analyze_csv.py reports/results.csv [--peak-bw-gbps 5300]

The CSV schema is verified on-device: fixed columns
Index,KernelName,gpu-id,queue-id,queue-index,pid,tid,grd,wgr,lds,scr,
arch_vgpr,accum_vgpr,sgpr,wave_size,sig,obj
followed by the requested metrics in request order.
See reference/04-python-api.md.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict

FIXED = ["Index", "KernelName", "gpu-id", "queue-id", "queue-index", "pid", "tid",
         "grd", "wgr", "lds", "scr", "arch_vgpr", "accum_vgpr", "sgpr",
         "wave_size", "sig", "obj"]


def load_rows(path):
    with open(path, newline="") as f:
        # skip rocprof banner noise; the header starts with "Index,"
        lines = [l for l in f if not l.startswith(("RPL:", "ROCPRofiler:", "ROCProfiler"))]
    rows = [r for r in csv.DictReader(lines) if r.get("KernelName")]
    if not rows:
        sys.exit(f"no data rows in {path} (only header?) — collection failed or empty trace")
    return rows


def to_float(r, key):
    v = r.get(key, "")
    try:
        return float(v)
    except ValueError:
        return math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--peak-bw-gbps", type=float, default=5300.0,
                    help="measured peak bandwidth GB/s for the %% of peak column "
                         "(default 5300 — REPLACE with your own memset measurement)")
    args = ap.parse_args()

    rows = load_rows(args.csv_path)

    # aggregate metric sums per kernel (name prefix, sans clone suffix)
    agg = defaultdict(lambda: defaultdict(float))
    launches = defaultdict(int)
    for r in rows:
        k = r["KernelName"].split("(")[0].replace("[clone .kd]", "").strip()
        launches[k] += 1
        for m in r:
            if m not in FIXED:
                agg[k][m] += to_float(r, m)

    for k in sorted(agg):
        a = agg[k]
        print(f"\n=== {k}  ({launches[k]} dispatch(es)) ===")
        waves = a.get("Wavefronts", math.nan)
        print(f"  Wavefronts:      {waves:,.0f}")
        if not math.isnan(waves) and waves > 0:
            # waves per 1M work-items, quick shape sanity
            pass
        for m in ("VALUInsts", "SALUInsts", "SFetchInsts", "SQ_INSTS_MMOP",
                  "LDSInsts", "LDSBankConflict", "SQ_LDS_BANK_CONFLICT"):
            if m in a and not math.isnan(a[m]):
                print(f"  {m:<18}{a[m]:,.0f}")
        for m in ("VALUBusy", "SALUBusy", "MemUnitBusy", "GPUBusy",
                  "L1CacheHit", "L2CacheHit", "DCacheHit", "PartialWaveRate"):
            if m in a and not math.isnan(a[m]):
                print(f"  {m:<18}{a[m]:,.1f}")
        fetch_kb = a.get("FETCH_SIZE", 0.0) if not math.isnan(a.get("FETCH_SIZE", math.nan)) else 0.0
        write_kb = a.get("WRITE_SIZE", 0.0) if not math.isnan(a.get("WRITE_SIZE", math.nan)) else 0.0
        if not (math.isnan(fetch_kb) and math.isnan(write_kb)):
            total_gb = (fetch_kb + write_kb) / 1024 / 1024
            print(f"  FETCH_SIZE:      {fetch_kb:,.0f} KB")
            print(f"  WRITE_SIZE:      {write_kb:,.0f} KB")
            print(f"  total traffic:   {total_gb:.3f} GB "
                  f"(~{total_gb / 1024 / 1024 * 100 / 1:.0f}% of 1 GiB ref) "
                  f"[@{args.peak_bw_gbps} GB/s peak: {total_gb / args.peak_bw_gbps * 1e6:.1f} us floor]")

        # launch-config echo from the first row (they're identical for one kernel)
        r0 = next(r for r in rows if r["KernelName"].split("(")[0].split("[clone")[0].strip() == k)
        grd, wgr = int(r0["grd"]), int(r0["wgr"])
        wave_size = int(r0["wave_size"])
        ctas = grd // wgr if wgr else 0
        waves_expect = grd / wave_size
        print(f"  launch:          grid={grd:,} items, {ctas:,} CTAs x {wgr} threads, "
              f"wave_size={wave_size}, lds={r0['lds']} B/CTA")
        print(f"  vgpr={r0['arch_vgpr']} sgpr={r0['sgpr']}")
        if not math.isnan(waves):
            ok = abs(waves - waves_expect) / waves_expect < 0.02
            print(f"  consistency:     Wavefronts {waves:,.0f} vs grd/wave_size {waves_expect:,.0f} "
                  f"-> {'OK' if ok else 'MISMATCH — check harness launch!'}")

    print("\nNote: Busyness/hit metrics are per-run percentages; durations must come from "
          "a separate HIP-event timing run, never from the profiled run.")


if __name__ == "__main__":
    main()
