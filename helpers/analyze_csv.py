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
PERCENT_METRICS = {"VALUBusy", "SALUBusy", "MemUnitBusy", "GPUBusy",
                   "L1CacheHit", "L2CacheHit", "DCacheHit", "PartialWaveRate"}


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
    ap.add_argument("--peak-bw-gbps", type=float,
                    help="measured peak bandwidth GB/s; omit unless both read "
                         "and write bytes were collected")
    args = ap.parse_args()
    if args.peak_bw_gbps is not None and args.peak_bw_gbps <= 0:
        ap.error("--peak-bw-gbps must be positive")

    rows = load_rows(args.csv_path)

    # Counter values sum across dispatches. Percentage/hit-rate metrics do not.
    agg = defaultdict(lambda: defaultdict(float))
    percentages = defaultdict(lambda: defaultdict(list))
    grouped = defaultdict(list)
    for r in rows:
        k = r["KernelName"].split("(")[0].replace("[clone .kd]", "").strip()
        grouped[k].append(r)
        for m in r:
            if m not in FIXED:
                value = to_float(r, m)
                if math.isnan(value):
                    continue
                if m in PERCENT_METRICS:
                    percentages[k][m].append(value)
                else:
                    agg[k][m] += value

    for k in sorted(grouped):
        a = agg[k]
        kernel_rows = grouped[k]
        print(f"\n=== {k}  ({len(kernel_rows)} dispatch(es)) ===")
        waves = a.get("Wavefronts", math.nan)
        if not math.isnan(waves):
            print(f"  Wavefronts:      {waves:,.0f}")
        for m in ("VALUInsts", "SALUInsts", "SFetchInsts", "SQ_INSTS_MMOP",
                  "LDSInsts", "LDSBankConflict", "SQ_LDS_BANK_CONFLICT"):
            if m in a and not math.isnan(a[m]):
                print(f"  {m:<18}{a[m]:,.0f}")
        for m in sorted(percentages[k]):
            values = percentages[k][m]
            print(f"  {m:<18}{sum(values) / len(values):,.1f} "
                  f"(per-dispatch mean, n={len(values)})")
        if "FETCH_SIZE" in a:
            print(f"  FETCH_SIZE:      {a['FETCH_SIZE']:,.0f} KB")
        if "WRITE_SIZE" in a:
            print(f"  WRITE_SIZE:      {a['WRITE_SIZE']:,.0f} KB")
        if "FETCH_SIZE" in a and "WRITE_SIZE" in a:
            fetch_kb, write_kb = a["FETCH_SIZE"], a["WRITE_SIZE"]
            total_gb = (fetch_kb + write_kb) / 1024 / 1024
            print(f"  total traffic:   {total_gb:.3f} GB")
            if args.peak_bw_gbps is not None:
                print(f"  bandwidth floor: {total_gb / args.peak_bw_gbps * 1e6:.1f} us "
                      f"at measured {args.peak_bw_gbps} GB/s")
        elif "FETCH_SIZE" in a or "WRITE_SIZE" in a:
            print("  total traffic:   unavailable (read or write metric not collected)")

        # Echo the first config, but sum the expected waves across all dispatches.
        r0 = kernel_rows[0]
        grd, wgr = int(r0["grd"]), int(r0["wgr"])
        wave_size = int(r0["wave_size"])
        ctas = grd // wgr if wgr else 0
        waves_expect = sum(int(r["grd"]) / int(r["wave_size"])
                           for r in kernel_rows)
        print(f"  launch:          grid={grd:,} items, {ctas:,} CTAs x {wgr} threads, "
              f"wave_size={wave_size}, lds={r0['lds']} B/CTA"
              + (" (first of varying configs)" if any(
                  (r["grd"], r["wgr"], r["wave_size"], r["lds"]) !=
                  (r0["grd"], r0["wgr"], r0["wave_size"], r0["lds"])
                  for r in kernel_rows[1:]) else ""))
        print(f"  vgpr={r0['arch_vgpr']} sgpr={r0['sgpr']}")
        if not math.isnan(waves) and waves_expect > 0:
            ok = abs(waves - waves_expect) / waves_expect < 0.02
            print(f"  consistency:     Wavefronts {waves:,.0f} vs summed grd/wave_size "
                  f"{waves_expect:,.0f} "
                  f"-> {'OK' if ok else 'MISMATCH — check harness launch!'}")

    print("\nNote: Busyness/hit metrics are unweighted per-dispatch means; "
          "durations must come from "
          "a separate HIP-event timing run, never from the profiled run.")


if __name__ == "__main__":
    main()
