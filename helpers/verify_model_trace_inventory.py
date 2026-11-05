#!/usr/bin/env python3
"""Check TP1, DP=EP worker identity and complete CSV parsing, not performance validity."""
import argparse
import collections
import csv
import json
import pathlib
import re
import sys

csv.field_size_limit(1 << 28)
REQUIRED = {"PID", "GPU_ID", "Queue_ID", "Dispatch_ID", "Start_Timestamp", "End_Timestamp", "Kernel_Name"}


def inspect(path, filename_pid):
    result = {"file": path.name, "rows": 0, "pids": [], "gpu_ids": [],
              "unnamed_kernels": 0, "errors": []}
    pids, devices, dispatches = set(), set(), set()
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f, strict=True)
            if not REQUIRED <= set(reader.fieldnames or []):
                raise ValueError("Missing required kernel-trace columns.")
            for line, row in enumerate(reader, 2):
                if None in row or any(row.get(k) is None for k in REQUIRED):
                    raise ValueError(f"Incomplete CSV row at line {line}.")
                pid, gpu, dispatch = (int(row[k]) for k in ("PID", "GPU_ID", "Dispatch_ID"))
                start, end = (int(row[k]) for k in ("Start_Timestamp", "End_Timestamp"))
                if end <= start:
                    raise ValueError(f"Nonpositive kernel duration at line {line}.")
                queue = int(row["Queue_ID"])
                event_id = (gpu, queue, dispatch, start, end)
                if event_id in dispatches:
                    raise ValueError(f"Duplicate dispatch event {dispatch} at line {line}.")
                dispatches.add(event_id)
                pids.add(pid)
                devices.add(gpu)
                result["rows"] += 1
                result["unnamed_kernels"] += not bool(row["Kernel_Name"])
        if pids and pids != {filename_pid}:
            result["errors"].append("Filename PID and row PID differ.")
        if len(devices) > 1:
            result["errors"].append("One worker file contains multiple GPU IDs.")
    except (OSError, ValueError, csv.Error) as exc:
        result["errors"].append(str(exc))
    result["pids"], result["gpu_ids"] = sorted(pids), sorted(devices)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("server_log", type=pathlib.Path)
    p.add_argument("trace_dir", type=pathlib.Path)
    p.add_argument("--workers", type=int, required=True)
    args = p.parse_args()
    report = {
        "accepted": False,
        "scope": "CSV integrity and worker/GPU identity only; no phase, model-correctness or performance qualification.",
        "expected_workers": args.workers,
        "workers": {},
        "files": [],
        "errors": [],
    }
    try:
        if args.workers <= 0:
            raise ValueError("workers must be positive")
        log = args.server_log.read_text(errors="replace")
        roles = collections.defaultdict(set)
        for m in re.finditer(r"Worker_DP(\d+)_EP(\d+) pid=(\d+)", log):
            roles[int(m.group(1))].add((int(m.group(2)), int(m.group(3))))
        if set(roles) != set(range(args.workers)) or any(len(v) != 1 for v in roles.values()):
            raise ValueError("Worker ranks are missing, duplicated or have multiple observed PIDs.")
        workers = {next(iter(v))[1]: {"dp": dp, "ep": next(iter(v))[0]}
                   for dp, v in roles.items()}
        if len(workers) != args.workers:
            raise ValueError("Multiple ranks share a worker PID.")
        if {worker["ep"] for worker in workers.values()} != set(range(args.workers)):
            raise ValueError("EP ranks are missing or duplicated.")
        report["workers"] = workers
        if not args.trace_dir.is_dir():
            raise ValueError("Trace directory does not exist.")
        paths = sorted(args.trace_dir.glob("results_*.csv"))
        if not paths:
            raise ValueError("No kernel CSV files found.")
        for path in paths:
            match = re.fullmatch(r"results_(\d+)\.csv", path.name)
            if not match:
                report["errors"].append(f"{path.name}: filename does not contain a PID.")
                continue
            pid = int(match.group(1))
            item = inspect(path, pid)
            if item["rows"] and pid not in workers:
                item["errors"].append("Kernel rows belong to an unbound process.")
            report["files"].append(item)
        by_pid = {int(re.fullmatch(r"results_(\d+)\.csv", f["file"]).group(1)): f
                  for f in report["files"]}
        for pid in workers:
            item = by_pid.get(pid)
            if item is None or not item["rows"] or len(item["gpu_ids"]) != 1:
                report["errors"].append(f"Worker {pid} lacks a nonempty single-GPU CSV.")
        devices = [by_pid[pid]["gpu_ids"][0] for pid in workers
                   if pid in by_pid and len(by_pid[pid]["gpu_ids"]) == 1]
        if len(set(devices)) != args.workers:
            report["errors"].append("Worker GPU IDs are not all distinct.")
        if any(item["errors"] for item in report["files"]):
            report["errors"].append("At least one CSV failed its integrity check.")
    except (OSError, ValueError) as exc:
        report["errors"].append(str(exc))
    report["accepted"] = not report["errors"]
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
