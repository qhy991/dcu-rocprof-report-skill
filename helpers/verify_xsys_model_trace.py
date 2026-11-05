#!/usr/bin/env python3
"""Validate xsys HCC Task traces against served worker identities."""
import argparse
import collections
import csv
import json
import pathlib
import re
import sys


csv.field_size_limit(1 << 28)
REQUIRED = {"StartNs", "EndNs", "Operaion", "KernelName", "DeviceID", "PID"}
FILE = re.compile(r"hcc_ops_trace_(\d+)\.csv")


def inspect(path, expected_pid, required_names):
    result = {
        "file": path.name,
        "rows": 0,
        "task_rows": 0,
        "pids": [],
        "device_ids": [],
        "required_task_matches": {name: 0 for name in required_names},
        "errors": [],
    }
    pids, devices = set(), set()
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if not REQUIRED <= set(reader.fieldnames or []):
                raise ValueError("Missing required xsys HCC columns.")
            for line, row in enumerate(reader, 2):
                if None in row or any(row.get(key) is None for key in REQUIRED):
                    raise ValueError(f"Incomplete CSV row at line {line}.")
                start, end = int(row["StartNs"]), int(row["EndNs"])
                pid, device = int(row["PID"]), int(row["DeviceID"])
                if end <= start:
                    raise ValueError(f"Nonpositive event duration at line {line}.")
                pids.add(pid)
                devices.add(device)
                result["rows"] += 1
                if row["Operaion"] == "Task":
                    result["task_rows"] += 1
                    for name in required_names:
                        result["required_task_matches"][name] += name in row["KernelName"]
        if pids != {expected_pid}:
            result["errors"].append("Filename PID and row PIDs differ.")
        if len(devices) != 1:
            result["errors"].append("One worker file contains multiple device IDs.")
        if not result["task_rows"]:
            result["errors"].append("No graph Task nodes were recorded.")
    except (OSError, ValueError, csv.Error) as exc:
        result["errors"].append(str(exc))
    result["pids"] = sorted(pids)
    result["device_ids"] = sorted(devices)
    return result


def workers_from_log(path, expected_workers):
    roles = collections.defaultdict(set)
    pattern = r"Worker_DP(\d+)_EP(\d+) pid=(\d+)"
    for match in re.finditer(pattern, path.read_text(errors="replace")):
        roles[int(match.group(1))].add((int(match.group(2)), int(match.group(3))))
    if set(roles) != set(range(expected_workers)) or any(len(value) != 1 for value in roles.values()):
        raise ValueError("Worker DP ranks are missing, duplicated or have multiple observed PIDs.")
    workers = {next(iter(value))[1]: {"dp": dp, "ep": next(iter(value))[0]}
               for dp, value in roles.items()}
    if len(workers) != expected_workers:
        raise ValueError("Multiple ranks share a worker PID.")
    if {worker["ep"] for worker in workers.values()} != set(range(expected_workers)):
        raise ValueError("EP ranks are missing or duplicated.")
    return workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_log", type=pathlib.Path)
    parser.add_argument("trace_dir", type=pathlib.Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--require", action="append", default=[], metavar="KERNEL_SUBSTRING")
    args = parser.parse_args()
    report = {
        "accepted": False,
        "scope": "xsys HCC Task identity and graph-node coverage only; no phase, output-quality or performance qualification.",
        "expected_workers": args.workers,
        "required_task_substrings": args.require,
        "workers": {},
        "files": [],
        "errors": [],
    }
    try:
        if args.workers <= 0:
            raise ValueError("workers must be positive")
        workers = workers_from_log(args.server_log, args.workers)
        report["workers"] = workers
        if not args.trace_dir.is_dir():
            raise ValueError("Trace directory does not exist.")
        files = sorted(args.trace_dir.rglob("hcc_ops_trace_*.csv"))
        if not files:
            raise ValueError("No xsys HCC CSV files found.")
        by_pid = {}
        for path in files:
            match = FILE.fullmatch(path.name)
            if not match:
                report["errors"].append(f"{path.name}: filename does not contain a PID.")
                continue
            pid = int(match.group(1))
            if pid in by_pid:
                report["errors"].append(f"Multiple xsys HCC files for PID {pid}.")
                continue
            item = inspect(path, pid, args.require)
            if pid not in workers:
                item["errors"].append("Trace file belongs to an unbound process.")
            by_pid[pid] = item
            report["files"].append(item)
        for pid in workers:
            item = by_pid.get(pid)
            if item is None or not item["task_rows"] or len(item["device_ids"]) != 1:
                report["errors"].append(f"Worker {pid} lacks a nonempty single-device Task trace.")
        devices = [by_pid[pid]["device_ids"][0] for pid in workers
                   if pid in by_pid and len(by_pid[pid]["device_ids"]) == 1]
        if len(set(devices)) != args.workers:
            report["errors"].append("Worker device IDs are not all distinct.")
        for name in args.require:
            count = sum(item["required_task_matches"][name] for item in report["files"])
            if not count:
                report["errors"].append(f"Required Task kernel substring was not observed: {name}")
        if any(item["errors"] for item in report["files"]):
            report["errors"].append("At least one xsys HCC CSV failed its integrity check.")
    except (OSError, ValueError) as exc:
        report["errors"].append(str(exc))
    report["accepted"] = not report["errors"]
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
