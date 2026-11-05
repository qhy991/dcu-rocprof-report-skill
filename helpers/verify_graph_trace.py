#!/usr/bin/env python3
"""Verify the fixed serving_graph_probe.py kernel counts in one rocprof CSV."""
import argparse
import collections
import csv
import json
import pathlib
import re
import sys

EXPECTED_PER_OPERATION = (8 + 64) * (1 + 10)


def verify(path):
    result = {
        "source": str(path),
        "scope": "Single-process elementwise graph count only; no full-model or performance qualification.",
        "expected_mul": EXPECTED_PER_OPERATION,
        "expected_add": EXPECTED_PER_OPERATION,
        "rows": 0,
        "mul": 0,
        "add": 0,
        "other": 0,
        "pids": [],
        "gpu_ids": [],
        "errors": [],
    }
    pids, gpu_ids, dispatches = set(), set(), set()
    try:
        match = re.fullmatch(r"results_(\d+)\.csv", path.name)
        if not match:
            raise ValueError("Expected results_<PID>.csv; a fixed shared basename is unsafe.")
        filename_pid = int(match.group(1))
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"Kernel_Name", "Start_Timestamp", "End_Timestamp", "PID", "GPU_ID", "Dispatch_ID"}
            if not required <= set(reader.fieldnames or []):
                raise ValueError("Missing required kernel-trace columns.")
            for line, row in enumerate(reader, 2):
                if None in row or any(row.get(k) is None for k in required):
                    raise ValueError(f"Incomplete CSV row at line {line}.")
                name = row["Kernel_Name"]
                if not name:
                    raise ValueError(f"Missing kernel name at line {line}.")
                start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                pid, gpu, dispatch = int(row["PID"]), int(row["GPU_ID"]), int(row["Dispatch_ID"])
                if end <= start:
                    raise ValueError(f"Nonpositive kernel duration at line {line}.")
                if dispatch in dispatches:
                    raise ValueError(f"Duplicate Dispatch_ID {dispatch}.")
                dispatches.add(dispatch)
                pids.add(pid)
                gpu_ids.add(gpu)
                result["rows"] += 1
                if "::MulFunctor<float>" in name:
                    result["mul"] += 1
                elif "CUDAFunctorOnSelf_add<float>" in name:
                    result["add"] += 1
                else:
                    result["other"] += 1
        result["pids"], result["gpu_ids"] = sorted(pids), sorted(gpu_ids)
        if pids != {filename_pid}:
            result["errors"].append("Filename PID does not match the single recorded PID.")
        if len(gpu_ids) != 1:
            result["errors"].append("Expected exactly one recorded GPU.")
        if result["mul"] != EXPECTED_PER_OPERATION or result["add"] != EXPECTED_PER_OPERATION:
            result["errors"].append("Graph-body kernel counts do not match the frozen probe.")
    except (OSError, ValueError, csv.Error) as exc:
        result["errors"].append(str(exc))
    result["accepted"] = not result["errors"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=pathlib.Path)
    args = parser.parse_args()
    result = verify(args.csv)
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
