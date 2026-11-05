#!/usr/bin/env python3
"""CPU fixtures for the xsys HCC trace acceptance contract."""
import csv
import json
import pathlib
import subprocess
import sys
import tempfile


if not __debug__:
    raise SystemExit("Run this acceptance test without Python optimization.")


def run(command, expected):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != expected:
        raise RuntimeError(f"Expected exit {expected}, got {result.returncode}: {result.stderr}")
    return json.loads(result.stdout)


def write_trace(path, pid, device, name):
    fields = ["StartNs", "EndNs", "Operaion", "KernelName", "CopySize", "DeviceID",
              "PID", "TID", "CorrelationID", "StreamID", "CommandType"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"StartNs": 10, "EndNs": 20, "Operaion": "KernelExecution",
                         "KernelName": "setup", "DeviceID": device, "PID": pid})
        writer.writerow({"StartNs": 21, "EndNs": 30, "Operaion": "Task",
                         "KernelName": name, "DeviceID": device, "PID": pid})


def main():
    helper = pathlib.Path(__file__).with_name("verify_xsys_model_trace.py")
    with tempfile.TemporaryDirectory(prefix="bw1100-xsys-validator-") as temporary:
        root = pathlib.Path(temporary)
        server = root / "server.log"
        server.write_text("".join(f"Worker_DP{i}_EP{i} pid={100 + i} fixture\n" for i in range(2)))
        trace = root / "trace" / "trace_fixture"
        trace.mkdir(parents=True)
        write_trace(trace / "hcc_ops_trace_100.csv", 100, 8, "target dispatch")
        write_trace(trace / "hcc_ops_trace_101.csv", 101, 9, "target combine")
        command = [sys.executable, str(helper), str(server), str(root / "trace"), "--workers", "2",
                   "--require", "dispatch", "--require", "combine"]
        assert run(command, 0)["accepted"] is True
        missing = trace / "hcc_ops_trace_101.csv"
        held = trace / "hcc_ops_trace_101.held"
        missing.rename(held)
        result = run(command, 1)
        assert any("Worker 101 lacks" in error for error in result["errors"])
        held.rename(missing)
        unmet = command[:-2] + ["--require", "not-present"]
        result = run(unmet, 1)
        assert any("not-present" in error for error in result["errors"])
        with (trace / "hcc_ops_trace_100.csv").open("a") as stream:
            stream.write('100,101,Task,"unterminated\n')
        result = run(command, 1)
        assert any(item["file"] == "hcc_ops_trace_100.csv" and item["errors"]
                   for item in result["files"])
    print(json.dumps({"accepted": True, "scope": "CPU fixture only; no GPU work.",
                      "positive_coverage_passed": True, "missing_rank_rejected": True,
                      "missing_required_kernel_rejected": True, "truncated_csv_rejected": True},
                     indent=2))


if __name__ == "__main__":
    main()
