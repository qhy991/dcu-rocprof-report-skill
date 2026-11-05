#!/usr/bin/env python3
"""Exercise positive and negative paths of the graph and ROCTX validators."""
import argparse
import csv
import json
import pathlib
import shutil
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference_trace", type=pathlib.Path)
    args = p.parse_args()
    helpers = pathlib.Path(__file__).parent
    with tempfile.TemporaryDirectory(prefix="bw1100-probe-validator-") as tmp:
        root = pathlib.Path(tmp)
        trace = root / args.reference_trace.name
        shutil.copyfile(args.reference_trace, trace)
        graph = [sys.executable, str(helpers / "verify_graph_trace.py"), str(trace)]
        assert run(graph, 0)["accepted"] is True
        with trace.open(newline="") as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        removed = False
        with trace.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                if not removed and "::MulFunctor<float>" in row["Kernel_Name"]:
                    removed = True
                    continue
                writer.writerow(row)
        assert removed and run(graph, 1)["accepted"] is False
        roctx = root / "roctx_trace_59.csv"
        with roctx.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Domain", "ROCTX_ID", "Message", "Timestamp"])
            writer.writerows([["ROCTX_DOMAIN", "1", "graph_count_n8", "11"],
                              ["ROCTX_DOMAIN", "1", "graph_count_n64", "22"]])
        command = [sys.executable, str(helpers / "verify_roctx_trace.py"), str(roctx),
                   "--pid", "59"]
        assert run(command, 0)["accepted"] is True
        roctx.write_text("Domain,ROCTX_ID,Message,Timestamp\nROCTX_DOMAIN,1,graph_count_n8,11\n")
        assert run(command, 1)["accepted"] is False
        server = root / "server.log"
        server.write_text("".join(f"(Worker_DP{i}_EP{i} pid={100+i}) fixture\n" for i in range(8)))
        model_dir = root / "model-trace"
        model_dir.mkdir()
        fields = ["PID", "GPU_ID", "Queue_ID", "Dispatch_ID", "Start_Timestamp", "End_Timestamp", "Kernel_Name"]
        for i in range(8):
            with (model_dir / f"results_{100+i}.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                writer.writerow([100+i, 8+i, 4, 1, 10, 20, "fixture_kernel"])
        inventory = [sys.executable, str(helpers / "verify_model_trace_inventory.py"),
                     str(server), str(model_dir), "--workers", "8"]
        assert run(inventory, 0)["accepted"] is True
        same_id = model_dir / "results_102.csv"
        with same_id.open("a") as f:
            f.write("102,10,4,1,21,30,fixture_kernel\n")
        assert run(inventory, 0)["accepted"] is True
        with same_id.open("a") as f:
            f.write("102,10,4,1,21,30,fixture_kernel\n")
        duplicate = run(inventory, 1)
        assert any(f["file"] == "results_102.csv" and f["errors"] for f in duplicate["files"])
        same_id.write_text("PID,GPU_ID,Queue_ID,Dispatch_ID,Start_Timestamp,End_Timestamp,Kernel_Name\n102,10,4,1,10,20,fixture_kernel\n")
        missing = model_dir / "results_107.csv"
        held = model_dir / "results_107.held"
        missing.rename(held)
        missing_result = run(inventory, 1)
        assert any("Worker 107 lacks" in e for e in missing_result["errors"])
        held.rename(missing)
        assert run(inventory, 0)["accepted"] is True
        broken = model_dir / "results_104.csv"
        with broken.open("a") as f:
            f.write('104,12,4,2,21,30,"unfinished')
        truncated = run(inventory, 1)
        assert any(f["file"] == "results_104.csv" and f["errors"] for f in truncated["files"])
        broken.write_text("PID,GPU_ID,Queue_ID,Dispatch_ID,Start_Timestamp,End_Timestamp,Kernel_Name\n104,12,4,1,10,20,fixture_kernel\n")
        assert run(inventory, 0)["accepted"] is True
        (model_dir / "results_k.csv").write_text("unused")
        shared = run(inventory, 1)
        assert any("results_k.csv" in e for e in shared["errors"])
    print(json.dumps({"accepted": True, "scope": "CPU fixture and reference CSV only; no GPU work.",
                      "graph_reference_passed": True, "missing_graph_node_rejected": True,
                      "roctx_both_markers_passed": True, "missing_roctx_marker_rejected": True,
                      "model_inventory_passed": True, "missing_rank_rejected": True,
                      "distinct_reused_id_passed": True, "duplicate_event_rejected": True,
                      "truncated_and_shared_name_rejected": True},
                     indent=2))


if __name__ == "__main__":
    main()
