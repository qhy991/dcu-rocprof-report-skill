#!/usr/bin/env python3
"""Check the two CPU ROCTX messages from serving_graph_probe.py."""
import argparse
import collections
import csv
import json
import pathlib
import re
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=pathlib.Path)
    p.add_argument("--pid", type=int, required=True)
    a = p.parse_args()
    result = {"source": str(a.csv), "accepted": False,
              "scope": "CPU marker delivery only; no GPU event-to-step correlation claim.",
              "messages": {}, "errors": []}
    try:
        if a.csv.name != f"roctx_trace_{a.pid}.csv":
            raise ValueError("ROCTX filename does not match the graph probe PID.")
        with a.csv.open(newline="") as f:
            reader = csv.DictReader(f)
            if not {"Domain", "ROCTX_ID", "Message", "Timestamp"} <= set(reader.fieldnames or []):
                raise ValueError("ROCTX CSV is missing required columns.")
            counts = collections.Counter()
            for row in reader:
                if None in row:
                    raise ValueError("Malformed ROCTX row.")
                if row.get("Message") in ("graph_count_n8", "graph_count_n64"):
                    if int(row["Timestamp"]) <= 0:
                        raise ValueError("ROCTX marker has a nonpositive timestamp.")
                    counts[row["Message"]] += 1
        result["messages"] = dict(counts)
        if not all(counts[k] > 0 for k in ("graph_count_n8", "graph_count_n64")):
            result["errors"].append("Both explicit graph markers were not found.")
    except (OSError, ValueError, csv.Error) as exc:
        result["errors"].append(str(exc))
    result["accepted"] = not result["errors"]
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
