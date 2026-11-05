#!/usr/bin/env python3
"""Check external timed events inside replayed HIP graphs on one HCU."""
import json
import pathlib
import statistics
import sys

out = pathlib.Path(sys.argv[1])
result = {
    "scope": "One HCU and two toy graphs only; no model or serving qualification.",
    "accepted": False,
    "graph_sizes": [8, 64],
    "replays_per_size": 5,
    "timings_ms": {},
}

try:
    import torch

    torch.manual_seed(7)
    x = torch.randn(1 << 16, device="cuda:0")
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    begin.record()
    end.record()
    torch.cuda.synchronize()
    if 'librocprofiler_tool.so' in pathlib.Path('/proc/self/maps').read_text():
        raise RuntimeError('Profiler preload is active; use a non-profiled event probe.')

    def work(tensor, count):
        y = tensor
        for i in range(count):
            y = y * 1.0001 + i
        return y

    graphs, outputs = {}, {}
    for count in (8, 64):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            begin.record()
            y = work(x, count)
            end.record()
        graphs[count], outputs[count] = graph, y

    timings = {8: [], 64: []}
    for iteration in range(5):
        for count in (8, 64):
            x.fill_(1.0 + iteration * 0.125)
            expected = work(x, count)
            graphs[count].replay()
            torch.cuda.synchronize()
            if not torch.equal(expected, outputs[count]):
                raise RuntimeError(f"Graph output mismatch at size {count}, replay {iteration}.")
            elapsed = begin.elapsed_time(end)
            if elapsed <= 0:
                raise RuntimeError(f"Event duration is not positive at size {count}.")
            timings[count].append(elapsed)
    fast, slow = statistics.median(timings[8]), statistics.median(timings[64])
    if slow <= 1.5 * fast:
        raise RuntimeError("The event duration did not distinguish the two graph workloads.")
    if 'librocprofiler_tool.so' in pathlib.Path('/proc/self/maps').read_text():
        raise RuntimeError('Profiler library was loaded during the event probe.')
    result.update({
        "accepted": True,
        "bitwise_outputs_match": True,
        "timings_ms": {str(k): v for k, v in timings.items()},
        "median_ms": {"8": fast, "64": slow},
        "qualification": "Replay updates external event timing for this toy workload only.",
    })
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"

out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
sys.exit(0 if result["accepted"] else 1)
