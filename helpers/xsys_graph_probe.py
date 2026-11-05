#!/usr/bin/env python3
"""Run one CUDA Graph replay probe on every visible HCU worker."""
import json
import multiprocessing as mp
import os
import time


def worker(rank):
    import torch

    torch.cuda.set_device(rank)
    value = torch.ones(65536, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = (value + 1) * 2
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = (value + 1) * 2
    started = time.monotonic_ns()
    for _ in range(25):
        graph.replay()
    torch.cuda.synchronize()
    ended = time.monotonic_ns()
    observed = result[0].item()
    if observed != 4:
        raise ValueError(f"rank {rank}: expected 4, got {observed}")
    print(json.dumps({"rank": rank, "pid": os.getpid(), "start_ns": started,
                      "end_ns": ended, "expected_task_nodes": 50, "value": observed}),
          flush=True)


def main():
    import torch

    workers = torch.cuda.device_count()
    if workers <= 0:
        raise SystemExit("No visible HCU devices.")
    context = mp.get_context("spawn")
    processes = [context.Process(target=worker, args=(rank,)) for rank in range(workers)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(90)
    failed = [(process.pid, process.exitcode) for process in processes if process.exitcode != 0]
    if failed:
        raise SystemExit(f"Graph probe workers failed: {failed}")


if __name__ == "__main__":
    main()
