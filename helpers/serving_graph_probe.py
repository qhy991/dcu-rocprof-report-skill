#!/usr/bin/env python3
"""Exercise 16- and 128-node graphs and emit CPU ROCTX markers."""
import json
import torch

x = torch.randn(1 << 16, device="cuda:0")
for n in (8, 64):
    with torch.cuda.nvtx.range(f"graph_count_n{n}"):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            y = x
            for i in range(n):
                y = y * 1.0001 + i
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = x
            for i in range(n):
                y = y * 1.0001 + i
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        print(json.dumps({"ops": n, "nodes_per_replay": 2 * n,
                          "replays": 10, "replay_nodes": 20 * n}), flush=True)
