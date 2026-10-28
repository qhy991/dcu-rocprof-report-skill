# Profiling a serving step (vLLM on DCU): which tool sees what

Kernel profiling (rocprof v1 counters, reference 03) answers "why is this kernel slow". Serving-step
profiling answers "where does one decode step go", under a real server with CUDA/HIP graphs, MTP and
data parallelism. This was learned on GLM-5.3-Flash, vLLM 0.28.1 + vllm_hcu (dtk2610), TP1 DP8 EP,
FULL_AND_PIECEWISE graphs, MTP k=3, on bw1100 (gfx938). Every row below was observed.

## 1. Tool coverage: most tools lose graph kernels

Probe: a HIP graph of n chained elementwise kernels, replayed under each tool
(`cells/probes/graph_profiler_cap*.py` in glm53-b1-decode-20261027, node4 HCU 5).

| tool | kernels recorded per graph replay (graph of 8 / 16 / 32 / 128 kernels, 2 per op) | verdict |
|---|---|---|
| torch.profiler (vLLM `--profiler-config`, `/start_profile`) | 3.0 / 3.3 / 4.0 / 6.0 of 8 / 16 / 32 / 128 | **drops most graph kernels** |
| `hipprof --hip-trace --stats` | ~231 of ~1600 (≈15 %) | **drops most** |
| `rocprofv2 --kernel-trace` | 1590 of 1584 expected (+ warmup) | **complete** |

Consequences seen in a real trace (torch.profiler, decode under MTP):
- zero MoE kernels in the trace while the server ran `DeepEPDeepGemmMaskedExperts`;
- ~800 µs "idle" gaps between layers that are dropped kernels, not idle time;
- durations pinned to the first graph node: `_fused_eh_norm_kernel` read 1068 µs at M=4 and
  2632 µs at M=16 in the trace, 4.3 µs in a standalone graph. It absorbed the dropped work.
- ≤ 8 events per `hipGraphLaunch` was the signature of the truncation.

**Rule.** Use `rocprofv2 --kernel-trace` for any per-op breakdown of a graph step. Use torch.profiler
only for eager-launched kernels, and only for counts. Before trusting any tool on a new stack, run the
graph-count probe above: a tool that cannot count a 128-kernel graph cannot rank ops.

## 2. The rocprofv2 recipe for a live server

rocprofv2 wraps the whole process, so it wraps the server:

```bash
/opt/dtk/rocprofiler/bin/rocprofv2 --kernel-trace -tp <DELAY_MS>:<ACTIVE_MS>:0 \
  -d <out>/rocprof -o k  vllm serve <model> <args...>
```

- Output: `<out>/rocprof/<pid>/results_k.csv` (one per process incl. EngineCore/Worker; check the
  directory layout on first run).
- `-tp` delays collection; `<ACTIVE_MS>` bounds the file size. A full DP8 server run is large: plan the
  window, do not trace the whole serve life.
- Columns: `Dispatch_ID, GPU_ID, Queue_ID, PID, GRD, WGR, LDS, SCR, Arch_VGPR, SGPR, Wave_Size,
  Kernel_Name, Start_Timestamp, End_Timestamp, Correlation_ID`. GPU_ID here is the device id from the
  KFD node list (8–15 on bw1100), not the HCU index; map with `hy-smi --showbus`.
- No step markers. Cut steps at a sampler kernel that runs exactly once per target step (on GLM-5.3 MTP:
  `_rejection_kernel`); `scripts/rocprof_step_table.py` in glm53-b1-decode-20261027 does this.
  Confirm the cut by counting: kernels per step must be identical across steady steps.
- Choosing the window: start the server, wait for steady decode, then delay. "Steady" must be
  measured, not assumed (see §3).

Budget: 1 server start ≈ 11 min (weights 60 s, graph capture 3 min, TileLang JIT 2–5 min), plus the
window. A profiling pass is ~20 min of node time; start it only after the native baseline exists.

## 3. Steady decode is not "the sweep has started"

- A request is `running` for its whole chunked prefill. With `--max-num-batched-tokens 128` a 4K prompt
  is 32 prefill steps; with 8 requests at c=8 the first decode step can come 30+ s after launch.
- A profiler window opened 40 s after launch caught only `execute_context_1(128)` steps (prefill) in
  every rank.
- Gate on `/metrics`: `vllm:num_requests_running >= c` AND `vllm:num_requests_waiting == 0` AND
  the step annotations of the target shape. Then wait a few seconds for MTP to settle.
- Check the window contents before analysing: count step kinds per rank. The trace was usable only
  after I found `execute_context_0(0)_generation_4(16)` (pure decode, M=16) in c=32 and
  `..._generation_1(4)` in c=8, both on some ranks only. Ranks without requests run dummy batches;
  pick a rank that has the shape you want.

## 4. Step boundaries under data parallelism

- vLLM 0.28 MRV2 (data2) emits `execute_*` annotations only on some ranks and only in some windows.
  Fallback boundary: the per-step `gloo:all_reduce` user annotation (DP token-count sync).
- The same kernel stream can be split two ways; compare the two before trusting a table.
- Each DP rank runs the full layer (TP=1): per-rank kernel tables are comparable directly; a rank with
  more requests has a longer step, so compare per request-count, not per rank.

## 5. Timing: profile for the list, time separately

- Kernel durations from the profiled run are usable for ranking; step wall is not (the profiler and
  per-kernel tracing stretch host gaps several-fold: a step measured 160 ms wall under torch.profiler
  against ~45 ms GPU work).
- Anything you quote as a speedup or a TPOT is measured without the profiler, in the same server
  window as its native control (see the E2E skill).
- Device time for a graph-replayed op: N copies per graph, replayed, divided by N (see
  op-fusion-campaign measurement.md). A single standalone kernel can be checked the same way.

## 6. Roofline check before a ranking is believed

For each top op: bytes it must move at the step's shape ÷ measured sustained bandwidth. Example from this
round: the lm_head GEMM `[M,4096] x [154880,4096]ᵀ` in a graph ran at 926 GB/s (M=4) and **232 GB/s
(M=16)**, while a plain 1.27 GB copy ran at 1761 GB/s read+write on the same HCU. A ratio like that is
a candidate, not a conclusion: check the GEMM's tile (M=16 falls off a tile cliff) before optimizing.

## 7. Failure catalogue (this round)

| symptom | cause | fix |
|---|---|---|
| trace has no MoE kernels, 800 µs gaps | torch/hipprof drop graph nodes | rocprofv2 `--kernel-trace` (§1) |
| `_fused_eh_norm_kernel` 1 ms in trace, 4 µs standalone | first graph node absorbs dropped time | same; never rank by graph-first-node |
| profile window shows only prefill | window opened before decode | gate on `/metrics` (§3) |
| `RuntimeError: No CUDA GPUs are available` in a bench client container | the image's hcu platform plugin imports FLA, which probes the GPU arch | `-e VLLM_PLUGINS=none` for device-less client containers |
| `docker exec` refused for the account | node docker hook | run client steps in `docker run` sidecars; stop the server via a `--pid host` sidecar |
| `--enforce-eager` server crashes at c≥8 with MTP | indexer draft-metadata build (`expanded_seq_lens_buffer` size mismatch) | do not use eager for MTP runs; keep the owner's graph mode |
| second server OOMs at warmup, model loads 103 GiB instead of 58 | unknown; same args and env as a good run | check `hy-smi --showpids` for a foreign process before any run; check `Model loading took` in the log |
| `pkill -f <script>` from ssh kills the ssh shell | pattern matches its own command line | kill by PID |
| stop a driver ⇒ server kept running for minutes | the driver's trap stops the server; `SIGTERM` to the driver is delayed | wait for `ps -p` to clear, then verify `hy-smi --showpids` is empty |
| 6 GB transfer over the MBA at 3 MB/s | relay through a laptop | direct LAN node-to-node (`10.17.176.x`), 58 s for 6.3 GB |
| node clocks differ by 8 min | no NTP across hosts | never compare timestamps across nodes; use the run dir name |

## 8. Checklist before reporting a profile

1. Tool proven on a graph-count probe on this stack (§1).
2. Window gated on steady decode of the target shape; step kinds counted per rank (§3).
3. Step boundary method stated; per-rank tables compared, not averaged (§4).
4. Ranking uses per-kernel device time from the complete trace; no wall or first-graph-node numbers.
5. Top ops checked against a byte roofline or a standalone graph timing (§5–6).
6. Every number carries the run directory and the node; the server's graph mode and `Model loading`
   line are recorded.
