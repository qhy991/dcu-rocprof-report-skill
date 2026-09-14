---
name: dcu-rocprof-report-skill
description: Profile HIP/CUDA kernels with rocprof on Hygon DCU (bw1100 / gfx938 / DTK). Use when the user asks to profile a kernel on DCU, analyze its performance, diagnose bottlenecks, read a rocprof report, or port a CUDA kernel to DCU — including variants in Chinese ("profile 一下", "为什么慢", "rocprof 报告", "DCU 移植").
---

# Skill: DCU Kernel Profiling (Hygon DCU / rocprof)

**When to use:** user asks to profile a kernel on Hygon DCU, analyze its performance, find its bottlenecks, port a CUDA kernel to DCU, or write an optimization plan based on rocprof data. Triggers include: "profile X on DCU", "为什么这个 kernel 在 DCU 上慢", "rocprof 报告说...", "这个 CUDA kernel 能不能在 DCU 上跑".

**Target hardware (verified on this machine):** Hygon DCU-3G "C-3000 HCU" (bw1100 node, 8 devices, 144 GiB HBM per card, gfx938 ISA `amdgcn-amd-amdhsa--gfx938:sramecc+:xnack-`). Toolchain: DTK 26.04 (ROCm fork) — `hipcc` 25.10.0 / clang 17. Most advice below is DTK-on-DCU-specific; generic CDNA/gfx9 advice is marked as such.

**Key mental model:** DTK is a **downstream fork of ROCm**. Tools are renamed and occasionally *broken in fork-specific ways* (see the `__shfl_xor_sync` VMFault in [09-common-issues](reference/09-common-issues.md)). When upstream ROCm docs disagree with what you observe, trust the probe on the machine.

---

## Golden rule

**Profile → Diagnose → Plan, in that order. Never guess.**

Most under-performing kernels are under-performing for exactly one reason that a 30-second rocprof run can tell you. Don't invent hypotheses before you have the counters. Don't start coding a fix before you've matched the observed pattern to a known diagnosis.

Unlike Nsight Compute, **rocprof has no rule engine and no "Est. Speedup" hints** — it returns raw counters only. The analysis ratios (IPC, bytes/metrics, occupancy) are *your* job; the helpers in [`helpers/`](helpers/) compute them.

---

## Environment facts (verified, do not re-derive)

- **The GPU toolchain lives in the DTK container, the runtime on the host.** Profile inside the vLLM/DTK image with `--device /dev/dri --device /dev/kfd -v /opt/hyhal:/opt/hyhal:ro`. Without the hyhal mount, `import torch` fails with `librocm_smi64.so.2` errors.
- **rocprof v1 CLI:** `rocprof --list-basic | --list-derived | -i input.txt | -o out.csv <app>`, located at `/opt/dtk/rocprofiler/bin/` (must add to PATH; `source /opt/dtk/env.sh` does not add it).
- **Metric definitions:** `/opt/dtk/rocprofiler/lib/rocprofiler/metrics.xml` (same content as `--list-basic` / `--list-derived`).
- **Compile:** `hipcc --offload-arch=gfx938 -O3` (or torch `cpp_extension.load` with `PYTORCH_ROCM_ARCH=gfx938 ROCM_HOME=HIP_PATH=CUDA_HOME=/opt/dtk`). `hipify-perconv` runs automatically in torch builds and **needs a writable source directory**.
- **CUDA→HIP source porting has exactly three known mechanical traps on DTK 26.04** — see [09-common-issues](reference/09-common-issues.md). Every one of them was hit and fixed on a real kernel port (RMSNorm h4096, 13.8× speedup verified).

---

## Quickstart (what to do when someone says "profile this kernel")

0. **Create a run directory first** under `profile/<run_name>/` — one directory per run, never reuse. Each run contains `harness/`, `reports/`, `analysis/`, and `REPORT.md`. See [`reference/00-directory-layout.md`](reference/00-directory-layout.md).

1. **Decide what you're profiling.** Which shapes, which dispatch path, what question. If inputs are variable-sized, pick representative shapes from the user's workload — never profile with arbitrary inputs.

2. **Build a standalone harness** unless profiling through an existing binary. Two supported routes: (a) `hipcc` standalone `.hip` binary — fastest, use [`helpers/harness_template.hip`](helpers/harness_template.hip); (b) torch `cpp_extension.load` for PyTorch-integrated kernels. See [`reference/02-harness-guide.md`](reference/02-harness-guide.md).

3. **Collect with rocprof using a pmc input file.** Keep the metric set within one hardware counter group (~6 counters) — larger sets abort with `Context Create failed` on DTK. Write to `reports/`. See [`reference/03-collection.md`](reference/03-collection.md).

4. **Parse the CSV with `helpers/analyze_csv.py`**, not by eye. See [`reference/04-python-api.md`](reference/04-python-api.md) for the exact column schema (verified).

5. **Work through the analysis dimensions** — wavefronts, VALU/SALU/MMOP pipe utilization, memory (FETCH_SIZE/WRITE_SIZE vs peak), LDS, occupancy. See [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md).

6. **Match the diagnosis playbook.** See [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md).

7. **Write the report** at `profile/<run_name>/REPORT.md`, recommendations ranked by expected impact. See [`reference/07-report-template.md`](reference/07-report-template.md).

---

## File index

### Reference docs

| File | Purpose |
|---|---|
| [`reference/00-directory-layout.md`](reference/00-directory-layout.md) | **Read first.** One run = one subdirectory, no cross-contamination |
| [`reference/01-workflow.md`](reference/01-workflow.md) | End-to-end checklist from request to final report |
| [`reference/02-harness-guide.md`](reference/02-harness-guide.md) | hipcc standalone vs torch cpp_extension; the writable-source hipify trap |
| [`reference/03-collection.md`](reference/03-collection.md) | rocprof pmc input files, single-group limit, rocprofv2 traces |
| [`reference/04-python-api.md`](reference/04-python-api.md) | Verified CSV schema + parsing patterns |
| [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md) | Six analysis dimensions on gfx938 |
| [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md) | Counter pattern → diagnosis → fix |
| [`reference/07-report-template.md`](reference/07-report-template.md) | Report structure |
| [`reference/08-gfx938-metric-names.md`](reference/08-gfx938-metric-names.md) | The verified metric vocabulary on DCU (297 derived + basic) |
| [`reference/09-common-issues.md`](reference/09-common-issues.md) | **The three CUDA→HIP port traps, the shfl VMFault, rocprof aborts, VMFault dump reading** |

### Helpers

| File | Purpose |
|---|---|
| [`helpers/harness_template.hip`](helpers/harness_template.hip) | Standalone HIP harness — paste kernel, fill allocation, compile with one command |
| [`helpers/analyze_csv.py`](helpers/analyze_csv.py) | Parse rocprof CSV, compute derived ratios (IPC, occupancy, bandwidth %) |
| [`helpers/profile_container.sh`](helpers/profile_container.sh) | Wrap any command in the DTK container with GPU devices + hyhal mounted |

---

## Critical lessons (don't skip)

1. **`__shfl_xor_sync` with a 32-bit mask fails to compile (static_assert demands 64-bit), and with the 64-bit mask it VMFaults at runtime.** Use `__shfl_xor(v, offset, 32)` instead. This was proven by bisection on real hardware — see [09-common-issues](reference/09-common-issues.md).

2. **Metric sets beyond one hardware counter group abort rocprof** (`Context Create failed`, core dumped) even though rocprof prints a "proposed metrics group set". Don't request 8 counters; request ≤6 per run and run twice if needed.

3. **The rocprof CSV `KernelName` carries a `[clone .kd]` suffix** on DTK. Filter kernels with a prefix/regex match, never exact equality.

4. **`gpu-id` in rocprof output is 8–15, not 0–7.** The DCU presents its 8 devices as GPU nodes 8–15 (host NUMA CPU nodes are 0–7). Don't "fix" this in analysis.

5. **hipify in torch builds writes `kernel.hip` next to the source.** A read-only source directory fails the build with "Failed to save kernel.hip". Copy sources to a writable directory first.

6. **Don't delegate understanding.** Name the two or three counter values that back your conclusion — e.g. "`FETCH_SIZE` at 40% of the 5.3 TB/s peak with `GPUBusy` near 100 and `Wavefronts` 4× the minimum" — never "the profile shows it's memory-bound".

---

## Related skills

- HYGON-AI's [SkillHub](https://github.com/HYGON-AI/SkillHub) has DCU-adjacent skills (torch trace operator profiler). The deployment-side reference is HYGON-AI's `inference-cookbook-das`.
