# dcu-rocprof-report-skill

A Claude Code / agent skill for profiling GPU kernels on **Hygon DCU** (bw1100-class
DCU-3G, gfx938, DTK 26.04) with **rocprof** — the rocprof/DCU counterpart of
[mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill)
(which targets NVIDIA B200 + Nsight Compute).

Everything in `reference/` and `helpers/` was **verified on real DCU hardware**:
the rocprof CSV schema, the metric vocabulary, the counter-group abort behavior,
and every entry in the common-issues list was observed or bisection-proven on
DCU-3G devices running DTK 26.04.

## What's different from the ncu skill

| ncu-report-skill (B200) | this skill (DCU) |
|---|---|
| `ncu --set full` sections | rocprof v1 `pmc:` input files, **one counter group per run** |
| Rule engine with "Est. Speedup" hints | raw counters only — the helpers compute the ratios |
| `ncu_report` Python module | plain CSV (`helpers/analyze_csv.py`, schema in reference/04) |
| CUDA harnesses | HIP harnesses (`hipcc --offload-arch=gfx938`) or torch cpp_extension + hipify |
| — | CUDA→HIP porting traps (the `__shfl_xor_sync` VMFault et al.) |

## Use

Install as a Claude Code skill (copy into `.claude/skills/` or point a plugin at
the repo), then trigger with e.g. *"profile this kernel on DCU"*, *"为什么这个
kernel 在 DCU 上慢"*, or *"帮我把这个 CUDA kernel 移植到 DCU"*.

Entry point: [`SKILL.md`](SKILL.md).

## Layout

```
SKILL.md                      # entry point: when to use, quickstart, golden rules
reference/00-09               # workflow, harness, collection, parsing, analysis,
                              # diagnosis playbook, report template, metric names, common issues
helpers/
  harness_template.hip        # standalone HIP harness (paste kernel, compile, profile)
  analyze_csv.py              # rocprof CSV parser + derived ratios + consistency checks
  profile_container.sh        # wrap any command in the DTK container with rocprof on PATH
```

## Provenance

- Toolchain/hardware facts verified on a bw1100 node (8× DCU-3G, 144 GiB each,
  Hygon C86 host, Kylin V10).
- The porting lessons come from a full CUDA→DCU port of a fused RMSNorm kernel
  (14/14 correctness workloads, 13.8× geomean vs eager PyTorch).
- HYGON-AI's [SkillHub](https://github.com/HYGON-AI/SkillHub) and
  [inference-cookbook-das](https://github.com/HYGON-AI/inference-cookbook-das)
  are the deployment-side references.

## License

MIT — see [LICENSE](LICENSE).
