# End-to-end workflow

From "user asks why a kernel is slow" to "final report". Follow in order; skip steps only with a stated reason.

## 1. Frame the question

- What kernel, what shape(s), what framework dispatch (raw hipcc binary / torch extension / vLLM operator)?
- What is the reference point — eager PyTorch, another GPU's baseline, a previous version of this kernel?
- If the user says "slow", get a number first: run the benchmark once and record it.

## 2. Get a working harness

- Existing binary you can launch → use it directly.
- PyTorch-integrated kernel → `torch.utils.cpp_extension.load` route (02-harness-guide).
- Anything else → standalone `.hip` harness from `helpers/harness_template.hip`.

Verify the harness produces **correct output** before profiling. A kernel that crashes or computes garbage has profiling data that means nothing (and on DCU, crashes often look like VMFaults — see 09-common-issues).

## 3. Collect

1. Write `prof_input.txt` with a single counter group (≤6 metrics first pass).
2. `rocprof -i prof_input.txt -o results.csv ./harness` inside the DTK container.
3. Sanity-check the CSV row count and `grd`/`wgr` against the launch config you intended.
4. If a second question needs other counters, **collect a second run** — don't widen one run past the counter-group limit.

## 4. Analyze

- `python3 helpers/analyze_csv.py reports/results.csv`
- Work through 05-analysis-dimensions; on any given kernel only 1–2 dimensions will dominate.
- Cross-check derived ratios for internal consistency (e.g. `Wavefronts` vs `grd/wgr × waves-per-CTA`).

## 5. Diagnose

Match counter patterns to 06-diagnosis-playbook. Each playbook entry names the counters that confirm *and* the counters that would refute it — check both.

## 6. Plan and report

- Rank fixes by expected impact, each tied to a specific counter value.
- State what to re-profile after each fix (the metric that should move).
- Write `REPORT.md` per 07-report-template.

## 7. If the kernel is a CUDA port

Read 09-common-issues **before** running anything — the three mechanical traps there produce compiler errors and VMFaults that waste hours if approached cold.
