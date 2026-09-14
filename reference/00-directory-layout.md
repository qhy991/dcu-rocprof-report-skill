# Directory layout

One profiling run = one directory. Never reuse, never mix runs.

```
profile/
└── <run_name>/            # e.g. 2026-09-14_rmsnorm_h4096_dcu
    ├── harness/           # .hip source, binary, build logs
    │   ├── kernel.hip
    │   ├── harness.hip
    │   └── build.log
    ├── reports/           # raw profiler output
    │   ├── prof_input.txt # the pmc metric input file (keep it — it IS the recipe)
    │   └── results.csv    # rocprof CSV output
    ├── analysis/          # parsed/derived data
    │   └── summary.md     # output of helpers/analyze_csv.py
    └── REPORT.md          # final report for the user
```

## Rules

1. **Create the directory before collecting anything.** Raw CSVs dumped to `/tmp` get lost when the `--rm` container exits — the container is ephemeral by design.
2. **Keep `prof_input.txt` next to `results.csv`.** On DTK the metric *set* is part of the measurement (counter-group limits, see 03-collection), so the input file is the only record of what was requested.
3. **Never edit a collected CSV.** If a collection was wrong, make a new run directory.
4. **Name runs `<date>_<kernel>_<variant>`** so two runs of the same kernel under different hypotheses sort correctly.
