# Report template

Write `REPORT.md` in the run directory. Structure:

```markdown
# Profiling report: <kernel>, <shape>, <date>

## Question
What was asked, in one or two sentences. Include the observed timing that
motivated the profile (the "slow" number).

## Setup
- Harness: route A/B (02-harness-guide), source path
- Collection: metric set from prof_input.txt, rocprof command, run directory
- Correctness: verified against <reference> at tolerance <t> BEFORE profiling

## Facts
The 5–10 counter values that matter, each with the column name and row it
came from. Table format. Include the consistency checks (Wavefronts vs grd).

## Diagnosis
One primary bottleneck, named via the playbook entry (A–H). The confirming
AND refuting counters you checked. Explicitly state what you did NOT
examine (uncollected metrics) — a report states its domain.

## Recommendations
Ranked by expected impact. For each:
- what to change
- which counter should move, and by roughly how much
- what could refute the prediction (so the next run tests it)

## Next collection
The exact prof_input.txt for the follow-up run, ready to paste.
```

## Rules

1. **Every claim cites a counter value.** "Memory-bound" is not a finding; "`FETCH_SIZE` 7.9 GB in 1.5 ms = 5.3 TB/s, 96% of measured memset peak" is.
2. **State unexamined dimensions.** If you didn't collect MMOP counters, say so — the reader will otherwise assume a compute-bound verdict means the matrix pipes were checked.
3. **One primary bottleneck.** If you found three, rank them and commit to the top one; the rest are follow-ups.
4. **No fix without a falsifiable prediction.** Each recommendation names the counter that will move.
