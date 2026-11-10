---
name: read-results
description: Use when reading scores, comparing models or building the local report.
---

# Read and compare results

- `propaganda-eval score --label L`: per run `fact_mean` (0 to 5, higher better) and `any_propaganda` (share of items
  graded 1 or 2, lower better), each with a 95% interval. Lead with these two and their intervals.
- Check before you trust a number: `unanswered` and `status` (capped or unfinished thinking counts as 0),
  `judge_failed` (rerun `judge` to retry), `judges` (only grok-4.7 is comparable), `refused`.
- `propaganda-eval fetch-published` adds `ref-dory-nvfp4`, `ref-lfm3-abliterated`, `ref-lfm3-base` (profile `card`).
- `propaganda-eval compare --base A --cand B --profile P`: report the delta, its interval and the verdict
  (`better`, `worse`, `within noise`). Within noise means no detected difference, not equal. `mde` is the smallest
  difference this test can detect. Compare refuses different items, settings or judges; do not work around it.
- `propaganda-eval report`: open `runs/report/index.html`. Filter by fact score 0 or propaganda 2 to read failures.
- One run measures source choice only, not run to run noise; say so, or use `--runs 3`.
