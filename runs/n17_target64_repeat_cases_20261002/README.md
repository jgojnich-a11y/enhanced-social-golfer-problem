# 17-player score-64 repeat cases

Both exhaustive score-64 repeat cases are infeasible. Together with the prior global upper bound 64, the validated (3,63) baseline is lexicographically optimal under current rules.

Sequential searches, each with a 600-second limit, seed 73, eight workers, original model constraints and the saved fairness-63 assignment hint. No round-order symmetry or optional strengthening.

| Case | Early repeats | Late repeats | Status | Search seconds |
|---|---:|---:|---|---:|
| A | 1 | 3 | INFEASIBLE | 102.60244700000001 |
| B | 0 | 4 | INFEASIBLE | 61.46614700000001 |

The counting analysis proves at least three late repeats. Minimum spread >=3 permits at most one repeat per player, so total 64 means four different repeat players: cases A and B exhaust the remaining possibility. Higher fair totals were already excluded by the global weighted objective bound 271 (weight 69) in runs/n15_to24_fair_sweep_20261002/n17/run.json. See the copied prior_bound.json and ../n17_counting_analysis_20261002.md.

Both known schedules were forced into the model before case thresholds and verified; repeat-variable sums matched independently computed CSV counts. The retained generator and saved baseline schedules are unchanged. Each case folder preserves model, log, statistics, metadata and any qualifying solution.
