# 19-player fairness targets 70 and 69

No validated improvement found. At least one target remains unresolved or failed; global fairness optimality remains unproven.

Sequential feasibility searches, 600 seconds each, seed 73, eight workers, original model constraints and the saved fairness-68 assignment hint. No optional strengthening or round-order symmetry.

| Case | Target total | Early repeats | Late repeats | Status | Search seconds |
|---|---:|---:|---:|---|---:|
| A | 70 | 0 | 6 | UNKNOWN | 600.037635 |
| B | 69 | 0 | 7 | UNKNOWN | 600.0312630000001 |

Minimum spread >=3 is required in both tests. Balance permits each player at most one triple appearance, so early repeats are zero. At least one same-late-slot player is required for every pair of rounds; at minimum spread 3 these are six distinct players, bounding total spread by 70. The two exact-total tests cover every improvement over the validated (3,68) baseline. See COUNTING_BOUND.md.

Both baseline schedules passed coverage, pair uniqueness, capacities, participation balance and forced-model validation. Repeat-variable totals matched direct CSV counts. The retained generator and original schedules are preserved. Models, logs, statistics, source and hint checksums, exact commands, and any qualifying solutions are saved here.
