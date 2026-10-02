# 19-player fairness targets with round-pair strengthening

**Updated bound: the optimal 19-player minimum-first fairness score is either
(3,68) or (3,69).** Total 70 is now proved infeasible; total 69 remains
unresolved. The counting bound excludes all totals above 70 and minimum
spread 4, so the total-70 proof tightens the fair upper bound to 69.

No validated improvement found. At least one target remains unresolved or failed; global fairness optimality remains unproven.

Sequential feasibility searches, 600 seconds each, seed 73, eight workers, original model constraints and the saved fairness-68 assignment hint. Explicit implied round-pair repeat constraints are added; no optional generator strengthening or round-order symmetry.

| Case | Target total | Early repeats | Late repeats | Status | Search seconds |
|---|---:|---:|---:|---|---:|
| A | 70 | 0 | 6 | INFEASIBLE | 131.594972 |
| B | 69 | 0 | 7 | UNKNOWN | 600.030662 |

Minimum spread >=3 is required in both tests. Balance permits each player at most one triple appearance, so early repeats are zero. At least one same-late-slot player is required for every pair of rounds; at minimum spread 3 these are six distinct players, bounding total spread by 70. The two exact-total tests cover every improvement over the validated (3,68) baseline. See COUNTING_BOUND.md.

Round-pair counts are explicitly constrained to six ones for total 70 or five ones and one two for total 69, retaining every qualifying schedule. Both saved schedules were checked before target restrictions; modeled counts matched all six independently computed intersections. Both baseline schedules passed coverage, pair uniqueness, capacities, participation balance and forced-model validation. Repeat-variable totals matched direct CSV counts. The retained generator and original schedules are preserved. Models, logs, statistics, source and hint checksums, exact commands, and any qualifying solutions are saved here.
