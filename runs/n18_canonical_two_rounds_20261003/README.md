# 18-player fairness optimality proved

The validated (minimum individual spread 3, total spread 65) schedule is
globally optimal under the current minimum-first fairness objective.

Every improvement must have two rounds sharing a tee slot for at most one
player. Exhaustive binary intersection enumeration gives exactly two relevant
symmetry classes. Fixing a complete representative of each first-two-round
class and leaving the final two rounds free rules out every improvement.
See SYMMETRY.md for the complete coverage argument.

| Same-slot players in first two rounds | Integer status / seconds | Retained status / seconds |
|---:|---|---|
| 0 | INFEASIBLE / 11.495 | INFEASIBLE / 17.214 |
| 1 | INFEASIBLE / 3.562 | INFEASIBLE / 6.041 |

Each test required every player to use at least three distinct slots and total
spread at least 66. These restrictions also cover any minimum-four schedule.
Both formulations used 1200-second limits, eight workers, seed 73. Neither
cross-check uses additional round-pair count constraints. Both models accept
the normalized saved baseline before the improvement restrictions.

Audit: enumeration.json, source snapshots, audit.json, dependencies.txt,
per-case exported models/results/statistics, solver.log and retained_solver.log.
Baseline: ../n18_20261001_214154/fair/solution_n18_sol.csv.
The retained root generator and original saved schedules were preserved.
All runs completed; no run is pending. Changes remain uncommitted.

This proves the chosen fairness objective, not the historical total-only
objective. The historical total 66 with minimum 2 remains a valid separate
reference. All fairness scores for player counts 15 through 24 are now proven.
