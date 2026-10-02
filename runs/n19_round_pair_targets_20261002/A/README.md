# 19-player fair-total-70 test

Proved fair total 70 infeasible. Combine both target statuses with the mathematical upper bound 70 to assess global optimality.

Status: INFEASIBLE; search seconds: 131.595.

Requires minimum spread >=3, total exactly 70, early repeats 0 and late repeats 6. Seed 73, eight workers, 600-second limit, fairness-68 assignment hint. Explicit same-late-slot round-pair counts: six ones for total 70, or five ones and one two for total 69. These constraints are implied by the current rules and minimum spread three. The counting argument bounds fair total spread by 70; both 70 and 69 must be resolved to establish a lower optimum. See model.pbtxt, run.json and solver.log.
