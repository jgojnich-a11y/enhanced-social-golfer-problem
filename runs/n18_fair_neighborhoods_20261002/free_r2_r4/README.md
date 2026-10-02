# 18-player fair-target-66 restricted neighborhood test

Proved no qualifying improvement exists in this restricted neighborhood. This does not establish global optimality.

Status: INFEASIBLE; elapsed: 4.328515 seconds.

This test removes the optimization objective and requires minimum spread >= 3 and total spread >= 66. All original constraints remain. Additional complete-round assignments restrict this neighborhood; see fixed_rounds and free_rounds in run.json. INFEASIBLE applies only to this neighborhood. Seed 73, eight workers, 400-second limit, saved six-hour fairness schedule hint. See run.json, model.pbtxt, solver.log and solver_snapshot.py for reproduction. An OPTIMAL status on this feasibility model means a valid threshold schedule was found; it does not establish maximum spread.
