# 18-player target-67 feasibility test

Unresolved: time limit or other stop without a feasible schedule or infeasibility proof.

Status: UNKNOWN; elapsed: 1000.0238820000001 seconds.

This test removes the optimization objective and requires total spread >= 67. All original constraints, including fixed first-round grouping, remain. Seed 73, eight workers, 1,000-second limit, recovered schedule hint. See run.json, model.pbtxt, solver.log and solver_snapshot.py for reproduction. An OPTIMAL status on this feasibility model means a valid threshold schedule was found; it does not establish maximum spread.
