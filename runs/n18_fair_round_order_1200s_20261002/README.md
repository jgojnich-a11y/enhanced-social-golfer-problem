# 18-player fair-target-66 round-order feasibility test

Unresolved: time limit or other stop without a feasible schedule or infeasibility proof.

Status: UNKNOWN; elapsed: 1200.028463 seconds.

This test removes the optimization objective and requires minimum spread >= 3 and total spread >= 66. All original constraints remain, plus lexicographic ordering of rounds 2-4. The hint is normalized by permuting these rounds. Both normalized saved baselines passed forced-model validation. This ordering preserves all schedules up to round permutation under the current round-invariant rules. Seed 73, eight workers, 1,200-second limit, saved six-hour fairness schedule hint. See run.json, model.pbtxt, solver.log and solver_snapshot.py for reproduction. An OPTIMAL status on this feasibility model means a valid threshold schedule was found; it does not establish maximum spread.
