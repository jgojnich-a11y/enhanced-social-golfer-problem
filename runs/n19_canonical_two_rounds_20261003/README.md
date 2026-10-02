# 19-player canonical first-two-round experiment

Proved score 69 infeasible in the complete canonical representative. Together with the prior total-70 proof, the validated (3,68) baseline is lexicographically optimal under current rules.

Independent confirmation: the retained generator also proved this canonical
score-69 test INFEASIBLE in **13.143 seconds**, without the extra round-pair
constraints. See retained_crosscheck.json and its model, statistics and log;
retained_crosscheck.py reproduces the check. The integer proof took 0.863 seconds.

Status: INFEASIBLE; search seconds: 0.863.

Minimum spread >=3, total exactly 69, canonical rounds 1 and 2 fixed, rounds 3 and 4 free. Seed 73, eight workers, 600-second limit. The normalized old (2,69) schedule supplies 76 integer hints; it does not satisfy the fairness threshold.

All twelve admissible late-intersection patterns were enumerated and verified to lie in a single orbit. Saved schedule variants covering every pattern normalized to the same first two rounds, preserved validity and spread metrics, and passed forced-model checks in both representations before the target restrictions. See SYMMETRY.md, normalization_validation.json, canonical_first_two_rounds.csv, source snapshots, model.pbtxt, run.json and solver.log. Prior model checks and the total-70 proof are preserved. The retained solver and original schedules are unchanged. No run from this experiment is pending.
