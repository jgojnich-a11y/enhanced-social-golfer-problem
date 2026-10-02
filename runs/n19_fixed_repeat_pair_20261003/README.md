# 19-player score-69 search with fixed exceptional round pair

Status: **UNKNOWN**, after 600.032 seconds. No qualifying schedule or infeasibility proof was found.

The best validated fairness result remains (minimum 3, total 68); the only unresolved improvement is (3,69). Total 70 was proved infeasible in the earlier round-pair audit.

Requires minimum spread >=3, total exactly 69, two late-repeat players for rounds 1–2, and one for every other round pair. Seed 73, eight workers, 600-second limit, normalized fairness-68 assignment hint (380 assignments).

All six pair choices were normalized in both saved baseline schedules: all twelve versions preserved schedule validity and scores, passed forced-model validation before target restrictions, and matched direct CSV round-pair counts. See SYMMETRY.md for why fixing the exceptional pair preserves every candidate up to round permutation and global player relabeling.

The search folder contains model.pbtxt, solver.log, solver_statistics.txt, run.json and the final report. Shared source and validator snapshots, original and normalized hints, normalization CSVs, dependency list and prior total-70 proof are in this folder. The retained generator and original saved schedules are unchanged.

The process exited successfully; final metadata agrees with statistics. No run is pending.
