# 15 players — optimal fairness schedule

Minimum distinct tee slots: **2**. Total spread: **45**. Four rounds; no repeated partners. Group sizes: 3, 4, 4, 4.

| Group | Tee time | Round 1 players | Round 2 players | Round 3 players | Round 4 players |
|---:|---|---|---|---|---|
| 1 | 10:00 | 13, 14, 15 | 3, 6, 11 | 4, 7, 12 | 2, 5, 10 |
| 2 | 10:07 | 9, 10, 11, 12 | 2, 7, 9, 15 | 2, 8, 11, 14 | 4, 6, 9, 14 |
| 3 | 10:14 | 5, 6, 7, 8 | 4, 8, 10, 13 | 1, 6, 10, 15 | 3, 8, 12, 15 |
| 4 | 10:21 | 1, 2, 3, 4 | 1, 5, 12, 14 | 3, 5, 9, 13 | 1, 7, 11, 13 |

| Player | Round 1 | Round 2 | Round 3 | Round 4 | Distinct slots | Three-player rounds | Four-player rounds |
|---:|---|---|---|---|---:|---:|---:|
| 1 | 10:21 | 10:21 | 10:14 | 10:21 | 2 | 0 | 4 |
| 2 | 10:21 | 10:07 | 10:07 | 10:00 | 3 | 1 | 3 |
| 3 | 10:21 | 10:00 | 10:21 | 10:14 | 3 | 1 | 3 |
| 4 | 10:21 | 10:14 | 10:00 | 10:07 | 4 | 1 | 3 |
| 5 | 10:14 | 10:21 | 10:21 | 10:00 | 3 | 1 | 3 |
| 6 | 10:14 | 10:00 | 10:14 | 10:07 | 3 | 1 | 3 |
| 7 | 10:14 | 10:07 | 10:00 | 10:21 | 4 | 1 | 3 |
| 8 | 10:14 | 10:14 | 10:07 | 10:14 | 2 | 0 | 4 |
| 9 | 10:07 | 10:07 | 10:21 | 10:07 | 2 | 0 | 4 |
| 10 | 10:07 | 10:14 | 10:14 | 10:00 | 3 | 1 | 3 |
| 11 | 10:07 | 10:00 | 10:07 | 10:21 | 3 | 1 | 3 |
| 12 | 10:07 | 10:21 | 10:00 | 10:14 | 4 | 1 | 3 |
| 13 | 10:00 | 10:14 | 10:21 | 10:21 | 3 | 1 | 3 |
| 14 | 10:00 | 10:21 | 10:07 | 10:07 | 3 | 1 | 3 |
| 15 | 10:00 | 10:07 | 10:14 | 10:14 | 3 | 1 | 3 |

Selected source: [runs/n15_to24_fair_sweep_20261002/n15/solution_n15_sol.csv](../../runs/n15_to24_fair_sweep_20261002/n15/solution_n15_sol.csv).
Optimality evidence: [runs/n15_to24_fair_sweep_20261002/n15/run.json](../../runs/n15_to24_fair_sweep_20261002/n15/run.json).
The CSV is headerless: round, group, player IDs. Player IDs are placeholders; assigning names preserves validity and fairness scores.
