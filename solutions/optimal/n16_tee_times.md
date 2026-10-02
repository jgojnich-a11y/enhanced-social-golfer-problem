# 16 players — optimal fairness schedule

Minimum distinct tee slots: **2**. Total spread: **44**. Four rounds; no repeated partners. Group sizes: 4, 4, 4, 4.

| Group | Tee time | Round 1 players | Round 2 players | Round 3 players | Round 4 players |
|---:|---|---|---|---|---|
| 1 | 10:00 | 13, 14, 15, 16 | 4, 7, 12, 15 | 3, 7, 9, 16 | 3, 5, 10, 15 |
| 2 | 10:07 | 9, 10, 11, 12 | 3, 8, 11, 14 | 2, 5, 12, 14 | 1, 8, 12, 16 |
| 3 | 10:14 | 5, 6, 7, 8 | 2, 6, 10, 16 | 1, 6, 11, 15 | 2, 7, 11, 13 |
| 4 | 10:21 | 1, 2, 3, 4 | 1, 5, 9, 13 | 4, 8, 10, 13 | 4, 6, 9, 14 |

| Player | Round 1 | Round 2 | Round 3 | Round 4 | Distinct slots | Three-player rounds | Four-player rounds |
|---:|---|---|---|---|---:|---:|---:|
| 1 | 10:21 | 10:21 | 10:14 | 10:07 | 3 | 0 | 4 |
| 2 | 10:21 | 10:14 | 10:07 | 10:14 | 3 | 0 | 4 |
| 3 | 10:21 | 10:07 | 10:00 | 10:00 | 3 | 0 | 4 |
| 4 | 10:21 | 10:00 | 10:21 | 10:21 | 2 | 0 | 4 |
| 5 | 10:14 | 10:21 | 10:07 | 10:00 | 4 | 0 | 4 |
| 6 | 10:14 | 10:14 | 10:14 | 10:21 | 2 | 0 | 4 |
| 7 | 10:14 | 10:00 | 10:00 | 10:14 | 2 | 0 | 4 |
| 8 | 10:14 | 10:07 | 10:21 | 10:07 | 3 | 0 | 4 |
| 9 | 10:07 | 10:21 | 10:00 | 10:21 | 3 | 0 | 4 |
| 10 | 10:07 | 10:14 | 10:21 | 10:00 | 4 | 0 | 4 |
| 11 | 10:07 | 10:07 | 10:14 | 10:14 | 2 | 0 | 4 |
| 12 | 10:07 | 10:00 | 10:07 | 10:07 | 2 | 0 | 4 |
| 13 | 10:00 | 10:21 | 10:21 | 10:14 | 3 | 0 | 4 |
| 14 | 10:00 | 10:07 | 10:07 | 10:21 | 3 | 0 | 4 |
| 15 | 10:00 | 10:00 | 10:14 | 10:00 | 2 | 0 | 4 |
| 16 | 10:00 | 10:14 | 10:00 | 10:07 | 3 | 0 | 4 |

Selected source: [runs/n15_to24_fair_sweep_20261002/n16/solution_n16_sol.csv](../../runs/n15_to24_fair_sweep_20261002/n16/solution_n16_sol.csv).
Optimality evidence: [runs/n15_to24_fair_sweep_20261002/n16/run.json](../../runs/n15_to24_fair_sweep_20261002/n16/run.json).
The CSV is headerless: round, group, player IDs. Player IDs are placeholders; assigning names preserves validity and fairness scores.
