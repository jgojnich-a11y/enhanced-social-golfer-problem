# 17 players — optimal fairness schedule

Minimum distinct tee slots: **3**. Total spread: **63**. Four rounds; no repeated partners. Group sizes: 3, 3, 3, 4, 4.

| Group | Tee time | Round 1 players | Round 2 players | Round 3 players | Round 4 players |
|---:|---|---|---|---|---|
| 1 | 10:00 | 15, 16, 17 | 8, 11, 14 | 4, 7, 16 | 1, 5, 12 |
| 2 | 10:07 | 12, 13, 14 | 3, 6, 15 | 2, 5, 10 | 3, 7, 9 |
| 3 | 10:14 | 9, 10, 11 | 4, 5, 13 | 1, 6, 14 | 2, 8, 17 |
| 4 | 10:21 | 5, 6, 7, 8 | 2, 9, 12, 16 | 3, 11, 12, 17 | 4, 10, 14, 15 |
| 5 | 10:28 | 1, 2, 3, 4 | 1, 7, 10, 17 | 8, 9, 13, 15 | 6, 11, 13, 16 |

| Player | Round 1 | Round 2 | Round 3 | Round 4 | Distinct slots | Three-player rounds | Four-player rounds |
|---:|---|---|---|---|---:|---:|---:|
| 1 | 10:28 | 10:28 | 10:14 | 10:00 | 3 | 2 | 2 |
| 2 | 10:28 | 10:21 | 10:07 | 10:14 | 4 | 2 | 2 |
| 3 | 10:28 | 10:07 | 10:21 | 10:07 | 3 | 2 | 2 |
| 4 | 10:28 | 10:14 | 10:00 | 10:21 | 4 | 2 | 2 |
| 5 | 10:21 | 10:14 | 10:07 | 10:00 | 4 | 3 | 1 |
| 6 | 10:21 | 10:07 | 10:14 | 10:28 | 4 | 2 | 2 |
| 7 | 10:21 | 10:28 | 10:00 | 10:07 | 4 | 2 | 2 |
| 8 | 10:21 | 10:00 | 10:28 | 10:14 | 4 | 2 | 2 |
| 9 | 10:14 | 10:21 | 10:28 | 10:07 | 4 | 2 | 2 |
| 10 | 10:14 | 10:28 | 10:07 | 10:21 | 4 | 2 | 2 |
| 11 | 10:14 | 10:00 | 10:21 | 10:28 | 4 | 2 | 2 |
| 12 | 10:07 | 10:21 | 10:21 | 10:00 | 3 | 2 | 2 |
| 13 | 10:07 | 10:14 | 10:28 | 10:28 | 3 | 2 | 2 |
| 14 | 10:07 | 10:00 | 10:14 | 10:21 | 4 | 3 | 1 |
| 15 | 10:00 | 10:07 | 10:28 | 10:21 | 4 | 2 | 2 |
| 16 | 10:00 | 10:21 | 10:00 | 10:28 | 3 | 2 | 2 |
| 17 | 10:00 | 10:28 | 10:21 | 10:14 | 4 | 2 | 2 |

Selected source: [runs/n15_to24_fair_sweep_20261002/n17/solution_n17_sol.csv](../../runs/n15_to24_fair_sweep_20261002/n17/solution_n17_sol.csv).
Optimality evidence: [runs/n17_target64_repeat_cases_20261002/README.md](../../runs/n17_target64_repeat_cases_20261002/README.md).
The CSV is headerless: round, group, player IDs. Player IDs are placeholders; assigning names preserves validity and fairness scores.
