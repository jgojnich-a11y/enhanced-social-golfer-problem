# 19 players — optimal fairness schedule

Minimum distinct tee slots: **3**. Total spread: **68**. Four rounds; no repeated partners. Group sizes: 3, 4, 4, 4, 4.

| Group | Tee time | Round 1 players | Round 2 players | Round 3 players | Round 4 players |
|---:|---|---|---|---|---|
| 1 | 10:00 | 17, 18, 19 | 4, 8, 16 | 3, 5, 10 | 6, 11, 14 |
| 2 | 10:07 | 13, 14, 15, 16 | 3, 7, 11, 19 | 1, 8, 12, 15 | 2, 8, 9, 17 |
| 3 | 10:14 | 9, 10, 11, 12 | 2, 6, 10, 15 | 4, 7, 14, 17 | 3, 12, 16, 18 |
| 4 | 10:21 | 5, 6, 7, 8 | 1, 9, 14, 18 | 2, 11, 13, 18 | 4, 5, 15, 19 |
| 5 | 10:28 | 1, 2, 3, 4 | 5, 12, 13, 17 | 6, 9, 16, 19 | 1, 7, 10, 13 |

| Player | Round 1 | Round 2 | Round 3 | Round 4 | Distinct slots | Three-player rounds | Four-player rounds |
|---:|---|---|---|---|---:|---:|---:|
| 1 | 10:28 | 10:21 | 10:07 | 10:28 | 3 | 0 | 4 |
| 2 | 10:28 | 10:14 | 10:21 | 10:07 | 4 | 0 | 4 |
| 3 | 10:28 | 10:07 | 10:00 | 10:14 | 4 | 1 | 3 |
| 4 | 10:28 | 10:00 | 10:14 | 10:21 | 4 | 1 | 3 |
| 5 | 10:21 | 10:28 | 10:00 | 10:21 | 3 | 1 | 3 |
| 6 | 10:21 | 10:14 | 10:28 | 10:00 | 4 | 1 | 3 |
| 7 | 10:21 | 10:07 | 10:14 | 10:28 | 4 | 0 | 4 |
| 8 | 10:21 | 10:00 | 10:07 | 10:07 | 3 | 1 | 3 |
| 9 | 10:14 | 10:21 | 10:28 | 10:07 | 4 | 0 | 4 |
| 10 | 10:14 | 10:14 | 10:00 | 10:28 | 3 | 1 | 3 |
| 11 | 10:14 | 10:07 | 10:21 | 10:00 | 4 | 1 | 3 |
| 12 | 10:14 | 10:28 | 10:07 | 10:14 | 3 | 0 | 4 |
| 13 | 10:07 | 10:28 | 10:21 | 10:28 | 3 | 0 | 4 |
| 14 | 10:07 | 10:21 | 10:14 | 10:00 | 4 | 1 | 3 |
| 15 | 10:07 | 10:14 | 10:07 | 10:21 | 3 | 0 | 4 |
| 16 | 10:07 | 10:00 | 10:28 | 10:14 | 4 | 1 | 3 |
| 17 | 10:00 | 10:28 | 10:14 | 10:07 | 4 | 1 | 3 |
| 18 | 10:00 | 10:21 | 10:21 | 10:14 | 3 | 1 | 3 |
| 19 | 10:00 | 10:07 | 10:28 | 10:21 | 4 | 1 | 3 |

Selected source: [runs/n15_to24_fair_sweep_20261002/n19/solution_n19_sol.csv](../../runs/n15_to24_fair_sweep_20261002/n19/solution_n19_sol.csv).
Optimality evidence: [runs/n19_canonical_two_rounds_20261003/README.md](../../runs/n19_canonical_two_rounds_20261003/README.md).
The CSV is headerless: round, group, player IDs. Player IDs are placeholders; assigning names preserves validity and fairness scores.
