# 18 players — optimal fairness schedule

Minimum distinct tee slots: **3**. Total spread: **65**. Four rounds; no repeated partners. Group sizes: 3, 3, 4, 4, 4.

| Group | Tee time | Round 1 players | Round 2 players | Round 3 players | Round 4 players |
|---:|---|---|---|---|---|
| 1 | 10:00 | 16, 17, 18 | 8, 12, 15 | 1, 9, 14 | 5, 11, 13 |
| 2 | 10:07 | 13, 14, 15 | 4, 7, 10 | 3, 6, 11 | 2, 8, 9 |
| 3 | 10:14 | 9, 10, 11, 12 | 1, 6, 13, 16 | 2, 7, 15, 17 | 4, 6, 14, 17 |
| 4 | 10:21 | 5, 6, 7, 8 | 2, 11, 14, 18 | 4, 5, 12, 16 | 3, 10, 15, 16 |
| 5 | 10:28 | 1, 2, 3, 4 | 3, 5, 9, 17 | 8, 10, 13, 18 | 1, 7, 12, 18 |

| Player | Round 1 | Round 2 | Round 3 | Round 4 | Distinct slots | Three-player rounds | Four-player rounds |
|---:|---|---|---|---|---:|---:|---:|
| 1 | 10:28 | 10:14 | 10:00 | 10:28 | 3 | 1 | 3 |
| 2 | 10:28 | 10:21 | 10:14 | 10:07 | 4 | 1 | 3 |
| 3 | 10:28 | 10:28 | 10:07 | 10:21 | 3 | 1 | 3 |
| 4 | 10:28 | 10:07 | 10:21 | 10:14 | 4 | 1 | 3 |
| 5 | 10:21 | 10:28 | 10:21 | 10:00 | 3 | 1 | 3 |
| 6 | 10:21 | 10:14 | 10:07 | 10:14 | 3 | 1 | 3 |
| 7 | 10:21 | 10:07 | 10:14 | 10:28 | 4 | 1 | 3 |
| 8 | 10:21 | 10:00 | 10:28 | 10:07 | 4 | 2 | 2 |
| 9 | 10:14 | 10:28 | 10:00 | 10:07 | 4 | 2 | 2 |
| 10 | 10:14 | 10:07 | 10:28 | 10:21 | 4 | 1 | 3 |
| 11 | 10:14 | 10:21 | 10:07 | 10:00 | 4 | 2 | 2 |
| 12 | 10:14 | 10:00 | 10:21 | 10:28 | 4 | 1 | 3 |
| 13 | 10:07 | 10:14 | 10:28 | 10:00 | 4 | 2 | 2 |
| 14 | 10:07 | 10:21 | 10:00 | 10:14 | 4 | 2 | 2 |
| 15 | 10:07 | 10:00 | 10:14 | 10:21 | 4 | 2 | 2 |
| 16 | 10:00 | 10:14 | 10:21 | 10:21 | 3 | 1 | 3 |
| 17 | 10:00 | 10:28 | 10:14 | 10:14 | 3 | 1 | 3 |
| 18 | 10:00 | 10:21 | 10:28 | 10:28 | 3 | 1 | 3 |

Selected source: [runs/n15_to24_fair_sweep_20261002/n18/solution_n18_sol.csv](../../runs/n15_to24_fair_sweep_20261002/n18/solution_n18_sol.csv).
Optimality evidence: [runs/n18_canonical_two_rounds_20261003/README.md](../../runs/n18_canonical_two_rounds_20261003/README.md).
The CSV is headerless: round, group, player IDs. Player IDs are placeholders; assigning names preserves validity and fairness scores.
