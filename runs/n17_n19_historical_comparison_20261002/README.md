# Historical comparison: 17–19 players

Subsequent result: the exhaustive repeat-case tests in
`runs/n17_target64_repeat_cases_20261002/` establish (3,63) as the 17-player
lexicographic fairness optimum. The comparison below records the evidence
available before those tests; the 18- and 19-player fair totals remain unproven.

Compared saved four-round schedules in the project root, solutions and sol_1 against the new 60-second fairness sweep. Every candidate is checked for coverage, group sizes and order, pair uniqueness, and balanced four-player participation. No schedule was modified and no solver search was launched.

| Players | Main saved reference: minimum / total | New sweep: minimum / total | Max appearances in one slot: saved → new |
|---:|---|---|---|
| 17 | 2 / 64 | 3 / 63 | 2 → 2 |
| 18 | 2 / 66 | 3 / 65 | 3 → 2 |
| 19 | 2 / 69 | 3 / 68 | 3 → 2 |

## Findings

- Main saved references score (2,64), (2,66) and (2,69). The new sweep raises the minimum to 3, reduces the worst same-slot count from 3 to 2, and gives up one total distinct slot in each case.
- The saved 18-player files sol_1/run_1_n18_sol.csv and sol_1/run_2_n18_sol.csv already score (3,65) and pass validation. The new sweep matches their fairness score; this score is not a new historical record.
- Among the saved files inspected, the best minimum-first scores for 17 and 19 players are (2,64) and (2,69). The sweep improves those minimum-first scores to (3,63) and (3,68).
- Filename prefixes such as opt_solution do not establish an optimality proof. The new 17–19 runs remain FEASIBLE, with total spread at minimum 3 unproven.
- Comparing total spread alone favors the main historical references. Comparing minimum first favors the new 17- and 19-player schedules and ties the already-saved minimum-3 schedules for 18 players.

## All inspected files

| Source | File | Valid | Minimum | Total | Distinct-slot histogram |
|---|---|---|---:|---:|---|
| historical_saved | sol_1/run_10_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_11_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_12_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_13_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_14_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_15_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_16_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_17_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_18_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_19_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_1_n17_sol.csv | True | 2 | 64 | {2: 1, 3: 2, 4: 14} |
| historical_saved | sol_1/run_1_n18_sol.csv | True | 3 | 65 | {3: 7, 4: 11} |
| historical_saved | sol_1/run_1_n19_sol.csv | True | 2 | 69 | {2: 2, 3: 3, 4: 14} |
| historical_saved | sol_1/run_20_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_21_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_22_n18_sol.csv | True | 2 | 64 | {2: 1, 3: 6, 4: 11} |
| historical_saved | sol_1/run_23_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_24_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_25_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_26_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_27_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_28_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_29_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_2_n18_sol.csv | True | 3 | 65 | {3: 7, 4: 11} |
| historical_saved | sol_1/run_30_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_31_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_3_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_4_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_66_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_67_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | sol_1/run_9_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | solution_n18_sol_20251120_052000.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | solution_n18_sol_60m.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | solutions/opt_solution_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | solutions/opt_solution_n19_sol.csv | True | 2 | 69 | {2: 2, 3: 3, 4: 14} |
| historical_saved | solutions/solution_n17_sol.csv | True | 2 | 64 | {2: 1, 3: 2, 4: 14} |
| historical_saved | solutions/solution_n17_sol_tour.csv | True | 2 | 49 | {2: 4, 3: 11, 4: 2} |
| historical_saved | solutions/solution_n18_sol.csv | True | 2 | 66 | {2: 1, 3: 4, 4: 13} |
| historical_saved | solutions/solution_n19_sol.csv | True | 2 | 69 | {2: 2, 3: 3, 4: 14} |
| historical_saved | solutions/v2_n17_sol.csv | True | 2 | 64 | {2: 1, 3: 2, 4: 14} |
| new_sweep | runs/n15_to24_fair_sweep_20261002/n17/solution_n17_sol.csv | True | 3 | 63 | {3: 5, 4: 12} |
| new_sweep | runs/n15_to24_fair_sweep_20261002/n18/solution_n18_sol.csv | True | 3 | 65 | {3: 7, 4: 11} |
| new_sweep | runs/n15_to24_fair_sweep_20261002/n19/solution_n19_sol.csv | True | 3 | 68 | {3: 8, 4: 11} |

Histograms map distinct-slot count to number of players. comparison.csv provides all summary metrics; comparison.json also includes player-level metrics and file checksums. metrics_snapshot.py and solver_snapshot.py preserve the validation code and group-size policy used.
