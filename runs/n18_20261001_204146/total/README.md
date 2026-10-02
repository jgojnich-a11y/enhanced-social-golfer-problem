# 18-player run: total

Started: 2026-10-01T20:41:47.045182+10:00

Finished: 2026-10-01T21:41:47.765879+10:00

Limit: 3600 seconds; seed: 1; workers: 8; four rounds; no warm start.

Status: FEASIBLE.

| Metric | Value |
|---|---:|
| validation | passed |
| repeated_player_pairs | 0 |
| minimum_distinct_slots | 2 |
| total_spread | 66 |
| average_distinct_slots | 3.6666666666666665 |
| average_hogging | 0.3333333333333333 |
| worst_hogging | 0.75 |
| max_same_slot_appearances | 3 |

See `run.json` for the exact command, `solver.log` for solver statistics, and `players.csv` for individual metrics. FEASIBLE does not establish optimality. The fair objective is 73 × minimum distinct slots + total spread. Runs execute concurrently, each with eight workers; timing is not an isolated hardware benchmark. Average start-time offsets assume seven-minute intervals from 10:00.
