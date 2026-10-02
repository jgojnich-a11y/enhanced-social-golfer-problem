# Fairness sweep: 15–24 players

Four rounds; 60 seconds per player count; seed 73; eight workers; no hints; sequential searches.

Original three-player-before-four-player tee-time policy, unique partners and balanced four-player participation. Fairness maximizes minimum distinct tee slots, then total spread. Tee times start at 10:00 every seven minutes.

| Players | Group sizes | Last tee time | Status | Minimum slots | Total spread | Seconds |
|---:|---|---|---|---:|---:|---:|
| 15 | [3, 4, 4, 4] | 10:21 | OPTIMAL | 2 | 45 | 0.15 |
| 16 | [4, 4, 4, 4] | 10:21 | OPTIMAL | 2 | 44 | 49.56 |
| 17 | [3, 3, 3, 4, 4] | 10:28 | FEASIBLE | 3 | 63 | 60.02 |
| 18 | [3, 3, 4, 4, 4] | 10:28 | FEASIBLE | 3 | 65 | 60.06 |
| 19 | [3, 4, 4, 4, 4] | 10:28 | FEASIBLE | 3 | 68 | 60.02 |
| 20 | [4, 4, 4, 4, 4] | 10:28 | OPTIMAL | 4 | 80 | 0.21 |
| 21 | [3, 3, 3, 4, 4, 4] | 10:35 | OPTIMAL | 4 | 84 | 0.29 |
| 22 | [3, 3, 4, 4, 4, 4] | 10:35 | OPTIMAL | 4 | 88 | 0.28 |
| 23 | [3, 4, 4, 4, 4, 4] | 10:35 | OPTIMAL | 4 | 92 | 0.40 |
| 24 | [4, 4, 4, 4, 4, 4] | 10:35 | OPTIMAL | 4 | 96 | 1.37 |

FEASIBLE is a validated candidate, OPTIMAL proves lexicographic fairness optimality, INFEASIBLE proves no valid schedule under these rules, and UNKNOWN is unresolved. Raw total spread is not directly comparable across player counts; group counts also change. These short no-hint runs do not supersede better saved schedules.

Each n-folder contains model, log, statistics, metadata and any solution and individual player metrics. Shared source snapshot, dependency list and environment metadata are in this folder. The retained generator and historical schedules are unchanged.

## Interpretation

All ten player counts produced valid schedules. Fifteen players are feasible
under the retained model's [3,4,4,4] policy; however, guaranteeing every player
three distinct tee times is impossible under these rules. The same minimum
spread limitation holds for 16 players. The reported total scores 45 and 44
are optimal subject to the minimum-first objective, not claims about the
unrestricted total-spread objective.

For 17–19 players, the saved objective upper bounds already exclude minimum
spread 4. Together with the validated minimum-3 schedules, this establishes
the optimal minimum spread as 3, although the best total at that minimum
remains unproven. In this sweep the totals and upper bounds are respectively
63–64, 65–70 and 68–72. Prior longer n18 fairness runs have a tighter bound;
these short runs do not weaken or replace that earlier evidence.

For 20–24 players, each player receives four different tee times across four
rounds, the absolute maximum. Consequently total spread is 4 times the player
count and lexicographic spread fairness is proven optimal. Average tee time
and the distribution of early versus late starts are separate fairness
measures not optimized here.

All searches ran sequentially and completed successfully; no run is pending.

## Subsequent 17-player optimality proof

The later exhaustive repeat-case tests in
`runs/n17_target64_repeat_cases_20261002/` both proved INFEASIBLE. Together
with this sweep's upper bound and validated schedule, they establish (3,63)
as the 17-player lexicographic fairness optimum under the current rules.
The table above retains the original sweep's solver statuses; 18- and
19-player total spread at minimum 3 remain unproven.

## Subsequent 19-player counting bound and tests

A direct counting argument tightened the fair-total upper bound for 19
players from 72 to 70; see `runs/n19_counting_analysis_20261002.md`.
The authorized 600-second exact-total tests for 70 and 69 both ended UNKNOWN
in `runs/n19_targets70_69_20261002/`. The best validated fairness score remains
(3,68), with optimum total between 68 and 70. No run is pending.

The subsequent explicit round-pair tests in
`runs/n19_round_pair_targets_20261002/` proved total 70 infeasible in
131.595 seconds; total 69 remained UNKNOWN at 600 seconds. This tightens
the current 19-player fair optimum interval to **(3,68)–(3,69)**. The original
sweep statuses and earlier run reports above retain their historical results.

## Final 19-player optimality proof

On 3 October 2026, complete canonical first-two-round symmetry reduction
proved score 69 infeasible, independently in the integer and retained
formulations. Combined with the earlier exclusion of 70, this establishes
**(3,68) as the 19-player lexicographic fairness optimum**. See
`runs/n19_canonical_two_rounds_20261003/` and `runs/FAIRNESS_STATUS.md`.
At that stage only the 18-player fairness total remained unresolved.

## 18-player proof — 3 October 2026

The subsequent complete two-class canonical search proved fairness (3,65)
optimal. Both relevant intersection classes are INFEASIBLE at minimum >=3,
total >=66, independently confirmed by the retained solver. See
`../n18_canonical_two_rounds_20261003/` and `../FAIRNESS_STATUS.md`.
Every fairness score for 15–24 players is now proven; the table above preserves
the original sweep statuses.
