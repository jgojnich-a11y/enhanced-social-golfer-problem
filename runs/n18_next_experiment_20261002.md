# Review and next experiment — 2 October 2026

Working directory verified as `/Users/jgojnich/Projects/enhanced-social-golfer-problem`.
Existing uncommitted files and the retained solver were preserved.

## Evidence reviewed

- The historical spread-66 schedule is valid, but has no surviving optimality proof.
- The six-hour total run reports FEASIBLE, objective 66, upper bound 67.
- The six-hour fairness run reports FEASIBLE, minimum spread 3, total 65,
  weighted objective 284, upper bound 286 (weight 73).
- Both latest 1,000-second target-total-67 tests report UNKNOWN, with
  repaired hints and with logically implied strengthening respectively.
- Bounds and conclusions apply to the current three-player-before-four-player
  tee-time policy. Spread counts distinct slots, not average start time.

## Exhaustive equal-size group reordering check

Enumerated slot relabelings independently in each round, allowing permutations
of the two three-player slots and the three four-player slots. There are 12
permutations per round. Fixing round 1 to identity leaves 1,728 combinations:
any unrestricted combination can be composed with the inverse of its round-1
permutation, preserving slot equality and therefore every player's spread.
Partners and three/four-player participation are unchanged.

| Saved grouping source | Best total | Best minimum, then total |
|---|---:|---:|
| `solutions/opt_solution_n18_sol.csv` | 66 | (3, 63) |
| `runs/n18_20261001_214154/total/solution_n18_sol.csv` | 66 | (3, 63) |
| `runs/n18_20261001_214154/fair/solution_n18_sol.csv` | 65 | (3, 65) |

This exhausts reordering for these saved groupings only. Improving the saved
fairness result requires changing at least some partner groups.

## Proposed next solver experiment

Test feasibility of **minimum individual spread >= 3 and total spread >= 66**.
This asks whether the known total score can coexist with the known minimum
fairness, rather than repeating the broader total-67 test.

Use the retained generator through a separate audited driver, with its
spread variables, objective cleared, and the two explicit thresholds added.
Start from the six-hour fairness schedule, seed 73, eight workers, repaired
hints, original constraints, and a 1,000-second limit. Preserve source,
model, dependency versions, hint checksum, full log and final status as in
the latest audit. Run in isolation to avoid concurrent CPU contention.

- FEASIBLE/OPTIMAL: validate and save a fairness improvement to at least (3, 66).
- INFEASIBLE: proves (3, 65) is the best total among schedules with minimum
  spread at least 3 under the current rules. Combined with the recorded
  fairness upper bound 286, which excludes minimum spread 4, this would
  establish the lexicographic fairness optimum.
- UNKNOWN: leaves that narrower question unresolved; do not describe it as
  evidence of infeasibility.

## Experiment result

Following user approval, the experiment ran on 2 October 2026 from 15:39:34
to 15:56:14 Australia/Sydney, with all settings above and all 360 assignment
hints loaded. The model passed validation. It ended **UNKNOWN** after
1,000.219 seconds; no qualifying schedule or infeasibility proof was found.
The question remains unresolved, and the saved fairness baseline remains
minimum spread 3, total 65.

Artifacts: `runs/n18_fair_target66_20261002/`. The retained solver and saved
schedules were unchanged. No run remains pending from this experiment.

## Repeat with a 1,200-second limit

At the user's request, a fresh search ran from 15:59:23 to 16:19:23
Australia/Sydney on 2 October 2026. Only the runtime changed to 1,200 seconds.
The model, hint, solver source, validator and dependency list were confirmed
byte-identical to the 1,000-second test; the comparison is saved alongside
the new run. Seed 73, eight workers, original constraints and the complete
360-assignment fairness hint were retained.

The repeat ended **UNKNOWN** after 1,200.026 seconds. No qualifying schedule
or infeasibility proof was found. Minimum spread 3 with total spread 66
remains unresolved; the best saved fairness result remains (3, 65).

Artifacts: `runs/n18_fair_target66_1200s_20261002/`. The process exited
successfully and no run from this experiment is pending. Existing files
and uncommitted changes were preserved.

## Two-round neighborhood experiment

Following user approval, three sequential tests each received a 400-second
limit, seed 73 and eight workers, starting from the saved (3,65) schedule.
Round 1 and one later round were fixed; the other two rounds could change
freely. Each test required minimum spread >=3 and total spread >=66.

| Free rounds | Fixed rounds | Status | Search seconds |
|---|---|---|---:|
| 2, 3 | 1, 4 | INFEASIBLE | 6.684 |
| 2, 4 | 1, 3 | INFEASIBLE | 4.329 |
| 3, 4 | 1, 2 | INFEASIBLE | 0.143 |

The baseline passed independent schedule validation and a forced-assignment
feasibility check in each driver before the improvement thresholds were
added. All restricted models validated; all processes exited successfully.

These are proofs for the three restricted neighborhoods only. With round 1
held fixed, any qualifying improvement must differ from this particular
baseline in each of rounds 2, 3 and 4. This does not prove global optimality.
The solver stopped early on proofs; the full 1,200-second budget was unnecessary.

Artifacts: `runs/n18_fair_neighborhoods_20261002/`. No new schedule was found;
the best saved fairness result remains (3,65). Retained solver and existing
schedules are unchanged. No run is pending.

## Unrestricted search with round-order symmetry

After the restricted neighborhoods were ruled out, the user authorized the
next proposed experiment. All three later rounds were free, with exact
lexicographic ordering of their player tee-slot vectors to eliminate equivalent
round permutations. The hint was normalized accordingly. The ordering
preserves all feasible schedules up to round permutation under current rules;
its justification is recorded in the run's `SYMMETRY.md`.

Before imposing minimum spread >=3 and total >=66, both normalized saved
baselines passed schedule validation and forced-model feasibility checks.
The fairness hint supplied all 360 assignment variables. Seed 73, eight
workers and a 1,200-second limit were retained, with original constraints
and no optimization objective. The retained generator was unchanged.

The run started at 16:27:48 and finished at 16:47:48 Australia/Sydney on
2 October 2026. It ended **UNKNOWN** after 1,200.028 seconds: no qualifying
schedule or infeasibility proof. The global fairness threshold remains
unresolved and the best saved fairness score remains (3,65).

Artifacts: `runs/n18_fair_round_order_1200s_20261002/`. Final metadata and
solver statistics agree; the process exited successfully. No run is pending.
