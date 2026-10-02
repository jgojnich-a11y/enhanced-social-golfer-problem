# Counting bound and proposed tests for 19 players

Current rules: four rounds, group sizes [3,4,4,4,4], unique partners,
balanced four-player participation, and the three-player group at 10:00.
The remaining four tee times run from 10:07 to 10:28.

## Participation and early-slot repetition

There are 64 four-player appearances. Balance assigns 12 players to fours
three times and seven players four times. Thus each player attends the
three-player group at most once: nobody repeats the 10:00 slot. All repeated
tee-slot appearances occur in four-player groups.

## At least one repeat for every pair of rounds

Take any two rounds. Their three-player groups are disjoint, because no
player attends a triple twice. Six players therefore attend a triple in
one of those rounds, leaving 19-6=13 players in fours in both rounds.

The intersections of the four fours in one round with the four fours in
the other form a 4x4 array. Every cell contains at most one player: two
players would have met in both rounds. Thirteen cells must be occupied.
Only twelve cells lie off the diagonal, so at least one diagonal cell is
occupied. That player repeats a four-player tee slot across these rounds.

There are six pairs of rounds. Under minimum distinct-slot spread >=3,
each player has at most one repeated appearance across four rounds, so
each repeated player contributes exactly one diagonal intersection for
exactly one round pair. The six round pairs therefore require at least six
different repeat players.

Total spread is consequently at most 4x19-6=70 for schedules with minimum
spread >=3. Minimum spread 4 is impossible, independently of the solver.
The validated minimum-3 baseline therefore attains the best possible minimum.

This bound is stronger than the short sweep's fair weighted objective
bound 303 (weight 77), which allowed total 72 at minimum 3. The counting
bound reduces the remaining total question to 68, 69 or 70.

## Proposed exhaustive improvement tests

Run the retained generator through separate audited drivers, with minimum
spread >=3 and its optimization objective cleared:

- First test total exactly 70: six different late-repeat players, with
  exactly one repeated player for each of the six round pairs.
- Then test total exactly 69: seven different late-repeat players, with
  five round pairs having one repeat each and the remaining pair having two.

Suggested limits: 600 seconds each, sequentially, seed 73, eight workers,
saved fairness-68 assignment hint and original constraints. No structural
case is omitted: the proof excludes all totals above 70, and these tests
cover all improvements over 68.

If 70 is feasible, it is optimal. If 70 is infeasible and 69 is feasible,
69 is optimal. If both are infeasible, the validated (3,68) is optimal.
UNKNOWN leaves the corresponding case unresolved. Finding 69 while 70
remains unresolved improves the baseline but does not prove optimality.

## Checks against saved schedules

The new sweep's (3,68) schedule has no early repeats and eight late repeated
appearances. The old main (2,69) schedule has no early repeats and seven
late repeated appearances, but its repeat-pair incidence count is nine
because two players each appear three times in one slot. The minimum-3 requirement
prevents that concentration.

All six 4x4 intersection arrays were calculated for both saved schedules:
each has exactly 13 occupied cells, at most one player per cell, and at
least one occupied diagonal cell, as the argument requires.

This is a direct counting derivation for the current model. The tests are
proposed, not launched. Existing schedules and the retained generator are
unchanged; no run is pending from this analysis.

## Authorized target-test results

The user subsequently authorized both exact-total tests. They ran sequentially
with 600-second limits, seed 73, eight workers, original constraints, and
the validated fairness-68 schedule supplying all 380 assignment hints.
Both known baselines passed forced-model validation with the added repeat
definitions before target constraints; repeat sums matched direct CSV counts.

Both total 70 and total 69 ended **UNKNOWN**. Neither a qualifying schedule
nor an infeasibility proof was found. The optimal minimum remains 3, and
the best total at that minimum remains bounded between 68 and 70. The
counting upper bound 70 still applies; timeouts do not exclude either target.

Audit: `runs/n19_targets70_69_20261002/`. Both processes exited successfully,
and final metadata agrees with solver statistics. Original schedules and
the retained generator are unchanged. No run is pending.

## Explicit round-pair strengthening result

The user authorized repeating both targets with the implied round-pair
repeat patterns encoded explicitly. Total 70 requires all six pair counts
to equal one. Total 69 requires five ones and one two; every choice of the
exceptional pair remains available. A separate driver implements exact
AND indicators and these counts without modifying the retained generator.

Both saved baselines passed forced-model checks before target restrictions,
and all six modeled counts matched independently computed CSV intersections.
The same hint, seed 73, eight workers and 600-second limits were retained.

- Total 70: **INFEASIBLE** in 131.595 seconds.
- Total 69: **UNKNOWN** after 600.031 seconds.

Combined with the mathematical upper bound, this rules out every fair
total >=70. The optimal minimum remains 3, and the total at that minimum
is now bounded between **68 and 69**. The validated (3,68) baseline remains
the best known. Its optimality is not yet proved.

Audit: `runs/n19_round_pair_targets_20261002/`, including the encoding
justification. Both processes exited successfully and final statistics
agree with metadata. The hint and solver source match the earlier tests;
existing schedules are unchanged. No run is pending.

## Fixed exceptional round-pair symmetry test — 3 October 2026

The user authorized a score-69 test with two repeats fixed to rounds 1–2
and one to every other round pair. Round permutation followed by global
player renaming restores the retained model's canonical first-round
partition, so every candidate is represented under current rules.

All six pair choices in both saved baselines were normalized; all twelve
versions preserved validity and scores and passed forced-model checks before
target restrictions. Modeled round-pair counts matched CSV intersections.
The normalized fairness-68 hint supplied all 380 assignment hints.

The 600-second test ended **UNKNOWN**, with seed 73 and eight workers.
No qualifying schedule or infeasibility proof was found. The best known
fairness result remains (3,68), and the sole unresolved improvement remains
(3,69). The earlier total-70 infeasibility proof is unaffected.

Audit: `runs/n19_fixed_repeat_pair_20261003/`. Source, model, normalization
checks, dependencies, hint, log and final statistics are preserved. Existing
solver and schedules are unchanged; no run is pending.

## Integer tee-slot formulation — 3 October 2026

The user authorized the planned alternative formulation. The new
`cp_sat_caseB_integer_experimental.py` gives each player an integer tee slot
per round and reifies player-pair meetings as equality of slots. It reduces
19-player pair-meeting indicators from 3,420 to 684 while preserving the
retained generator's capacities, unique partners, participation balance,
first-round grouping and spread definitions. The retained generator is
unchanged. A constraint-by-constraint equivalence argument is saved with
the experiment.

Cross-checks matched status/objective results for 8, 9 and 12 players;
both encodings accepted each other's nine-player schedules and rejected a
repeated-partner mutation. Saved 17-, 18- and 19-player schedules passed
forced-model checks in both encodings, and the new spread and round-pair
counts matched direct schedule calculations.

The same normalized score-68 hint supplied 76 integer slot values. The
score-69 test retained minimum spread >=3, the fixed exceptional round pair,
seed 73, eight workers and a 600-second limit. It ended **UNKNOWN** after
600.248 seconds. No qualifying schedule or infeasibility proof was found.
The fair optimum remains either (3,68) or (3,69).

Audit: `runs/n19_integer_target69_20261003/`, including source snapshots,
equivalence explanation, validation results, model, log and final statistics.
Final metadata agrees with statistics and source checksums; existing
schedules are unchanged. No run is pending.

## Final canonical two-round proof — 3 October 2026

The authorized next experiment fixed canonical first and second rounds,
covering the single orbit of all twelve admissible exceptional-pair
intersection patterns. See `runs/n19_canonical_two_rounds_20261003/SYMMETRY.md`
for the full counting and normalization argument. Saved schedule variants
covering all twelve patterns passed independent validation and forced-model
checks in both encodings before the target restrictions.

The integer model proved fair total 69 **INFEASIBLE in 0.863 seconds**.
A direct check in the retained generator, fixing the same two rounds but
adding no round-pair strengthening, independently proved **INFEASIBLE in
13.143 seconds**. Both processes exited successfully.

Together with the earlier total-70 exclusion and counting bound, this
establishes the 19-player lexicographic fairness optimum as **minimum
individual spread 3, total spread 68**. The existing sweep schedule attains
that score. This is not an optimality claim for the separate total-only
objective or for average tee-time fairness.

Audit: `runs/n19_canonical_two_rounds_20261003/`. Canonical rounds, complete
orbit enumeration, normalization checks, source snapshots, original and
normalized hints, models, logs and both final reports are preserved. The
retained generator and original schedules are unchanged. No run is pending.
