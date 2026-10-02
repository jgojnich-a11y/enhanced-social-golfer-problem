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
