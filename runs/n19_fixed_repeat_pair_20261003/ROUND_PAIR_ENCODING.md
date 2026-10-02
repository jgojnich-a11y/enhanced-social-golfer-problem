# Implied round-pair constraints

The separate driver retains the original generator and models the same
minimum-spread-3, exact-total-70 or exact-total-69 feasibility questions as
the previous audit. It adds exact count variables for late-slot repetitions
between each of the six pairs of rounds.

For rounds r,s, slot g and player p, a Boolean is the AND of X[r,g,p] and
X[s,g,p]. Three linear inequalities enforce both directions. Summing these
Booleans over the four late slots and all players gives the round-pair count.

Balanced participation means a player attends the sole triple at most once.
For any pair of rounds, their triples are disjoint, leaving 13 players in
fours in both rounds. Unique partners permit at most one player in each
cell of the 4x4 group-intersection array. Only twelve cells have different
slot indices, so at least one player repeats a late slot across the rounds.
All six round-pair counts are therefore at least one.

With minimum spread >=3, each player repeats at most once. A repeated
player uses one slot exactly twice and contributes exactly one to exactly
one round-pair count. The sum of all six counts therefore equals the
number of repeated appearances, 76 minus total spread.

- Total 70: count sum six, so all six counts equal one.
- Total 69: count sum seven, so five counts equal one and one equals two.

These restrictions exclude no qualifying schedule. The identity of the
pair with two repeats is not fixed: six extra-pair Booleans with sum one
permit every possible choice. No new round-order symmetry is imposed.

Before imposing target-specific or minimum-spread constraints, both saved
baselines were forced into the augmented model. All six modeled counts
matched independent set intersections computed from their CSVs. This
checks the definitions even for the old minimum-2 schedule, whose pair-count
sum differs from its number of repeated appearances. The aggregate equality
is added only after the minimum-3 constraint, where it is valid.

Source snapshot, original hint, seed 73, eight workers and 600-second limit
are preserved from the prior audit. The optional generator strengthening
remains off; the added constraints live exclusively in the driver. Models,
full logs, final statistics, baseline checks and checksums are saved per case.
