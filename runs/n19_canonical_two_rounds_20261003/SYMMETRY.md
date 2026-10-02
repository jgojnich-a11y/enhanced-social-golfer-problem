# Complete canonical first-two-round representative

## Why one intersection pattern suffices

For 19 players the groups are [3,4,4,4,4]. Balanced participation permits
each player at most one triple appearance. Thus triples in two rounds are
disjoint, and their four-by-four late-group intersection array contains
13 players. Each cell contains at most one player by partner uniqueness.

Every late row has three or four occupied cells: its four players can send
at most one member to the other round's triple. The same applies to every
column. Therefore the three empty late cells lie in three distinct rows
and three distinct columns.

A fair score-69 schedule has seven different repeat players, so the six
round-pair counts are five ones and one two. Permute rounds to move the
unique pair with two to rounds 1–2. Its four-by-four late array has two
occupied diagonal cells and therefore two empty diagonal cells. The third
empty cell is off-diagonal, in the remaining two indices.

There are twelve possible empty-cell patterns. Under a simultaneous
permutation of rows and columns (a global permutation of the four late tee
slots) they form one orbit, represented by late zero cells (0,0), (1,1),
(2,3). Exhaustive enumeration of the three-cell patterns verifies this.

The triple row and column are then determined: the triple–triple cell is
empty, and each late row/column with three occupied late cells has exactly
one player in the other round's triple. The entire five-by-five occupied
intersection pattern is therefore fixed.

Each occupied cell contains exactly one player. Globally rename that player
to the ID in the corresponding canonical cell. This restores the retained
first-round grouping and makes round 2 exactly the saved canonical grouping.
The two fixed rounds are recorded in canonical_first_two_rounds.csv.

Round permutation, common late-slot permutation, and global player renaming
preserve partner uniqueness, capacities, balanced group-size participation,
minimum distinct-slot spread, and total spread. The triple slot remains
fixed; all permuted slots contain fours. The current model has no identity-
dependent or round-dependent restrictions. It also has no average-start-time
objective that would distinguish these global late-slot permutations.

Consequently every qualifying score-69 schedule has a representative with
these exact first two rounds. This is an exhaustive symmetry reduction,
not a neighborhood restriction around an arbitrary saved schedule.

## Implementation checks

The old valid (2,69) schedule has one round pair with two diagonal repeats.
Its round order was changed to [3,4,1,2]. Global late-slot permutations
produced saved-schedule variants covering all twelve patterns. Every variant
normalized to the same canonical first two rounds, preserved its validity
and spread metrics, and was accepted by both the integer and retained
models when all its assignments were forced before the target restrictions.
These positive controls verify that the canonical partial schedule is
compatible with the underlying scheduling rules.

The normalized old schedule is only a search hint: it fails minimum spread
three. No baseline was treated as a qualifying solution.

## Independent target checks

The integer model proved minimum >=3 and total 69 infeasible with the
canonical first two rounds and implied round-pair constraints in 0.863
seconds. A separate check directly in the retained generator imposed only
minimum >=3, total 69, and the same fixed first two rounds. It imposed no
additional round-pair constraints and independently proved INFEASIBLE in
13.143 seconds. See retained_crosscheck.json, model and log.

Together with the earlier mathematical upper bound 70, the retained-model
proof excluding 70, and the validated (3,68) schedule, these establish the
19-player lexicographic fairness optimum as minimum spread 3, total 68.
The result is scoped to the existing four-round rules and spread definition;
it does not claim an optimum for the separate total-only objective.
