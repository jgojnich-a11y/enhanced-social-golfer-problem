# Equivalence of the experimental integer tee-slot model

The retained generator is unchanged. The experimental source is saved as
model_snapshot.py, with the retained source saved as solver_snapshot.py.

## Constraint correspondence

Each player has one integer slot per round, with domain covering every group
index. Membership Booleans are equivalent to equality between that integer
and a slot index, so exactly one membership is true per player and round.
Each group's membership sum equals its original three/four-player capacity.

For each unordered player pair and round, one Boolean is equivalent to
equality of their integer slots. At most one such Boolean may be true across
the four rounds. In the retained formulation, each pair has a separate AND
indicator per round and group; since a player occupies exactly one group,
at most one such indicator can be true within a round. The new equality
Boolean is exactly their per-round meeting indicator. Both encodings impose
the same no-repeat-partner rule in both directions.

Each player's four-group count lies between lower and lower+1, where lower
is floor(total four-group appearances / player count). Exact capacities fix
the aggregate count, and therefore the number receiving lower+1 is exactly
the original remainder. This reproduces the retained balance constraints.

Group sizes use the identical three-before-four rule, and the identical
canonical first-round assignments are fixed. Used-slot Booleans are exact
ORs of memberships across rounds; individual distinct-slot counts are their
sums. The optional minimum-first weighted objective matches the retained
objective, including its priority weight.

Thus every schedule feasible in either representation is feasible in the
other, with the same spread metrics. This follows from the correspondence;
the checks below are implementation cross-checks, not exhaustive enumeration
of all schedules for all player counts.

## Round-pair counts and target

The experimental model counts same-slot players between rounds using one
reified integer equality per player and round pair, rather than a separate
AND for each slot. For 19 players, balanced participation prevents any
player from attending the triple twice, so these equalities all represent
late-slot repeats. The fixed exceptional-pair argument in SYMMETRY.md applies
unchanged: count 2 for rounds 1–2, count 1 for the other five pairs.

The test requires minimum spread >=3 and total exactly 69, with no
optimization objective. Seed 73, eight workers, probing level 1 and a
600-second limit match the prior test. The same normalized hint is encoded
as 76 integer slot values instead of 380 membership values; the hint fixes
no assignments and remains a nonqualifying score-68 search guide.

## Validation

- Eight and twelve players: both encodings proved the base rules infeasible.
- Nine players: both proved the same minimum-first weighted objective 77;
  each solver's schedule was forced successfully into the other model.
- A nine-player mutation repeats round 1 as round 2, preserving group sizes
  and coverage while breaking partner uniqueness. Both models rejected it.
- Saved fairness schedules for 17, 18 and 19 players, and the old total-69
  schedule for 19 players: all passed independent schedule validation and
  forced-model checks in both encodings. Integer-model distinct counts and
  all six round-pair counts matched direct calculations from the CSVs.
- Any qualifying integer-model solution is independently checked and then
  forced into the retained fairness model before being reported as an
  improvement. The required weighted score there is 77x3+69=300.

For 19 players, pair-meeting indicators fall from 3,420 to 684. This does not
guarantee improved runtime; it changes how the solver sees the same problem.
Models, sources, dependencies, validation results, checksums, hint, logs and
final reports are preserved in this audit directory.
