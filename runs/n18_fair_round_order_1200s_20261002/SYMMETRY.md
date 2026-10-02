# Round-order symmetry justification

The experiment retains all original constraints and the fixed first-round
partition. For each later round, its vector is the tee-slot index of players
1 through 18. The driver requires round 2's vector <= round 3's vector <=
round 4's vector in lexicographic order.

Pair uniqueness, per-round capacities, balanced group-size participation,
minimum distinct slots and total distinct slots are invariant under a
permutation of rounds 2-4. Every feasible schedule therefore has a sorted
representative with the same scores. No improvement is excluded by this
ordering under the current rules. Round-dependent availability or other
future rules would require reviewing this justification.

Prefix booleans record whether all preceding vector entries agree. When
that prefix is true, the next left entry must be <= the right entry; the
prefix continues only when the entries are equal. This implements exact
lexicographic nondecreasing order without large integer coefficients.

The fairness hint was normalized to original round order [1,3,4,2], and all
360 assignment hints were supplied. Before adding improvement thresholds,
both the normalized fairness baseline (3,65) and historical baseline (2,66)
passed independent schedule validation and forced-assignment feasibility
checks in the ordered model. These checks validate compatibility, not global
optimality. The original CSVs and retained generator are unchanged.

The test then requires minimum individual spread >=3 and total spread >=66,
with no optimization objective, seed 73, eight workers and a 1,200-second
search limit. INFEASIBLE would rule out that threshold globally under the
current rules, unlike the earlier restricted-neighborhood tests. UNKNOWN
leaves the question unresolved. A qualifying schedule would be independently
validated and saved by the driver.
