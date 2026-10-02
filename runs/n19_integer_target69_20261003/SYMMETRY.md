# Fixing the exceptional round pair

A fair score-69 schedule has seven different repeat players. The six
round-pair late-repeat counts are at least one, so their pattern is five
ones and one two. This experiment fixes the unique pair with two to rounds
1 and 2, and fixes all other counts to one.

This loses no qualifying schedule under current identity-independent rules:

1. Permute the candidate's four rounds to move the exceptional pair to 1–2.
2. For each tee slot in the new first round, globally rename its players
   to the canonical player IDs used by the retained generator for that slot.
3. Apply this bijection to every round.

Round permutation and global player renaming preserve capacities, coverage,
unique partners, balanced four-player participation, all tee-slot repeat
counts, minimum spread, and total spread. No tee-slot permutation is used.
The renaming restores the fixed first-round grouping without adding any
identity-based restriction. Each candidate therefore has a representative
in this symmetry-restricted model.

For each of the two saved baseline schedules, all six possible pairs were
moved to rounds 1–2. All twelve normalized schedules passed independent
schedule validation, and every normalized schedule was forced into the
model before target-specific constraints. All checks were feasible, modeled
round-pair counts matched direct CSV intersections, and all summary metrics
were preserved. These checks support the implementation; the argument above
establishes symmetry coverage.

The saved fairness-68 hint was normalized by moving its most repeated pair
to rounds 1–2, then applying the canonical player renaming. This remains a
nonqualifying hint, not a score-69 solution. All 380 assignment hints were
supplied. Original saved schedules and the retained generator are unchanged.

The separate search requires minimum spread >=3 and total exactly 69, with
seed 73, eight workers and a 600-second limit. The prior total-70 infeasibility
report is copied here; the counting bound excludes higher totals. Thus a
qualifying result proves (3,69) optimal, while INFEASIBLE proves the validated
(3,68) baseline optimal. UNKNOWN leaves the remaining one-point gap open.
