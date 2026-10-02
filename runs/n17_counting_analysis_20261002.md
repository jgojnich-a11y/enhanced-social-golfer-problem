# A counting obstruction for 17 players

This derivation concerns four rounds, group sizes [3,3,3,4,4], fixed group-size
tee ordering, no repeated partners, and balanced four-player participation.
It is a direct argument for the current model, not an optimality certificate
for total spread 63.

## Four-player participation

There are 4 x 2 x 4 = 32 four-player appearances. Balance requires one or two
per player, so 15 players have two and two players have one. A player meets
8 plus their four-player count distinct partners across four rounds: 9 or 10,
well below the 16 available. The basic partner-budget bound does not prevent
these schedules.

## At least three late-slot repetitions

Call the two four-player tee slots A (10:21) and B (10:28). A player who uses
both slots belongs to one A group in round i and one B group in round j,
where i != j. There are 4 x 3 = 12 such ordered round pairs.

Each A_i / B_j group intersection contains at most one player: two players
in the intersection would play together in both rounds, violating unique
partners. Therefore at most 12 players can use both late slots.

Of the 15 players who attend two four-player groups, at least three must
therefore attend the same late slot twice. These are three distinct players
and at least three repeated tee-slot appearances. Across 68 player-round
appearances, total distinct-slot spread is consequently at most 65. In
particular, minimum spread 4 is mathematically impossible.

This argument is independent of solver search. It does not by itself exclude
total spread 65 or distinguish fair totals 63 and 64. The saved 60-second
fairness solver run supplies the tighter upper bound 64 at minimum spread 3:
its weighted objective bound is 271, with weight 69, so total <=271-69x3=64.
The schedule found has minimum 3 and total 63.

## Two possible repeat patterns for fair total 64

With minimum spread at least 3, each player either uses four distinct slots
or has exactly one repeated appearance and three distinct slots. Total 64
therefore requires four different players with one repeat each. Since at
least three repeats occur in late slots, only these patterns are possible:

1. Three late-repeat players and one early-repeat player, with disjoint sets.
2. Four late-repeat players and no early-repeat players.

The occupancy counts sharpen these cases. Let a and b be the numbers of
players repeating A and B, d=a+b, and let s_A and s_B be the numbers of
one-four-player-round players assigned to each slot. There are 15-d players
using both slots. Slot capacities give

16 = (15-d) + 2a + s_A, and 16 = (15-d) + 2b + s_B.

For d=3, a,b are 1,2 in either order and both one-four-player-round players
use the slot with only one repeat player. For d=4, a=b=2 and the two
one-four-player-round players split between the late slots.

These are necessary conditions, not constructions or proofs of feasibility.
They suggest splitting the target-minimum-3/total-64 test into two structural
cases, rather than searching without that explanation.

## Saved schedule comparison

- New n17 sweep: three late repeats plus two early repeats, total 63.
  The late- and early-repeat player sets are disjoint, so minimum spread is 3.
- Main old n17 schedule: three late repeats plus one early repeat, total 64.
  The early-repeat player also repeats a late slot, leaving that player with
  only two distinct slots. Its total is higher, but its minimum is lower.

Thus the old score-64 schedule already has the minimum possible late-repeat
count. The remaining fairness question is whether its four repeats can be
distributed over four different players while maintaining all partner rules,
or whether the four-late-repeat alternative can work.

No new solver search was launched for this analysis. Existing schedules and
the retained generator are unchanged.

## Subsequent case-test result

Following user authorization, both necessary score-64 cases were tested with
600-second limits, seed 73 and eight workers. Case A (three late repeats,
one early repeat) proved INFEASIBLE in 102.602 seconds. Case B (four late
repeats, no early repeats) proved INFEASIBLE in 61.466 seconds.

The validated (3,63) baseline, the earlier global fair upper bound 64, and
these exhaustive case proofs establish lexicographic fairness optimum
**minimum spread 3, total spread 63** under the retained four-round rules.
This does not establish an optimum for the separate total-only objective.

Audit: `runs/n17_target64_repeat_cases_20261002/`. Both baseline schedules
passed forced-model validation with the added repeat definitions, and model
repeat counts matched direct CSV calculations. Both processes exited
successfully; final statistics agree with metadata. No run is pending.
