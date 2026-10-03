# Proposed benchmark: 18-player mixed-size tee-time rotation

This is a proposed standalone benchmark case, not a result from a run of
GroupMixer. The reference optimum is established under the rules below.

## Case definition

18 players attend all four rounds. Each round contains five ordered groups,
with exact sizes **3, 3, 4, 4, 4**. Groups correspond to tee times **10:00,
10:07, 10:14, 10:21 and 10:28**. The size pattern is identical in every round;
sizes vary between groups, not between rounds. Player order within a group
does not matter.

Hard constraints:

- Every player belongs to exactly one group in every round.
- Every unordered player pair shares a group at most once across all rounds.
- Four-player-group participation is balanced: every player has two or three
  four-player rounds. Capacities force six players to have two and twelve
  to have three; these identities are not prescribed.
- The two three-player groups occupy the first two tee slots in every round.

There are no player-specific restrictions, absences or prescribed partners.
A fixed first-round player partition is a permissible symmetry reduction,
not an additional restriction on the underlying case.

## Objective and reference result

Let d(p) be the number of distinct tee slots occupied by player p over the
four rounds. Maximize, lexicographically:

1. min d(p);
2. sum d(p).

Equivalently maximize **73 × min d(p) + sum d(p)**. Total spread is at most
72, so the weight gives exact minimum-first priority.

The proven optimum is **minimum 3, total 65**, weighted score **284**.
Every attaining schedule gives seven players three distinct tee slots and
eleven players four. Average start time is not part of this objective.

The historical total-only reference has minimum 2 and total 66. It ranks
below (3,65) under this objective; optimality of total 66 without the
minimum-first objective is not established here.

Unique contacts are fixed by the hard constraints: each round supplies
2 × C(3,2) + 3 × C(4,2) = 24 pairs, giving **96 distinct met pairs** over
four rounds and **57 unmet pairs** out of C(18,2) = 153. Six players meet
ten partners and twelve meet eleven. Minimizing unmet pairs would therefore
be constant on every feasible schedule for this case.

## Evidence

- [Validated reference schedule](solutions/optimal/n18_sol.csv).
- [Readable tee times](solutions/optimal/n18_tee_times.md).
- [Collection validation record](solutions/optimal/validation.json).
- [Complete optimality argument and audit](runs/n18_canonical_two_rounds_20261003/README.md).
- [Symmetry coverage argument](runs/n18_canonical_two_rounds_20261003/SYMMETRY.md).

Any improvement must have a round pair sharing a tee slot for at most one
player. Exhaustive binary intersection-matrix enumeration gives exactly two
relevant symmetry classes. Both canonical cases, with minimum at least 3
and total at least 66, were proved INFEASIBLE by the experimental integer
formulation and independently confirmed by the retained Boolean formulation.
This also excludes minimum four. The saved feasible reference attains (3,65).

## Relationship to GroupMixer benchmarks

GroupMixer already publishes constrained cases, including a sailing case
with changing capacities and other mixed constraints. Mixed sizes alone
are therefore not a claim of novelty. This case's distinguishing combination
is exact (3,3,4,4,4) capacities, hard partner uniqueness, balanced size
participation, ordered tee slots, and a proven lexicographic slot objective.

Sources reviewed on 3 October 2026:

- [Benchmark cases](https://groupmixer.app/benchmarks).
- [Sailing case](https://groupmixer.app/benchmarks/cases/sailing-trip-57p-5g-5s).
- [Methodology](https://groupmixer.app/benchmarks/methodology).

Their methodology permits case-specific formulations and symmetry, but
forbids embedded answers and cross-solver hints; it counts input preparation,
solving and output conversion within the deadline. A compliant timing
comparison would need fresh no-hint runs under their declared resource limits
and seeds, independently validated schedules, and the exact objective above.
Our existing proof timings are audit evidence, not a compliant cold-start
performance comparison: the integer searches used saved-schedule hints and
our setup did not implement their complete execution contract.

## Fresh local benchmark results

Five fresh no-hint runs were completed on 3 October 2026 using the retained
solver with `--fair_spread` and `--strengthen`, eight workers, and seeds
101, 202, 303, 404 and 505. All five returned independently valid schedules
attaining (3,65), weighted score 284. Median weighted score: 284.

| Seed | Minimum / total | Solver status | End-to-end seconds |
|---:|---|---|---:|
| 101 | 3 / 65 | FEASIBLE | 59.090 |
| 202 | 3 / 65 | FEASIBLE | 59.115 |
| 303 | 3 / 65 | FEASIBLE | 59.087 |
| 404 | 3 / 65 | FEASIBLE | 59.112 |
| 505 | 3 / 65 | FEASIBLE | 59.135 |

Each process had an external 60-second deadline. Import and preparation time
were included; search was budgeted to leave approximately one second for
conversion and validation. No saved schedule, hint or reference-score
constraint entered search. All five solver upper bounds remained 288: these
attempts found the known optimum but did not prove it within the deadline.
Optimality comes from the separate exhaustive proof linked above.

[Local benchmark evidence](runs/n18_cold_benchmark_20261003/README.md)
includes source snapshots, configuration, platform, hashes, full logs and
validated output schedules. Measurements used macOS ARM64 and OR-Tools
9.14.6206, without CPU pinning or an 8 GiB process limit; they are not a
direct performance comparison with GroupMixer's published results.

The reference CSV may be supplied as evaluation evidence, but must not be
embedded in a benchmarked solver. No GroupMixer execution or submission has
been performed as part of drafting this description.
