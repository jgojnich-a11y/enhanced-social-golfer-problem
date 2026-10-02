# 18-player model audit and target-67 comparison

Both tests use a 1,000-second limit, seed 73, eight workers, the same
historical spread-66 schedule, and the repaired 360-assignment hint.
`repaired_hints/` uses the original constraints. `strengthened/` adds
first-round clique exclusions and redundant unique-partner count equalities.
Runs execute concurrently; compare status and search statistics rather than
interpreting times as isolated performance benchmarks.

## Audit findings

- Complete rows now specify every player assignment; partial rows supply
  listed memberships without treating unlisted players as absent.
- Malformed, duplicate, out-of-range, over-capacity or missing hints fail
  explicitly, rather than silently changing the experiment.
- First-round co-members cannot share a later group: direct AtMostOne
  clique constraints are implied by existing pair uniqueness.
- For each player, the count of unique partners equals the sum of
  group-size-minus-one over rounds, because partners cannot repeat.
- Fixed first-round grouping loses no schedule quality under the current
  rules: globally renaming players maps any first-round partition to it,
  preserving pair uniqueness and both spread objectives. This justification
  requires that players have no individual identity-dependent constraints.
- Placing three-player groups before four-player groups is an actual
  tee-time/group-size policy: conclusions remain scoped to that policy.
- Later rounds can be permuted, but no new symmetry-breaking constraints
  were added; introducing them would require further hint normalization.

## Validation

Forced complete assignments verified that the 16-player spread-52 schedule,
the historical 18-player spread-66 schedule, and the six-hour fairness
spread-65 schedule all remain feasible in the strengthened model.
The repaired partial super_hints assignments were also honored in a valid
18-player schedule. All revised models passed OR-Tools validation.

The tests require spread >= 67 and remove optimization. Finding a schedule
improves 66. INFEASIBLE proves 66 optimal under these rules. UNKNOWN leaves
the question unresolved. The score-66 hint does not meet the threshold;
it supplies search guidance rather than a qualifying solution.

Each subfolder stores its exact solver source, full model, dependencies,
hint checksum, command, timestamps, solver log and final report.
