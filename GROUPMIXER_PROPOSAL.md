# Benchmark proposal: 18-player mixed-size tee-time rotation

Hello GroupMixer team,

Would you be interested in an additional constrained benchmark case based
on a four-round golf rotation with a proven fairness optimum?

The case has 18 players and five ordered groups per round, with exact sizes
(3,3,4,4,4). The group-size pattern repeats in every round. The groups are
tee slots at 10:00, 10:07, 10:14, 10:21 and 10:28.

Every player attends every round, no unordered player pair meets more than
once, and four-player-group participation is balanced: six players have
two four-player rounds and twelve have three, without prescribing identities.

The objective first maximizes the minimum number of distinct tee slots
received by any player, then maximizes total distinct slots across players.
The proven optimum is (minimum 3, total 65). Equivalently, maximize
73 × minimum + total, with optimal value 284.

The project provides a validated reference schedule and an exhaustive
symmetry-based exclusion of every improvement, independently checked in
two OR-Tools formulations. Five additional fresh, no-hint local runs with
seeds 101/202/303/404/505 each found (3,65) within a 60-second end-to-end
deadline. Those attempts were FEASIBLE, not solver-proven OPTIMAL; optimality
comes from the separate exhaustive argument. These local ARM64/macOS timings
are not a direct comparison with your benchmark: CPUs were not pinned and
an 8 GiB process limit was not imposed.

Public reference material:

- Repository: https://github.com/jgojnich-a11y/enhanced-social-golfer-problem
- Schedule: https://github.com/jgojnich-a11y/enhanced-social-golfer-problem/blob/main/solutions/optimal/n18_sol.csv
- Proof and audit: https://github.com/jgojnich-a11y/enhanced-social-golfer-problem/tree/main/runs/n18_canonical_two_rounds_20261003
- Full case definition: https://github.com/jgojnich-a11y/enhanced-social-golfer-problem/blob/main/N18_BENCHMARK_CASE.md
- Fresh no-hint evidence: https://github.com/jgojnich-a11y/enhanced-social-golfer-problem/tree/main/runs/n18_cold_benchmark_20261003

The optimum and reference schedule are evaluation evidence, not embedded
answers or hints for a benchmarked solver.

Does this fit your benchmark scope, and can your scenario format express
the exact lexicographic tee-slot objective and group-size participation
balance? If so, what format would you prefer for contributing the case?

John Gojnich

---

Published on 3 October 2026 as John Gojnich (guest):

https://groupmixer.app/community/p/thread_223622e1403764efb52e595f026d61596b79/benchmark-proposal-18-player-mixed-size-tee-time-rotation

Publication was confirmed on the live thread page. The posted wording follows this proposal with minor formatting changes. Supporting evidence was published to GitHub before posting, commit 8632c7f. Screenshot: runs/n18_cold_benchmark_20261003/groupmixer_submission.png. Guest editing is tied to the posting browser; automatic email updates require a signed-in profile.
