# Local no-hint 18-player benchmark

{
  "attempts": 5,
  "valid": 5,
  "reference_optimum_hits": 5,
  "best": [
    3,
    65
  ],
  "median_weighted_score": 284.0,
  "solver_optimal_count": 0,
  "no_pending_runs": true
}

Five fresh processes, seeds 101/202/303/404/505, eight workers and a 60-second external deadline per attempt. Retained solver snapshot with minimum-first fairness and implied strengthening. No saved schedules, hints or target score constraints enter search; reference score is checked only after solving. Every admitted schedule was independently validated without repair.

This is local macOS evidence, not an execution of GroupMixer or a directly comparable published benchmark. CPU affinity and the 8 GiB process limit were not imposed. See environment.json, results.json, summary.json, source snapshot and per-attempt logs/results. All attempts have ended.
