# Enhanced Social Golfer Problem

Four-round golf schedules with unique partners, balanced participation in three- and four-player groups, and fair tee-slot allocation. Tee times start at 10:00 at seven-minute intervals; three-player groups precede four-player groups.

Fairness maximizes the minimum number of distinct tee slots received by any player, then the total across all players. Average start time is not part of this objective. The older hogging metric is retained only in historical comparisons.

## Proven results

The fairness optima for every player count from 15 through 24 are proven. The [schedule collection](solutions/README.md) contains ten independently validated schedules, readable tee-time tables, and proof links. See [proof status](runs/FAIRNESS_STATUS.md) for the methods and evidence.

For 18 players the optimum is minimum spread 3 and total spread 65. The historical total-only score 66 has minimum 2 and therefore ranks below it under the current objective.

## Running the solver

Create a local Python environment and install `requirements.txt`; see the [setup instructions for macOS, Linux and Windows](SETUP.md). No particular processor architecture is required. The retained generator is `cp_sat_caseB_v1_1.py`. It supports total spread (`--maximize_spread`), minimum-first fairness (`--fair_spread`), optional implied constraints (`--strengthen`), and complete or partial CSV warm-start hints. CSV hints do not provide checkpoint/resume. The separate integer formulation is experimental.

Write new experiments under `runs/` and preserve commands, source versions, dependencies, solver logs, and output schedules together. `solutions/optimal/` is the curated collection; `solutions/archive/2026-10-03/` preserves older schedules and comparisons with checksum manifests. Historical root files and `sol_1/` are in its `workspace_legacy/` subdirectory. `solutions.txt` is archived there too; closed Vim recovery files are preserved in the archive’s `editor_recovery/` directory.

[Historical run notes](HISTORICAL_RUNS.md) record earlier experiments. Comparison utilities remain available, but historical performance claims are not optimality evidence.

Author: John Gojnich.

### References and related work

These sources provide background and comparison schedules. Their constraints
and objectives may differ from this project's participation balance and
group-position spread measures.

- **CSPLib, Problem 010: Social Golfers Problem.**
  [Problem specification](https://www.csplib.org/Problems/prob010/),
  [results](https://www.csplib.org/Problems/prob010/results/), and
  [nine-round schedule](https://www.csplib.org/Problems/prob010/results/9weeksol.txt.html).
  The published nine-round schedule matches `solutions/archive/2026-10-03/N32R9_n32_csp.csv`;
  `solutions/archive/2026-10-03/N32R7_n32_csp.csv` contains its first seven rounds.
- **Warwick Harvey, Warwick's Results Page for the Social Golfer Problem.**
  [Archived page, captured 8 March 2005](https://web.archive.org/web/20050308115423/http://www.icparc.ic.ac.uk/~wh/golf/).
  A historical collection of schedules and upper/lower bounds, with results
  last updated 16 September 2002.
- **Martin Mariusz Lester (2021), Scheduling Reach Mahjong Tournaments Using
  Pseudoboolean Constraints.** In *Theory and Applications of Satisfiability
  Testing – SAT 2021*, LNCS 12831, pp. 349–358.
  [Paper / DOI](https://doi.org/10.1007/978-3-030-80223-3_24).
  Related scheduling work and a possible source of
  `solutions/archive/2026-10-03/N24R5_n24_mahjong.csv`; that CSV's exact source is unconfirmed.
- **Martin Mariusz Lester (2021), CoMaToSe: Constraint Mahjong Tournament Scheduler.**
  [Software archive / DOI](https://doi.org/10.5281/zenodo.4764650).
  Software referenced by the Mahjong paper.
- **Roiqk7, SocialGolfersProblem.**
  [GitHub project](https://github.com/Roiqk7/SocialGolfersProblem) and
  [example output](https://github.com/Roiqk7/SocialGolfersProblem/blob/main/data/out/result.txt).
  A project visited during the original research; no match to a local
  schedule has been established.

See [solution source notes](SOLUTION_SOURCES.md) for provenance findings and
recorded browser visit dates. Group-position spread in this project refers
to group/table indices, rather than seats within a group.
