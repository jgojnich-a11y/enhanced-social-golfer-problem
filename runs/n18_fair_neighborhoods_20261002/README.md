# Two-round neighborhood searches

Three sequential searches; 400 seconds per search, seed 73, eight workers. Each starts from the saved (3,65) fairness schedule and requires minimum spread >=3 and total >=66. Round 1 and one later round are fixed to the source.

| Free rounds | Fixed rounds | Status | Seconds |
|---|---|---|---:|
| [2, 3] | [1, 4] | INFEASIBLE | 6.6838630000000006 |
| [2, 4] | [1, 3] | INFEASIBLE | 4.328515 |
| [3, 4] | [1, 2] | INFEASIBLE | 0.14258400000000002 |

INFEASIBLE rules out only the corresponding neighborhood, not global improvement. UNKNOWN leaves that neighborhood unresolved. See each subfolder for model, sources, hint, dependencies, log, validation and metadata.
