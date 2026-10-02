# Current fairness status — 3 October 2026

Rules: four rounds, unique partners, balanced three/four-player participation,
triples before fours, and tee times seven minutes apart from 10:00. Fairness
maximizes minimum individual distinct tee slots, then total distinct slots.
No average-start-time objective or player-specific restrictions apply.

| Players | Best minimum / total | Fairness optimality |
|---:|---|---|
| 15 | 2 / 45 | Proven by sweep OPTIMAL status |
| 16 | 2 / 44 | Proven by sweep OPTIMAL status |
| 17 | 3 / 63 | Proven by global bound and exhaustive score-64 repeat cases |
| 18 | 3 / 65 | Proven by exhaustive two-class canonical exclusion of every fairness improvement |
| 19 | 3 / 68 | Proven by counting bound, total-70 exclusion and complete canonical score-69 exclusion |
| 20 | 4 / 80 | Proven; four distinct slots for everyone |
| 21 | 4 / 84 | Proven; four distinct slots for everyone |
| 22 | 4 / 88 | Proven; four distinct slots for everyone |
| 23 | 4 / 92 | Proven; four distinct slots for everyone |
| 24 | 4 / 96 | Proven; four distinct slots for everyone |

Validated attaining schedules for every count are in
`runs/n15_to24_fair_sweep_20261002/nN/solution_nN_sol.csv`.

Proof references:

- Original sweep: `runs/n15_to24_fair_sweep_20261002/`.
- 17-player proof: `runs/n17_target64_repeat_cases_20261002/`.
- 18-player fair upper bound: `runs/n18_20261001_214154/fair/run.json`,
  weighted bound 286 with weight 73, hence total <=67 at minimum 3.
- 18-player complete optimality proof: `runs/n18_canonical_two_rounds_20261003/`.
  Every improvement has a round pair with diagonal trace zero or one; exhaustive
  intersection-matrix enumeration gives one symmetry class for each. Both
  classes are INFEASIBLE with minimum >=3 and total >=66 in both formulations.
  This also excludes minimum four. Integer times: 11.495 and 3.562 seconds;
  retained formulation: 17.214 and 6.041 seconds.
- 19-player total-70 proof: `runs/n19_round_pair_targets_20261002/A/`.
- 19-player complete canonical score-69 proof:
  `runs/n19_canonical_two_rounds_20261003/`; integer formulation INFEASIBLE
  in 0.863 seconds, independently confirmed by the retained formulation in
  13.143 seconds without additional round-pair strengthening.

Historical total-only scores remain separate: the older 17-, 18- and
19-player references score 64, 66 and 69 with minimum 2. Higher total alone
does not outrank the proven minimum-first fairness scores above.

The retained generator `cp_sat_caseB_v1_1.py` and original schedules remain
unchanged by these latest experiments. The experimental integer formulation
is separate. Existing changes and all audits remain uncommitted. No run is
pending from these experiments.
