# Setup and running

The project does not require a particular processor architecture. Use Python
and OR-Tools packages compatible with your operating system and processor.
Python 3.13 with the versions pinned in `requirements.txt` is the tested setup.
Create your own virtual environment after downloading or cloning the project;
virtual environments should not be copied between machines.

Run these commands from the project directory on macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The examples below use `.venv/bin/python`. On Windows, substitute
`.\.venv\Scripts\python.exe`. No environment activation is required.
Keep `requirements.txt` to reproduce the dependency versions used in the
documented experiments. Solver runtimes and selected schedules can differ
between machines; validate generated schedules independently.

### Optional Apple Silicon note

For native execution on an Apple Silicon Mac, create `.venv` with an ARM
Python installation. You can inspect the interpreter architecture with:

```sh
.venv/bin/python -c 'import platform; print(platform.machine())'
```

`arm64` indicates native ARM execution. This is a local performance check,
not a project requirement; Intel/AMD machines use their own compatible Python.

## Generate a four-round schedule

Choose a new output prefix under `runs/` for each run to avoid overwriting saved results. The curated optimal schedules and tee-time summaries are indexed in [solutions/README.md](solutions/README.md); older files are in `solutions/archive/2026-10-03/`.

```sh
.venv/bin/python cp_sat_caseB_v1_1.py --n 18 --time_limit 60 --maximize_spread --seed 1 --workers 8 --out_prefix runs/trial_01
```

This writes `runs/trial_01_n18_sol.csv` when a solution
is found. The time limit applies to solver search. A run may end without a
solution. Parallel search can produce different schedules even with a fixed
seed; use `--workers 1` for more repeatable comparisons.

## Compare saved schedules

```sh
.venv/bin/python compare_spread_v3.py solutions/optimal/n18_sol.csv runs/trial_01_n18_sol.csv --rounds 4
```

## Retained solver

`cp_sat_caseB_v1_1.py` is the retained generator. It fixes the number of
rounds at four, enforces pair uniqueness and balanced four-group
participation, and optionally maximizes distinct group positions.
It supports CSV warm-start hints rather than full checkpoint/resume.

The separate integer formulation is experimental. Historical solution CSVs
are preserved in `solutions/archive/`, including schedules with more than
four rounds; v1.1 cannot reproduce those longer schedules as it stands.

For future runs, preserve the full command, generator commit, dependency
versions, solver log, and output CSV together before committing to Git.

## Prioritize tee-time fairness

Use `--fair_spread` to maximize the minimum distinct tee-time slots any
player receives, then maximize total spread. This option includes spread
optimization; `--maximize_spread` is not required. Pair uniqueness and
four-group participation balance remain enforced. Without `--fair_spread`,
the existing solver behavior is retained.

```sh
.venv/bin/python cp_sat_caseB_v1_1.py --n 16 --time_limit 30 --fair_spread --seed 1 --workers 8 --out_prefix runs/my_fair_trial
```

This objective does not directly balance average start time or minimize the
maximum number of appearances in a particular slot. A time-limited FEASIBLE
result is a candidate, not proof of optimal fairness.
