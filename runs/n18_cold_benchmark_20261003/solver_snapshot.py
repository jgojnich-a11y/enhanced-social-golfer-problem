#!/usr/bin/env python3
"""
cp_sat_caseB_v1_1.py

v1.1: stable CP-SAT solver (based on v1) with:
 - 3-before-4 structural ordering (per-round group sizes)
 - explicit pair indicator booleans (robust pair uniqueness)
 - mild symmetry break (player 1 -> group 0 of round 0)
 - optional position-spread objective (--maximize_spread)

Requires: ortools
pip install ortools
"""
import argparse
import itertools
import math
import csv, os
from ortools.sat.python import cp_model
from datetime import datetime

def load_hints_from_csv(path, X):
    """Load complete group rows or positive-only partial membership hints."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hint file not found: {path}")
    N = max(p for r, g, p in X)
    sizes = group_distribution(N)
    rows = {}
    known = {}
    with open(path, newline="") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row:
                continue
            try:
                r, g, *players = map(int, row)
            except ValueError:
                if line == 1 and row[0].strip().lower() == "round":
                    continue
                raise ValueError(f"Invalid hint CSV row {line}")
            r, g = r - 1, g - 1
            if (r, g, 1) not in X or not players:
                raise ValueError(f"Invalid round/group or empty membership at row {line}")
            if len(set(players)) != len(players) or len(players) > sizes[g]:
                raise ValueError(f"Invalid group membership at row {line}")
            if (r, g) in rows:
                raise ValueError(f"Duplicate round/group at row {line}")
            for player in players:
                if (r, g, player) not in X or (r, player) in known:
                    raise ValueError(f"Invalid or duplicate player at row {line}")
                known[r, player] = g
            rows[r, g] = set(players)
    hints = {}
    for (r, g), players in rows.items():
        for player in range(1, N + 1):
            if player in players:
                hints[r, g, player] = 1
            elif len(players) == sizes[g]:
                hints[r, g, player] = 0
    # A known membership also implies absence from every other group.
    for (r, player), assigned_group in known.items():
        for rr, g, pp in X:
            if rr == r and pp == player:
                hints[rr, g, pp] = int(g == assigned_group)
    print(f"[INFO] Loaded {len(hints)} assignment hints from {path}")
    return hints

def group_distribution(N):
    # your rule: prefer 4-groups unless remainder forces some 3-groups
    g3 = (4 - (N % 4)) % 4
    g4 = (N - 3*g3) // 4
    # per-round group sizes: 3-groups first, then 4-groups
    sizes = [3]*g3 + [4]*g4
    return sizes

def build_model(N, time_limit=60, maximize_spread=False, enforce_balance=True, seed=None, workers=8, fair_spread=False, strengthen=False):
    model = cp_model.CpModel()
    ROUNDS = 4
    per_round_sizes = group_distribution(N)
    G = len(per_round_sizes)  # groups per round (same each round)

    # --- Decision variables: X[(r,g,p)] boolean: player p in group g of round r ---
    X = {}
    for r in range(ROUNDS):
        for g in range(G):
            for p in range(1, N+1):
                X[(r,g,p)] = model.NewBoolVar(f"X_r{r}_g{g}_p{p}")

    # --- Structural constraints (3-before-4 handled by per_round_sizes ordering) ---
    # each group has exact capacity
    for r in range(ROUNDS):
        for g, size in enumerate(per_round_sizes):
            model.Add(sum(X[(r,g,p)] for p in range(1, N+1)) == size)

    # each player appears exactly once per round
    for r in range(ROUNDS):
        for p in range(1, N+1):
            model.Add(sum(X[(r,g,p)] for g in range(G)) == 1)

    # --- Pair indicators: y[(i,j,r,g)] boolean if i and j share group g in round r ---
    # We'll create y for each unordered pair, each round & group and constrain it with AddBoolAnd / AddBoolOr
    y = {}
    for i, j in itertools.combinations(range(1, N+1), 2):
        y[(i,j)] = []
        for r in range(ROUNDS):
            for g in range(G):
                var = model.NewBoolVar(f"y_{i}_{j}_r{r}_g{g}")
                # var => (X[r,g,i] and X[r,g,j])
                model.AddBoolAnd([X[(r,g,i)], X[(r,g,j)]]).OnlyEnforceIf(var)
                # not var => at least one of them is false
                model.AddBoolOr([X[(r,g,i)].Not(), X[(r,g,j)].Not()]).OnlyEnforceIf(var.Not())
                y[(i,j)].append(var)
        # pair uniqueness: sum over all rounds/groups ≤ 1
        model.Add(sum(y[(i,j)]) <= 1)

    # --- Balance 4-group participation (optional but recommended) ---
    # Count total 4-group "slots" across all rounds and balance among players
    four_group_indices = [g for g,s in enumerate(per_round_sizes) if s == 4]
    if enforce_balance and len(four_group_indices) > 0:
        S = ROUNDS * (len(four_group_indices) )  # number of times a 4-group exists (per-round groups count)
        # BUT we need the number of 4-group *slots* (each 4-group holds 4 players)
        S_slots = ROUNDS * (len(four_group_indices) * 4)
        t = S_slots // N
        r_rem = S_slots - t * N
        # count4[p] = number of times player p is in a 4-group (i.e., number of 4-group slots they occupy)
        count4 = {}
        z = {}
        for p in range(1, N+1):
            count4[p] = model.NewIntVar(0, ROUNDS * len(four_group_indices), f"count4_{p}")
            model.Add(count4[p] == sum(X[(r,g,p)] for r in range(ROUNDS) for g in four_group_indices))
            # enforce count4 equals either t or t+1 using boolean z
            z[p] = model.NewBoolVar(f"z_extra4_{p}")
            model.Add(count4[p] == t + z[p])
        # exactly r_rem players should have t+1
        model.Add(sum(z[p] for p in range(1, N+1)) == r_rem)

    # --- (Mild) symmetry breaker: pin player 1 into group 0 of round 0 (helps solver) ---
    # --- model.Add(X[(0, 0, 1)] == 1)
    # --- (Hot) Fixed Round 1 pattern (generic version) ---
    # Works for any N and arbitrary group_sizes, e.g. [3,3,3,4,4]
    start = N
    fixed_round1 = []
    for size in per_round_sizes:
        group_players = list(range(start - size + 1, start + 1))
        fixed_round1.append(group_players)
        start -= size

    # Now enforce those assignments in round 0
    for g, players in enumerate(fixed_round1):
        for p in range(1, N + 1):
            if p in players:
                model.Add(X[(0, g, p)] == 1)
            else:
                model.Add(X[(0, g, p)] == 0)

    if strengthen:
        # Every first-round group is a clique of already-met players.
        # At most one member can occupy any later group. This is implied
        # by pair uniqueness but gives the solver the whole clique at once.
        for players in fixed_round1:
            for r in range(1, ROUNDS):
                for g in range(G):
                    model.AddAtMostOne(X[r, g, p] for p in players)
        # Unique partners equal the sum of group-size-minus-one per round.
        # Redundant equality couples pair indicators to group-size balance.
        for p in range(1, N + 1):
            partner_indicators = [v for (i, j), values in y.items()
                                  if p in (i, j) for v in values]
            model.Add(sum(partner_indicators) ==
                      sum((size - 1) * X[r, g, p]
                          for r in range(ROUNDS)
                          for g, size in enumerate(per_round_sizes)))

    # --- Optional: position-spread variables and objective (maximize distinct group-indices per player) ---
    distinct_pos = None
    if maximize_spread or fair_spread:
        # p_used[(p, g_index)] indicates player p ever occupies group index g across any round
        p_used = {}
        for p in range(1, N+1):
            for g in range(G):
                p_used[(p,g)] = model.NewBoolVar(f"posUsed_p{p}_g{g}")
                # p_used >= X[r,g,p] for every round r
                for r in range(ROUNDS):
                    model.Add(p_used[(p,g)] >= X[(r,g,p)])
                # optional tightness:
                model.Add(p_used[(p,g)] <= sum(X[(r,g,p)] for r in range(ROUNDS)))

        # distinct position count per player
        distinct_pos = {}
        for p in range(1, N+1):
            distinct_pos[p] = model.NewIntVar(0, min(ROUNDS, G), f"distinct_pos_{p}")
            model.Add(distinct_pos[p] == sum(p_used[(p,g)] for g in range(G)))

        total_spread = sum(distinct_pos.values())
        if fair_spread:
            minimum_spread = model.NewIntVar(0, min(ROUNDS, G), "minimum_spread")
            model.AddMinEquality(minimum_spread, list(distinct_pos.values()))
            # One extra slot for the least-served player outweighs every
            # possible change in total spread: exact lexicographic priority.
            priority_weight = N * min(ROUNDS, G) + 1
            model.Maximize(priority_weight * minimum_spread + total_spread)
        else:
            model.Maximize(total_spread)

    # solver setup
    solver = cp_model.CpSolver()
    if seed is not None:
        solver.parameters.random_seed = seed
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    # show some progress
    solver.parameters.log_search_progress = True
    solver.parameters.cp_model_probing_level = 1

    return model, solver, X, per_round_sizes, ROUNDS

def solve_and_output(
    N, time_limit=60, maximize_spread=False,
    enforce_balance=True, seed=None, workers=8,
    out_prefix=None, input_csv=None, fair_spread=False, strengthen=False
):
    model, solver, X, group_sizes, ROUNDS = build_model(
        N, time_limit, maximize_spread, enforce_balance, seed, workers, fair_spread, strengthen
    )

    # --- OPTIONAL: Add hints from existing CSV ---
    if input_csv:
        hints = load_hints_from_csv(input_csv, X)
        if hints:
            for (r, g, p), val in hints.items():
                model.AddHint(X[(r, g, p)], val)
            print(f"[INFO] Added {len(hints)} variable hints from {input_csv}")
        else:
            print(f"[WARN] No hints loaded from {input_csv}")

    # --- SOLVE ---
    start = datetime.now()
    status = solver.Solve(model)
    elapsed = (datetime.now() - start).total_seconds()
    status_name = solver.StatusName(status)
    print(f"Solver status: {status_name}, elapsed {elapsed:.2f}s")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # extract solution into rounds -> groups -> members
        G = len(group_sizes)
        sol = []
        for r in range(ROUNDS):
            round_groups = []
            for g in range(G):
                members = [p for p in range(1, N + 1) if solver.Value(X[(r, g, p)]) == 1]
                round_groups.append(sorted(members))
            sol.append(round_groups)

        # print succinct summary
        for r, rnd in enumerate(sol, start=1):
            print(f"Round {r}:")
            for g, grp in enumerate(rnd, start=1):
                print(f"  Group {g} (size {len(grp)}): {grp}")

        # save CSV if requested
        if out_prefix:
            fname = f"{out_prefix}_n{N}_sol.csv"
            with open(fname, "w", newline="") as fh:
                w = csv.writer(fh)
                for r, rnd in enumerate(sol, start=1):
                    for g, grp in enumerate(rnd, start=1):
                        w.writerow([r, g] + grp)
            print(f"Saved solution CSV to {fname}")
        return sol
    else:
        print("❌ No solution (status=%s)" % status_name)
        return None

def solve_and_output_old(N, time_limit=60, maximize_spread=False, enforce_balance=True, seed=None, workers=8, out_prefix=None):
    model, solver, X, group_sizes, ROUNDS = build_model(N, time_limit, maximize_spread, enforce_balance, seed, workers)
    start = datetime.now()
    status = solver.Solve(model)
    elapsed = (datetime.now() - start).total_seconds()
    status_name = solver.StatusName(status)
    print(f"Solver status: {status_name}, elapsed {elapsed:.2f}s")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # extract solution into rounds -> groups -> members
        G = len(group_sizes)
        sol = []
        for r in range(ROUNDS):
            round_groups = []
            for g in range(G):
                members = [p for p in range(1, N+1) if solver.Value(X[(r,g,p)]) == 1]
                round_groups.append(sorted(members))
            sol.append(round_groups)

        # print succinct summary
        for r, rnd in enumerate(sol, start=1):
            print(f"Round {r}:")
            for g, grp in enumerate(rnd, start=1):
                print(f"  Group {g} (size {len(grp)}): {grp}")
        # save csv if requested
        if out_prefix:
            fname = f"{out_prefix}_n{N}_sol.csv"
            with open(fname, "w", newline='') as fh:
                w = csv.writer(fh)
                for r, rnd in enumerate(sol, start=1):
                    for g, grp in enumerate(rnd, start=1):
                        w.writerow([r, g] + grp)
            print(f"Saved solution CSV to {fname}")
        return sol
    else:
        print("❌ No solution (status=%s)" % status_name)
        return None

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=17, help="Number of players")
    p.add_argument("--time_limit", type=int, default=120, help="Seconds solver time limit")
    p.add_argument("--maximize_spread", action="store_true", help="Maximize distinct group-index spread per player")
    p.add_argument("--strengthen", action="store_true", help="Add implied first-round clique and partner-count constraints")
    p.add_argument("--fair_spread", action="store_true", help="Maximize minimum distinct tee-time slots first, then total spread")
    p.add_argument("--enforce_balance", action="store_true", default=True, help="Enforce exact 4-group slot balance (t or t+1)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out_prefix", type=str, default="solution", help="prefix for saved CSV")
    p.add_argument("--input_csv", help="Optional existing solution CSV for AddHint warm start")
    args = p.parse_args()
    if args.input_csv:
        print(f"Using {args.input_csv} as warm-start hint")

    print(f"Starting cp_sat_caseB_v1_1.py N={args.n}, maximize_spread={args.maximize_spread}, enforce_balance={args.enforce_balance}")
    #sol = solve_and_output(args.n, args.time_limit, args.maximize_spread, args.enforce_balance, args.seed, args.workers, args.out_prefix)
    sol = solve_and_output(
        args.n,
        args.time_limit,
        args.maximize_spread,
        args.enforce_balance,
        args.seed,
        args.workers,
        args.out_prefix,
        input_csv=args.input_csv,
        fair_spread=args.fair_spread,
        strengthen=args.strengthen
    )
