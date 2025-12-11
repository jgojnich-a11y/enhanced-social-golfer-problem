"""
summarise_unmet_pairs.py

Usage:
    python3 summarise_unmet_pairs.py solutions/ --summary
"""
import csv
import os
from itertools import combinations

def parse_solution_csv(path):
    """Reads CSV: round, group, players…  → (round_list, N)"""
    rounds = {}
    max_p = 0

    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].isdigit():
                continue

            r = int(row[0])
            g = int(row[1])
            players = list(map(int, row[2:]))

            max_p = max(max_p, max(players))

            if r not in rounds:
                rounds[r] = {}
            rounds[r][g] = players

    R = []
    for r in sorted(rounds.keys()):
        groups = [rounds[r][g] for g in sorted(rounds[r].keys())]
        R.append(groups)

    return R, max_p


def compute_unmet_pairs(R, N):
    """Returns sorted list of pairs that never meet."""
    all_players = range(1, N+1)
    all_pairs = set(combinations(all_players, 2))
    met_pairs = set()

    for round_groups in R:
        for group in round_groups:
            for pair in combinations(group, 2):
                met_pairs.add(tuple(sorted(pair)))

    unmet = sorted(all_pairs - met_pairs)
    return unmet


def summarise_directory(dir_path):
    files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
    for fname in sorted(files):
        full = os.path.join(dir_path, fname)

        R, N = parse_solution_csv(full)
        unmet = compute_unmet_pairs(R, N)

        total_pairs = N*(N-1)//2
        print(f"\n=== {fname} ===")
        print(f"N = {N}, Rounds = {len(R)}")
        print(f"Total pairs = {total_pairs}")
        print(f"Unmet pairs = {len(unmet)}")
        print("Pairs that never meet:")
        print(", ".join(str(p) for p in unmet))


def summarise_one_line(dir_path):
    rows = []

    files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
    for fname in files:
        full = os.path.join(dir_path, fname)

        R, N = parse_solution_csv(full)
        unmet = compute_unmet_pairs(R, N)

        total_pairs = N*(N-1)//2
        rounds = len(R)
        unmet_ct = len(unmet)

        pct = 100 * unmet_ct / total_pairs

        rows.append((pct, fname, N, rounds, total_pairs, unmet_ct))

    # sort by % unmet ascending
    rows.sort()

    for pct, fname, N, rounds, total_pairs, unmet_ct in rows:
        print(f"{fname},{N},{rounds},{total_pairs},{unmet_ct},{pct:.1f}%")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="Directory containing CSV solutions")
    ap.add_argument("--summary", action="store_true",
                    help="Print one-line summary per solution")
    args = ap.parse_args()

    if args.summary:
        summarise_one_line(args.directory)
    else:
        summarise_directory(args.directory)
