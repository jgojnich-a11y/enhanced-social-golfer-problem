import csv
import argparse
from collections import defaultdict, Counter

def load_positions(filename):
    """Load player positions (group indices) per round."""
    data = defaultdict(list)
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            round_num, group_num, *players = row
            for p in players:
                data[int(p)].append(int(group_num))
    return data

def calc_distinct_and_hogging(pos_data, total_rounds=4):
    """
    Returns:
      distinct_counts[p]: number of distinct groups a player used
      hogging_index[p]: fraction of rounds spent in their most common group
    """
    distinct_counts = {}
    hogging_index = {}

    for p, groups in pos_data.items():
        c = Counter(groups)
        distinct_counts[p] = len(c)
        hogging_index[p] = max(c.values()) / total_rounds  # 1.0 = total hogging
    return distinct_counts, hogging_index

def compare_files(file1, file2, total_rounds=4):
    pos1 = load_positions(file1)
    pos2 = load_positions(file2)

    d1, h1 = calc_distinct_and_hogging(pos1, total_rounds)
    d2, h2 = calc_distinct_and_hogging(pos2, total_rounds)

    players = sorted(set(d1.keys()) | set(d2.keys()))
    avg_d1 = sum(d1[p] for p in players) / len(players)
    avg_d2 = sum(d2[p] for p in players) / len(players)
    avg_h1 = sum(h1[p] for p in players) / len(players)
    avg_h2 = sum(h2[p] for p in players) / len(players)

    print(f"{file1}: avg distinct = {avg_d1:.3f}, avg hogging = {avg_h1:.3f}")
    print(f"{file2}: avg distinct = {avg_d2:.3f}, avg hogging = {avg_h2:.3f}")
    print(f"Δ distinct = {avg_d2 - avg_d1:+.3f}, Δ hogging = {avg_h2 - avg_h1:+.3f}\n")

    print("Per-player comparison:")
    print("Player | Distinct  | Hogging  | ΔDistinct | ΔHogging")
    print("-------|------------|----------|------------|----------")
    for p in players:
        print(f"{p:6d} | {d1.get(p,0):>4d} → {d2.get(p,0):<4d} | "
              f"{h1.get(p,0):.2f} → {h2.get(p,0):.2f} | "
              f"{d2.get(p,0)-d1.get(p,0):+8d} | {h2.get(p,0)-h1.get(p,0):+8.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two N18 spread solutions.")
    parser.add_argument(
        "fileA",
        nargs="?",
        default="solution_n18_sol_60m.csv",
        help="Reference/base solution (default: solution_n18_sol_60m.csv)",
    )
    parser.add_argument(
        "fileB",
        nargs="?",
        default="solution_n18_sol_20251120_052000.csv",
        help="Solution to compare against baseline (default: previous 60m file)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="Total rounds (default: 4)",
    )

    args = parser.parse_args()

    compare_files(args.fileA, args.fileB, total_rounds=args.rounds)
