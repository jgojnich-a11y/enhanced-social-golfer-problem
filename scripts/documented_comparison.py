#!/usr/bin/env python3
"""Run and document both 18-player objectives using a snapshot of the solver."""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import platform
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ZONE = ZoneInfo('Australia/Sydney')

def now():
    return datetime.now(ZONE).isoformat()

def metrics(path):
    rounds = collections.defaultdict(list)
    groups = collections.defaultdict(dict)
    slots = collections.defaultdict(list)
    pairs = collections.Counter()
    fours = collections.Counter()
    with path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            r, g, *players = map(int, row)
            assert g not in groups[r], 'Duplicate group row'
            groups[r][g] = players
            rounds[r].extend(players)
            pairs.update(itertools.combinations(sorted(players), 2))
            if len(players) == 4:
                fours.update(players)
            for p in players:
                slots[p].append((r, g))
    assert set(rounds) == {1, 2, 3, 4}, 'Wrong rounds'
    assert all(sorted(ps) == list(range(1, 19)) for ps in rounds.values()), 'Player coverage'
    assert all(set(gs) == {1, 2, 3, 4, 5} and [len(gs[g]) for g in range(1, 6)] == [3, 3, 4, 4, 4] for gs in groups.values()), 'Group sizes'
    assert max(pairs.values()) == 1, 'Repeated player pairs'
    assert all(fours[p] in (2, 3) for p in range(1, 19)), 'Unbalanced four-player participation'
    records = []
    for p, visits in sorted(slots.items()):
        times = [g for r, g in sorted(visits)]
        distinct = len(set(times))
        most = max(collections.Counter(times).values())
        records.append(dict(player=p, round_slots=times, distinct_slots=distinct,
                            repeat_appearances=4-distinct, max_same_slot_appearances=most,
                            hogging_index=most/4, average_minutes_after_1000=sum((g-1)*7 for g in times)/4,
                            four_player_rounds=fours[p]))
    summary = dict(validation='passed', repeated_player_pairs=0,
                   minimum_distinct_slots=min(r['distinct_slots'] for r in records),
                   total_spread=sum(r['distinct_slots'] for r in records),
                   average_distinct_slots=sum(r['distinct_slots'] for r in records)/18,
                   average_hogging=sum(r['hogging_index'] for r in records)/18,
                   worst_hogging=max(r['hogging_index'] for r in records),
                   max_same_slot_appearances=max(r['max_same_slot_appearances'] for r in records))
    return summary, records

def save_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)

def run(mode, folder, limit):
    dest = folder / mode
    dest.mkdir()
    snapshot = folder / 'solver_snapshot.py'
    flag = '--fair_spread' if mode == 'fair' else '--maximize_spread'
    command = [sys.executable, str(snapshot), '--n', '18', '--time_limit', str(limit),
               flag, '--seed', '1', '--workers', '8', '--out_prefix', str(dest/'solution')]
    metadata = dict(mode=mode, state='running', started_at=now(), command=command,
                    players=18, rounds=4, time_limit_seconds=limit, seed=1, workers=8,
                    warm_start=None, tee_times=['10:00','10:07','10:14','10:21','10:28'])
    save_json(dest/'run.json', metadata)
    try:
        with (dest/'solver.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        metadata.update(finished_at=now(), exit_code=result.returncode)
        logtext = (dest/'solver.log').read_text()
        for key, pattern in [('solver_status',r'^Solver status: (\w+)'),
                             ('solver_elapsed_seconds',r'^Solver status: \w+, elapsed ([\d.]+)s'),
                             ('objective',r'^objective: ([-\d.e+]+)'),
                             ('best_bound',r'^best_bound: ([-\d.e+]+)')]:
            match = re.search(pattern, logtext, re.M)
            if match:
                metadata[key] = match.group(1) if key == 'solver_status' else float(match.group(1))
        solution = dest/'solution_n18_sol.csv'
        if result.returncode != 0:
            raise RuntimeError(f'Solver exited {result.returncode}; see solver.log')
        if solution.exists():
            summary, records = metrics(solution)
            metadata.update(summary=summary, state='complete', solution_sha256=hashlib.sha256(solution.read_bytes()).hexdigest())
            with (dest/'players.csv').open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['player','round_1_slot','round_2_slot','round_3_slot','round_4_slot','distinct_slots','repeat_appearances','max_same_slot_appearances','hogging_index','average_minutes_after_1000','four_player_rounds'])
                for r in records:
                    w.writerow([r['player'],*r['round_slots'],*[r[k] for k in ['distinct_slots','repeat_appearances','max_same_slot_appearances','hogging_index','average_minutes_after_1000','four_player_rounds']]])
        else:
            metadata['state'] = 'no_solution'
        text = f'# 18-player run: {mode}\n\nStarted: {metadata["started_at"]}\n\nFinished: {metadata["finished_at"]}\n\nLimit: {limit} seconds; seed: 1; workers: 8; four rounds; no warm start.\n\nStatus: {metadata.get("solver_status","unknown")}.\n\n'
        if metadata.get('summary'):
            text += '| Metric | Value |\n|---|---:|\n'+''.join(f'| {k} | {v} |\n' for k,v in metadata['summary'].items())
        text += '\nSee `run.json` for the exact command, `solver.log` for solver statistics, and `players.csv` for individual metrics. FEASIBLE does not establish optimality. The fair objective is 73 × minimum distinct slots + total spread. Runs execute concurrently, each with eight workers; timing is not an isolated hardware benchmark. Average start-time offsets assume seven-minute intervals from 10:00.\n'
        (dest/'README.md').write_text(text)
    except Exception as exc:
        metadata.update(state='failed', finished_at=now(), error=str(exc))
    save_json(dest/'run.json', metadata)
    print(mode, metadata['state'], metadata.get('solver_status'), metadata.get('summary'), flush=True)
    return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time-limit', type=int, default=3600)
    parser.add_argument('--modes', nargs='+', choices=['total','fair'], default=['total','fair'])
    args = parser.parse_args()
    if args.time_limit <= 0:
        parser.error('--time-limit must be positive')
    folder = ROOT/'runs'/datetime.now(ZONE).strftime('n18_%Y%m%d_%H%M%S')
    folder.mkdir(parents=True)
    shutil.copy2(ROOT/'cp_sat_caseB_v1_1.py', folder/'solver_snapshot.py')
    shutil.copy2(ROOT/'requirements.txt', folder/'requirements.txt')
    freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True)
    (folder/'installed_dependencies.txt').write_text(freeze)
    githead = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    save_json(folder/'environment.json', dict(started_at=now(), python=sys.version, executable=sys.executable,
              architecture=platform.machine(), platform=platform.platform(), git_head=githead,
              solver_sha256=hashlib.sha256((folder/'solver_snapshot.py').read_bytes()).hexdigest(),
              note='Solver snapshot includes uncommitted changes; git HEAD alone does not identify this code.'))
    print('Run folder:', folder, flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, mode, folder, args.time_limit) for mode in args.modes]
        results = [f.result() for f in futures]
    save_json(folder/'comparison.json', results)
    lines = ['# 18-player comparison', '', f'Time limit per mode: {args.time_limit} seconds.', '',
             '| Mode | Status | Minimum slots | Total spread | Average hogging | Worst hogging |',
             '|---|---|---:|---:|---:|---:|']
    for r in results:
        s=r.get('summary',{})
        lines.append(f'| {r["mode"]} | {r.get("solver_status",r["state"])} | {s.get("minimum_distinct_slots","—")} | {s.get("total_spread","—")} | {s.get("average_hogging","—")} | {s.get("worst_hogging","—")} |')
    lines += ['', 'Each mode has its own solution CSV, per-player metrics, run metadata, solver log, and report. Both use the preserved solver snapshot in this folder.', '', 'Runs execute concurrently; compare solution quality rather than treating elapsed times as isolated speed measurements.']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')
    if any(r['state']=='failed' for r in results):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
