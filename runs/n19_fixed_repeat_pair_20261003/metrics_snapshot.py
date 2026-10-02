"""Audited short fairness sweep with the retained four-round model."""
import collections
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ortools.sat.python import cp_model
from solver_snapshot import build_model, group_distribution

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
ZONE = ZoneInfo('Australia/Sydney')


def now():
    return datetime.now(ZONE).isoformat()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def validate(schedule, n, sizes):
    pairs = collections.Counter()
    fours = collections.Counter()
    slots = collections.defaultdict(list)
    assert len(schedule) == 4
    for groups in schedule:
        assert [len(group) for group in groups] == sizes
        assert sorted(itertools.chain.from_iterable(groups)) == list(range(1, n+1))
        for g, group in enumerate(groups):
            pairs.update(itertools.combinations(sorted(group), 2))
            if len(group) == 4:
                fours.update(group)
            for p in group:
                slots[p].append(g)
    assert max(pairs.values()) == 1
    lower, remainder = divmod(4 * sizes.count(4) * 4, n)
    assert all(fours[p] in (lower, lower+1) for p in range(1, n+1))
    assert sum(fours[p] == lower+1 for p in range(1, n+1)) == remainder
    players = [dict(player=p, round_slots=[g+1 for g in slots[p]],
                    distinct_slots=len(set(slots[p])),
                    max_same_slot_appearances=max(collections.Counter(slots[p]).values()),
                    four_player_rounds=fours[p],
                    average_minutes_after_1000=sum(slots[p])*7/4)
               for p in range(1, n+1)]
    summary = dict(validation='passed', repeated_player_pairs=0,
                   minimum_distinct_slots=min(p['distinct_slots'] for p in players),
                   total_spread=sum(p['distinct_slots'] for p in players),
                   max_same_slot_appearances=max(p['max_same_slot_appearances'] for p in players),
                   four_player_rounds_min=min(fours[p] for p in range(1, n+1)),
                   four_player_rounds_max=max(fours[p] for p in range(1, n+1)))
    return summary, players


def one(n):
    folder = BASE / f'n{n}'
    folder.mkdir()
    model, solver, assignments, sizes, rounds = build_model(
        n, time_limit=60, fair_spread=True, seed=73, workers=8)
    assert not model.Validate(), model.Validate()
    model.ExportToFile(str(folder/'model.pbtxt'))
    record = dict(n=n, state='running', started_at=now(), rounds=rounds,
                  group_sizes=sizes, time_limit_seconds=60, seed=73, workers=8,
                  fair_spread=True, strengthen=False, hint=None,
                  tee_times=[f'{600+g*7:04d}' for g in range(len(sizes))])
    record['tee_times'] = [f'{(600+g*7)//60:02d}:{(600+g*7)%60:02d}' for g in range(len(sizes))]
    save(folder/'run.json', record)
    with (folder/'solver.log').open('w') as log:
        solver.parameters.log_to_stdout = False
        solver.log_callback = lambda message: log.write(message + '\n')
        status = solver.Solve(model)
    record.update(state='complete', finished_at=now(),
                  solver_status=solver.StatusName(status), solver_elapsed_seconds=solver.WallTime())
    (folder/'solver_statistics.txt').write_text(solver.ResponseStats())
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        schedule = [[[p for p in range(1,n+1) if solver.Value(assignments[r,g,p])]
                     for g in range(len(sizes))] for r in range(rounds)]
        summary, players = validate(schedule, n, sizes)
        record['summary'] = summary
        record['objective'] = solver.ObjectiveValue()
        record['best_bound'] = solver.BestObjectiveBound()
        assert round(record['objective']) == (4*n+1)*summary['minimum_distinct_slots']+summary['total_spread']
        solution = folder/f'solution_n{n}_sol.csv'
        with solution.open('w', newline='') as stream:
            writer = csv.writer(stream)
            for r, groups in enumerate(schedule,1):
                for g, group in enumerate(groups,1):
                    writer.writerow([r,g]+group)
        record['solution_sha256'] = hashlib.sha256(solution.read_bytes()).hexdigest()
        save(folder/'players.json', players)
        record['conclusion'] = ('Proven lexicographic fairness optimum under this model.'
                                if status == cp_model.OPTIMAL else
                                'Validated feasible schedule; optimal fairness is unproven.')
    elif status == cp_model.INFEASIBLE:
        record['conclusion'] = 'Proven infeasible under the current four-round scheduling rules.'
    else:
        record['conclusion'] = 'Unresolved: no schedule or infeasibility proof within the search limit.'
    save(folder/'run.json', record)
    return record


def report(records):
    save(BASE/'comparison.json', records)
    lines = ['# Fairness sweep: 15–24 players', '',
             'Four rounds; 60 seconds per player count; seed 73; eight workers; no hints; sequential searches.', '',
             'Original three-player-before-four-player tee-time policy, unique partners and balanced four-player participation. Fairness maximizes minimum distinct tee slots, then total spread. Tee times start at 10:00 every seven minutes.', '',
             '| Players | Group sizes | Last tee time | Status | Minimum slots | Total spread | Seconds |',
             '|---:|---|---|---|---:|---:|---:|']
    for record in records:
        summary = record.get('summary', {})
        lines.append(f"| {record['n']} | {record['group_sizes']} | {record['tee_times'][-1]} | {record['solver_status']} | {summary.get('minimum_distinct_slots','—')} | {summary.get('total_spread','—')} | {record['solver_elapsed_seconds']:.2f} |")
    lines += ['', 'FEASIBLE is a validated candidate, OPTIMAL proves lexicographic fairness optimality, INFEASIBLE proves no valid schedule under these rules, and UNKNOWN is unresolved. Raw total spread is not directly comparable across player counts; group counts also change. These short no-hint runs do not supersede better saved schedules.', '',
              'Each n-folder contains model, log, statistics, metadata and any solution and individual player metrics. Shared source snapshot, dependency list and environment metadata are in this folder. The retained generator and historical schedules are unchanged.']
    (BASE/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    save(BASE/'environment.json', dict(started_at=now(), python=sys.version,
         architecture=platform.machine(), platform=platform.platform(),
         git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         solver_sha256=hashlib.sha256((BASE/'solver_snapshot.py').read_bytes()).hexdigest(),
         driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         command=[sys.executable,str(Path(__file__).resolve())],
         note='Snapshot includes uncommitted changes; HEAD alone does not identify source.'))
    (BASE/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    records = []
    for n in range(15,25):
        record = one(n)
        records.append(record)
        report(records)
        print(n,record['solver_status'],record.get('summary'),flush=True)
    save(BASE/'completion.json', dict(state='complete', finished_at=now(), run_count=len(records)))


if __name__ == '__main__':
    main()
