"""Validate and compare saved schedules against the 15–24 fairness sweep."""
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

from metrics_snapshot import validate
from solver_snapshot import group_distribution

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def analyze(path, source):
    n = int(re.search(r'n(17|18|19)', path.name).group(1))
    result = dict(source=source, n=n, path=str(path.relative_to(ROOT)),
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    try:
        groups = collections.defaultdict(dict)
        with path.open() as stream:
            for line, row in enumerate(csv.reader(stream), 1):
                if not row:
                    continue
                r, g, *players = map(int, row)
                assert g not in groups[r], f'Duplicate round/group at line {line}'
                groups[r][g] = players
        sizes = group_distribution(n)
        assert set(groups) == {1,2,3,4}, 'Expected exactly four rounds'
        assert all(set(gs) == set(range(1,len(sizes)+1)) for gs in groups.values()), 'Wrong group indices'
        schedule = [[groups[r][g] for g in range(1,len(sizes)+1)] for r in range(1,5)]
        summary, players = validate(schedule, n, sizes)
        summary['distinct_slot_histogram'] = dict(sorted(collections.Counter(p['distinct_slots'] for p in players).items()))
        summary['average_distinct_slots'] = summary['total_spread']/n
        summary['worst_hogging'] = summary['max_same_slot_appearances']/4
        summary['average_hogging'] = sum(p['max_same_slot_appearances']/4 for p in players)/n
        summary['earliest_player_average_minutes'] = min(p['average_minutes_after_1000'] for p in players)
        summary['latest_player_average_minutes'] = max(p['average_minutes_after_1000'] for p in players)
        result.update(valid=True, summary=summary, players=players)
    except (AssertionError, ValueError, IndexError) as error:
        result.update(valid=False, error=str(error) or type(error).__name__)
    return result


def main():
    historical = list(ROOT.glob('*n18*.csv'))
    historical += [p for p in (ROOT/'solutions').glob('*.csv')
                   if re.match(r'(?:opt_solution|solution|v2)_n(?:17|18|19)_',p.name)
                   and 'summary' not in p.name]
    historical += [p for p in (ROOT/'sol_1').glob('*.csv')
                   if re.search(r'_n(?:17|18|19)_',p.name)]
    results = [analyze(p,'historical_saved') for p in sorted(historical)]
    for n in range(17,20):
        results.append(analyze(ROOT/f'runs/n15_to24_fair_sweep_20261002/n{n}/solution_n{n}_sol.csv','new_sweep'))
    (BASE/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    fields = ['source','n','path','valid','minimum_distinct_slots','total_spread','max_same_slot_appearances','four_player_rounds_min','four_player_rounds_max','average_hogging','earliest_player_average_minutes','latest_player_average_minutes','error']
    with (BASE/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {field:result.get(field,'') for field in fields}
            row.update({field:result.get('summary',{}).get(field,row[field]) for field in fields})
            writer.writerow(row)
    reference = {17:'solutions/solution_n17_sol.csv',18:'solutions/opt_solution_n18_sol.csv',19:'solutions/opt_solution_n19_sol.csv'}
    lines = ['# Historical comparison: 17–19 players','',
             'Compared saved four-round schedules in the project root, solutions and sol_1 against the new 60-second fairness sweep. Every candidate is checked for coverage, group sizes and order, pair uniqueness, and balanced four-player participation. No schedule was modified and no solver search was launched.','',
             '| Players | Main saved reference: minimum / total | New sweep: minimum / total | Max appearances in one slot: saved → new |','|---:|---|---|---|']
    comparisons=[]
    for n in range(17,20):
        old=next(r for r in results if r['path']==reference[n])
        new=next(r for r in results if r['n']==n and r['source']=='new_sweep')
        assert old['valid'] and new['valid']
        a,b=old['summary'],new['summary']
        lines.append(f"| {n} | {a['minimum_distinct_slots']} / {a['total_spread']} | {b['minimum_distinct_slots']} / {b['total_spread']} | {a['max_same_slot_appearances']} → {b['max_same_slot_appearances']} |")
        historical_n=[r for r in results if r['n']==n and r['source']=='historical_saved' and r['valid']]
        best=max((r['summary']['minimum_distinct_slots'],r['summary']['total_spread']) for r in historical_n)
        comparisons.append(dict(n=n,best_saved_lexicographic_score=best,
                               paths=[r['path'] for r in historical_n if (r['summary']['minimum_distinct_slots'],r['summary']['total_spread'])==best],
                               new_score=[b['minimum_distinct_slots'],b['total_spread']]))
    (BASE/'best_saved_comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    lines += ['', '## Findings','',
              '- Main saved references score (2,64), (2,66) and (2,69). The new sweep raises the minimum to 3, reduces the worst same-slot count from 3 to 2, and gives up one total distinct slot in each case.',
              '- The saved 18-player files sol_1/run_1_n18_sol.csv and sol_1/run_2_n18_sol.csv already score (3,65) and pass validation. The new sweep matches their fairness score; this score is not a new historical record.',
              '- Among the saved files inspected, the best minimum-first scores for 17 and 19 players are (2,64) and (2,69). The sweep improves those minimum-first scores to (3,63) and (3,68).',
              '- Filename prefixes such as opt_solution do not establish an optimality proof. The new 17–19 runs remain FEASIBLE, with total spread at minimum 3 unproven.',
              '- Comparing total spread alone favors the main historical references. Comparing minimum first favors the new 17- and 19-player schedules and ties the already-saved minimum-3 schedules for 18 players.',
              '', '## All inspected files','', '| Source | File | Valid | Minimum | Total | Distinct-slot histogram |','|---|---|---|---:|---:|---|']
    for r in results:
        s=r.get('summary',{})
        lines.append(f"| {r['source']} | {r['path']} | {r['valid']} | {s.get('minimum_distinct_slots','—')} | {s.get('total_spread','—')} | {s.get('distinct_slot_histogram',r.get('error',''))} |")
    lines += ['', 'Histograms map distinct-slot count to number of players. comparison.csv provides all summary metrics; comparison.json also includes player-level metrics and file checksums. metrics_snapshot.py and solver_snapshot.py preserve the validation code and group-size policy used.']
    (BASE/'README.md').write_text('\n'.join(lines)+'\n')
    print('Saved schedules inspected:',len(historical),'valid:',sum(r['valid'] for r in results if r['source']=='historical_saved'))
    print(json.dumps(comparisons,indent=2))
    for r in results:
        if not r['valid']: print('INVALID',r['path'],r['error'])


if __name__=='__main__':
    main()
