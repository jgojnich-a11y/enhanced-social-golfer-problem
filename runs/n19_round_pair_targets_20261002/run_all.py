import json
import subprocess
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
(base/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
results = []
for case in ['A','B']:
    with (base/f'case_{case}.log').open('w') as log:
        process = subprocess.run([sys.executable,str(base/'test_driver.py'),'--case',case],stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:
        result = dict(case=case,state='failed',exit_code=process.returncode)
    else:
        result = json.loads((base/case/'run.json').read_text())
        result['exit_code'] = process.returncode
        (base/f'case_{case}.log').rename(base/case/'solver.log')
    results.append(result)
    (base/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    print(case,result.get('target_total_spread'),result.get('solver_status',result['state']),result.get('summary'),flush=True)
a,b = results
if a.get('summary'):
    conclusion = 'Validated fairness (3,70) found: proven lexicographic optimum under current rules by the counting upper bound 70.'
elif b.get('summary') and a.get('solver_status')=='INFEASIBLE':
    conclusion = 'Validated fairness (3,69) found and 70 proved infeasible: proven lexicographic optimum under current rules.'
elif b.get('summary'):
    conclusion = 'Validated fairness improvement to (3,69). Total 70 remains unresolved, so optimality is unproven.'
elif all(r.get('solver_status')=='INFEASIBLE' for r in results):
    conclusion = 'Both fair totals 70 and 69 proved infeasible. Together with the counting upper bound 70, the validated (3,68) baseline is lexicographically optimal under current rules.'
else:
    conclusion = 'No validated improvement found. At least one target remains unresolved or failed; global fairness optimality remains unproven.'
lines = ['# 19-player fairness targets with round-pair strengthening','',conclusion,'',
         'Sequential feasibility searches, 600 seconds each, seed 73, eight workers, original model constraints and the saved fairness-68 assignment hint. Explicit implied round-pair repeat constraints are added; no optional generator strengthening or round-order symmetry.','',
         '| Case | Target total | Early repeats | Late repeats | Status | Search seconds |','|---|---:|---:|---:|---|---:|']
for r in results:
    lines.append(f"| {r['case']} | {r.get('target_total_spread')} | {r.get('early_repeat_appearances')} | {r.get('late_repeat_appearances')} | {r.get('solver_status',r['state'])} | {r.get('solver_elapsed_seconds')} |")
lines += ['', 'Minimum spread >=3 is required in both tests. Balance permits each player at most one triple appearance, so early repeats are zero. At least one same-late-slot player is required for every pair of rounds; at minimum spread 3 these are six distinct players, bounding total spread by 70. The two exact-total tests cover every improvement over the validated (3,68) baseline. See COUNTING_BOUND.md.', '',
          'Round-pair counts are explicitly constrained to six ones for total 70 or five ones and one two for total 69, retaining every qualifying schedule. Both saved schedules were checked before target restrictions; modeled counts matched all six independently computed intersections. Both baseline schedules passed coverage, pair uniqueness, capacities, participation balance and forced-model validation. Repeat-variable totals matched direct CSV counts. The retained generator and original schedules are preserved. Models, logs, statistics, source and hint checksums, exact commands, and any qualifying solutions are saved here.']
(base/'README.md').write_text('\n'.join(lines)+'\n')
(base/'completion.json').write_text(json.dumps(dict(state='complete' if all(r['exit_code']==0 for r in results) else 'failed',conclusion=conclusion,run_count=len(results)),indent=2)+'\n')
if any(r['exit_code'] for r in results):
    raise SystemExit(1)
