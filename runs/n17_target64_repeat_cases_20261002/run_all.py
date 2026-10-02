import json
import subprocess
import sys
from pathlib import Path

base=Path(__file__).resolve().parent
(base/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
results=[]
for case in ['A','B']:
    # Driver owns creation of its case directory; log stays at base level.
    with (base/f'case_{case}.log').open('w') as log:
        process=subprocess.run([sys.executable,str(base/'test_driver.py'),'--case',case],stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:
        result=dict(case=case,state='failed',exit_code=process.returncode)
    else:
        result=json.loads((base/case/'run.json').read_text())
        result['exit_code']=process.returncode
        (base/f'case_{case}.log').rename(base/case/'solver.log')
    results.append(result)
    (base/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    print(case,result.get('solver_status',result['state']),result.get('summary'),flush=True)
if any(r.get('summary') for r in results):
    conclusion='Validated (3,64) found: lexicographic fairness optimal under current rules, using the prior global upper bound 64.'
elif all(r.get('solver_status')=='INFEASIBLE' for r in results):
    conclusion='Both exhaustive score-64 repeat cases are infeasible. Together with the prior global upper bound 64, the validated (3,63) baseline is lexicographically optimal under current rules.'
else:
    conclusion='No validated improvement found; at least one case remains unresolved or failed. Global fairness optimality remains unproven.'
lines=['# 17-player score-64 repeat cases','',conclusion,'',
       'Sequential searches, each with a 600-second limit, seed 73, eight workers, original model constraints and the saved fairness-63 assignment hint. No round-order symmetry or optional strengthening.','',
       '| Case | Early repeats | Late repeats | Status | Search seconds |','|---|---:|---:|---|---:|']
for r in results:
    lines.append(f"| {r['case']} | {r.get('early_repeat_appearances')} | {r.get('late_repeat_appearances')} | {r.get('solver_status',r['state'])} | {r.get('solver_elapsed_seconds')} |")
lines += ['', 'The counting analysis proves at least three late repeats. Minimum spread >=3 permits at most one repeat per player, so total 64 means four different repeat players: cases A and B exhaust the remaining possibility. Higher fair totals were already excluded by the global weighted objective bound 271 (weight 69) in runs/n15_to24_fair_sweep_20261002/n17/run.json. See the copied prior_bound.json and ../n17_counting_analysis_20261002.md.', '', 'Both known schedules were forced into the model before case thresholds and verified; repeat-variable sums matched independently computed CSV counts. The retained generator and saved baseline schedules are unchanged. Each case folder preserves model, log, statistics, metadata and any qualifying solution.']
(base/'README.md').write_text('\n'.join(lines)+'\n')
(base/'completion.json').write_text(json.dumps(dict(state='complete' if all(r['exit_code']==0 for r in results) else 'failed',conclusion=conclusion,run_count=len(results)),indent=2)+'\n')
if any(r['exit_code'] for r in results): raise SystemExit(1)
