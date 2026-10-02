from pathlib import Path
import json, subprocess, sys
base=Path(__file__).resolve().parent
results=[]
for name in ['free_r2_r3','free_r2_r4','free_r3_r4']:
 folder=base/name
 with (folder/'solver.log').open('w') as log:
  process=subprocess.run([sys.executable,str(folder/'test_driver.py')],stdout=log,stderr=subprocess.STDOUT)
 if process.returncode:
  results.append(dict(neighborhood=name,state='failed',exit_code=process.returncode))
 else:
  result=json.loads((folder/'run.json').read_text())
  results.append(dict(neighborhood=name,exit_code=process.returncode,**result))
 (base/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
 print(name, results[-1].get('solver_status',results[-1]['state']),flush=True)
lines=['# Two-round neighborhood searches','', 'Three sequential searches; 400 seconds per search, seed 73, eight workers. Each starts from the saved (3,65) fairness schedule and requires minimum spread >=3 and total >=66. Round 1 and one later round are fixed to the source.','', '| Free rounds | Fixed rounds | Status | Seconds |','|---|---|---|---:|']
for r in results:
 lines.append(f"| {r.get('free_rounds')} | {r.get('fixed_rounds')} | {r.get('solver_status',r['state'])} | {r.get('solver_elapsed_seconds')} |")
lines += ['', 'INFEASIBLE rules out only the corresponding neighborhood, not global improvement. UNKNOWN leaves that neighborhood unresolved. See each subfolder for model, sources, hint, dependencies, log, validation and metadata.']
(base/'README.md').write_text('\n'.join(lines)+'\n')
if any(r['exit_code'] for r in results): raise SystemExit(1)
