import sys,subprocess,json,hashlib,platform,shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path.cwd();folder=root/'runs/n18_hint_1000s_20261002'
sys.path.insert(0,str(root))
from scripts.documented_comparison import metrics
hint=folder/'hint.csv';shutil.copy2(root/'solutions/solution_n18_sol.csv',hint)
command=[sys.executable,str(folder/'solver_snapshot.py'),'--n','18','--time_limit','1000','--maximize_spread','--seed','73','--workers','8','--input_csv',str(hint),'--out_prefix',str(folder/'solution')]
now=lambda:datetime.now(ZoneInfo('Australia/Sydney')).isoformat()
d=dict(state='running',started_at=now(),command=command,python=sys.version,architecture=platform.machine(),hint_sha256=hashlib.sha256(hint.read_bytes()).hexdigest(),solver_sha256=hashlib.sha256((folder/'solver_snapshot.py').read_bytes()).hexdigest())
def save(): (folder/'run.json').write_text(json.dumps(d,indent=2)+'\n')
save()
(folder/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
with (folder/'solver.log').open('w') as f: result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
d.update(state='complete' if result.returncode==0 else 'failed',finished_at=now(),exit_code=result.returncode)
p=folder/'solution_n18_sol.csv'
if p.exists():d['summary'],d['players']=metrics(p)
log=(folder/'solver.log').read_text()
d['solver_statistics']=[line for line in log.splitlines() if line.startswith(('Solver status:','objective:','best_bound:'))]
save()
(folder/'README.md').write_text('# 18-player hinted run: 1,000 seconds\n\nOriginal total-spread objective; seed 73; eight workers.\n\n'+'\n\n'.join(d['solver_statistics'])+'\n\nSee run.json for command, timestamps, source checksums, validation and individual metrics.\n')
print(json.dumps(d.get('summary',{})),flush=True)
print('\n'.join(d['solver_statistics']),flush=True)
