#!/usr/bin/env python3
"""Wait for a documented comparison, then extend modes not proven optimal."""
import json
import subprocess
import sys
import time
from pathlib import Path

folder = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parents[1]
policy = folder/'extension_policy.json'
data = dict(initial_time_limit_seconds=3600, extension_time_limit_seconds=21600,
            trigger='Valid solution not proven OPTIMAL, or no solution found',
            state='waiting_for_initial_runs', warm_start=False,
            note='Extension starts a fresh search; both attempts are preserved.')
policy.write_text(json.dumps(data, indent=2)+'\n')
while not (folder/'comparison.json').exists():
    states=[]
    for mode in ('total','fair'):
        p=folder/mode/'run.json'
        if p.exists():
            states.append(json.loads(p.read_text()).get('state'))
    if len(states)==2 and all(s in ('failed','no_solution','complete') for s in states):
        time.sleep(2)
        if not (folder/'comparison.json').exists():
            data.update(state='failed',error='Initial runner finished without comparison.json')
            policy.write_text(json.dumps(data,indent=2)+'\n')
            sys.exit(1)
    time.sleep(10)
results=json.loads((folder/'comparison.json').read_text())
failed=[r['mode'] for r in results if r['state']=='failed']
modes=[r['mode'] for r in results if r['state']!='failed' and r.get('solver_status')!='OPTIMAL']
data.update(unresolved_modes=modes, failed_modes=failed)
if modes:
    data['state']='extension_running'
    policy.write_text(json.dumps(data,indent=2)+'\n')
    command=[sys.executable,str(root/'scripts/documented_comparison.py'),'--time-limit','21600','--modes',*modes]
    with (folder/'extension_driver.log').open('w') as log:
        code=subprocess.call(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    data.update(state='extension_complete' if code==0 else 'extension_failed',exit_code=code)
    for line in (folder/'extension_driver.log').read_text().splitlines():
        if line.startswith('Run folder: '):
            data['extension_folder']=line.removeprefix('Run folder: ')
else:
    data['state']='no_extension_needed' if not failed else 'initial_runs_failed'
policy.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2),flush=True)
