from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[1]
base=root/'runs/n18_target67_audit_20261002'
def run(mode):
 folder=base/mode
 with (folder/'solver.log').open('w') as log:
  result=subprocess.run([sys.executable,str(folder/'test_driver.py')],stdout=log,stderr=subprocess.STDOUT,cwd=root)
 print(mode,'exit',result.returncode,flush=True)
 return dict(mode=mode,exit_code=result.returncode)
with ThreadPoolExecutor(max_workers=2) as executor:
 results=list(executor.map(run,['repaired_hints','strengthened']))
(base/'completion.json').write_text(json.dumps(results,indent=2)+'\n')
