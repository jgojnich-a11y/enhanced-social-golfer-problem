from pathlib import Path
import subprocess,sys,json
base=Path(__file__).resolve().parent
(base/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
with (base/'solver.log').open('w') as log:
 result=subprocess.run([sys.executable,str(base/'test_driver.py')],stdout=log,stderr=subprocess.STDOUT)
if result.returncode: raise SystemExit(result.returncode)
(base/'solver.log').rename(base/'search/solver.log')
report=json.loads((base/'search/run.json').read_text())
(base/'completion.json').write_text(json.dumps(dict(state='complete',exit_code=result.returncode,solver_status=report['solver_status'],conclusion=report['conclusion']),indent=2)+'\n')
print(report['solver_status'],report['conclusion'],flush=True)
