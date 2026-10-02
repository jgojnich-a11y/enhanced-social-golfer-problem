import csv,hashlib,json,platform,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from ortools.sat.python import cp_model
folder=Path(__file__).resolve().parent
root=folder.parents[1]
sys.path.insert(0,str(folder))
from solver_snapshot import build_model,load_hints_from_csv
sys.path.insert(0,str(root))
from metrics_snapshot import metrics
now=lambda:datetime.now(ZoneInfo('Australia/Sydney')).isoformat()
hint=folder/'hint.csv';shutil.copy2(root/'runs/n18_20261001_214154/fair/solution_n18_sol.csv',hint)
m,s,X,sizes,R=build_model(18,time_limit=1000,maximize_spread=True,seed=73,workers=8,strengthen=False)
# Use the snapshot's exact distinct-position variables, then remove optimization:
# any feasible schedule at or above the threshold answers this test.
proto=m.Proto()
counts=[m.GetIntVarFromProtoIndex(i) for i,v in enumerate(proto.variables) if v.name.startswith('distinct_pos_')]
assert len(counts)==18
m.Add(sum(counts)>=66)
for count in counts: m.Add(count>=3)
m.ClearObjective()
hints=load_hints_from_csv(str(hint),X)
for k,v in hints.items():m.AddHint(X[k],v)
assert not m.Validate(),m.Validate()
m.ExportToFile(str(folder/'model.pbtxt'))
d=dict(state='running',strengthen=False,started_at=now(),target_spread=66,target_minimum_spread=3,time_limit_seconds=1000,seed=73,workers=8,
 objective='feasibility only: minimum distinct tee slots >= 3 and total >= 66',python=sys.version,architecture=platform.machine(),
 command=[sys.executable,str(Path(__file__).resolve())],
 hint_sha256=hashlib.sha256(hint.read_bytes()).hexdigest(),
 solver_sha256=hashlib.sha256((folder/'solver_snapshot.py').read_bytes()).hexdigest(),hint_assignment_count=len(hints))
d['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
d['note']='Snapshot preserves uncommitted solver changes; git HEAD alone does not identify this source.'
d['metrics_sha256']=hashlib.sha256((folder/'metrics_snapshot.py').read_bytes()).hexdigest()
def save(): (folder/'run.json').write_text(json.dumps(d,indent=2)+'\n')
save()
(folder/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
status=s.Solve(m)
d.update(state='complete',finished_at=now(),solver_status=s.StatusName(status),solver_elapsed_seconds=s.WallTime())
(folder/'solver_statistics.txt').write_text(s.ResponseStats())
if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
 p=folder/'solution_n18_sol.csv'
 with p.open('w',newline='') as f:
  w=csv.writer(f)
  for r in range(R):
   for g in range(len(sizes)):
    w.writerow([r+1,g+1]+[player for player in range(1,19) if s.Value(X[r,g,player])])
 d['summary'],d['players']=metrics(p)
 assert d['summary']['total_spread']>=66
 assert d['summary']['minimum_distinct_slots']>=3
 d['solution_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 d['conclusion']='Found a schedule improving fairness to minimum spread at least 3 and total at least 66; this does not prove maximum fairness.'
elif status==cp_model.INFEASIBLE:
 d['conclusion']='Proved no schedule has minimum spread at least 3 and total at least 66 under this model; the saved (3,65) is optimal within minimum-spread-at-least-3 schedules.'
else:
 d['conclusion']='Unresolved: time limit or other stop without a feasible schedule or infeasibility proof.'
save()
(folder/'README.md').write_text('# 18-player fair-target-66 feasibility test\n\n'+d['conclusion']+'\n\nStatus: '+d['solver_status']+'; elapsed: '+str(d['solver_elapsed_seconds'])+' seconds.\n\nThis test removes the optimization objective and requires minimum spread >= 3 and total spread >= 66. All original constraints, including fixed first-round grouping, remain. Seed 73, eight workers, 1,000-second limit, saved six-hour fairness schedule hint. See run.json, model.pbtxt, solver.log and solver_snapshot.py for reproduction. An OPTIMAL status on this feasibility model means a valid threshold schedule was found; it does not establish maximum spread.\n')
print(d['conclusion'],flush=True)
