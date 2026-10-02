import csv,hashlib,json,platform,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from ortools.sat.python import cp_model
folder=Path(__file__).resolve().parent
root=folder.parents[2]
sys.path.insert(0,str(folder))
from solver_snapshot import build_model,load_hints_from_csv
sys.path.insert(0,str(root))
from metrics_snapshot import metrics
now=lambda:datetime.now(ZoneInfo('Australia/Sydney')).isoformat()
hint=folder/'hint.csv';shutil.copy2(root/'runs/n18_20261001_214154/fair/solution_n18_sol.csv',hint)
m,s,X,sizes,R=build_model(18,time_limit=400,maximize_spread=True,seed=73,workers=8,strengthen=False)
# Use the snapshot's exact distinct-position variables, then remove optimization:
# any feasible schedule at or above the threshold answers this test.
proto=m.Proto()
counts=[m.GetIntVarFromProtoIndex(i) for i,v in enumerate(proto.variables) if v.name.startswith('distinct_pos_')]
assert len(counts)==18
free_rounds = [2, 4]
fixed_rounds = [r for r in range(1, R+1) if r not in free_rounds]
baseline_summary, _ = metrics(hint)
assert (baseline_summary['minimum_distinct_slots'], baseline_summary['total_spread']) == (3,65)
hints = load_hints_from_csv(str(hint), X)
assert len(hints) == len(X) == 360
# Verify the source remains feasible before imposing improvement thresholds.
check = m.Clone()
check.ClearObjective()
for key, value in hints.items():
 check.Add(check.GetBoolVarFromProtoIndex(X[key].Index()) == value)
checker = cp_model.CpSolver()
checker.parameters.max_time_in_seconds = 10
checker.parameters.num_search_workers = 1
check_status = checker.Solve(check)
assert check_status in (cp_model.FEASIBLE, cp_model.OPTIMAL), checker.StatusName(check_status)
for (r,g,p), value in hints.items():
 if r+1 in fixed_rounds: m.Add(X[r,g,p] == value)
m.Add(sum(counts)>=66)
for count in counts: m.Add(count>=3)
m.ClearObjective()
hints=load_hints_from_csv(str(hint),X)
for k,v in hints.items():m.AddHint(X[k],v)
assert not m.Validate(),m.Validate()
m.ExportToFile(str(folder/'model.pbtxt'))
d=dict(state='running',strengthen=False,started_at=now(),target_spread=66,target_minimum_spread=3,time_limit_seconds=400,seed=73,workers=8,
 objective='feasibility only: minimum distinct tee slots >= 3 and total >= 66',python=sys.version,architecture=platform.machine(),
 command=[sys.executable,str(Path(__file__).resolve())],
 hint_sha256=hashlib.sha256(hint.read_bytes()).hexdigest(),
 solver_sha256=hashlib.sha256((folder/'solver_snapshot.py').read_bytes()).hexdigest(),hint_assignment_count=len(hints))
d['free_rounds']=free_rounds
d['fixed_rounds']=fixed_rounds
d['baseline_validation']=dict(summary=baseline_summary,forced_model_status=checker.StatusName(check_status))
d['restricted_neighborhood']=True
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
 assert all(s.Value(X[key]) == value for key,value in hints.items() if key[0]+1 in fixed_rounds)
 d['solution_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 d['conclusion']='Found a schedule improving fairness to minimum spread at least 3 and total at least 66; this does not prove maximum fairness.'
elif status==cp_model.INFEASIBLE:
 d['conclusion']='Proved no qualifying improvement exists in this restricted neighborhood. This does not establish global optimality.'
else:
 d['conclusion']='Unresolved: time limit or other stop without a feasible schedule or infeasibility proof.'
save()
(folder/'README.md').write_text('# 18-player fair-target-66 restricted neighborhood test\n\n'+d['conclusion']+'\n\nStatus: '+d['solver_status']+'; elapsed: '+str(d['solver_elapsed_seconds'])+' seconds.\n\nThis test removes the optimization objective and requires minimum spread >= 3 and total spread >= 66. All original constraints remain. Additional complete-round assignments restrict this neighborhood; see fixed_rounds and free_rounds in run.json. INFEASIBLE applies only to this neighborhood. Seed 73, eight workers, 400-second limit, saved six-hour fairness schedule hint. See run.json, model.pbtxt, solver.log and solver_snapshot.py for reproduction. An OPTIMAL status on this feasibility model means a valid threshold schedule was found; it does not establish maximum spread.\n')
print(d['conclusion'],flush=True)
