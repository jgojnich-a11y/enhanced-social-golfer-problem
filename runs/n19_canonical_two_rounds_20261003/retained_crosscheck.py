"""Reproduce the direct retained-model check without round-pair strengthening."""
import hashlib
import json
from pathlib import Path
from ortools.sat.python import cp_model
from solver_snapshot import build_model
from helpers_snapshot import fix_reference

base=Path(__file__).resolve().parent
record=json.loads((base/'run.json').read_text())
model,search,X,_,_=build_model(19,time_limit=30,fair_spread=True,seed=73,workers=8)
names={v.name:i for i,v in enumerate(model.Proto().variables)}
counts=[model.GetIntVarFromProtoIndex(names[f'distinct_pos_{p}']) for p in range(1,20)]
minimum=model.GetIntVarFromProtoIndex(names['minimum_spread'])
model.ClearObjective()
model.Add(sum(counts)==69)
model.Add(minimum>=3)
fix_reference(model,X,record['canonical_first_two_rounds'])
assert not model.Validate(),model.Validate()
model.ExportToFile(str(base/'retained_crosscheck_model.pbtxt'))
search.parameters.log_to_stdout=False
with (base/'retained_crosscheck.log').open('w') as log:
    search.log_callback=lambda message:log.write(message+'\n')
    status=search.Solve(model)
(base/'retained_crosscheck_statistics.txt').write_text(search.ResponseStats())
result=dict(solver_status=search.StatusName(status),solver_elapsed_seconds=search.WallTime(),
            time_limit_seconds=30,seed=73,workers=8,
            objective='feasibility only: minimum >=3 and total exactly 69',fixed_rounds=[1,2],
            round_pair_strengthening=False,
            retained_source_sha256=hashlib.sha256((base/'solver_snapshot.py').read_bytes()).hexdigest())
(base/'retained_crosscheck.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
assert status==cp_model.INFEASIBLE,search.StatusName(status)
