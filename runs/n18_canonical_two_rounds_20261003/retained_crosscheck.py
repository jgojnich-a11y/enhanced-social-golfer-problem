from pathlib import Path
from ortools.sat.python import cp_model
from solver_snapshot import build_model
from helpers_snapshot import save, solver, fix_reference, now
import json

BASE=Path(__file__).resolve().parent
results=[]
for trace in (0,1):
    case=BASE/f'trace{trace}'
    original=json.loads((case/'run.json').read_text())
    model,_,X,_,_=build_model(18,fair_spread=True)
    model.ClearObjective()
    fix_reference(model,X,original['canonical_first_two_rounds'])
    variables={v.name:model.GetIntVarFromProtoIndex(i) for i,v in enumerate(model.Proto().variables)}
    distinct=[variables[f'distinct_pos_{p}'] for p in range(1,19)]
    for v in distinct:model.Add(v>=3)
    model.Add(sum(distinct)>=66)
    assert not model.Validate()
    model.ExportToFile(str(case/'retained_model.pbtxt'))
    search=solver(limit=1200,workers=8);search.parameters.log_search_progress=True
    print('START RETAINED TRACE',trace,flush=True)
    status=search.Solve(model)
    record=dict(trace=trace,status=search.StatusName(status),elapsed_seconds=search.WallTime(),finished_at=now(),time_limit_seconds=1200,workers=8,seed=73)
    save(case/'retained_crosscheck.json',record)
    (case/'retained_statistics.txt').write_text(search.ResponseStats())
    results.append(record);save(BASE/'retained_results.json',results)
    print('RESULT RETAINED',record,flush=True)
