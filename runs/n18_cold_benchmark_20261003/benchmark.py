"""Local cold-start benchmark; no schedules or answer constraints enter solving."""
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def attempt(seed):
    started=time.monotonic()
    from solver_snapshot import build_model
    from ortools.sat.python import cp_model
    folder=BASE/f'seed_{seed}';folder.mkdir(exist_ok=True)
    model,search,X,sizes,rounds=build_model(18,fair_spread=True,strengthen=True,seed=seed,workers=8)
    assert not model.Validate()
    assert not model.Proto().solution_hint.vars
    search.parameters.max_time_in_seconds=max(0.01,59-time.monotonic()+started)
    with (folder/'solver.log').open('w') as log:
        search.parameters.log_to_stdout=False
        search.log_callback=lambda message:log.write(message+'\n')
        status=search.Solve(model)
    record=dict(seed=seed,status=search.StatusName(status),solver_seconds=search.WallTime(),search_limit_seconds=search.parameters.max_time_in_seconds,workers=8,hints=0,strengthen=True)
    (folder/'statistics.txt').write_text(search.ResponseStats())
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        schedule=[[[p for p in range(1,19) if search.Value(X[r,g,p])] for g in range(5)] for r in range(4)]
        with (folder/'solution.csv').open('w',newline='') as stream:
            writer=csv.writer(stream)
            for r,groups in enumerate(schedule,1):
                for g,players in enumerate(groups,1):writer.writerow([r,g]+players)
        # Independent validator intentionally runs only after the solver returns.
        import itertools
        from collections import Counter,defaultdict
        pairs=Counter();fours=Counter();slots=defaultdict(set)
        for groups in schedule:
            assert list(map(len,groups))==[3,3,4,4,4]
            assert sorted(p for group in groups for p in group)==list(range(1,19))
            for g,group in enumerate(groups):
                pairs.update(itertools.combinations(sorted(group),2))
                for p in group:slots[p].add(g);fours[p]+=int(len(group)==4)
        assert max(pairs.values())==1 and len(pairs)==96
        assert Counter(fours.values())=={2:6,3:12}
        minimum=min(map(len,slots.values()));total=sum(map(len,slots.values()))
        assert search.ObjectiveValue()==73*minimum+total
        record.update(validation='passed',minimum_spread=minimum,total_spread=total,objective=search.ObjectiveValue(),solver_bound=search.BestObjectiveBound(),attains_reference_optimum=(minimum,total)==(3,65))
    record['attempt_seconds']=time.monotonic()-started
    save(folder/'result.json',record)
    print(json.dumps(record),flush=True)

def main():
    import ortools
    save(BASE/'environment.json',dict(started_at=datetime.now(ZoneInfo('Australia/Sydney')).isoformat(),python=sys.version,ortools=ortools.__version__,platform=platform.platform(),architecture=platform.machine(),source_sha256=hashlib.sha256((BASE/'solver_snapshot.py').read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),git_head=subprocess.check_output(['/usr/bin/git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),deadline_seconds=60,seeds=[101,202,303,404,505],configuration='retained fair_spread + strengthen, first round fixed by player symmetry, no hints',limitations=['Local macOS measurements, not GroupMixer execution contract','Eight solver workers; CPUs not pinned; no 8 GiB process limit','Import/startup and output overhead included in external wall time; one second reserved after model preparation']))
    results=[]
    for seed in (101,202,303,404,505):
        folder=BASE/f'seed_{seed}';folder.mkdir(exist_ok=True)
        begin=time.monotonic()
        with (folder/'process.log').open('w') as log:
            try:
                proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),str(seed)],stdout=log,stderr=subprocess.STDOUT,timeout=60)
                path=folder/'result.json'
                record=json.loads(path.read_text()) if proc.returncode==0 and path.exists() else dict(seed=seed,status='ERROR',returncode=proc.returncode)
            except subprocess.TimeoutExpired:
                record=dict(seed=seed,status='EXTERNAL_DEADLINE',note='No result admitted; process exceeded end-to-end deadline.')
        record['external_seconds']=time.monotonic()-begin
        results.append(record);save(BASE/'results.json',results)
        print('COMPLETED',seed,record,flush=True)
    valid=[r for r in results if r.get('validation')=='passed']
    best=max(((r['minimum_spread'],r['total_spread']) for r in valid),default=None)
    scores=[r['objective'] for r in valid]
    summary=dict(attempts=5,valid=len(valid),reference_optimum_hits=sum(r.get('attains_reference_optimum',False) for r in valid),best=best,median_weighted_score=statistics.median(scores) if scores else None,solver_optimal_count=sum(r['status']=='OPTIMAL' for r in results),no_pending_runs=True)
    save(BASE/'summary.json',summary)
    (BASE/'README.md').write_text('# Local no-hint 18-player benchmark\n\n'+json.dumps(summary,indent=2)+'\n\nFive fresh processes, seeds 101/202/303/404/505, eight workers and a 60-second external deadline per attempt. Retained solver snapshot with minimum-first fairness and implied strengthening. No saved schedules, hints or target score constraints enter search; reference score is checked only after solving. Every admitted schedule was independently validated without repair.\n\nThis is local macOS evidence, not an execution of GroupMixer or a directly comparable published benchmark. CPU affinity and the 8 GiB process limit were not imposed. See environment.json, results.json, summary.json, source snapshot and per-attempt logs/results. All attempts have ended.\n')
    print('SUMMARY',summary,flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:attempt(int(sys.argv[1]))
    else:main()
