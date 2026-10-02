import collections
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ortools.sat.python import cp_model
from model_snapshot import build_model, add_round_pair_counts
from solver_snapshot import build_model as reference_model, group_distribution
from metrics_snapshot import validate

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def now():
    return datetime.now(ZoneInfo('Australia/Sydney')).isoformat()


def save(path,value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def solver(limit=10,workers=1):
    result = cp_model.CpSolver()
    result.parameters.max_time_in_seconds = limit
    result.parameters.num_search_workers = workers
    result.parameters.random_seed = 73
    result.parameters.cp_model_probing_level = 1
    return result


def read(path):
    groups = collections.defaultdict(dict)
    with path.open() as stream:
        for row in csv.reader(stream):
            r,g,*players = map(int,row)
            assert g not in groups[r]
            groups[r][g] = players
    assert set(groups) == {1,2,3,4}
    return [[groups[r][g] for g in sorted(groups[r])] for r in range(1,5)]


def extract(data,search):
    return [[[p for p in range(1,data['n']+1) if search.Value(data['slots'][r,p])==g]
             for g in range(len(data['sizes']))] for r in range(4)]


def fix_integer(data,schedule):
    for r,groups in enumerate(schedule):
        for g,players in enumerate(groups):
            for p in players:
                data['model'].Add(data['slots'][r,p] == g)


def fix_reference(model,X,schedule):
    for r,groups in enumerate(schedule):
        for g,players in enumerate(groups):
            for p in range(1,max(p for group in groups for p in group)+1):
                model.Add(X[r,g,p] == int(p in players))


def pair_counts(schedule):
    return {f'{r+1}-{s+1}':sum(len(set(schedule[r][g]) & set(schedule[s][g]))
                             for g in range(len(schedule[r])))
            for r,s in itertools.combinations(range(4),2)}


def checks():
    results=[]
    for n in [8,9,12]:
        candidate=build_model(n,fair_spread=True)
        reference,rs,X,sizes,_=reference_model(n,time_limit=10,fair_spread=True,seed=73,workers=1)
        rs.parameters.log_search_progress=False
        assert candidate['sizes']==sizes==group_distribution(n)
        assert not candidate['model'].Validate()
        cs=solver()
        cstatus=cs.Solve(candidate['model'])
        rstatus=rs.Solve(reference)
        assert cstatus==rstatus and cstatus in (cp_model.OPTIMAL,cp_model.INFEASIBLE),(n,cs.StatusName(cstatus),rs.StatusName(rstatus))
        result=dict(n=n,integer_status=cs.StatusName(cstatus),reference_status=rs.StatusName(rstatus))
        if cstatus==cp_model.OPTIMAL:
            assert cs.ObjectiveValue()==rs.ObjectiveValue()
            result['objective']=cs.ObjectiveValue()
            cschedule=extract(candidate,cs)
            rschedule=[[[p for p in range(1,n+1) if rs.Value(X[r,g,p])]
                        for g in range(len(sizes))] for r in range(4)]
            validate(cschedule,n,sizes);validate(rschedule,n,sizes)
            # Cross-force each solver's result in the other representation.
            icheck=build_model(n);fix_integer(icheck,rschedule)
            assert solver().Solve(icheck['model'])==cp_model.OPTIMAL
            rcheck,_,rX,_,_=reference_model(n)
            fix_reference(rcheck,rX,cschedule)
            assert solver().Solve(rcheck)==cp_model.OPTIMAL
            # Repeating round 1 as round 2 preserves coverage/capacities,
            # but violates unique partners. Both encodings must reject it.
            mutant=[list(map(list,groups)) for groups in cschedule]
            mutant[1]=list(map(list,mutant[0]))
            bad=build_model(n);fix_integer(bad,mutant)
            assert solver().Solve(bad['model'])==cp_model.INFEASIBLE
            badref,_,bX,_,_=reference_model(n)
            fix_reference(badref,bX,mutant)
            assert solver().Solve(badref)==cp_model.INFEASIBLE
            result.update(cross_forced_checks='passed',repeated_partner_mutation='rejected by both')
        results.append(result)
    saved=[(17,ROOT/'runs/n15_to24_fair_sweep_20261002/n17/solution_n17_sol.csv'),
           (18,ROOT/'runs/n18_20261001_214154/fair/solution_n18_sol.csv'),
           (19,BASE/'hint.csv'),(19,ROOT/'solutions/opt_solution_n19_sol.csv')]
    for n,path in saved:
        schedule=read(path)
        data=build_model(n)
        summary,_=validate(schedule,n,data['sizes'])
        counts=add_round_pair_counts(data)
        fix_integer(data,schedule)
        cs=solver();status=cs.Solve(data['model'])
        assert status==cp_model.OPTIMAL,cs.StatusName(status)
        assert all(cs.Value(data['distinct'][p])==len({g for groups in schedule for g,players in enumerate(groups) if p in players}) for p in range(1,n+1))
        independent=pair_counts(schedule)
        assert {f'{r+1}-{s+1}':cs.Value(v) for (r,s),v in counts.items()}==independent
        reference,_,X,_,_=reference_model(n)
        fix_reference(reference,X,schedule)
        assert solver().Solve(reference)==cp_model.OPTIMAL
        results.append(dict(n=n,source=str(path.relative_to(ROOT)),summary=summary,
                            integer_forced_status=cs.StatusName(status),reference_forced_status='OPTIMAL',
                            round_pair_counts=independent))
    save(BASE/'validation.json',results)
    print('Equivalence cross-checks and saved-schedule validation passed.',flush=True)
    return results


def main():
    validation=checks()
    data=build_model(19)
    model=data['model']
    round_pairs=add_round_pair_counts(data)
    model.Add(sum(data['distinct'].values())==69)
    for count in data['distinct'].values():model.Add(count>=3)
    for pair,count in round_pairs.items():model.Add(count==(2 if pair==(0,1) else 1))
    schedule=read(BASE/'hint.csv')
    validate(schedule,19,data['sizes'])
    for r,groups in enumerate(schedule):
        for g,players in enumerate(groups):
            for p in players:model.AddHint(data['slots'][r,p],g)
    assert not model.Validate(),model.Validate()
    model.ExportToFile(str(BASE/'model.pbtxt'))
    record=dict(state='running',started_at=now(),n=19,rounds=4,group_sizes=data['sizes'],
                target_minimum_spread=3,target_total_spread=69,fixed_exceptional_round_pair=[1,2],
                time_limit_seconds=600,seed=73,workers=8,integer_hint_count=76,
                player_round_slot_variables=len(data['slots']),pair_meeting_indicators=len(data['meetings']),
                retained_pair_meeting_indicators=4*len(data['sizes'])*19*18//2,
                model_variable_count=len(model.Proto().variables),objective='feasibility only',
                validation=validation,python=sys.version,architecture=platform.machine(),
                command=[sys.executable,str(Path(__file__).resolve())],
                hint_sha256=hashlib.sha256((BASE/'hint.csv').read_bytes()).hexdigest(),
                integer_source_sha256=hashlib.sha256((BASE/'model_snapshot.py').read_bytes()).hexdigest(),
                retained_source_sha256=hashlib.sha256((BASE/'solver_snapshot.py').read_bytes()).hexdigest(),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                note='Snapshots preserve uncommitted code; HEAD alone does not identify source.')
    save(BASE/'run.json',record)
    (BASE/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    search=solver(limit=600,workers=8)
    search.parameters.log_search_progress=True
    status=search.Solve(model)
    record.update(state='complete',finished_at=now(),solver_status=search.StatusName(status),solver_elapsed_seconds=search.WallTime())
    (BASE/'solver_statistics.txt').write_text(search.ResponseStats())
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        schedule=extract(data,search)
        summary,players=validate(schedule,19,data['sizes'])
        independent=pair_counts(schedule)
        assert summary['minimum_distinct_slots']>=3 and summary['total_spread']==69
        assert independent=={'1-2':2,'1-3':1,'1-4':1,'2-3':1,'2-4':1,'3-4':1}
        # A qualifying integer-model solution must also satisfy the retained model.
        reference,_,X,_,_=reference_model(19,fair_spread=True)
        fix_reference(reference,X,schedule)
        verification=solver();verify_status=verification.Solve(reference)
        assert verify_status==cp_model.OPTIMAL,verification.StatusName(verify_status)
        assert verification.ObjectiveValue()==300
        path=BASE/'solution_n19_sol.csv'
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream)
            for r,groups in enumerate(schedule,1):
                for g,group in enumerate(groups,1):writer.writerow([r,g]+group)
        save(BASE/'players.json',players)
        record.update(summary=summary,round_pair_counts=independent,
                      retained_model_validation='OPTIMAL on forced assignment (weighted score 300)',
                      solution_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      conclusion='Validated (3,69) found and cross-checked in the retained model; prior exclusion of total 70 establishes lexicographic fairness optimality.')
    elif status==cp_model.INFEASIBLE:
        record['conclusion']='Score 69 proved infeasible in the equivalent integer model with valid round-pair symmetry. Together with the prior total-70 exclusion, the validated (3,68) schedule is lexicographically optimal under current rules.'
    else:
        record['conclusion']='Unresolved: no qualifying schedule or infeasibility proof. Best known (3,68), with (3,69) the sole remaining improvement.'
    save(BASE/'run.json',record)
    (BASE/'README.md').write_text('# 19-player integer tee-slot experiment\n\n'+record['conclusion']+f"\n\nStatus: {record['solver_status']}; elapsed: {record['solver_elapsed_seconds']:.3f} seconds.\n\nMinimum spread >=3, total exactly 69, exceptional repeat pair fixed to rounds 1-2. Seed 73, eight workers, 600-second limit, 76 integer hints representing the same normalized schedule as the prior fixed-pair test. Pair-meeting indicators: 684 rather than 3,420. This reduction is a representation change, not a guarantee of faster solving.\n\nSee validation.json for small-instance status/objective comparisons, bidirectional forced-schedule checks and rejection of a repeated-partner mutation, plus both encodings accepting saved schedules. See MODEL.md and SYMMETRY.md for equivalence and symmetry arguments, and run.json, model.pbtxt, solver.log and solver_statistics.txt for the audit. Source snapshots, dependency list and prior total-70 report are preserved. The retained generator and original schedules are unchanged. No run from this experiment is pending.\n")
    print(record['solver_status'],record['conclusion'],flush=True)


if __name__=='__main__':main()
