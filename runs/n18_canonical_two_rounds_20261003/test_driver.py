import itertools
from collections import Counter
from pathlib import Path
from ortools.sat.python import cp_model
from model_snapshot import build_model
from solver_snapshot import build_model as reference_model
from helpers_snapshot import now, save, solver, read, extract, fix_integer, fix_reference, pair_counts
from metrics_snapshot import validate

BASE=Path(__file__).resolve().parent
SIZES=(3,3,4,4,4)
PERMS=[a+b for a in itertools.permutations(range(2)) for b in itertools.permutations(range(2,5))]
FIRST=[[16,17,18],[13,14,15],[9,10,11,12],[5,6,7,8],[1,2,3,4]]

def transform(m,p,t):
    return tuple(m[p[c]*5+p[r]] if t else m[p[r]*5+p[c]] for r in range(5) for c in range(5))

def matrix(schedule):
    return tuple(len(set(a)&set(b)) for a in schedule[0] for b in schedule[1])

def canonical(m):
    second=[[] for _ in range(5)]
    for r,players in enumerate(FIRST):
        columns=[c for c in range(5) if m[r*5+c]]
        assert len(columns)==len(players)
        for p,c in zip(players,columns):second[c].append(p)
    return [FIRST,second]

def normalize(source,m):
    wanted=canonical(m)
    for a,b in itertools.permutations(range(4),2):
        order=[a,b]+[r for r in range(4) if r not in (a,b)]
        for p in PERMS:
            variant=[[source[r][p[g]] for g in range(5)] for r in order]
            if matrix(variant)!=m:continue
            mapping={}
            for r in range(5):
                for c in range(5):
                    actual=set(variant[0][r])&set(variant[1][c])
                    desired=set(wanted[0][r])&set(wanted[1][c])
                    if actual:mapping[next(iter(actual))]=next(iter(desired))
            assert len(mapping)==18
            return [[[mapping[q] for q in group] for group in rounds] for rounds in variant]
    raise AssertionError('No qualifying baseline pair')

def main():
    patterns=[]
    zeros=tuple(5-s for s in SIZES)
    for rows in itertools.product(*[list(itertools.combinations(range(5),z)) for z in zeros]):
        if tuple(sum(c in row for row in rows) for c in range(5))!=zeros:continue
        patterns.append(tuple(int(c not in rows[r]) for r in range(5) for c in range(5)))
    reps={min(transform(m,p,t) for p in PERMS for t in (False,True)) for m in patterns}
    trace=lambda m:sum(m[g*5+g] for g in range(5))
    selected=sorted(m for m in reps if trace(m)<=1)
    assert len(patterns)==258 and len(reps)==26 and Counter(map(trace,selected))=={0:1,1:1}
    save(BASE/'enumeration.json',dict(pattern_count=len(patterns),class_count=len(reps),classes_by_trace=dict(Counter(map(trace,reps))),patterns_by_trace=dict(Counter(map(trace,patterns))),representatives=[list(m) for m in sorted(reps)]))
    source=read(BASE.parent/'n18_20261001_214154/fair/solution_n18_sol.csv')
    original,_=validate(source,18,list(SIZES))
    results=[]
    for m in selected:
        case=BASE/f'trace{trace(m)}';case.mkdir(exist_ok=True)
        hint=normalize(source,m);summary,_=validate(hint,18,list(SIZES));assert summary==original
        controls={}
        for kind in ('integer','retained'):
            if kind=='integer':
                data=build_model(18);model=data['model'];fix_integer(data,hint)
            else:
                model,_,X,_,_=reference_model(18);fix_reference(model,X,hint)
            search=solver();status=search.Solve(model);assert status==cp_model.OPTIMAL
            controls[kind]=search.StatusName(status)
        record=dict(started_at=now(),state='running',trace=trace(m),canonical_first_two_rounds=canonical(m),target_minimum_spread=3,target_total_spread_at_least=66,time_limit_seconds=1200,workers=8,seed=73,baseline_validation=summary,forced_baseline_controls=controls)
        save(case/'run.json',record)
        data=build_model(18);model=data['model'];fix_integer(data,canonical(m))
        for v in data['distinct'].values():model.Add(v>=3)
        model.Add(sum(data['distinct'].values())>=66)
        for r,groups in enumerate(hint):
            for g,players in enumerate(groups):
                for q in players:model.AddHint(data['slots'][r,q],g)
        assert not model.Validate()
        model.ExportToFile(str(case/'model.pbtxt'))
        search=solver(limit=1200,workers=8);search.parameters.log_search_progress=True
        print('START TRACE',trace(m),flush=True)
        status=search.Solve(model)
        record.update(state='complete',finished_at=now(),status=search.StatusName(status),elapsed_seconds=search.WallTime())
        (case/'statistics.txt').write_text(search.ResponseStats())
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            schedule=extract(data,search);summary,players=validate(schedule,18,list(SIZES))
            assert summary['minimum_distinct_slots']>=3 and summary['total_spread']>=66
            reference,_,X,_,_=reference_model(18);fix_reference(reference,X,schedule)
            assert solver().Solve(reference)==cp_model.OPTIMAL
            save(case/'solution.json',schedule);save(case/'players.json',players);record['summary']=summary
        save(case/'run.json',record);results.append(record)
        save(BASE/'results.json',results)
        print('RESULT TRACE',trace(m),record['status'],record['elapsed_seconds'],flush=True)
    print('ALL RESULTS',[(r['trace'],r['status']) for r in results],flush=True)

if __name__=='__main__':main()
