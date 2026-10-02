import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from pathlib import Path

from ortools.sat.python import cp_model
from model_snapshot import build_model, add_round_pair_counts
from solver_snapshot import build_model as reference_model
from helpers_snapshot import now, save, solver, read, extract, fix_integer, fix_reference, pair_counts
from metrics_snapshot import validate

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
CANONICAL=[[[17,18,19],[13,14,15,16],[9,10,11,12],[5,6,7,8],[1,2,3,4]],
           [[5,9,13],[1,6,10,17],[2,7,14,18],[3,8,11,15],[4,12,16,19]]]


def matrix(first,second):
    return [[set(a)&set(b) for b in second] for a in first]


def permute(schedule,order,late_map):
    transformed=[]
    for r in order:
        groups=[[] for _ in range(5)]
        for g,players in enumerate(schedule[r]):
            groups[0 if g==0 else 1+late_map[g-1]]=list(players)
        transformed.append(groups)
    return transformed


def normalize(schedule):
    desired=matrix(*CANONICAL)
    for permutation in itertools.permutations(range(4)):
        transformed=permute(schedule,list(range(4)),permutation)
        actual=matrix(transformed[0],transformed[1])
        if [[len(cell) for cell in row] for row in actual] != [[len(cell) for cell in row] for row in desired]:
            continue
        mapping={}
        for r in range(5):
            for c in range(5):
                assert len(actual[r][c])<=1
                if actual[r][c]:mapping[next(iter(actual[r][c]))]=next(iter(desired[r][c]))
        assert sorted(mapping)==list(range(1,20)) and sorted(mapping.values())==list(range(1,20))
        result=[[[mapping[p] for p in players] for players in groups] for groups in transformed]
        assert all(set(result[r][g])==set(CANONICAL[r][g]) for r in range(2) for g in range(5))
        return result,dict(late_slot_mapping=[p+2 for p in permutation],player_mapping=mapping)
    raise AssertionError('No canonical representative found')


def write(path,schedule):
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream)
        for r,groups in enumerate(schedule,1):
            for g,players in enumerate(groups,1):writer.writerow([r,g]+sorted(players))


def checks():
    # Exhaust every admissible 4x4 late intersection pattern: three zeros,
    # in distinct rows/columns, exactly two of them diagonal.
    patterns={frozenset(cells) for cells in itertools.combinations(list(itertools.product(range(4),repeat=2)),3)
              if len({r for r,c in cells})==3 and len({c for r,c in cells})==3
              and sum(r==c for r,c in cells)==2}
    representative=frozenset([(0,0),(1,1),(2,3)])
    orbit={frozenset((p[r],p[c]) for r,c in representative) for p in itertools.permutations(range(4))}
    assert len(patterns)==12 and patterns==orbit
    source=read(BASE/'original_hint.csv')
    original_summary,_=validate(source,19,[3,4,4,4,4])
    selected=next(pair for pair in itertools.combinations(range(4),2) if pair_counts(source)[f'{pair[0]+1}-{pair[1]+1}']==2)
    order=list(selected)+[r for r in range(4) if r not in selected]
    ordered=permute(source,order,(0,1,2,3))
    # Produce a saved-schedule instance for each of the twelve structural
    # patterns, normalize it, and cross-check both solver representations.
    seen=set();checks=[]
    for permutation in itertools.permutations(range(4)):
        variant=permute(ordered,list(range(4)),permutation)
        intersections=matrix(variant[0],variant[1])
        zeros=frozenset((r,c) for r in range(4) for c in range(4) if not intersections[r+1][c+1])
        if zeros in seen:continue
        seen.add(zeros)
        normalized,info=normalize(variant)
        summary,_=validate(normalized,19,[3,4,4,4,4])
        assert summary==original_summary
        assert pair_counts(normalized)['1-2']==2
        data=build_model(19);fix_integer(data,normalized)
        assert solver().Solve(data['model'])==cp_model.OPTIMAL
        reference,_,X,_,_=reference_model(19);fix_reference(reference,X,normalized)
        assert solver().Solve(reference)==cp_model.OPTIMAL
        checks.append(dict(late_zero_pattern=sorted(zeros),summary=summary,
                           normalization=info,integer_forced_status='OPTIMAL',retained_forced_status='OPTIMAL'))
    assert seen==patterns
    hint,info=normalize(ordered)
    write(BASE/'hint.csv',hint)
    write(BASE/'canonical_first_two_rounds.csv',CANONICAL)
    result=dict(pattern_count=12,orbit_complete=True,original_round_order=[r+1 for r in order],
                hint_normalization=info,original_summary=original_summary,checks=checks)
    save(BASE/'normalization_validation.json',result)
    return result


def main():
    validation=checks()
    data=build_model(19);model=data['model']
    pairs=add_round_pair_counts(data)
    fix_integer(data,CANONICAL)
    model.Add(sum(data['distinct'].values())==69)
    for count in data['distinct'].values():model.Add(count>=3)
    for pair,count in pairs.items():model.Add(count==(2 if pair==(0,1) else 1))
    hint=read(BASE/'hint.csv')
    for r,groups in enumerate(hint):
        for g,players in enumerate(groups):
            for p in players:model.AddHint(data['slots'][r,p],g)
    assert not model.Validate(),model.Validate()
    model.ExportToFile(str(BASE/'model.pbtxt'))
    record=dict(state='running',started_at=now(),n=19,rounds=4,group_sizes=data['sizes'],
                target_minimum_spread=3,target_total_spread=69,fixed_rounds=[1,2],free_rounds=[3,4],
                fixed_exceptional_round_pair=[1,2],canonical_first_two_rounds=CANONICAL,
                time_limit_seconds=600,seed=73,workers=8,objective='feasibility only',integer_hint_count=76,
                hint_source='normalized historical (minimum 2,total 69) schedule; nonqualifying guidance',
                normalization_validation=validation,python=sys.version,architecture=platform.machine(),
                command=[sys.executable,str(Path(__file__).resolve())],
                hint_sha256=hashlib.sha256((BASE/'hint.csv').read_bytes()).hexdigest(),
                integer_source_sha256=hashlib.sha256((BASE/'model_snapshot.py').read_bytes()).hexdigest(),
                retained_source_sha256=hashlib.sha256((BASE/'solver_snapshot.py').read_bytes()).hexdigest(),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(BASE/'run.json',record)
    (BASE/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    search=solver(limit=600,workers=8);search.parameters.log_search_progress=True
    status=search.Solve(model)
    record.update(state='complete',finished_at=now(),solver_status=search.StatusName(status),solver_elapsed_seconds=search.WallTime())
    (BASE/'solver_statistics.txt').write_text(search.ResponseStats())
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        schedule=extract(data,search)
        summary,players=validate(schedule,19,data['sizes'])
        assert summary['minimum_distinct_slots']>=3 and summary['total_spread']==69
        assert pair_counts(schedule)=={'1-2':2,'1-3':1,'1-4':1,'2-3':1,'2-4':1,'3-4':1}
        assert all(set(schedule[r][g])==set(CANONICAL[r][g]) for r in range(2) for g in range(5))
        reference,_,X,_,_=reference_model(19,fair_spread=True);fix_reference(reference,X,schedule)
        verification=solver();assert verification.Solve(reference)==cp_model.OPTIMAL
        assert verification.ObjectiveValue()==300
        write(BASE/'solution_n19_sol.csv',schedule);save(BASE/'players.json',players)
        record.update(summary=summary,retained_model_validation='forced assignment OPTIMAL, objective 300',
                      conclusion='Validated (3,69) found; prior exclusion of total 70 establishes lexicographic fairness optimality.')
    elif status==cp_model.INFEASIBLE:
        record['conclusion']='Proved score 69 infeasible in the complete canonical representative. Together with the prior total-70 proof, the validated (3,68) baseline is lexicographically optimal under current rules.'
    else:
        record['conclusion']='Unresolved: no qualifying schedule or infeasibility proof. The fair optimum remains (3,68) or (3,69).'
    save(BASE/'run.json',record)
    (BASE/'README.md').write_text('# 19-player canonical first-two-round experiment\n\n'+record['conclusion']+f"\n\nStatus: {record['solver_status']}; search seconds: {record['solver_elapsed_seconds']:.3f}.\n\nMinimum spread >=3, total exactly 69, canonical rounds 1 and 2 fixed, rounds 3 and 4 free. Seed 73, eight workers, 600-second limit. The normalized old (2,69) schedule supplies 76 integer hints; it does not satisfy the fairness threshold.\n\nAll twelve admissible late-intersection patterns were enumerated and verified to lie in a single orbit. Saved schedule variants covering every pattern normalized to the same first two rounds, preserved validity and spread metrics, and passed forced-model checks in both representations before the target restrictions. See SYMMETRY.md, normalization_validation.json, canonical_first_two_rounds.csv, source snapshots, model.pbtxt, run.json and solver.log. Prior model checks and the total-70 proof are preserved. The retained solver and original schedules are unchanged. No run from this experiment is pending.\n")
    print(record['solver_status'],record['conclusion'],flush=True)


if __name__=='__main__':main()
