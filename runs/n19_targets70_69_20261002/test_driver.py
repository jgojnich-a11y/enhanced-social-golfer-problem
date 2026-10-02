import argparse
import collections
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ortools.sat.python import cp_model
from solver_snapshot import build_model, load_hints_from_csv
from metrics_snapshot import validate

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def now():
    return datetime.now(ZoneInfo('Australia/Sydney')).isoformat()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def read(path):
    groups = collections.defaultdict(dict)
    with path.open() as stream:
        for row in csv.reader(stream):
            r,g,*players = map(int,row)
            assert g not in groups[r]
            groups[r][g] = players
    assert set(groups) == {1,2,3,4}
    assert all(set(gs) == {1,2,3,4,5} for gs in groups.values())
    return [[groups[r][g] for g in range(1,6)] for r in range(1,5)]


def repeats(schedule):
    visits = collections.defaultdict(list)
    for groups in schedule:
        for g,group in enumerate(groups):
            for p in group:
                visits[p].append(g)
    early = {}
    late = {}
    for p,slots in visits.items():
        a = [g for g in slots if g < 1]
        b = [g for g in slots if g >= 1]
        early[p] = len(a)-len(set(a))
        late[p] = len(b)-len(set(b))
    return dict(early_repeat_appearances=sum(early.values()),
                late_repeat_appearances=sum(late.values()),
                early_repeat_players=sorted(p for p,count in early.items() if count),
                late_repeat_players=sorted(p for p,count in late.items() if count),
                overlapping_repeat_players=sorted(p for p in visits if early[p] and late[p]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['A','B'], required=True)
    case = parser.parse_args().case
    folder = BASE/case
    folder.mkdir()
    target = {'A':70,'B':69}[case]
    early_target,late_target = 0,76-target
    model,solver,X,sizes,rounds = build_model(19,time_limit=600,maximize_spread=True,seed=73,workers=8)
    names = {v.name:i for i,v in enumerate(model.Proto().variables)}
    counts = [model.GetIntVarFromProtoIndex(names[f'distinct_pos_{p}']) for p in range(1,20)]
    repeat_vars = {}
    for p in range(1,20):
        for g in range(5):
            used = model.GetBoolVarFromProtoIndex(names[f'posUsed_p{p}_g{g}'])
            repeat_vars[p,g] = model.NewIntVar(0,3,f'repeat_p{p}_g{g}')
            model.Add(repeat_vars[p,g] == sum(X[r,g,p] for r in range(rounds))-used)
    model.ClearObjective()
    # Confirm the repeat definitions reproduce both known schedules before
    # imposing the case or fairness thresholds.
    baseline_checks = []
    for label,path in [('fair68',BASE/'hint.csv'),('old69',ROOT/'solutions/opt_solution_n19_sol.csv')]:
        schedule = read(path)
        summary,_ = validate(schedule,19,sizes)
        check = model.Clone()
        assignments = load_hints_from_csv(str(path),X)
        assert len(assignments) == len(X) == 380
        for key,value in assignments.items():
            check.Add(check.GetBoolVarFromProtoIndex(X[key].Index()) == value)
        checker = cp_model.CpSolver()
        checker.parameters.max_time_in_seconds = 10
        checker.parameters.num_search_workers = 1
        check_status = checker.Solve(check)
        assert check_status in (cp_model.FEASIBLE,cp_model.OPTIMAL),checker.StatusName(check_status)
        independent = repeats(schedule)
        modeled_early = sum(checker.Value(check.GetIntVarFromProtoIndex(v.Index())) for (p,g),v in repeat_vars.items() if g<1)
        modeled_late = sum(checker.Value(check.GetIntVarFromProtoIndex(v.Index())) for (p,g),v in repeat_vars.items() if g>=1)
        assert (modeled_early,modeled_late) == (independent['early_repeat_appearances'],independent['late_repeat_appearances'])
        baseline_checks.append(dict(source=label,summary=summary,repeat_summary=independent,
                                    forced_model_status=checker.StatusName(check_status)))
    model.Add(sum(counts)==target)
    for count in counts:
        model.Add(count>=3)
    model.Add(sum(v for (p,g),v in repeat_vars.items() if g<1) == early_target)
    model.Add(sum(v for (p,g),v in repeat_vars.items() if g>=1) == late_target)
    hints = load_hints_from_csv(str(BASE/'hint.csv'),X)
    for key,value in hints.items():
        model.AddHint(X[key],value)
    assert not model.Validate(),model.Validate()
    model.ExportToFile(str(folder/'model.pbtxt'))
    record = dict(state='running',case=case,started_at=now(),players=19,rounds=4,
        group_sizes=sizes,target_minimum_spread=3,target_total_spread=target,
        early_repeat_appearances=early_target,late_repeat_appearances=late_target,
        time_limit_seconds=600,seed=73,workers=8,strengthen=False,
        round_order_symmetry=False,objective='feasibility only',
        hint_assignment_count=len(hints),baseline_validation=baseline_checks,
        python=sys.version,architecture=platform.machine(),command=[sys.executable,str(Path(__file__).resolve()),'--case',case],
        hint_sha256=hashlib.sha256((BASE/'hint.csv').read_bytes()).hexdigest(),
        solver_sha256=hashlib.sha256((BASE/'solver_snapshot.py').read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        note='Source snapshot includes uncommitted changes; HEAD alone does not identify source.')
    save(folder/'run.json',record)
    status = solver.Solve(model)
    record.update(state='complete',finished_at=now(),solver_status=solver.StatusName(status),solver_elapsed_seconds=solver.WallTime())
    (folder/'solver_statistics.txt').write_text(solver.ResponseStats())
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        schedule = [[[p for p in range(1,20) if solver.Value(X[r,g,p])]
                     for g in range(5)] for r in range(rounds)]
        summary,players = validate(schedule,19,sizes)
        repeat_summary = repeats(schedule)
        assert summary['minimum_distinct_slots']>=3 and summary['total_spread']==target
        assert repeat_summary['early_repeat_appearances']==early_target
        assert repeat_summary['late_repeat_appearances']==late_target
        assert not repeat_summary['overlapping_repeat_players']
        assert len(set(repeat_summary['early_repeat_players']+repeat_summary['late_repeat_players']))==late_target
        solution = folder/'solution_n19_sol.csv'
        with solution.open('w',newline='') as stream:
            writer = csv.writer(stream)
            for r,groups in enumerate(schedule,1):
                for g,group in enumerate(groups,1): writer.writerow([r,g]+group)
        save(folder/'players.json',players)
        record.update(summary=summary,repeat_summary=repeat_summary,
                      solution_sha256=hashlib.sha256(solution.read_bytes()).hexdigest(),
                      conclusion=f'Found validated fairness (3,{target}). Total 70 proves optimality by the counting bound; total 69 is optimal only if total 70 is excluded.')
    elif status==cp_model.INFEASIBLE:
        record['conclusion']=f'Proved fair total {target} infeasible. Combine both target statuses with the mathematical upper bound 70 to assess global optimality.'
    else:
        record['conclusion']='Repeat case unresolved: no qualifying schedule or infeasibility proof within the limit.'
    save(folder/'run.json',record)
    (folder/'README.md').write_text(f"# 19-player fair-total-{target} test\n\n{record['conclusion']}\n\nStatus: {record['solver_status']}; search seconds: {record['solver_elapsed_seconds']:.3f}.\n\nRequires minimum spread >=3, total exactly {target}, early repeats {early_target} and late repeats {late_target}. Seed 73, eight workers, 600-second limit, fairness-68 assignment hint. The counting argument bounds fair total spread by 70; both 70 and 69 must be resolved to establish a lower optimum. See model.pbtxt, run.json and solver.log.\n")
    print(case,record['solver_status'],record.get('summary'),flush=True)


if __name__=='__main__':
    main()
