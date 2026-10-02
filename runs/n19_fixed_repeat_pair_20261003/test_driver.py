import argparse
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


def round_pair_repeats(schedule):
    result = {}
    for r,s in itertools.combinations(range(4),2):
        result[f'{r+1}-{s+1}'] = sum(len(set(schedule[r][g]) & set(schedule[s][g])) for g in range(1,5))
    return result


def normalize(schedule, pair):
    order = list(pair) + [r for r in range(4) if r not in pair]
    mapping = {}
    end = 19
    for group in schedule[order[0]]:
        canonical = list(range(end-len(group)+1,end+1))
        mapping.update(zip(sorted(group),canonical))
        end -= len(group)
    normalized = [[[mapping[p] for p in group] for group in schedule[r]] for r in order]
    return normalized,dict(original_round_order=[r+1 for r in order],player_mapping=mapping)


def write_schedule(path,schedule):
    with path.open('w',newline='') as stream:
        writer = csv.writer(stream)
        for r,groups in enumerate(schedule,1):
            for g,players in enumerate(groups,1): writer.writerow([r,g]+sorted(players))


def main():
    case = 'fixed_1_2'
    folder = BASE/'search'
    folder.mkdir()
    target = 69
    normalized_checks = []
    for label,path in [('fair68',BASE/'original_hint.csv'),('old69',ROOT/'solutions/opt_solution_n19_sol.csv')]:
        original = read(path)
        original_summary,_ = validate(original,19,[3,4,4,4,4])
        original_pairs = round_pair_repeats(original)
        for pair in itertools.combinations(range(4),2):
            normalized,info = normalize(original,pair)
            summary,_ = validate(normalized,19,[3,4,4,4,4])
            assert summary == original_summary
            assert round_pair_repeats(normalized)['1-2'] == original_pairs[f'{pair[0]+1}-{pair[1]+1}']
            destination = BASE/f'{label}_pair_{pair[0]+1}_{pair[1]+1}.csv'
            write_schedule(destination,normalized)
            normalized_checks.append((label,destination,info))
        # Align the most repeated pair in each baseline with rounds 1-2.
        selected = max(itertools.combinations(range(4),2),key=lambda pair:original_pairs[f'{pair[0]+1}-{pair[1]+1}'])
        chosen,info = normalize(original,selected)
        write_schedule(BASE/('hint.csv' if label=='fair68' else 'old_normalized.csv'),chosen)
        if label=='fair68': hint_normalization=info
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
    round_pair_counts = {}
    for r,s in itertools.combinations(range(rounds),2):
        same_slot = []
        for g in range(1,5):
            for p in range(1,20):
                common = model.NewBoolVar(f'late_repeat_r{r}_r{s}_g{g}_p{p}')
                # Exact AND: this player uses this late slot in both rounds.
                model.Add(common <= X[r,g,p])
                model.Add(common <= X[s,g,p])
                model.Add(common >= X[r,g,p] + X[s,g,p] - 1)
                same_slot.append(common)
        count = model.NewIntVar(0,19,f'late_repeat_pair_{r}_{s}')
        model.Add(count == sum(same_slot))
        # The disjoint triples leave 13 common four-group players in a 4x4
        # intersection array with only 12 off-diagonal cells.
        model.Add(count >= 1)
        round_pair_counts[r,s] = count
    model.ClearObjective()
    # Confirm the repeat definitions reproduce both known schedules before
    # imposing the case or fairness thresholds.
    baseline_checks = []
    for label,path,normalization in normalized_checks:
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
        independent_pairs = round_pair_repeats(schedule)
        modeled_pairs = {f'{r+1}-{s+1}': checker.Value(check.GetIntVarFromProtoIndex(v.Index())) for (r,s),v in round_pair_counts.items()}
        assert modeled_pairs == independent_pairs
        assert all(value >= 1 for value in independent_pairs.values())
        independent['round_pair_repeat_counts'] = independent_pairs
        baseline_checks.append(dict(source=label,normalization=normalization,summary=summary,repeat_summary=independent,
                                    forced_model_status=checker.StatusName(check_status)))
    model.Add(sum(counts)==target)
    for count in counts:
        model.Add(count>=3)
    model.Add(sum(v for (p,g),v in repeat_vars.items() if g<1) == early_target)
    model.Add(sum(v for (p,g),v in repeat_vars.items() if g>=1) == late_target)
    # Minimum spread three permits only one repeat per player, so each
    # repeat contributes to exactly one round-pair intersection.
    model.Add(sum(round_pair_counts.values()) == late_target)
    # Any candidate's unique double-repeat pair can be moved to rounds 1-2.
    # Globally relabeling players restores the canonical first-round partition.
    for pair,count in round_pair_counts.items():
        model.Add(count == (2 if pair==(0,1) else 1))
    hints = load_hints_from_csv(str(BASE/'hint.csv'),X)
    for key,value in hints.items():
        model.AddHint(X[key],value)
    assert not model.Validate(),model.Validate()
    model.ExportToFile(str(folder/'model.pbtxt'))
    record = dict(state='running',case=case,started_at=now(),players=19,rounds=4,
        group_sizes=sizes,target_minimum_spread=3,target_total_spread=target,
        early_repeat_appearances=early_target,late_repeat_appearances=late_target,
        time_limit_seconds=600,seed=73,workers=8,strengthen=False,
        round_order_symmetry=False,fixed_exceptional_round_pair=[1,2],hint_normalization=hint_normalization,round_pair_strengthening=True,
        round_pair_repeat_pattern=([1]*6 if target==70 else [1]*5+[2]),objective='feasibility only',
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
        independent_pairs = round_pair_repeats(schedule)
        assert independent_pairs == {'1-2':2,'1-3':1,'1-4':1,'2-3':1,'2-4':1,'3-4':1}
        repeat_summary['round_pair_repeat_counts'] = independent_pairs
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
                      conclusion='Found validated (3,69); combined with the prior total-70 infeasibility proof and counting bound, lexicographic fairness optimality is established.')
    elif status==cp_model.INFEASIBLE:
        record['conclusion']='Proved fair total 69 infeasible up to round permutation and player relabeling. Combined with the prior total-70 proof, the validated (3,68) baseline is lexicographically optimal under current rules.'
    else:
        record['conclusion']='Repeat case unresolved: no qualifying schedule or infeasibility proof within the limit.'
    save(folder/'run.json',record)
    (folder/'README.md').write_text(f"# 19-player fair-total-{target} test\n\n{record['conclusion']}\n\nStatus: {record['solver_status']}; search seconds: {record['solver_elapsed_seconds']:.3f}.\n\nRequires minimum spread >=3, total exactly {target}, early repeats {early_target} and late repeats {late_target}. Seed 73, eight workers, 600-second limit, fairness-68 assignment hint. Explicit pair counts: two for rounds 1-2 and one for each other pair. All twelve normalized baseline versions passed forced-model validation. The round and global player relabeling preserves every candidate up to symmetry under current identity-independent rules. These constraints are implied by the current rules and minimum spread three. The counting argument bounds fair total spread by 70; both 70 and 69 must be resolved to establish a lower optimum. See model.pbtxt, run.json and solver.log.\n")
    print(case,record['solver_status'],record.get('summary'),flush=True)


if __name__=='__main__':
    main()
