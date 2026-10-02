"""Experimental four-round model with one integer tee slot per player/round.

Retains the rules of cp_sat_caseB_v1_1.py. Pair meetings are reified slot
equalities per round, rather than indicators for each round and group.
"""
import itertools
from ortools.sat.python import cp_model


def group_distribution(n):
    triples = (4 - n % 4) % 4
    return [3]*triples + [4]*((n-3*triples)//4)


def build_model(n, fair_spread=False):
    model = cp_model.CpModel()
    sizes = group_distribution(n)
    assert sizes and sum(sizes) == n
    rounds = 4
    slots = {(r,p): model.NewIntVar(0,len(sizes)-1,f'slot_r{r}_p{p}')
             for r in range(rounds) for p in range(1,n+1)}
    membership = {}
    for r in range(rounds):
        for g,size in enumerate(sizes):
            for p in range(1,n+1):
                member = model.NewBoolVar(f'in_r{r}_g{g}_p{p}')
                model.Add(slots[r,p] == g).OnlyEnforceIf(member)
                model.Add(slots[r,p] != g).OnlyEnforceIf(member.Not())
                membership[r,g,p] = member
            model.Add(sum(membership[r,g,p] for p in range(1,n+1)) == size)

    meetings = {}
    for i,j in itertools.combinations(range(1,n+1),2):
        for r in range(rounds):
            meet = model.NewBoolVar(f'meet_{i}_{j}_r{r}')
            model.Add(slots[r,i] == slots[r,j]).OnlyEnforceIf(meet)
            model.Add(slots[r,i] != slots[r,j]).OnlyEnforceIf(meet.Not())
            meetings[i,j,r] = meet
        model.Add(sum(meetings[i,j,r] for r in range(rounds)) <= 1)

    four_groups = [g for g,size in enumerate(sizes) if size == 4]
    lower = rounds*4*len(four_groups)//n
    for p in range(1,n+1):
        participation = sum(membership[r,g,p] for r in range(rounds) for g in four_groups)
        # Exact capacities fix the number receiving lower+1 automatically.
        model.AddLinearConstraint(participation,lower,lower+1)

    end = n
    for g,size in enumerate(sizes):
        for p in range(end-size+1,end+1):
            model.Add(slots[0,p] == g)
        end -= size

    used = {}
    distinct = {}
    for p in range(1,n+1):
        for g in range(len(sizes)):
            used[p,g] = model.NewBoolVar(f'used_p{p}_g{g}')
            model.AddMaxEquality(used[p,g],[membership[r,g,p] for r in range(rounds)])
        distinct[p] = model.NewIntVar(0,min(rounds,len(sizes)),f'distinct_p{p}')
        model.Add(distinct[p] == sum(used[p,g] for g in range(len(sizes))))
    if fair_spread:
        minimum = model.NewIntVar(0,min(rounds,len(sizes)),'minimum_spread')
        model.AddMinEquality(minimum,list(distinct.values()))
        weight = n*min(rounds,len(sizes))+1
        model.Maximize(weight*minimum + sum(distinct.values()))
    return dict(model=model,n=n,rounds=rounds,sizes=sizes,slots=slots,
                membership=membership,meetings=meetings,distinct=distinct)


def add_round_pair_counts(data):
    """Count same-slot players across rounds; for n19 these are all late slots."""
    model = data['model']
    pair_counts = {}
    for r,s in itertools.combinations(range(data['rounds']),2):
        equalities = []
        for p in range(1,data['n']+1):
            equal = model.NewBoolVar(f'same_slot_r{r}_r{s}_p{p}')
            model.Add(data['slots'][r,p] == data['slots'][s,p]).OnlyEnforceIf(equal)
            model.Add(data['slots'][r,p] != data['slots'][s,p]).OnlyEnforceIf(equal.Not())
            equalities.append(equal)
        count = model.NewIntVar(0,data['n'],f'round_pair_count_{r}_{s}')
        model.Add(count == sum(equalities))
        pair_counts[r,s] = count
    data['round_pair_counts'] = pair_counts
    return pair_counts
