"""Validate and curate the proven four-round fairness schedules; preserve originals."""
import csv
import hashlib
import itertools
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOL=ROOT/'solutions'
ARCHIVE=SOL/'archive/2026-10-03'
OUT=SOL/'optimal'
EXPECTED={15:(2,45),16:(2,44),17:(3,63),18:(3,65),19:(3,68),**{n:(4,4*n) for n in range(20,25)}}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def validate(path,n):
    triples=(4-n%4)%4;sizes=[3]*triples+[4]*((n-3*triples)//4)
    rounds={r:{} for r in range(1,5)};pairs=Counter();slots={p:[] for p in range(1,n+1)};fours=Counter()
    with path.open(newline='') as stream:
        for row in csv.reader(stream):
            r,g,*players=map(int,row)
            assert r in rounds and 1<=g<=len(sizes) and g not in rounds[r],(path,row)
            assert len(players)==sizes[g-1] and len(set(players))==len(players)
            assert all(1<=p<=n for p in players)
            rounds[r][g]=players
            pairs.update(itertools.combinations(sorted(players),2))
            for p in players:slots[p].append(g);fours[p]+=int(len(players)==4)
    for groups in rounds.values():
        assert sorted(groups)==list(range(1,len(sizes)+1))
        assert sorted(p for group in groups.values() for p in group)==list(range(1,n+1))
    assert max(pairs.values())==1
    appearances=4*sum(s for s in sizes if s==4);lower,remainder=divmod(appearances,n)
    assert all(lower<=fours[p]<=lower+int(remainder>0) for p in slots)
    assert sum(fours[p]==lower+1 for p in slots)==remainder
    score=(min(len(set(s)) for s in slots.values()),sum(len(set(s)) for s in slots.values()))
    assert score==EXPECTED[n],(path,score)
    return rounds,slots,fours,sizes,score

def time(g):
    minutes=600+7*(g-1);return f'{minutes//60:02d}:{minutes%60:02d}'

def main():
    assert not ARCHIVE.exists() and not OUT.exists(),'Collection already created; review before rerunning.'
    # Validate every chosen source before moving any original file.
    sources={n:ROOT/f'runs/n15_to24_fair_sweep_20261002/n{n}/solution_n{n}_sol.csv' for n in EXPECTED}
    checked={n:validate(p,n) for n,p in sources.items()}
    originals=sorted(p for p in SOL.iterdir() if p.is_file())
    manifest=[dict(original_path=str(p.relative_to(ROOT)),archive_path=str((ARCHIVE/p.name).relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in originals]
    ARCHIVE.mkdir(parents=True);OUT.mkdir()
    for p in originals:shutil.move(str(p),str(ARCHIVE/p.name))
    for item in manifest:assert digest(ROOT/item['archive_path'])==item['sha256']
    (ARCHIVE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ARCHIVE/'README.md').write_text('# Original solutions archive — 3 October 2026\n\nAll '+str(len(manifest))+' files previously at the top level of `solutions/` were moved here without changing their contents. `manifest.json` records original paths, archived paths, sizes and SHA-256 checksums; every checksum was verified after moving. Historical names and original solver formats are preserved. These include older objectives, experimental outputs and schedules outside the current four-round 15–24-player scope. Historical logs elsewhere retain their original path references; resolve those paths through this manifest.\n')
    index=['# Optimal fairness schedules — 3 October 2026','','Four rounds; unique partners; balanced three/four-player participation. Tee times start at 10:00, seven minutes apart, with three-player groups before four-player groups. Fairness first maximizes each player’s minimum number of distinct tee slots, then the sum across players. Every selected schedule was independently revalidated from its CSV. Average tee time is not an optimization objective.','','| Players | Minimum distinct slots | Total spread | Schedule | Tee times | Proof |','|---:|---:|---:|---|---|---|']
    records=[]
    for n,source in sources.items():
        rounds,slots,fours,sizes,score=checked[n]
        dest=OUT/f'n{n}_sol.csv';shutil.copy2(source,dest);assert digest(source)==digest(dest)
        validate(dest,n)
        if n==17:proof='runs/n17_target64_repeat_cases_20261002/README.md'
        elif n in (18,19):proof=f'runs/n{n}_canonical_two_rounds_20261003/README.md'
        else:proof=f'runs/n15_to24_fair_sweep_20261002/n{n}/run.json'
        md=[f'# {n} players — optimal fairness schedule','',f'Minimum distinct tee slots: **{score[0]}**. Total spread: **{score[1]}**. Four rounds; no repeated partners. Group sizes: '+', '.join(map(str,sizes))+'.','','| Group | Tee time | Round 1 players | Round 2 players | Round 3 players | Round 4 players |','|---:|---|---|---|---|---|']
        for g in range(1,len(sizes)+1):md.append('| '+str(g)+' | '+time(g)+' | '+' | '.join(', '.join(map(str,sorted(rounds[r][g]))) for r in range(1,5))+' |')
        md+=['','| Player | Round 1 | Round 2 | Round 3 | Round 4 | Distinct slots | Three-player rounds | Four-player rounds |','|---:|---|---|---|---|---:|---:|---:|']
        for p,s in slots.items():md.append(f'| {p} | '+' | '.join(time(g) for g in s)+f' | {len(set(s))} | {4-fours[p]} | {fours[p]} |')
        md+=['',f'Selected source: [{source.relative_to(ROOT)}](../../{source.relative_to(ROOT)}).',f'Optimality evidence: [{proof}](../../{proof}).','The CSV is headerless: round, group, player IDs. Player IDs are placeholders; assigning names preserves validity and fairness scores.','']
        (OUT/f'n{n}_tee_times.md').write_text('\n'.join(md))
        index.append(f'| {n} | {score[0]} | {score[1]} | [CSV](optimal/n{n}_sol.csv) | [Summary](optimal/n{n}_tee_times.md) | [Evidence](../{proof}) |')
        records.append(dict(players=n,minimum_distinct_slots=score[0],total_spread=score[1],group_sizes=sizes,validation='passed',schedule=str(dest.relative_to(ROOT)),source=str(source.relative_to(ROOT)),sha256=digest(dest),proof=proof))
    index+=['','The 15–16-player proofs come from solver OPTIMAL results; 17–19 use exhaustive infeasibility proofs plus validated attaining schedules; 20–24 attain the absolute maximum of four distinct slots per player. See [full proof status](../runs/FAIRNESS_STATUS.md).','','Older schedules, logs and comparisons are preserved in [the dated archive](archive/2026-10-03/README.md). The historical 18-player total-only score 66 has minimum 2; the selected minimum-first fairness optimum is (3,65). Detailed experimental audits remain in `runs/`.','','Run new experiments under `runs/` to keep this collection clean. Regeneration script: `scripts/organize_solutions.py` (one-time migration; refuses to overwrite an existing collection).','']
    (SOL/'README.md').write_text('\n'.join(index))
    (OUT/'validation.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(dict(archived_files=len(manifest),validated_schedules=len(records),scores={str(n):EXPECTED[n] for n in EXPECTED}),indent=2))

if __name__=='__main__':main()
