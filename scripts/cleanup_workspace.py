"""One-time authorized cleanup, preserving source data and active editor files."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'solutions/archive/2026-10-03'
OUT=BASE/'workspace_legacy'
assert not OUT.exists()
OUT.mkdir()
records=[]
for name in ['sol_1','solution_n18_sol_20251120_052000.csv','solution_n18_sol_60m.csv','super_hints.csv','solutions.txt']:
    source=ROOT/name
    files=sorted(source.rglob('*')) if source.is_dir() else [source]
    entries=[(p,hashlib.sha256(p.read_bytes()).hexdigest()) for p in files if p.is_file()]
    # solutions.txt has an active Vim session: preserve its live pathname.
    if name=='solutions.txt':shutil.copy2(source,OUT/name)
    else:shutil.move(str(source),str(OUT/name))
    for p,sha in entries:
        dest=OUT/p.relative_to(ROOT)
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==sha
        records.append(dict(original_path=str(p.relative_to(ROOT)),archive_path=str(dest.relative_to(ROOT)),sha256=sha,operation='copy' if name=='solutions.txt' else 'move'))
(OUT/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
(OUT/'README.md').write_text('# Remaining historical workspace files\n\nOriginal sol_1 schedules, two root-level 18-player schedules, and partial hints were moved here unchanged. solutions.txt is a snapshot; the root original remains because it has an active Vim session. Every file checksum was verified after archiving; see manifest.json. Historical run notes retain original paths, which can be resolved through the manifests.\n')
# Guard the deletion against confusing the obsolete root environment with .venv.
old=(ROOT/'pyvenv.cfg').read_text()
assert '/usr/local/' in old
assert '/opt/homebrew/' in (ROOT/'.venv/pyvenv.cfg').read_text()
assert not (ROOT/'historical_scripts').is_symlink()
(ROOT/'historical_scripts').rmdir()
for name in ('bin','include','lib'):
    path=ROOT/name
    assert path.is_dir() and not path.is_symlink()
    shutil.rmtree(path)
(ROOT/'pyvenv.cfg').unlink()
(BASE/'cleanup.json').write_text(json.dumps(dict(archived_file_count=len(records),removed=['historical_scripts/','bin/','include/','lib/','pyvenv.cfg'],preserved_active_editor_files=['solutions.txt','.solutions.txt.swp','.HISTORICAL_RUNS.md.swp'],retained_environment='.venv'),indent=2)+'\n')
print(f'Archived and checksum-verified {len(records)} files; removed obsolete root environment and empty directory.')
