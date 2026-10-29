"""Create a fixed challenge set and exclude its entire species cores thereafter."""
import copy
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from champions_rl.engine import Engine
from champions_rl.holdouts import CORES,held_out_core
from champions_rl.teams import TeamPool,fingerprint
from champions_rl.persistence import atomic_json
from champions_rl.knowledge import knowledge

path=ROOT/'data/teams_validation_cores.json'
if path.exists():raise FileExistsError(path)
pool=TeamPool();rng=random.Random(518731);teams=[];seen=set()
for core in CORES:
    while sum(tuple(r['core'])==core for r in teams)<24:
        names=list(core)+rng.sample([n for n in pool.curated if n not in core],3)
        team=[];items=set()
        for name in names:
            options=pool.curated.get(name,pool.by_species[name])
            options=[r for r in options if not r['set']['item'] or r['set']['item'] not in items]
            if not options:break
            s=copy.deepcopy(rng.choice(options)['set']);team.append(s);items.add(s['item'])
        if len(team)!=6:continue
        rng.shuffle(team);digest=fingerprint(team)
        if digest in seen:continue
        with Engine() as engine:errors=engine.request('validate',teams=[team])[0]
        if errors:continue
        seen.add(digest);teams.append({'id':digest,'core':core,'team':team})
assert all(held_out_core(r['team']) for r in teams)
atomic_json(path,{'split':'validation_cores','seed':518731,'knowledge_sha256':knowledge().sha256,
                 'exclusion':'Prospective for v4 training. Transferred v3 weights may have encountered these cores historically.',
                 'cores':CORES,'teams':teams})
print(json.dumps({'teams':len(teams),'cores':CORES}))
