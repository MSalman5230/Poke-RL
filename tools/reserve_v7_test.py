"""Create a prospective v7 test manifest; never run its battles during selection."""
from pathlib import Path
import hashlib
import json
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from champions_rl.engine import Engine
from champions_rl.teams import TeamPool,fingerprint
from champions_rl.knowledge import knowledge


def reserve():
    destination=ROOT/'data/teams_test_v7.json'
    if destination.exists():raise ValueError('Prospective test already reserved; refusing replacement')
    forbidden=set()
    for path in (ROOT/'data').glob('teams_*.json'):
        forbidden.update(fingerprint(t['team']) for t in json.loads(path.read_text())['teams'])
    for path in (ROOT/'runtime_data/distill_v6_to_v7').glob('v6_policy_*.json'):
        forbidden.update(json.loads(path.read_text()).get('team_fingerprints',[]))
    pool=TeamPool();rng=random.Random(491731);teams=[]
    while len(teams)<128:
        team=pool.sample(rng,'diverse');sha=fingerprint(team)
        if sha in forbidden:continue
        forbidden.add(sha);teams.append({'id':sha,'team':team})
    with Engine() as engine:
        if any(engine.request('validate',teams=[t['team'] for t in teams])):raise ValueError('Invalid reserved team')
    content={'split':'test_v7','seed':491731,'battle_seed':290731,'knowledge_sha256':knowledge().sha256,
             'purpose':'Prospective test: no outcome-based selection; v6 test outcomes are already known.',
             'generator':'diverse: teammate-guided + authored templates + broad coverage','teams':teams}
    destination.write_bytes((json.dumps(content,indent=2)+'\n').encode())
    print(json.dumps({'path':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'teams':len(teams),'battle_seed':290731}))


if __name__=='__main__':reserve()
