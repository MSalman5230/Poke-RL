import copy
import gzip
import hashlib
import json
from pathlib import Path
import pytest
from champions_rl.arena import play_match
from champions_rl.engine import Engine
from champions_rl.features import Encoder
from champions_rl.policies import ScriptedBot
from champions_rl.replays import parse_replay,import_replay
from champions_rl.teams import TeamPool
from champions_rl.training import load_demonstrations


@pytest.fixture(scope='module')
def replay():
    team=TeamPool().data['reference_team'];frames=[]
    with Engine() as engine:
        encoder=Encoder(engine=engine)
        bots=[ScriptedBot('balanced',encoder=encoder),ScriptedBot('hyperoffense',encoder=encoder)]
        result=play_match(engine,[team,team],bots,81372,capture=lambda side,obs,action:frames.append({'observation':copy.deepcopy(obs),'action':action}))
    assert not result['truncated']
    return {'formatid':'gen9championsbssregmc','log':'\n'.join(result['trace'])},frames,team


def test_public_replay_action_labels_do_not_leak_future_moves(replay,tmp_path):
    payload,_,_=replay;parsed=parse_replay(payload)
    assert parsed['ended'] and parsed['examples']
    row=parsed['examples'][0];side=row['actor_side']
    active=next(m for m in row['public_state']['teams'][side] if m['active'])
    assert row['opponent_action']['move'] not in active['moves']
    assert not row['imitation_ready'] and row['viewer_side']==1-side
    assert row['viewer_outcome']==-row['actor_outcome']
    assert not any('set' in m or 'stats' in m for team in row['public_state']['teams'] for m in team)
    source=tmp_path/'replay.json';source.write_text(json.dumps(payload),encoding='utf-8')
    report=import_replay(source,tmp_path/'import')
    assert report['public_action_examples']>0 and report['imitation_samples']==report['complete_teams']==0
    assert import_replay(source,tmp_path/'import')==report
    with pytest.raises(ValueError,match='format'):parse_replay({**payload,'formatid':'gen9ou'})


def test_captured_own_masks_and_explicit_teams_enable_verified_imitation(replay,tmp_path):
    payload,frames,team=replay;parsed=parse_replay(payload)
    row=parsed['examples'][0]
    frame=next(f for f in frames if f['observation']['turn']==row['turn'] and f['observation']['side']==row['actor_side'])
    source=tmp_path/'replay.json';source.write_text(json.dumps(payload),encoding='utf-8')
    companion={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'complete_teams':{'p1':team},'decision_frames':[frame]}
    sidecar=tmp_path/'companion.json';sidecar.write_text(json.dumps(companion),encoding='utf-8')
    report=import_replay(source,tmp_path/'good',sidecar)
    assert report['imitation_samples']==report['complete_teams']==1
    data=load_demonstrations(Path(report['output']))
    assert len(data['labels'])==1 and data['mask'][0,data['labels'][0]]
    companion['decision_frames'][0]['observation']['opponent'][0]['set']=team[0]
    sidecar.write_text(json.dumps(companion),encoding='utf-8')
    with pytest.raises(ValueError,match='Private opponent'):import_replay(source,tmp_path/'bad',sidecar)
