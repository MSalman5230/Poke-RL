"""Import offline Champions replay JSON without inventing hidden sets or masks."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .engine import ROOT,Engine
from .features import Encoder,FEATURE_VERSION
from .knowledge import knowledge,canonical
from .persistence import atomic_json,durable_npz
from .provenance import feature_signature
from .rewards import potential
from .teams import fingerprint
from .opponent_labels import switch_offset


def parse_replay(payload):
    result=subprocess.run([shutil.which('node') or 'node',str(ROOT/'simulator/replays.cjs')],
                          input=json.dumps(payload),text=True,encoding='utf-8',capture_output=True,timeout=60,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise ValueError(result.stderr.strip())
    return json.loads(result.stdout)


def import_replay(source,output,companion=None):
    source=Path(source);output=Path(output);content=source.read_bytes()
    payload=json.loads(content);parsed=parse_replay(payload)
    sha=hashlib.sha256(content).hexdigest();folder=output/sha[:20]
    provenance={'feature_signature':feature_signature(),
                'parser_sha256':hashlib.sha256((ROOT/'simulator/replays.cjs').read_bytes()).hexdigest(),
                'importer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'opponent_labels_sha256':hashlib.sha256((ROOT/'champions_rl/opponent_labels.py').read_bytes()).hexdigest()}
    if (folder/'manifest.json').exists():
        existing=json.loads((folder/'manifest.json').read_text())
        companion_sha=hashlib.sha256(Path(companion).read_bytes()).hexdigest() if companion else None
        if existing['source_sha256']!=sha or existing.get('companion_sha256')!=companion_sha:
            raise ValueError('Existing replay import differs; use a new output folder')
        if any(existing.get(key)!=value for key,value in provenance.items()):
            raise ValueError('Existing replay import has incompatible parser/features; use a new output folder')
        return existing
    sidecar=json.loads(Path(companion).read_text()) if companion else {}
    if sidecar and sidecar.get('source_sha256')!=sha:raise ValueError('Companion must bind to the exact replay SHA256')
    teams=[];samples=[];labels=[];outcomes=[];enemy=[];raw=[];seen_frames=set()
    if sidecar.get('decision_frames') and parsed['warnings']:
        raise ValueError('Identity/Transform replay needs specialized reconstruction before imitation')
    with Engine() as engine:
        for side,team in sidecar.get('complete_teams',{}).items():
            if side not in ('p1','p2'):raise ValueError('Team side must be p1 or p2')
            index=int(side[-1])-1
            expected={canonical(m['species']) for m in parsed['team_evidence'][index]}
            if {canonical(m['species']) for m in team}!=expected:raise ValueError('Companion roster differs from the public preview')
            errors=engine.request('validate',teams=[team])[0]
            if errors:raise ValueError(str(errors))
            by_species={canonical(m['species']):m for m in team}
            for evidence in parsed['team_evidence'][index]:
                full=by_species[canonical(evidence['species'])]
                if not set(evidence['moves'])<=set(map(canonical,full['moves'])):
                    raise ValueError('Observed moves conflict with the supplied complete team')
            teams.append({'side':side,'id':fingerprint(team),'team':team,'source':'Explicit companion complete team; never inferred from marginals'})
        # Optional observations are independently captured client/adapter frames,
        # not reconstructed legal masks guessed from a spectator replay.
        encoder=Encoder(engine=engine)
        for frame in sidecar.get('decision_frames',[]):
            obs=frame['observation'];side=obs['side'];action=frame['action']
            if side not in (0,1) or obs.get('format')!=parsed['format']:raise ValueError('Invalid decision-frame side or format')
            frame_key=(obs['turn'],side)
            if frame_key in seen_frames:raise ValueError('Duplicate captured decision for the same turn and side')
            seen_frames.add(frame_key)
            if any(any(k in m for k in ('set','stats','selected','current_hp','max_hp')) for m in obs['opponent']):
                raise ValueError('Private opponent information in imitation frame')
            if not all(m.get('set') for m in obs['own']):raise ValueError('Own team must be known in captured frames')
            legal=next((a for a in obs['legal'] if a['id']==action),None)
            if legal is None:raise ValueError('Imitation label is not in the captured legal-action mask')
            matching=[e for e in parsed['examples'] if e['turn']==obs['turn'] and e['actor_side']==side]
            def agrees(row):
                label=row['opponent_action']
                return legal['kind']==label['kind'] and (canonical(legal.get('move'))==label.get('move') if label['kind']=='move' else legal['slot']==label['slot'])
            matching=[r for r in matching if agrees(r)]
            if len(matching)!=1 or matching[0]['actor_outcome'] is None:
                raise ValueError('Captured choice lacks a unique matching completed replay action')
            if obs['phase']!='move':raise ValueError('Public replay import cannot verify preview or forced replacement labels')
            expected=matching[0]['public_state']['teams'][1-side]
            if [canonical(m['base_species']) for m in obs['opponent']]!=[canonical(m['base_species']) for m in expected]:
                raise ValueError('Captured opponent roster differs from replay')
            for mon,public in zip(obs['opponent'],expected):
                moves=[canonical(m['id'] if isinstance(m,dict) else m) for m in mon.get('moves',[])]
                if not set(moves)<=set(public['moves']):raise ValueError('Future opponent moves leaked into a captured frame')
                for field in ('item','ability'):
                    if mon.get(field) is not None and mon[field]!=public[field]:raise ValueError('Future opponent reveal in captured frame')
                for field in ('seen','active','fainted','status','boosts','types'):
                    if mon.get(field)!=public.get(field):raise ValueError('Captured public state differs from replay: '+field)
                if abs(mon.get('hp',1)-public['hp'])>.011:raise ValueError('Captured opponent HP differs from replay')
            samples.append(encoder.encode(obs));labels.append(action)
            outcome=matching[0]['actor_outcome']
            # A reply is a target, never an input. Use only the other side's
            # ordinary action from this simultaneous turn, not a later reveal
            # or a guessed choice when that Pokemon could not act.
            replies=[r['opponent_action'] for r in parsed['examples']
                     if r['turn']==obs['turn'] and r['actor_side']==1-side]
            enemy_label=-100
            if len(replies)==1:
                reply=replies[0]
                if reply['kind']=='move':
                    token=knowledge().token('moves',reply['move'])
                    if token>1:enemy_label=token
                elif reply['kind']=='switch' and 0<=reply.get('slot',-1)<6:
                    enemy_label=switch_offset()+reply['slot']
            outcomes.append(outcome-.15*potential(obs));enemy.append(enemy_label)
            raw.append({'observation':obs,'expert_action':action,'student_action':None,
                        'outcome':outcome,'enemy_label':enemy_label})
    folder.mkdir(parents=True,exist_ok=True)
    with gzip.open(folder/'public_examples.json.gz','wt',encoding='utf-8') as stream:json.dump(parsed,stream)
    atomic_json(folder/'complete_teams.json',teams)
    if samples:
        arrays={k:np.stack([s[k] for s in samples]) for k in samples[0]}
        arrays.update(labels=np.asarray(labels,dtype=np.int64),outcomes=np.asarray(outcomes,dtype=np.float32),
                      enemy_labels=np.asarray(enemy,dtype=np.int64),student_actions=np.full(len(labels),-100,dtype=np.int64))
        stem=folder/('replay_'+sha[:20])
        durable_npz(stem.with_suffix('.npz'),**arrays)
        atomic_json(stem.with_suffix('.json'),{'teacher':'replay','seed':int(sha[:8],16),'samples':len(samples),
                    'source_sha256':sha,'feature_version':FEATURE_VERSION,'feature_signature':feature_signature(),
                    'knowledge_sha256':knowledge().sha256,'reference_fallbacks':0})
        with gzip.open(stem.with_suffix('.observations.json.gz'),'wt',encoding='utf-8') as stream:
            json.dump({'rows':raw,'outcome':None,'feature_signature':feature_signature()},stream)
    report={'source_sha256':sha,'companion_sha256':hashlib.sha256(Path(companion).read_bytes()).hexdigest() if companion else None,
            'format':parsed['format'],'public_action_examples':len(parsed['examples']),'imitation_samples':len(samples),
            'opponent_labeled_samples':sum(label!=-100 for label in enemy),
            'complete_teams':len(teams),'ended':parsed['ended'],'output':str(folder.resolve()),
            **provenance,
            'limitations':parsed['limitations'],'automatically_added_to_training':False}
    atomic_json(folder/'manifest.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',dest='source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--companion',type=Path,help='SHA-bound complete teams and/or captured public decision frames')
    print(json.dumps(import_replay(**vars(parser.parse_args())),indent=2))
