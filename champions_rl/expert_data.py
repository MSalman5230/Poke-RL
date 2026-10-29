"""Resumable expert trajectories and DAgger labels on learner-visited positions."""
import argparse
import hashlib
import json
import gzip
import multiprocessing as mp
from multiprocessing.util import Finalize
from pathlib import Path
import random
import time
import traceback
import numpy as np
from .arena import load_teams,STYLES
from .engine import Engine,ROOT
from .features import Encoder,FEATURE_VERSION
from .knowledge import knowledge
from .opponent_labels import action_label
from .persistence import atomic_json,durable_npz,append_json
from .policies import ScriptedBot
from .provenance import feature_signature
from .rewards import potential
from .teams import TeamPool,fingerprint
from .holdouts import held_out_core


def initialize(settings):
    global _settings,_engine,_encoder,_teams,_saved,_student
    _settings=settings
    _engine=Engine()
    _encoder=Encoder(engine=_engine)
    _teams=[t for t in load_teams('train') if not held_out_core(t)]
    _saved=TeamPool().data['reference_team']
    _student=None
    if settings.get('student'):
        import torch
        from .model import BattleModel,NeuralBot
        torch.set_num_threads(1)
        model,_=BattleModel.load(settings['student'])
        _student=NeuralBot(model,_encoder,deterministic=False)
    Finalize(None,_engine.close,exitpriority=10)


def game(index):
    settings=_settings
    seed=settings['seed']+index
    rng=random.Random(seed)
    if _student:
        import torch
        torch.manual_seed(seed)
    teacher=settings['teacher']
    if teacher=='mixture':teacher=rng.choice(['defensive','defensive','balanced','setup','damage'])
    stem=Path(settings['output'])/f"{settings['teacher']}_{settings['seed']}_{index:06d}"
    common={'teacher':teacher,'seed':seed,'game':index,'feature_version':FEATURE_VERSION,
            'feature_signature':feature_signature(),'knowledge_sha256':knowledge().sha256,
            'student_sha256':settings.get('student_sha256'),'beta':settings['beta'],
            'team_distribution':settings['teams'],'query_probability':settings['query_probability']}
    started=time.monotonic()
    try:
        if teacher=='search':
            from .search import SearchBot
            from .model import PrivilegedCritic
            if _student is None:raise ValueError('Search distillation requires --student')
            expert=SearchBot(_student.model,_encoder,seed=seed,confidence=1.,
                             critic=PrivilegedCritic.load_training_state(settings['critic_state']))
        elif teacher=='reference':
            from .reference import ReferenceBot
            expert=ReferenceBot(seed,strict=True)
        else:expert=ScriptedBot(teacher,seed,_encoder)
        opponent_name='reference' if settings.get('opponents')=='reference' else rng.choice(STYLES[1:])
        if opponent_name=='reference':
            from .reference import ReferenceBot
            opponent=ReferenceBot(seed+719,strict=True)
        else:opponent=ScriptedBot(opponent_name,seed+719,_encoder)
        seat=index%2
        teams=[rng.choice(_teams),rng.choice(_teams)]
        if settings['teams']=='saved':teams=[_saved,_saved]
        if settings['teams']=='cross':teams[seat]=_saved
        state=_engine.request('new',game='demonstration',teams=teams,seed=[rng.randrange(65536) for _ in range(4)])
        samples=[];raw=[]
        count=0
        while not (state['ended'] or state['truncated']):
            own=state['views'][seat];other=state['views'][1-seat]
            enemy=opponent.choose(other) if other['legal'] else None
            label=student_action=None
            if own['legal']:
                query=_student is None or rng.random()<settings['query_probability']
                if _student:student_action=_student.choose(own)
                if query:label=expert.choose(own)
                action=label if _student is None or (query and rng.random()<settings['beta']) else student_action
                if label is not None and (teacher!='search' or expert.last_search.get('used')):
                    samples.append({'obs':_encoder.encode(own),'labels':label,'student_actions':-100 if student_action is None else student_action,
                                    'enemy_labels':action_label(other,enemy),'potential':potential(own)})
                    raw.append({'observation':own,'expert_action':label,'student_action':student_action})
            else:action=None
            actions=[None,None];actions[seat]=action;actions[1-seat]=enemy
            state=_engine.request('step',game='demonstration',actions=actions)
            count+=1
            if count>2520:raise RuntimeError('Demonstration decision limit exceeded')
        if state['truncated']:raise RuntimeError('Truncated demonstration excluded from terminal value supervision')
        outcome=0 if state['winner'] is None else 1 if state['winner']==seat else -1
        payload={k:np.stack([r['obs'][k] for r in samples]) for k in samples[0]['obs']} if samples else {}
        if not samples:raise RuntimeError('No queried expert decisions')
        payload.update(labels=np.asarray([r['labels'] for r in samples],dtype=np.int64),
                       outcomes=np.asarray([outcome-.15*r['potential'] for r in samples],dtype=np.float32),
                       enemy_labels=np.asarray([r['enemy_labels'] for r in samples],dtype=np.int64),
                       student_actions=np.asarray([r['student_actions'] for r in samples],dtype=np.int64))
        audits=getattr(expert,'audit',[])
        if getattr(expert,'fallbacks',0):raise RuntimeError('Degraded expert game')
        durable_npz(stem.with_suffix('.npz'),**payload)
        # Keep public observations so future feature revisions can be rebuilt
        # without pretending an old encoded tensor has the new semantics.
        with gzip.open(stem.with_suffix('.observations.json.gz'),'wt',encoding='utf-8') as stream:
            json.dump({'rows':raw,'outcome':outcome,'feature_signature':feature_signature()},stream)
        metadata={**common,'samples':len(samples),'outcome':outcome,'opponent':opponent_name,'reference_fallbacks':0,
                  'reference_trials':sorted({a['trials'] for a in audits}),'turns':state['views'][0]['turn'],
                  'expert_disagreements':sum(r['student_actions']>=0 and r['student_actions']!=r['labels'] for r in samples),
                  'search_statistics':dict(getattr(expert,'statistics',{})),
                  'team_fingerprints':[fingerprint(t) for t in teams],'seconds':time.monotonic()-started,'error':None}
        atomic_json(stem.with_suffix('.json'),metadata)
        return metadata
    except Exception:
        metadata={**common,'samples':0,'error':traceback.format_exc(),'seconds':time.monotonic()-started}
        atomic_json(stem.with_suffix('.json'),metadata)
        return metadata
    finally:
        _engine.request('close',game='demonstration')


def collect(output,games=128,teacher='reference',workers=4,seed=51001,teams='saved',student=None,beta=.25,query_probability=1.,opponents='mixture',search_certificate=None,critic_state=None):
    if games<1 or workers<1 or not 0<=beta<=1 or not 0<query_probability<=1:raise ValueError('Invalid collection settings')
    if teacher=='search':
        # A high imitation agreement or a search-score margin is insufficient.
        # Require independent paired battle evidence for these exact weights.
        if search_certificate is None:raise ValueError('Search distillation is disabled until paired battle validation demonstrates an improvement')
        certificate=json.loads(Path(search_certificate).read_text(encoding='utf-8'))
        if not student or not critic_state or certificate.get('critic_sha256')!=hashlib.sha256(Path(critic_state).read_bytes()).hexdigest() or certificate.get('checkpoint_sha256')!=hashlib.sha256(Path(student).read_bytes()).hexdigest() or certificate.get('feature_signature')!=feature_signature() or certificate.get('games',0)<256 or certificate.get('paired_improvement_ci',[0])[0]<=0 or certificate.get('errors',1):
            raise ValueError('Search validation certificate does not establish an improvement for this checkpoint')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings={'output':str(output.resolve()),'games':games,'teacher':teacher,'seed':seed,'teams':teams,'beta':beta,
              'query_probability':query_probability,'student':str(Path(student).resolve()) if student else None,
              'opponents':opponents,
              'search_certificate':str(search_certificate) if search_certificate else None,
              'critic_state':str(critic_state) if critic_state else None,
              'student_sha256':hashlib.sha256(Path(student).read_bytes()).hexdigest() if student else None,
              'feature_signature':feature_signature()}
    manifest=output/f'collection_{teacher}_{seed}.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text());previous.setdefault('opponents','mixture');previous.setdefault('search_certificate',None);previous.setdefault('critic_state',None)
        if previous!=settings:raise ValueError('Collection settings changed; use a new seed or directory')
    atomic_json(manifest,settings)
    rows=[];jobs=[]
    for i in range(games):
        path=output/f'{teacher}_{seed}_{i:06d}.json'
        if path.exists():
            row=json.loads(path.read_text())
            if row['feature_signature']!=feature_signature():raise ValueError('Incompatible existing shard')
            if row.get('error') or path.with_suffix('.npz').exists():rows.append(row);continue
        jobs.append(i)
    context=mp.get_context('spawn')
    with context.Pool(workers,initializer=initialize,initargs=(settings,)) as pool:
        for row in pool.imap_unordered(game,jobs,chunksize=1):
            rows.append(row)
            append_json(output/f'progress_{teacher}_{seed}.jsonl',row)
            print(json.dumps({'event':'expert_game','completed':len(rows),'games':games,**row}),flush=True)
        pool.close();pool.join()
    report={'games':len(rows),'valid_games':sum(not r.get('error') for r in rows),'errors':sum(bool(r.get('error')) for r in rows),
            'samples':sum(r['samples'] for r in rows),'expert_disagreements':sum(r.get('expert_disagreements',0) for r in rows),'settings':settings}
    atomic_json(output/f'summary_{teacher}_{seed}.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--games',type=int,default=128)
    parser.add_argument('--teacher',choices=['reference','mixture','defensive','balanced','setup','damage','search'],default='reference')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=51001)
    parser.add_argument('--teams',choices=['saved','cross','train'],default='saved')
    parser.add_argument('--student')
    parser.add_argument('--opponents',choices=['mixture','reference'],default='mixture')
    parser.add_argument('--beta',type=float,default=.25)
    parser.add_argument('--query-probability',type=float,default=1.)
    parser.add_argument('--search-certificate',type=Path)
    parser.add_argument('--critic-state',type=Path)
    print(json.dumps(collect(**vars(parser.parse_args()))),flush=True)
