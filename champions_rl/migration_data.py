"""Preserve a frozen policy's behavior on fresh training teams during migration.

These are policy-distillation labels, not newly discovered expert actions.
All held-out manifests and designated cores are excluded from team generation.
"""
import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
from multiprocessing.util import Finalize
from pathlib import Path
import random
import traceback
import numpy as np
import torch
from .engine import ROOT,Engine
from .features import Encoder,FEATURE_VERSION
from .knowledge import knowledge
from .legacy import load_frozen_policy
from .model import tensor_batch
from .teams import TeamPool,fingerprint
from .holdouts import held_out_core
from .policies import ScriptedBot,TRAINING_STYLES
from .opponent_labels import action_label
from .persistence import atomic_json,durable_npz
from .provenance import feature_signature
from .rewards import potential
from .power import keep_awake


def initialize(settings):
    global _settings,_engine,_encoder,_old_encoder,_model,_pool,_forbidden
    _settings=settings;torch.set_num_threads(1)
    _engine=Engine();_encoder=Encoder(engine=_engine);_pool=TeamPool()
    _model,_,_old_encoder,_=load_frozen_policy(settings['checkpoint'],settings['runtime'],engine=_engine)
    _forbidden=set()
    for path in (ROOT/'data').glob('teams_*.json'):
        if path.name=='teams_train.json':continue
        _forbidden.update(fingerprint(t['team']) for t in json.loads(path.read_text())['teams'])
    Finalize(None,_engine.close,exitpriority=10)


def sample(rng):
    for attempt in range(1000):
        team=_pool.sample(rng,'diverse')
        if fingerprint(team) not in _forbidden and not held_out_core(team):return team
    raise RuntimeError('Training holdout rejection budget exceeded')


def game(index):
    seed=_settings['seed']+index;rng=random.Random(seed);torch.manual_seed(seed)
    output=Path(_settings['output'])/f'v6_policy_{_settings["seed"]}_{index:06d}'
    meta={'teacher':'frozen_v6_policy','seed':seed,'game':index,'source_sha256':_settings['source_sha256'],
          'knowledge_sha256':knowledge().sha256,'feature_version':FEATURE_VERSION,'feature_signature':feature_signature(),
          'reference_fallbacks':0,'label_source':'Frozen policy behavior, not search or a new expert certificate'}
    try:
        teams=[sample(rng),sample(rng)];seat=index%2;distribution='diverse'
        draw=rng.random()
        if draw<.2:teams=[_pool.data['reference_team']]*2;distribution='saved_mirror'
        elif draw<.35:teams[seat]=_pool.data['reference_team'];distribution='saved_vs_varied'
        style=rng.choice([*TRAINING_STYLES[1:],'frozen_v6_policy','frozen_v6_policy'])
        bot=None if style=='frozen_v6_policy' else ScriptedBot(style,seed,_encoder)
        state=_engine.request('new',game='migration',teams=teams,seed=[rng.randrange(65536) for _ in range(4)])
        rows=[];raw=[]
        while not (state['ended'] or state['truncated']):
            views=state['views'];actions=[None,None]
            slots=[i for i in (seat,1-seat) if views[i]['legal'] and (i==seat or bot is None)]
            if slots:
                with torch.no_grad():
                    predictions=_model.act(tensor_batch([_old_encoder.encode(views[i]) for i in slots]),deterministic=True)[0].tolist()
                for i,choice in zip(slots,predictions):actions[i]=choice
            if bot and views[1-seat]['legal']:actions[1-seat]=bot.choose(views[1-seat])
            if views[seat]['legal']:
                rows.append({'obs':_encoder.encode(views[seat]),'label':actions[seat],
                             'enemy':action_label(views[1-seat],actions[1-seat]),'potential':potential(views[seat])})
                raw.append({'observation':views[seat],'expert_action':actions[seat],'student_action':None})
            state=_engine.request('step',game='migration',actions=actions)
        if state['truncated']:raise RuntimeError('Truncated trajectory excluded from distillation')
        outcome=0 if state['winner'] is None else 1 if state['winner']==seat else -1
        payload={k:np.stack([r['obs'][k] for r in rows]) for k in rows[0]['obs']}
        payload.update(labels=np.array([r['label'] for r in rows],dtype=np.int64),
                       enemy_labels=np.array([r['enemy'] for r in rows],dtype=np.int64),
                       outcomes=np.array([outcome-.15*r['potential'] for r in rows],dtype=np.float32),
                       student_actions=np.full(len(rows),-100,dtype=np.int64))
        durable_npz(output.with_suffix('.npz'),**payload)
        with gzip.open(output.with_suffix('.observations.json.gz'),'wt',encoding='utf-8') as stream:
            json.dump({'rows':raw,'outcome':outcome,'feature_signature':feature_signature()},stream)
        meta.update(samples=len(rows),outcome=outcome,opponent=style,team_distribution=distribution,
                    team_fingerprints=list(map(fingerprint,teams)),error=None)
    except Exception:meta.update(samples=0,error=traceback.format_exc())
    finally:_engine.request('close',game='migration')
    atomic_json(output.with_suffix('.json'),meta)
    return meta


@keep_awake()
def collect(checkpoint,runtime,output,games=512,workers=8,seed=873123):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings={'checkpoint':str(Path(checkpoint).resolve()),'runtime':str(Path(runtime).resolve()),
              'output':str(output.resolve()),'seed':seed,'source_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
    results=[];pending=[]
    for index in range(games):
        meta=output/f'v6_policy_{seed}_{index:06d}.json'
        if meta.exists():
            row=json.loads(meta.read_text())
            if row['source_sha256']!=settings['source_sha256'] or row['feature_signature']!=feature_signature():
                raise ValueError('Existing distillation shard differs from current run')
            results.append(row)
        else:pending.append(index)
    with mp.get_context('spawn').Pool(workers,initializer=initialize,initargs=(settings,)) as pool:
        for row in pool.imap_unordered(game,pending):
            results.append(row)
            if len(results)%32==0:print(json.dumps({'games':len(results),'samples':sum(r['samples'] for r in results),'errors':sum(bool(r['error']) for r in results)}),flush=True)
    report={**settings,'games':games,'samples':sum(r['samples'] for r in results),
            'valid_games':sum(not r['error'] for r in results),'errors':sum(bool(r['error']) for r in results),
            'feature_signature':feature_signature()}
    atomic_json(output/'collection.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','runtime','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--games',type=int,default=512);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=873123)
    collect(**vars(parser.parse_args()))
