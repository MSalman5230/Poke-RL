"""Held-out terminal-result calibration for sampled or deterministic play."""
import argparse
import hashlib
import json
import multiprocessing as mp
from multiprocessing.util import Finalize
from pathlib import Path
import random
import time
import numpy as np
import torch
from .arena import load_teams
from .engine import Engine
from .features import Encoder,FEATURE_VERSION
from .model import BattleModel,PrivilegedCritic,tensor_batch,encode_for_model
from .policies import ScriptedBot
from .provenance import feature_signature
from .persistence import atomic_json
from .rewards import potential
from .teams import TeamPool


def initialize(settings):
    global _settings,_model,_critic,_encoder,_engine,_teams
    _settings=settings;torch.set_num_threads(1)
    _model,_=BattleModel.load(settings['checkpoint'])
    data=torch.load(settings['critic_state'],map_location='cpu',weights_only=True)
    if data.get('feature_signature')!=feature_signature() or data.get('feature_version')!=FEATURE_VERSION:
        raise ValueError('Critic inputs differ from this audit')
    config=data['config']
    _critic=PrivilegedCritic(config['width'],config['layers'],config['heads'])
    _critic.load_state_dict(data['critic']);_critic.eval()
    _engine=Engine();_encoder=Encoder(engine=_engine);_teams=load_teams(settings['split'])
    Finalize(None,_engine.close,exitpriority=10)


def game(index):
    seed=_settings['seed']+index//2
    rng=random.Random(seed);torch.manual_seed(seed+index%2)
    teams=[rng.choice(_teams),rng.choice(_teams)]
    if _settings['roster']=='saved':teams=[TeamPool().data['reference_team']]*2
    seat=index%2;style=_settings['opponents'][(index//2)%len(_settings['opponents'])]
    opponent=ScriptedBot(style,seed,_encoder)
    state=_engine.request('new',game='value_audit',teams=teams,seed=[rng.randrange(65536) for _ in range(4)],training=True)
    rows=[];steps=0
    while not (state['ended'] or state['truncated']):
        own=state['views'][seat];other=state['views'][1-seat]
        actions=[None,None]
        if own['legal']:
            public=encode_for_model(_model,_encoder,own)
            private=_encoder.encode(state['training']['critic_views'][seat],entities_only=True)
            with torch.no_grad():
                action,_,value=_model.act(tensor_batch([public]),deterministic=_settings['deterministic'])
                private_value=_critic(tensor_batch([private]))
            actions[seat]=int(action.item())
            rows.append({'game':index,'turn':own['turn'],'opponent':style,'public':float(value),
                         'private':float(private_value),'potential':potential(own)})
        if other['legal']:actions[1-seat]=opponent.choose(other)
        state=_engine.request('step',game='value_audit',actions=actions);steps+=1
        if steps>2520:raise RuntimeError('Audit decision limit exceeded')
    _engine.request('close',game='value_audit')
    if state['truncated']:return {'game':index,'truncated':True,'rows':[]}
    outcome=0 if state['winner'] is None else 1 if state['winner']==seat else -1
    for row in rows:row['outcome']=outcome
    return {'game':index,'truncated':False,'rows':rows}


def diagnostics(rows):
    result={'positions':len(rows),'games':len({r['game'] for r in rows})}
    outcome=np.asarray([r['outcome'] for r in rows]);target=outcome-.15*np.asarray([r['potential'] for r in rows])
    for name in ('public','private'):
        value=np.asarray([r[name] for r in rows]);probability=np.clip((value+.15*np.asarray([r['potential'] for r in rows])+1)/2,0,1)
        result[name]={'return_mse':float(np.mean((value-target)**2)),
                      'explained_variance':float(1-np.var(target-value)/np.var(target)) if np.var(target)>0 else None,
                      'brier':float(np.mean((probability-(outcome+1)/2)**2)),
                      'calibration':[]}
        for low,high in zip(np.arange(0,1,.2),np.arange(.2,1.01,.2)):
            mask=(probability>=low)&((probability<high) if high<.99 else (probability<=1))
            if mask.any():result[name]['calibration'].append({'count':int(mask.sum()),'predicted':float(probability[mask].mean()),'observed':float(((outcome[mask]+1)/2).mean())})
    return result


def run(checkpoint,critic_state,output,games=256,workers=4,split='validation',roster='manifest',seed=823104,
        opponents=('balanced','hyperoffense','setup_sweeper','attrition','pivot'),deterministic=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in locals().copy().items() if k!='output'}
    settings['opponents']=list(opponents)
    atomic_json(output/'settings.json',settings)
    started=time.monotonic();completed=[]
    with mp.get_context('spawn').Pool(workers,initializer=initialize,initargs=(settings,)) as pool:
        for row in pool.imap_unordered(game,range(games)):
            completed.append(row)
            if len(completed)%16==0:print(json.dumps({'event':'value_audit','completed':len(completed),'games':games}),flush=True)
    rows=[r for g in completed for r in g['rows']]
    report={'settings':settings,'feature_signature':feature_signature(),
            'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'critic_sha256':hashlib.sha256(Path(critic_state).read_bytes()).hexdigest(),
            'truncated_games':sum(g['truncated'] for g in completed),'seconds':time.monotonic()-started,
            'overall':diagnostics(rows),'by_turn':{},
            'limits':'Held-out team/seed trajectories in the recorded action-selection mode. The critic is trained on sampled actions; deterministic play changes that policy. '
                     'Correlated positions are descriptive. This does not validate ranking counterfactual search leaves.'}
    for label,low,high in [('preview',0,0),('1-5',1,5),('6-10',6,10),('11-20',11,20),('21+',21,10000)]:
        selected=[r for r in rows if low<=r['turn']<=high]
        if selected:report['by_turn'][label]=diagnostics(selected)
    atomic_json(output/'positions.json',rows);atomic_json(output/'report.json',report)
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','critic-state','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--games',type=int,default=256);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--split',default='validation');parser.add_argument('--roster',choices=['saved','manifest'],default='manifest')
    parser.add_argument('--deterministic',action=argparse.BooleanOptionalAction,default=True,
                        help='Use --no-deterministic to audit the sampled policy used in PPO rollouts')
    parser.add_argument('--seed',type=int,default=823104)
    run(**vars(parser.parse_args()))
