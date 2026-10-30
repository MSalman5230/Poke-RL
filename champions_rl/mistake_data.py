"""Bounded fresh loss collection and public-teacher corrections; no optimization."""
import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
from multiprocessing.util import Finalize
import os
from pathlib import Path
import random
import numpy as np
import torch
from .engine import Engine,ROOT
from .features import Encoder,FEATURE_VERSION
from .knowledge import knowledge
from .legacy import load_frozen_policy
from .model import NeuralBot
from .opponent_labels import action_label
from .persistence import atomic_json,durable_npz
from .provenance import feature_signature
from .reference import ReferenceBot
from .rewards import potential
from .teams import TeamPool
from .power import keep_awake


def initialize(settings):
    global _settings,_engine,_student,_teacher,_encoder,_team
    _settings=settings;torch.set_num_threads(1);os.environ['PCA_PROBE_WORKERS']='1'
    _engine=Engine();_encoder=Encoder(engine=_engine);_team=TeamPool().data['reference_team']
    bots=[]
    for prefix in ('student','teacher'):
        model,_,encoder,_=load_frozen_policy(settings[prefix],settings[prefix+'_runtime'],engine=_engine)
        bots.append(NeuralBot(model,encoder,deterministic=True))
    _student,_teacher=bots
    Finalize(None,_engine.close,exitpriority=10)
    # Multiprocessing workers do not reliably run ordinary atexit hooks.
    from opponents.autoplay_reference.bot.simulator.bridge import get
    Finalize(None,get().close,exitpriority=10)


def category(obs,student_action,teacher_action):
    if obs['phase']=='preview':return 'team_selection'
    selected={a['id']:a for a in obs['legal']}
    a,b=selected[student_action],selected[teacher_action]
    if a['kind']=='switch':
        mon=obs['own'][a['slot']]
        if _encoder.hazard_cost(mon,obs['conditions'][0])>0:return 'switch_into_hazards'
        return 'switch_or_preserve_teammate'
    if b['kind']=='switch':return 'preserve_active_or_switch'
    move=knowledge().moves.get(a.get('move'),{})
    if move.get('boosts') and move.get('target')=='self':return 'setup_or_recovery_timing'
    return 'attack_or_status_choice'


def game(index):
    seed=_settings['seed']+index;seat=index%2
    stem=Path(_settings['output'])/'raw'/f'loss_{seed}_{index:06d}'
    if stem.with_suffix('.json').exists():return json.loads(stem.with_suffix('.json').read_text())
    enemy=ReferenceBot(seed+519,strict=True);rng=random.Random(seed);rows=[]
    state=_engine.request('new',game='loss_collection',teams=[_team,_team],seed=[rng.randrange(65536) for _ in range(4)])
    try:
        while not(state['ended'] or state['truncated']):
            own,other=state['views'][seat],state['views'][1-seat]
            action=_student.choose(own) if own['legal'] else None
            reply=enemy.choose(other) if other['legal'] else None
            if action is not None:rows.append({'observation':own,'student_action':action,'enemy_label':action_label(other,reply)})
            choices=[None,None];choices[seat]=action;choices[1-seat]=reply
            state=_engine.request('step',game='loss_collection',actions=choices)
        lost=state['winner']==1-seat and not state['truncated']
        if lost:
            for row in rows:
                obs=row['observation']
                assert not any('set' in m or 'stats' in m for m in obs['opponent'])
                row['expert_action']=_teacher.choose(obs)
                assert row['expert_action'] in [a['id'] for a in obs['legal']]
                row['category']=category(obs,row['student_action'],row['expert_action'])
                row['outcome']=-1.
            with gzip.open(stem.with_suffix('.observations.json.gz'),'wt',encoding='utf-8') as stream:json.dump({'rows':rows,'outcome':-1.},stream)
        report={'game':index,'seed':seed,'seat':seat,'loss':lost,'truncated':state['truncated'],
                'positions':len(rows) if lost else 0,'teacher_disagreements':sum(r['expert_action']!=r['student_action'] for r in rows) if lost else 0,
                'raw':str(stem.with_suffix('.observations.json.gz')) if lost else None,
                'reference_decisions':len(enemy.audit),'reference_fallbacks':enemy.fallbacks}
        if any(a['errors'] or a['trials']!=32 or a['deadline_passed'] for a in enemy.audit):raise ValueError('Degraded reference game')
        atomic_json(stem.with_suffix('.json'),report)
        return report
    except (RuntimeError,TimeoutError,ValueError) as error:
        report={'game':index,'seed':seed,'seat':seat,'loss':False,'positions':0,'teacher_disagreements':0,
                'error':repr(error),'reference_diagnostics':getattr(enemy,'last_diagnostics',{}),'raw':None}
        attempt=0;failure=stem.with_name(stem.name+'_error.json')
        while failure.exists():attempt+=1;failure=stem.with_name(stem.name+f'_error_{attempt}.json')
        atomic_json(failure,report)
        return report
    finally:
        try:_engine.request('close',game='loss_collection')
        except (RuntimeError,TimeoutError):pass


def verify_teacher(settings):
    reports={prefix:json.loads(Path(settings[prefix+'_validation']).read_text()) for prefix in ('student','teacher')}
    for prefix,report in reports.items():
        if report['checkpoint_sha256']!=hashlib.sha256(Path(settings[prefix]).read_bytes()).hexdigest():raise ValueError('Validation weights differ')
        row=report['results']['reference']
        if not report['complete'] or row['games']<128 or row['errors'] or row['fallbacks'] or row['truncations']:raise ValueError('Incomplete/degraded teacher evidence')
        if report['reference_teams']!='saved' or not report['deterministic']:raise ValueError('Teacher evidence does not match this narrow collection domain')
    for key in ('seed','reference_manifest_sha256','knowledge_sha256'):
        if reports['student'][key]!=reports['teacher'][key]:raise ValueError('Teacher evidence schedules differ')
    teacher=reports['teacher']['results']['reference'];student=reports['student']['results']['reference']
    if teacher['confidence'][0]<=student['confidence'][1]:raise ValueError('Teacher advantage is not established in this domain')
    if settings['seed'] in (92341,190731,290731):raise ValueError('Use a fresh training seed')
    return {'teacher':teacher,'student':student,
            'scope':'Saved-roster mirror against frozen reference only. This does not establish a better teacher on broad teams.',
            'label_meaning':'Public-state preferences of a separately validated teacher, not proofs that every alternative wins.'}


@keep_awake()
def run(output,games=128,positions=160,workers=2,seed=682917):
    output=Path(output);(output/'raw').mkdir(parents=True,exist_ok=True)
    settings={'output':str(output),'games':games,'positions':positions,'seed':seed,
      'student':'runtime_data/ppo_v7_mix_neural/step_000262244.pt','student_runtime':'runtime_data/baselines/v7_e3e2d8a',
      'teacher':'runtime_data/candidates/v6_2m_review/policy.pt','teacher_runtime':'runtime_data/baselines/v6_3cc5468',
      'student_validation':'runtime_data/review_v7_mix_neural/reference/report.json',
      'teacher_validation':'runtime_data/eval_v6_ppo_final_reference/report.json','feature_signature':feature_signature()}
    evidence=verify_teacher(settings)
    manifest=output/'settings.json'
    if manifest.exists() and json.loads(manifest.read_text())!=settings:raise ValueError('Use a fresh output folder for changed settings')
    atomic_json(manifest,settings);completed=[]
    with mp.get_context('spawn').Pool(workers,initializer=initialize,initargs=(settings,),maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(game,range(games)):
            completed.append(row)
            print(json.dumps({'event':'fresh_loss_collection','completed':len(completed),'games':games,
                              'losses':sum(r['loss'] for r in completed),'positions':sum(r['positions'] for r in completed)}),flush=True)
    # Select whole-game chronological blocks until the bounded review budget.
    # Only teacher disagreements are imitation labels; agreements stay in the
    # review file, so agreement is never counted as a discovered mistake.
    inspected=[];shards=[];output_data=output/'corrections';output_data.mkdir(exist_ok=True)
    with Engine() as engine:
        encoder=Encoder(engine=engine)
        for result in sorted(completed,key=lambda r:r['game']):
            if not result['loss'] or len(inspected)>=positions:continue
            with gzip.open(result['raw'],'rt',encoding='utf-8') as stream:raw=json.load(stream)
            rows=raw['rows'][:positions-len(inspected)]
            inspected.extend({'game':result['game'],**row} for row in rows)
            corrections=[r for r in rows if r['expert_action']!=r['student_action']]
            if not corrections:continue
            stem=output_data/f'dagger_v6_{seed}_{result["game"]:06d}'
            encoded=[encoder.encode(row['observation']) for row in corrections]
            payload={key:np.stack([r[key] for r in encoded]) for key in encoded[0]}
            payload.update(labels=np.array([r['expert_action'] for r in corrections]),
                student_actions=np.array([r['student_action'] for r in corrections]),
                enemy_labels=np.array([r['enemy_label'] for r in corrections]),
                outcomes=np.array([-1.-.15*potential(r['observation']) for r in corrections],dtype=np.float32))
            durable_npz(stem.with_suffix('.npz'),**payload)
            # A small loss corpus needs a deliberate game-level holdout. Use
            # chronology only, never loss magnitude or teacher agreement.
            heldout=len(shards)%5==4
            meta={'teacher':'validated_v6','student_sha256':hashlib.sha256(Path(settings['student']).read_bytes()).hexdigest(),
                  'teacher_sha256':hashlib.sha256(Path(settings['teacher']).read_bytes()).hexdigest(),
                  'samples':len(corrections),'feature_version':FEATURE_VERSION,'feature_signature':feature_signature(),
                  'knowledge_sha256':knowledge().sha256,'seed':result['seed'],'reference_fallbacks':0,
                  'verification':evidence,'source_game':result['game'],'heldout':heldout}
            atomic_json(stem.with_suffix('.json'),meta)
            with gzip.open(stem.with_suffix('.observations.json.gz'),'wt',encoding='utf-8') as stream:json.dump({'rows':corrections,'outcome':-1.},stream)
            shards.append({'game':result['game'],'samples':len(corrections),'heldout':heldout})
    with gzip.open(output/'investigated_positions.json.gz','wt',encoding='utf-8') as stream:json.dump(inspected,stream)
    from collections import Counter
    report={'settings':settings,'verification':evidence,'games':len(completed),'loss_games':sum(r['loss'] for r in completed),
            'invalid_attempts':sum(bool(r.get('error')) for r in completed),
            'investigated_positions':len(inspected),'correction_samples':sum(r['samples'] for r in shards),
            'heldout_correction_samples':sum(r['samples'] for r in shards if r['heldout']),
            'categories':dict(Counter(r['category'] for r in inspected if r['expert_action']!=r['student_action'])),
            'shards':shards,'validation_partition':'Every fifth correction game in chronological order; no turns from a held-out game train the policy.',
            'training_performed':False,'earlier_examples_replaced':False}
    atomic_json(output/'report.json',report);print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--games',type=int,default=128);parser.add_argument('--positions',type=int,default=160)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--seed',type=int,default=682917)
    run(**vars(parser.parse_args()))
