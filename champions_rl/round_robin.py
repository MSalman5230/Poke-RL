"""Paired policy round-robin with frozen worker inputs and seed sensitivity."""
import atexit
from collections import defaultdict
import hashlib
from itertools import combinations_with_replacement
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
import numpy as np
from .arena import load_teams,play_match
from .engine import ROOT,Engine
from .features import Encoder
from .policies import ScriptedBot
from .reference import ReferenceBot
from .teams import TeamPool,fingerprint
from .persistence import atomic_json,append_json


def _initialize(ready,teams):
    global _engine,_encoder,_teams
    _encoder=Encoder()
    _teams=teams
    _engine=Engine()
    _engine.request('new',game='warmup',teams=[teams[0],teams[0]],seed=[1,2,3,4])
    atexit.register(_engine.close)
    from multiprocessing.util import Finalize
    Finalize(None,_engine.close,exitpriority=10)
    ready.put(True)


def _pair(job):
    a,b,index,seed,mirror=job
    rng=random.Random(seed+index)
    teams=[_teams[0],_teams[0]] if mirror else [rng.choice(_teams),rng.choice(_teams)]
    rows=[]
    for leg in (0,1):
        names=[a,b] if leg==0 else [b,a]
        policies=[ReferenceBot(seed+index*13+int(n=='reference'),strict=True) if n=='reference'
                  else ScriptedBot(n,seed+index*13+list(STYLES).index(n),_encoder) for n in names]
        try:
            result=play_match(_engine,teams,policies,seed+index,game='round-robin')
            winner=result['winner']
            score=.5 if winner is None and not result['truncated'] else float(winner==leg)
            audits=[d for p in policies for d in getattr(p,'audit',[])]
            rows.append({'leg':leg,'score_a':score,'winner_seat':winner,'truncated':result['truncated'],
                         'turns':result['turns'],'seconds':result['seconds'],
                         'reference_decisions':len(audits),'trials':sorted({d['trials'] for d in audits}),
                         'fallbacks':sum(getattr(p,'fallbacks',0) for p in policies),'error':None})
        except Exception as exc:
            rows.append({'leg':leg,'score_a':0.,'error':repr(exc),'truncated':False,'fallbacks':0})
    return {'a':a,'b':b,'pair':index,'seed':seed+index,'teams':[fingerprint(t) for t in teams],'legs':rows}


STYLES=('random','damage','balanced','setup','defensive','stochastic','reference')


def summarize(pairs,styles):
    grouped=defaultdict(list)
    for row in pairs:
        grouped[row['a'],row['b']].append(row)
    results={}
    totals=defaultdict(lambda:[0.,0])
    for (a,b),games in grouped.items():
        scores=np.array([sum(x['score_a'] for x in row['legs'])/2 for row in games])
        rng=np.random.default_rng(901)
        boot=scores[rng.integers(len(scores),size=(2000,len(scores)))].mean(1)
        legs=[x for row in games for x in row['legs']]
        results[f'{a} vs {b}']={'games':len(legs),'score_a':float(scores.mean()),
            'paired_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
            'between_seed_pair_score_sd':float(scores.std()),
            'split_pairs':sum(0<s<1 for s in scores).item(),
            'errors':sum(bool(x['error']) for x in legs),'truncations':sum(x['truncated'] for x in legs),
            'fallbacks':sum(x['fallbacks'] for x in legs)}
        if a!=b:
            totals[a][0]+=sum(scores)*2
            totals[a][1]+=len(legs)
            totals[b][0]+=sum(1-s for s in scores)*2
            totals[b][1]+=len(legs)
    ranking=sorted([{'policy':name,'score':s/max(1,n),'games':n} for name,(s,n) in totals.items()],key=lambda x:-x['score'])
    return results,ranking


def round_robin(output,games=64,workers=8,seed=41173,styles=STYLES,roster='saved',resume=False,retry_errors=False):
    if games<2 or games%2 or not 1<=workers<=16:
        raise ValueError('Use an even game count and 1–16 workers')
    if 'reference' in styles and roster!='saved':
        raise ValueError('Strict reference baseline currently requires its supported saved roster')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=resume)
    paths=[*sorted((ROOT/'champions_rl').glob('*.py')),*sorted((ROOT/'simulator').glob('*.cjs')),
           ROOT/'data/sets.json',ROOT/'data/teams_validation.json',ROOT/'data/knowledge.json']
    hashes={}
    for path in paths:
        relative=path.relative_to(ROOT)
        target=output/'code_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        content=path.read_bytes()
        if resume and target.exists():
            if relative.name not in ('round_robin.py','persistence.py') and target.read_bytes()!=content:
                raise ValueError(f'Resume must use the frozen baseline code/data: {relative}')
            content=target.read_bytes()
        else:
            target.write_bytes(content)
        hashes[relative.as_posix()]=hashlib.sha256(content).hexdigest()
    teams=[TeamPool().data['reference_team']] if roster=='saved' else load_teams('validation')
    jobs=[(a,b,i,seed,roster=='saved') for a,b in combinations_with_replacement(styles,2) for i in range(games//2)]
    # Interleave expensive reference pairings with ordinary games.
    random.Random(seed).shuffle(jobs)
    context=mp.get_context('spawn')
    ready=context.Queue()
    pairs=[]
    if resume and (output/'pairs.jsonl').exists():
        raw=(output/'pairs.jsonl').read_bytes()
        seen=set()
        for line in raw.splitlines():
            try:
                row=json.loads(line)
                key=(row['a'],row['b'],row['pair'])
                if retry_errors and any(x['error'] for x in row['legs']):
                    append_json(output/'invalid_attempts.jsonl',row)
                    continue
                if key not in seen:
                    seen.add(key);pairs.append(row)
            except (ValueError,KeyError,UnicodeError):
                continue
        backup=output/'pairs.before_recovery.jsonl'
        if not backup.exists():backup.write_bytes(raw)
        with (output/'pairs.jsonl').open('w',encoding='utf-8') as handle:
            for row in pairs:handle.write(json.dumps(row)+'\n')
        jobs=[j for j in jobs if (j[0],j[1],j[2]) not in seen]
    total_jobs=len(pairs)+len(jobs)
    if not jobs:
        return json.loads((output/'report.json').read_text())
    atomic_json(output/'run.json',{'games':games,'workers':workers,'seed':seed,'styles':list(styles),'roster':roster,'source_sha256':hashes})
    started=time.monotonic()
    with context.Pool(workers,initializer=_initialize,initargs=(ready,teams)) as pool:
        for _ in range(workers):
            ready.get(timeout=120)
        print(json.dumps({'event':'round_robin_workers_frozen','workers':workers,'pair_jobs':len(jobs)}),flush=True)
        for result in pool.imap_unordered(_pair,jobs,chunksize=1):
            pairs.append(result)
            append_json(output/'pairs.jsonl',result)
            rows,ranking=summarize(pairs,styles)
            report={'roster':roster,'seed':seed,'games_per_pair':games,'policies':list(styles),'source_sha256':hashes,
                    'pair_jobs_complete':len(pairs),'pair_jobs_total':total_jobs,'seconds_this_invocation':time.monotonic()-started,
                    'complete':len(pairs)==total_jobs,'results':rows,'ranking':ranking,
                    'invalid_attempts_archive':'invalid_attempts.jsonl' if (output/'invalid_attempts.jsonl').exists() else None,
                    'interpretation':'Paired seeds, same teams, swapped policies; diagonal policies are calibration. Seed variation is not an upper bound on achievable skill. Ranking is provisional if incomplete or any games failed.'}
            atomic_json(output/'report.json',report)
            if len(pairs)%16==0 or len(pairs)==total_jobs:
                print(json.dumps({'event':'round_robin_progress','complete':len(pairs),'total':total_jobs,'ranking':ranking}),flush=True)
        pool.close()
        pool.join()
    ready.close()
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--games',type=int,default=64)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--styles',default=','.join(STYLES))
    parser.add_argument('--roster',choices=['saved','validation'],default='saved')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-errors',action='store_true')
    args=parser.parse_args()
    round_robin(args.output,args.games,args.workers,styles=args.styles.split(','),roster=args.roster,resume=args.resume,retry_errors=args.retry_errors)
