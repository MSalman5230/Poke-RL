"""Native mechanics queries over explicitly sampled, public-compatible worlds."""
from collections import OrderedDict
import json
from .engine import Engine
from .knowledge import canonical


class NativeTactics:
    def __init__(self,engine=None):
        self.engine=engine
        self.cache=OrderedDict()

    def calculate(self,queries):
        if self.engine is None:self.engine=Engine()
        keys=[json.dumps(q,sort_keys=True,separators=(',',':')) for q in queries]
        missing=list(dict.fromkeys(k for k in keys if k not in self.cache))
        if missing:
            rows=self.engine.request('calculate',queries=[json.loads(k) for k in missing])
            for key,row in zip(missing,rows):
                if any(row.get(n) is None for n in ('min','max','mean','ko')):
                    raise ValueError(f'Nonfinite native tactical result: {row}; query={key}')
            self.cache.update(zip(missing,rows))
        result=[self.cache[k] for k in keys]
        while len(self.cache)>4096:self.cache.popitem(last=False)
        return result

    @staticmethod
    def query(actor,target,move,obs,mega=False,reverse=False):
        def public(m):
            return {k:m[k] for k in ('set','species','hp','status','boosts','item','ability','types','stats') if k in m}
        return {'actor':public(actor),'target':public(target),'move':move,'mega':mega,
                'weather':obs.get('weather',''),'terrain':obs.get('terrain',''),'pseudo':obs.get('pseudo',{}),
                'conditions':list(reversed(obs['conditions'])) if reverse else obs['conditions'],
                'faints':[sum(p.get('fainted',False) for p in team) for team in
                          ([obs['opponent'],obs['own']] if reverse else [obs['own'],obs['opponent']]) ]}

    def enrich(self,encoder,obs,encoded):
        if obs['phase']=='preview' or not obs['legal']:return
        active=next(p for p in obs['own'] if p['active'])
        target=next((p for p in obs['opponent'] if p['active']),None)
        if target is None:return
        scenarios=encoder.beliefs.scenarios(target)[:3]
        total=sum(w for _,w in scenarios)
        scenarios=[(m,w/total) for m,w in scenarios]
        queries=[];jobs=[]
        for action in obs['legal']:
            if action['kind']!='move':continue
            for enemy,weight in scenarios:
                queries.append(self.query(active,enemy,action['move'],obs,action['mega']))
                jobs.append((action['id'],weight))
        grouped={}
        for (action,weight),row in zip(jobs,self.calculate(queries) if queries else []):
            grouped.setdefault(action,[]).append((row,weight))
        ix={name:i for i,name in enumerate(encoder.action_names)}
        for action,rows in grouped.items():
            f=encoded['actions'][action]
            for key,out in [('mean','estimated_damage'),('ko','ko_probability')]:f[ix[out]]=sum(r[key]*w for r,w in rows)
            f[ix['damage_min']]=min(r['min'] for r,_ in rows)
            f[ix['damage_max']]=max(r['max'] for r,_ in rows)
            f[ix['ko_margin']]=max(-1,min(1,f[ix['estimated_damage']]-target['hp']))
            reverse='trickroom' in obs['pseudo']
            f[ix['move_first_probability']]=sum(w*(.5 if r['speed'][0]==r['speed'][1] else float((r['speed'][0]<r['speed'][1]) if reverse else (r['speed'][0]>r['speed'][1]))) for r,w in rows)
            f[ix['native_tactics']]=1.
        # Conservative survival: each hidden-set hypothesis can select its
        # most threatening damaging move. This is a risk feature, not a promise
        # that the opponent samples every move uniformly or always attacks.
        queries=[];jobs=[]
        for action in obs['legal']:
            if action.get('revive'):continue
            actor=obs['own'][action['slot']] if action['kind']=='switch' else active
            for world,(enemy,weight) in enumerate(scenarios):
                for move in enemy['set']['moves']:
                    if encoder.k.moves.get(canonical(move),{}).get('category')=='Status':continue
                    query=self.query(enemy,actor,move,obs,reverse=True)
                    query['target_mega']=bool(action.get('mega'))
                    queries.append(query);jobs.append((action['id'],world,weight))
        threats={}
        for (action,world,weight),row in zip(jobs,self.calculate(queries) if queries else []):
            threats.setdefault(action,{}).setdefault(world,[]).append((row,weight))
        for action,worlds in threats.items():
            f=encoded['actions'][action]
            f[ix['native_tactics']]=1.
            f[ix['incoming_damage']]=sum(max(r['mean'] for r,_ in rows)*rows[0][1] for rows in worlds.values())
            f[ix['survival_probability']]=1-sum(max(r['ko'] for r,_ in rows)*rows[0][1] for rows in worlds.values())
            if action in grouped:
                probability=0.
                for world,(own,weight) in enumerate(grouped[action]):
                    replies=worlds.get(world,[])
                    for enemy,_ in replies:
                        first=.5 if own['speed'][0]==own['speed'][1] else float((own['speed'][0]<own['speed'][1]) if 'trickroom' in obs['pseudo'] else (own['speed'][0]>own['speed'][1]))
                        if own['priority']!=enemy['priority']:first=float(own['priority']>enemy['priority'])
                        probability+=weight*first/max(1,len(replies))
                f[ix['move_first_probability']]=probability
