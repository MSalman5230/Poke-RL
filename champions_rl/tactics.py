"""Native mechanics queries over explicitly sampled, public-compatible worlds."""
from collections import OrderedDict,Counter
import json
import math
from .engine import Engine
from .knowledge import canonical


class NativeTactics:
    def __init__(self,engine=None):
        self.engine=engine
        self.cache=OrderedDict()
        self.dependencies=None
        self.metrics=Counter()

    def cache_query(self,query):
        q=dict(query)
        sensitive=canonical(q['move']) in self.dependencies['moves']
        from .knowledge import knowledge
        k=knowledge()
        sensitive=sensitive or bool(k.moves.get(canonical(q['move']),{}).get('multihit'))
        for key in ('actor','target'):
            m=q[key]
            ability=canonical(m.get('ability') if m.get('ability') is not None else m['set'].get('ability'))
            if q.get('mega' if key=='actor' else 'target_mega'):
                stone=k.items.get(canonical(m.get('item') or m['set'].get('item')),{}).get('megaStone')
                if isinstance(stone,dict):stone=stone.get(m['set']['species'])
                ability=canonical(k.species_info(stone).get('abilities',{}).get('0',ability)) if stone else ability
            sensitive=sensitive or ability in self.dependencies['abilities']
            sensitive=sensitive or canonical(m.get('item') or m['set'].get('item')) in self.dependencies['items']
        for key in ('actor','target'):
            m=dict(q[key]);s=dict(m['set']);s.pop('name',None)
            # Other moves do not affect this conditional hit calculation.
            # Party/move-history-dependent moves are explicitly unsupported.
            s['moves']=['tackle']
            s['item']=canonical(m.get('item') if m.get('item') is not None else s.get('item'))
            s['ability']=canonical(m.get('ability') if m.get('ability') is not None else s.get('ability'))
            if not q.get('mega' if key=='actor' else 'target_mega'):
                # Damage depends on the resulting stats, not the many spreads
                # and natures which can produce those same stats.
                stats=k.mon_stats(m)
                m['max_hp']=stats['hp'];m['stats']={n:stats[n] for n in ('atk','def','spa','spd','spe')}
                s={n:s[n] for n in ('species','item','ability','moves')}
                s.update(nature='Serious',evs={},level=50)
            m['set']=s
            if key!='actor' or canonical(q['move'])!='ragefist':m.pop('times_attacked',None)
            m['boosts']={k:v for k,v in m.get('boosts',{}).items() if v}
            if not sensitive:m['hp']=1.
            # Durations do not enter this conditional damage calculation. Keep
            # effect identities and known substitute HP, not their start turn.
            m['volatiles']={k:{f:v for f,v in row.items() if f!='turn'} for k,row in m.get('volatiles',{}).items()}
            q[key]=m
        q['conditions']=[{k:{'layers':v.get('layers',0)} for k,v in side.items()} for side in q.get('conditions',[{},{}])]
        q['pseudo']={k:{} for k in q.get('pseudo',{})}
        q['mega']=bool(q.get('mega'));q['target_mega']=bool(q.get('target_mega'))
        if all(m['set']['ability']!='supremeoverlord' for m in (q['actor'],q['target'])):q['faints']=[0,0]
        return q

    @staticmethod
    def summarize(row,query):
        maximum=row['maxhp']
        hp=max(1,math.floor(query['target'].get('hp',1.)*maximum+.5))
        survive=row['survive_full'] and hp==maximum
        rolls=[(min(d,hp-1) if survive else d,w) for d,w in row['rows']]
        return {'min':min(d for d,_ in rolls)/maximum,'max':max(d for d,_ in rolls)/maximum,
                'mean':sum(d*w for d,w in rolls)/maximum,
                'ko':row['accuracy']*sum(w for d,w in rolls if d>=hp),
                **{k:row[k] for k in ('speed','priority','method','reliable','unsupported')}}

    def calculate(self,queries):
        if self.engine is None:self.engine=Engine()
        if self.dependencies is None:
            meta=self.engine.request('tactics_metadata')
            if meta['version']!=4:raise ValueError('Native tactics metadata version mismatch')
            self.dependencies={k:set(meta[k]) for k in ('moves','abilities','items')}
        normalized=[self.cache_query(q) for q in queries]
        keys=[json.dumps(q,sort_keys=True,separators=(',',':')) for q in normalized]
        missing={k:q for k,q in zip(keys,normalized) if k not in self.cache}
        self.metrics['queries']+=len(keys)
        self.metrics['misses']+=len(missing)
        self.metrics['hits_or_batch_duplicates']+=len(keys)-len(missing)
        if missing:
            self.metrics['requests']+=1
            rows=self.engine.request('calculate',queries=list(missing.values()))
            for key,row in zip(missing,rows):
                if not row.get('rows') or any(not math.isfinite(d) or not math.isfinite(w) for d,w in row['rows']):
                    raise ValueError(f'Nonfinite native tactical result: {row}; query={key}')
            self.cache.update(zip(missing,rows))
        result=[self.summarize(self.cache[k],q) for k,q in zip(keys,queries)]
        for k in keys:self.cache.move_to_end(k)
        while len(self.cache)>32768:self.cache.popitem(last=False)
        return result

    @staticmethod
    def query(actor,target,move,obs,mega=False,reverse=False):
        def public(m):
            result={k:m[k] for k in ('set','species','hp','status','boosts','item','ability','types','stats','times_attacked','volatiles') if k in m}
            result['volatiles']={**{k:{} for k in m.get('volatile_ids',[])},**m.get('volatiles',{})}
            return result
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
        outgoing_count=len(queries)
        threat_jobs=[]
        for action in obs['legal']:
            if action.get('revive'):continue
            actor=obs['own'][action['slot']] if action['kind']=='switch' else active
            for world,(enemy,weight) in enumerate(scenarios):
                for move in enemy['set']['moves']:
                    if encoder.k.moves.get(canonical(move),{}).get('category')=='Status':continue
                    query=self.query(enemy,actor,move,obs,reverse=True)
                    query['target_mega']=bool(action.get('mega'))
                    queries.append(query);threat_jobs.append((action['id'],world,weight))
        calculated=self.calculate(queries) if queries else []
        grouped={}
        for (action,weight),row in zip(jobs,calculated[:outgoing_count]):
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
            f[ix['native_tactics']]=float(all(r['reliable'] for r,_ in rows))
        # Conservative survival: each hidden-set hypothesis can select its
        # most threatening damaging move. This is a risk feature, not a promise
        # that the opponent samples every move uniformly or always attacks.
        threats={}
        for (action,world,weight),row in zip(threat_jobs,calculated[outgoing_count:]):
            threats.setdefault(action,{}).setdefault(world,[]).append((row,weight))
        for action,worlds in threats.items():
            f=encoded['actions'][action]
            reliable=all(r['reliable'] for rows in worlds.values() for r,_ in rows)
            f[ix['native_tactics']]=min(f[ix['native_tactics']],float(reliable)) if action in grouped else float(reliable)
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
