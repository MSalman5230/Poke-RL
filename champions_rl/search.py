"""Optional one-ply policy/value search over public-compatible hidden sets."""
from collections import Counter
import copy
import random
import numpy as np
import torch
from .features import Encoder
from .knowledge import knowledge,canonical
from .model import NeuralBot,encode_for_model,tensor_batch
from .opponent_labels import switch_offset
from .rewards import potential


def hypothetical_world(obs,beliefs,rng,replies):
    """Reconstruct a full party from public evidence; own sets remain exact."""
    k=knowledge()
    if obs['phase']!='move':raise ValueError('Search supports simultaneous move decisions')
    if any(m.get('set') or m.get('stats') or 'selected' in m for m in obs['opponent']):
        raise ValueError('Search was passed privileged opponent information')
    for event in obs['history']:
        if obs['turn']-event['turn']<=3 and event['move'] in ('wish','futuresight','doomdesire','healingwish','lunardance'):
            raise ValueError('Delayed effect is outside the prototype reconstruction')
    teams=[copy.deepcopy([m['set'] for m in obs['own']]),[beliefs.sample(m,rng) for m in obs['opponent']]]
    for team in teams:
        for i,s in enumerate(team):s['name']='unit'+str(i)
    picked=[i for i,m in enumerate(obs['opponent']) if m['seen'] or m['active']]
    if len(picked)>3:raise ValueError('Unresolved public party identity')
    remaining=[i for i in range(6) if i not in picked]
    picked+=rng.sample(remaining,3-len(picked))
    selections=[[i for i,m in enumerate(obs['own']) if m['selected']],picked]
    sides=[]
    for side,roster in enumerate((obs['own'],obs['opponent'])):
        indices=sorted(selections[side],key=lambda i:not roster[i]['active'])
        active=roster[indices[0]]
        if not active['active'] or active['hp']<=0:raise ValueError('Active combatant unavailable')
        members=[]
        for i in indices:
            mon=roster[i];s=copy.deepcopy(teams[side][i])
            if mon.get('volatiles',{}).get('transform'):raise ValueError('Transform reconstruction unsupported')
            if mon.get('item') is not None:s['item']=mon['item']
            if mon.get('ability') is not None:s['ability']=mon['ability']
            stats=k.mon_stats({**mon,'set':s})
            stats={stat:int(value) for stat,value in stats.items()}
            hp=mon.get('current_hp',max(1,round(mon['hp']*stats['hp'])) if mon['hp']>0 else 0)
            pp={m['id']:m['pp'] for m in mon.get('moves',[]) if isinstance(m,dict)}
            s['species']=mon['species']
            members.append({'identity':mon['base_species'],'set':s,'stats':stats,'hp':hp,'maxhp':stats['hp'],
                'types':mon['types'],'status':mon['status'],'progress':mon.get('status_progress',{}),
                'ability':s['ability'],'base_ability':teams[side][i]['ability'],'item':s['item'],
                'moves':[{'id':canonical(m),'pp':pp.get(canonical(m),max(0,k.moves[canonical(m)]['pp']-mon.get('pp_used',{}).get(canonical(m),0)))} for m in s['moves']],
                'times_attacked':mon.get('times_attacked',0),'mega_evolved':'-Mega' in mon['species']})
        effects=[]
        for key,row in active.get('volatiles',{}).items():
            if row.get('single') and row['turn']<obs['turn']:continue
            if key.startswith('fallen'):continue  # restored from public totalFainted
            effects.append({'id':key,'elapsed':max(0,obs['turn']-row['turn']),**({'move':row['move']} if row.get('move') else {})})
        if canonical(members[0]['item']) in ('choicescarf','choiceband','choicespecs') and active.get('last_move'):
            effects.append({'id':'choicelock','move':active['last_move']})
        # Active own-only condition IDs are available in its normal request.
        if side==0 and 'mustrecharge' in active.get('volatile_ids',[]):effects.append({'id':'mustrecharge'})
        conditions={}
        for key,row in obs['conditions'][side].items():
            if key in ('spikes','toxicspikes','stealthrock','stickyweb'):conditions[key]=row['layers']
            else:
                duration=4 if key=='tailwind' else 5
                if key in ('reflect','lightscreen','auroraveil') and any(canonical(m['item'])=='lightclay' for m in members):duration=8
                conditions[key]=max(1,duration-(obs['turn']-row['turn']))
        sides.append({'pokemon':members,'boosts':active['boosts'],'volatiles':effects,'observation':{},
                      'fresh':active.get('entry_turn')==obs['turn'],'last_move':active.get('last_move'),
                      'mega_used':obs['mega'][side],'total_fainted':len(obs['faints'][side]),'conditions':conditions})
    field={'weather':obs['weather'],'weather_turns':max(1,5-obs['turn']+obs.get('weather_since',obs['turn'])),
           'terrain':obs['terrain'],'terrain_turns':max(1,5-obs['turn']+obs.get('terrain_since',obs['turn'])),
           'trickroom':max(1,5-obs['turn']+obs['pseudo']['trickroom']['turn']) if 'trickroom' in obs['pseudo'] else 0,
           'extra':{key:{'remaining':max(1,5-obs['turn']+row['turn'])} for key,row in obs['pseudo'].items() if key!='trickroom'}}
    return {'teams':teams,'input':{'format':obs['format'],'kind':'turn','turn':obs['turn'],'sides':sides,
                                  'field':field,'wishes':{},'future_moves':{}},
            'seed':[rng.randrange(65536) for _ in range(4)],
            'rollout_seeds':[[rng.randrange(65536) for _ in range(4)] for _ in range(replies)]}


def paired_search_margin(scores,policy,rng,min_worlds=16,margin=.02):
    """World-block bootstrap; paired score confidence is not battle validation."""
    if len(next(iter(scores.values())))<min_worlds:return policy,{}
    n=len(scores[policy]);sample=rng.integers(0,n,size=(4096,n))
    alpha=.05/max(1,len(scores)-1)
    intervals={};eligible=[]
    for action,values in scores.items():
        if action==policy:continue
        delta=np.asarray(values)-scores[policy]
        low,high=np.quantile(delta[sample].mean(1),[alpha/2,1-alpha/2])
        intervals[action]=[float(low),float(high)]
        if low>margin:eligible.append(action)
    return max(eligible,key=lambda a:np.mean(scores[a])) if eligible else policy,intervals


class SearchBot(NeuralBot):
    def __init__(self,model,encoder=None,worlds=16,replies=2,candidates=3,seed=101,confidence=.95,critic=None,min_worlds=16):
        super().__init__(model,encoder,deterministic=True,name='one_ply_search')
        self.worlds=worlds;self.replies=replies;self.candidates=candidates
        self.rng=random.Random(seed);self.confidence=confidence;self.critic=critic;self.min_worlds=min_worlds
        self.statistics=Counter();self.last_search={}

    def choose(self,obs):
        encoded=encode_for_model(self.model,self.encoder,obs)
        with torch.no_grad():
            logits,_,enemy=self.model(tensor_batch([encoded],self.device))
            probabilities=logits[0].softmax(-1).cpu().numpy()
            enemy=enemy[0].softmax(-1).cpu().numpy()
        policy=int(probabilities.argmax())
        self.statistics['decisions']+=1
        if self.critic is None or obs['phase']!='move' or len(obs['legal'])<2 or probabilities[policy]>=self.confidence:
            self.statistics['policy_only']+=1
            self.last_search={'used':False,'reason':'private critic unavailable, phase, forced action, or high policy confidence'}
            return policy
        try:
            worlds=[hypothetical_world(obs,self.encoder.beliefs,self.rng,self.replies) for _ in range(self.worlds)]
            options=sorted((a['id'] for a in obs['legal']),key=lambda i:-probabilities[i])[:self.candidates]
            k=knowledge()
            reply_probabilities={'move:'+mid:float(enemy[k.token('moves',mid)]) for mid in k.moves}
            reply_probabilities.update({'switch:'+str(i):float(enemy[switch_offset()+i]) for i in range(6)})
            if self.encoder.native is None:raise ValueError('Search requires a native engine')
            if self.encoder.native.engine is None:
                from .engine import Engine
                self.encoder.native.engine=Engine()
            result=self.encoder.native.engine.request('lookahead',observation=obs,worlds=worlds,candidates=options,
                                                       replies=self.replies,reply_probabilities=reply_probabilities)
            # Compare candidates on the same completed worlds and replies. Never
            # let a failed rollout disappear only from a candidate's denominator.
            groups={}
            for row in result['rows']:groups.setdefault((row['world'],row['reply']),{})[row['action']]=row
            # A world is retained only if every reply/candidate completed. This
            # prevents selective failures from favoring one candidate.
            complete=[g for (world,reply),g in groups.items() if
                      all(set(groups.get((world,r),{}))==set(options) for r in range(self.replies))]
            if not complete:raise ValueError(str(result['errors'] or 'No complete candidate group'))
            rows=[g[a] for g in complete for a in options]
            with torch.no_grad():
                values=[]
                for start in range(0,len(rows),128):
                    batch=tensor_batch([self.encoder.encode(r['critic_observation'],entities_only=True) for r in rows[start:start+128]],self.device)
                    values.extend(self.critic(batch).cpu().tolist())
            by_world={}
            for row,value in zip(rows,values):
                outcome=0 if row['winner'] is None else 1 if row['winner']==0 else -1
                after=0 if row['ended'] else potential(row['observation'])
                score=(outcome if row['ended'] else float(value))+.15*(after-potential(obs))
                by_world.setdefault(row['world'],{}).setdefault(row['action'],[]).append(score)
            scores={a:np.asarray([np.mean(world[a]) for world in by_world.values()]) for a in options}
            means={a:float(np.mean(v)) for a,v in scores.items()}
            action,intervals=paired_search_margin(scores,policy,np.random.default_rng(self.rng.randrange(2**31)),self.min_worlds)
            self.statistics['searched']+=1;self.statistics['changed']+=action!=policy
            self.statistics['world_errors']+=len(result['errors'])
            self.last_search={'used':True,'policy':policy,'action':action,'scores':means,'groups':len(complete),
                              'worlds':len(by_world),'paired_score_intervals':intervals,'private_critic':True,'errors':result['errors']}
            return action
        except (ValueError,RuntimeError) as error:
            self.statistics['fallbacks']+=1
            self.last_search={'used':False,'reason':str(error)}
            return policy
