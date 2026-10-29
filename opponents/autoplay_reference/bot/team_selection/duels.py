"""TS-11/12: fixed-attack, damage-and-Speed preview estimates (not win odds)."""
import copy
import hashlib
import json
import math
from pathlib import Path
from opponents.autoplay_reference.bot.domain import catalog, moves
from opponents.autoplay_reference.bot.decision.opponent_action import CONTEXTUAL

MODEL = 'preview-direct-race-v2'
MAX_ACTIONS = 32
REACTIVE = frozenset(('counter','mirrorcoat','metalburst','comeuppance','beakblast','shelltrap'))
PRIOR_REFERENCE = REACTIVE | frozenset(('payback','avalanche','revenge','assurance'))
ASSUMPTIONS = ('full starting HP; native entry abilities and supplied Mega forms',
    '92.5% damage; other non-accuracy random rolls use a fixed midpoint',
    'accuracy scales expected damage/cost, not a measured KO probability',
    'first and subsequent attacks retain their measured effects; subsequent profile held fixed',
    'fixed attacks; no voluntary switching, setup, status-only plans or future field changes',
    'Hero activation costs one opposing action, offset by usable Zero-form Flip Turn damage',
    'Flip Turn chip changes enemy HP, never heals Palafin; receiver/return sequence remains abstract')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def scoring_identity():
    return {'model':MODEL,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def member_input(name, member, form=None):
    return {'identity':name,'form':form or member['species'],
        'activation':int(bool(form and form.endswith('-Hero') and member['species']!=form)),
        'set':{'name':name,'species':member['species'],'ability':member['ability'],'item':member['item'],
               'nature':member['nature'],'evs':copy.deepcopy(member['sp']),'level':50,
               'moves':[m['name'] for m in member['moves']]},
        'pp':{catalog.canonical(m['name']):m['pp'] for m in member['moves']}}


def options(member, other):
    references=[catalog.canonical(m['name']) for m in other['moves'] if m['pp']>0]
    return [{'name':catalog.canonical(m['name']),
             'references':references if catalog.canonical(m['name']) in CONTEXTUAL else [],
             'prior_reference':catalog.canonical(m['name']) in PRIOR_REFERENCE}
            for m in member['moves'] if m['pp']>0 and moves.damaging(m['name'])]


def task(ours, enemy, *, opening=False):
    a,b=ours,enemy
    def member(value):
        return {'moves':[{'name':name,'pp':value['pp'][catalog.canonical(name)]} for name in value['set']['moves']]}
    am,bm=member(a),member(b)
    members=copy.deepcopy([a,b])
    modeled=[am,bm]
    copied=[]
    for side,value in enumerate(members):
        if value['set']['species']=='Ditto':
            modeled[side]=member(members[1-side])
            for move in modeled[side]['moves']:
                move['pp']=min(5,catalog.get().move(move['name'])['pp'])
            value['activation']=int(value['set']['ability']!='Imposter')
            copied.append(side)
    result={'seed':[1,2,3,4],'members':members,'options':[options(modeled[0],modeled[1]),options(modeled[1],modeled[0])]}
    if copied:
        result.update(copy_context='opening' if opening else 'active',copy_sides=copied)
    return result


def _profile(rows, name, reference):
    return next((r for r in rows if r['name']==name and r['reference']==reference),None) or \
           next((r for r in rows if r['name']==name and r['reference'] is None),None)


def activation_tasks(input_task):
    """Raw Zero-form dependencies for a usable pre-Hero Flip Turn."""
    members=copy.deepcopy(input_task['members'])
    for member in members:
        if member.get('activation',0) and member.get('set',{}).get('species')=='Palafin':
            member['form']='Palafin'
            member['activation']=0
    result={}
    for side,member in enumerate(input_task['members']):
        if (member.get('activation',0) and member.get('pp',{}).get('flipturn',0)>0 and
                member.get('set',{}).get('species')=='Palafin'):
            result[side]=task(*members,opening=input_task.get('copy_context')=='opening')
    return result


def _race(a,b,sides,activation,first_side,activation_profiles=None,activation_replies=None):
    hp=[float(s['hp']) for s in sides]
    maxhp=[float(s['maxhp']) for s in sides]
    used=[0,0]
    waiting=list(activation)
    charging=[0,0]
    # Compare exact action priority, including native fractional priority, before Speed.
    profiles=[a,b]
    activation_profiles=activation_profiles or [None,None]
    activation_replies=activation_replies or profiles
    activation_facts=[{'required':bool(activation[i]),'flip_turn_executed':False,
        'flip_turn_damage':0.,'incoming_damage':0.,'self_cost':0.,'net_cost_hp_percent':0.} for i in (0,1)]
    def first_actor(current):
        def key(p):
            row=p['first']
            return row['priority'],row.get('fractional_priority',0),row['speed']
        order_a,order_b=(key(p) for p in current)
        return 0 if order_a>order_b else 1 if order_b>order_a else first_side
    opening_first=None
    exchanges=0
    for exchanges in range(1,MAX_ACTIONS+1):
        starting_hp=list(hp)
        activating=[bool(v) for v in waiting]
        current=list(profiles)
        for side in (0,1):
            if activating[side] and activation_profiles[side] is not None:
                current[side]=activation_profiles[side]
            elif not activating[side] and activating[1-side] and activation_profiles[1-side] is not None:
                current[side]=activation_replies[side]
        first=first_actor(current)
        if opening_first is None:
            opening_first=first
        order=[first,1-first]
        flinched=[False,False]
        for side in order:
            foe=1-side
            if hp[side]<=0 or hp[foe]<=0:
                break
            if waiting[side]:
                waiting[side]-=1
                if activation_profiles[side] is None or flinched[side]:
                    continue
                # This is the initial Zero-form move, not a Hero attack or heal.
                row=activation_profiles[side]['first']
                activation_facts[side]['flip_turn_executed']=True
            else:
                if flinched[side]:
                    continue
                row=current[side]['first' if used[side]==0 else 'repeat']
                cost=max(1,row['actions'])
                charging[side]+=1
                if charging[side]<cost:
                    continue
                charging[side]=0
                used[side]+=1
            damage=max(0.,row['damage'])*row['accuracy']
            loss=row['source_loss']*row['accuracy']
            if (current[side]['name'] in ('suckerpunch','upperhand') and side!=first or
                current[side]['name'] in ('counter','mirrorcoat','metalburst','comeuppance','shelltrap') and side==first or
                current[side]['name']=='focuspunch' and side!=first and current[foe]['first']['damage']>0):
                damage=0.
                loss=0.
            before_foe,before_self=hp[foe],hp[side]
            hp[foe]=max(0.,hp[foe]-damage)
            hp[side]=max(0.,min(maxhp[side],hp[side]-loss))
            if activating[foe]:
                activation_facts[foe]['incoming_damage']+=before_foe-hp[foe]
            if activating[side]:
                activation_facts[side]['flip_turn_damage']+=before_foe-hp[foe]
                activation_facts[side]['self_cost']+=before_self-hp[side]
            if row.get('flinch') and row['accuracy']>=1:
                flinched[foe]=True
        for side in (0,1):
            if activating[side]:
                activation_facts[side]['net_cost_hp_percent']=100*(starting_hp[side]-hp[side])/maxhp[side] - \
                    100*(starting_hp[1-side]-hp[1-side])/maxhp[1-side]
        if min(hp)<=0:
            break
        if all(max(0,p['repeat']['damage'])==0 for p in profiles) and all(used):
            break
    # Advantage in [-1,1], with a full survivor advantage before HP margin.
    score=((hp[0]>0)-(hp[1]>0)+(hp[0]/maxhp[0]-hp[1]/maxhp[1]))/2
    for side,facts in enumerate(activation_facts):
        facts['incoming_hp_percent']=100*facts['incoming_damage']/maxhp[side]
        facts['flip_turn_hp_percent']=100*facts['flip_turn_damage']/maxhp[1-side]
        facts['self_cost_hp_percent']=100*facts['self_cost']/maxhp[side]
    return {'score':score,'remaining_hp':hp,'exchanges':exchanges,
            'capped':exchanges==MAX_ACTIONS and min(hp)>0,'first_side':opening_first,
            'activation':activation_facts}


def score(profile, input_task, *, activation_lookup=None):
    sides=profile['sides']
    if len(sides)!=2 or any(s['maxhp']<=0 or not math.isfinite(s['speed']) for s in sides):
        raise ValueError('Invalid preview profile sides')
    names=[list(dict.fromkeys(p['name'] for p in rows)) for rows in profile['profiles']]
    table=[]
    activation=[m.get('activation',0) for m in input_task['members']]
    activation_raw={}
    dependencies=activation_tasks(input_task)
    for side,dependency in dependencies.items():
        if activation_lookup is None:
            raise ValueError('Zero-form Flip Turn profile is required for Hero activation')
        raw=activation_lookup(dependency)
        validate(raw,dependency)
        activation_raw[side]=raw
    for own in names[0]:
        replies=[]
        for enemy in names[1]:
            a=_profile(profile['profiles'][0],own,enemy)
            b=_profile(profile['profiles'][1],enemy,own)
            if a is None or b is None:
                raise ValueError('Incomplete contextual preview profile')
            names_by_side=[own,enemy]
            activation_profiles=[None,None]
            activation_replies=[a,b]
            for side,raw in activation_raw.items():
                activation_profiles[side]=_profile(raw['profiles'][side],'flipturn',names_by_side[1-side])
                if activation_profiles[side] is None:
                    raise ValueError('Incomplete Zero-form Flip Turn activation profile')
                if 1-side not in activation_raw:
                    activation_replies[1-side]=_profile(profile['profiles'][1-side],names_by_side[1-side],'flipturn')
                    if activation_replies[1-side] is None:
                        raise ValueError('Incomplete opponent reply against activation Flip Turn')
            outcomes=[_race(a,b,sides,activation,side,activation_profiles,activation_replies) for side in (0,1)]
            replies.append({'our_move':own,'enemy_move':enemy,'score':sum(o['score'] for o in outcomes)/2,
                            'orders':outcomes})
        if replies:
            table.append({'our_move':own,'worst_reply':min(replies,key=lambda r:r['score']),'replies':replies})
    if not table:
        return {'score':0.,'our_move':None,'enemy_move':None,'comparisons':[],
                'limitation':'one side has no modeled damaging attack','assumptions':ASSUMPTIONS}
    best=max(table,key=lambda r:r['worst_reply']['score'])
    return {'score':best['worst_reply']['score'],'our_move':best['our_move'],
            'enemy_move':best['worst_reply']['enemy_move'],'comparisons':table,'assumptions':ASSUMPTIONS,
            'activation_profiles':{str(side):sha(t) for side,t in dependencies.items()}}


def validate(profile, input_task):
    if not isinstance(profile,dict) or len(profile.get('sides',[]))!=2 or len(profile.get('profiles',[]))!=2:
        raise ValueError('Malformed preview profile')
    for side,rows in enumerate(profile['profiles']):
        expected={(o['name'],ref) for o in input_task['options'][side] for ref in (o['references'] or [None])}
        if profile.get('copy_failures',{}).get(str(side)):
            expected=set()
        actual={(r['name'],r['reference']) for r in rows}
        if expected!=actual or len(rows)!=len(expected):
            raise ValueError('Incomplete/duplicate preview profile options')
        for row in rows:
            for phase in ('first','repeat'):
                value=row[phase]
                if any(not isinstance(value.get(k),(int,float)) or not math.isfinite(value[k])
                       for k in ('damage','source_loss','accuracy','actions','speed','priority')) or not 0<=value['accuracy']<=1:
                    raise ValueError('Invalid preview attack measurement')
    return profile
