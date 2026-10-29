import copy
import json
import subprocess
import numpy as np
from champions_rl.features import Encoder
from champions_rl.knowledge import knowledge,canonical
from champions_rl.teams import TeamPool


def mon(name,**kwargs):
    return {'species':name,'base_species':name,'types':knowledge().species_info(name)['types'],
            'moves':[],'item':None,'ability':None,'hp':1.,'boosts':{},**kwargs}


def test_one_revealed_status_move_does_not_erase_hidden_attacks():
    e=Encoder()
    g=mon('Gholdengo')
    before=e.best_damage(g,mon('Garchomp'))
    g['moves']=['nastyplot']
    assert {'nastyplot','shadowball','makeitrain'}<=set(e.attacks(g))
    assert before>0 and e.best_damage(g,mon('Garchomp'))>0.2
    for candidate,_ in e.beliefs.candidates(g):
        assert 'nastyplot' in map(canonical,candidate['moves'])
    g['moves']=['nastyplot','recover','thunderbolt','shadowball']
    assert set(e.attacks(g))==set(g['moves'])


def test_effective_speed_scarf_boost_paralysis_weather_and_tailwind():
    k=knowledge()
    s=next(s for s in TeamPool().data['reference_team'] if s['species']=='Garchomp')
    g=mon('Garchomp',set=s,item=canonical(s['item']),ability=canonical(s['ability']))
    assert k.mon_stats(g)['spe']==154
    assert k.effective_speed(g)==231
    assert k.effective_speed(g,conditions={'tailwind':{}})==462
    assert k.effective_speed({**g,'status':'par'})==115
    assert k.effective_speed({**g,'boosts':{'spe':1}})==346
    assert k.effective_speed({**g,'ability':'swiftswim'},'raindance')==462


def test_usage_stats_and_public_memory_are_explicit():
    e=Encoder()
    g=mon('Gholdengo',moves=['shadowball'],pp_used={'shadowball':4},status='slp',status_progress={'sleepElapsed':2,'sleepSource':'rest'})
    assert e.k.mon_stats(g)['spa']>e.k.stats('Gholdengo')['spa']
    x=e.mon_features(g,False,{'turn':4,'weather':'','conditions':[{},{}]})
    assert x[e.mon_names.index('sleep_elapsed')]==2/3
    assert x[e.mon_names.index('rest_sleep')]==1
    assert x[e.mon_names.index('pp_0')]<1


def test_native_damage_uses_correct_offensive_stats_and_fixed_damage():
    from champions_rl.tactics import NativeTactics
    from champions_rl.engine import Engine
    team=TeamPool().data['reference_team']
    def known(s):return mon(s['species'],set=s,item=canonical(s['item']),ability=canonical(s['ability']))
    a,d=known(team[-1]),known(team[1])
    with Engine() as engine:
        native=NativeTactics(engine)
        def calc(move,attacker=a,defender=d):
            return native.calculate([{'actor':attacker,'target':defender,'move':move,'conditions':[{},{}]}])[0]
        body=calc('bodypress')['mean']
        assert calc('bodypress',{**a,'boosts':{'atk':6}})['mean']==body
        assert calc('bodypress',{**a,'boosts':{'def':2}})['mean']>body*1.8
        foul=calc('foulplay')['mean']
        assert calc('foulplay',defender={**d,'boosts':{'atk':2}})['mean']>foul*1.8
        fixed=calc('seismictoss')
        assert np.isclose(fixed['min'],50/knowledge().mon_stats(d)['hp'])
        assert fixed['min']==fixed['max']
        assert calc('earthquake')['speed']==[231,167]
        hits=calc('iciclespear')
        assert hits['max']>hits['min']*2 and 0<=hits['ko']<=1


def known(name,item='',ability=None,**kwargs):
    k=knowledge()
    ability=ability or next(iter(k.species_info(name)['abilities'].values()))
    s={'species':name,'item':item,'ability':ability,'nature':'Serious','evs':{},'moves':['tackle']}
    return mon(name,set=s,item=canonical(item),ability=canonical(ability),**kwargs)


def test_damage_distributions_match_resolved_moves_and_restore_state():
    from champions_rl.engine import Engine,ROOT
    from champions_rl.tactics import NativeTactics
    def q(a,d,m):return {'actor':a,'target':d,'move':m,'conditions':[{},{}]}
    queries=[
        q(known('Archaludon',boosts={'spa':6}),known('Mimikyu',ability='Disguise'),'flashcannon'),
        q(known('Annihilape',times_attacked=5),known('Volcarona'),'ragefist'),
        q(known('Archaludon'),known('Garchomp',volatiles={'substitute':{'hp':30}}),'dragonpulse'),
        q(known('Archaludon',boosts={'spa':6}),known('Garchomp',item='Focus Sash'),'dragonpulse'),
        q(known('Archaludon',boosts={'spa':6}),known('Archaludon',ability='Sturdy'),'aurasphere'),
        q(known('Archaludon',ability='Mold Breaker',boosts={'spa':6}),known('Archaludon',ability='Sturdy'),'aurasphere'),
        q(known('Volcarona'),known('Gholdengo',item='Occa Berry'),'flamethrower'),
        q(known('Volcarona',hp=.4),known('Garchomp'),'eruption'),
        q(known('Garchomp'),known('Dragonite',ability='Multiscale'),'dragonclaw'),
        q(known('Garchomp'),known('Dragonite',ability='Multiscale',hp=.8),'dragonclaw'),
        q(known('Primarina'),known('Garchomp',hp=.4),'brine'),
    ]
    queries[0]['target']['set']['evs']={'hp':2}
    actual=json.loads(subprocess.run(['node',str(ROOT/'tests/native_outcomes.cjs')],input=json.dumps(queries),
                                    text=True,encoding='utf-8',capture_output=True,check=True).stdout)
    with Engine() as engine:
        native=NativeTactics(engine)
        estimates=native.calculate(queries)
        # Reversing compatible queries cannot change berry consumption or formes.
        assert estimates==list(reversed(native.calculate(list(reversed(queries)))))
        uncached=engine.request('calculate',queries=queries)
        assert estimates==[native.summarize(r,q) for r,q in zip(uncached,queries)]
    for query,estimated,rolls in zip(queries,estimates,actual):
        for key,actual_roll in zip(('min','max'),rolls):
            predicted=min(round(estimated[key]*actual_roll['maxhp']),actual_roll['initial'])
            assert predicted==actual_roll['loss'],(query['move'],query['target']['species'],estimated,actual_roll)
    assert estimates[0]['ko']==0 and actual[0][0]['hp']==116
    assert estimates[1]['mean']>estimates[1]['min']>1
    assert estimates[2]['ko']==0 and estimates[2]['max']==0
    assert estimates[3]['ko']==estimates[4]['ko']==0 and estimates[5]['ko']==1


def test_hp_free_cache_reuses_only_safe_distributions():
    from champions_rl.engine import Engine
    from champions_rl.tactics import NativeTactics
    with Engine() as engine:
        native=NativeTactics(engine)
        q={'actor':known('Garchomp'),'target':known('Garchomp',item='Focus Sash'),'move':'outrage'}
        full=native.calculate([q])[0];misses=native.metrics['misses']
        low=copy.deepcopy(q);low['target']['hp']=.1;low['actor']['hp']=.5
        damaged=native.calculate([low])[0]
        assert native.metrics['misses']==misses and full['ko']==0 and damaged['ko']==1
        for move,ability in [('eruption','Rough Skin'),('dragonclaw','Multiscale'),('brine','Rough Skin')]:
            q.update(move=move,target=known('Dragonite',ability=ability))
            before=native.calculate([q])[0];misses=native.metrics['misses']
            low=copy.deepcopy(q);low['target']['hp']=.4;low['actor']['hp']=.2
            after=native.calculate([low])[0]
            assert native.metrics['misses']==misses+1
        q.update(move='iciclespear',target=known('Garchomp',volatiles={'substitute':{}}))
        row=native.calculate([q])[0]
        assert not row['reliable'] and 'unknown_substitute_hp' in row['unsupported']
        q.update(move='earthquake',target=known('Garchomp',volatiles={'unknownmechanic':{}}))
        assert not native.calculate([q])[0]['reliable']


def test_cache_normalization_matches_uncached_varied_native_queries():
    import random
    from champions_rl.engine import Engine
    from champions_rl.tactics import NativeTactics
    pool=TeamPool();rng=random.Random(80013);queries=[]
    for _ in range(96):
        a,d=[copy.deepcopy(rng.choice(pool.rows)['set']) for _ in range(2)]
        def exact(s):
            return mon(s['species'],set=s,item=canonical(s['item']),ability=canonical(s['ability']),
                       hp=rng.choice([1.,.5,.8]),boosts={'atk':rng.choice([0,1,2]),'spa':rng.choice([0,1])})
        query={'actor':exact(a),'target':exact(d),'move':rng.choice(a['moves']),'conditions':[{},{}],
               'weather':rng.choice(['','raindance','sunnyday']),'faints':[rng.randrange(3),rng.randrange(3)]}
        queries.append(query)
    with Engine() as engine:
        native=NativeTactics(engine)
        cached=native.calculate(queries)
        direct=engine.request('calculate',queries=queries)
        for q,actual,row in zip(queries,cached,direct):
            expected=native.summarize(row,q)
            assert actual==expected,(q['actor']['species'],q['target']['species'],q['move'],actual,expected)
