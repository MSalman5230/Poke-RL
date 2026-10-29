import copy
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
