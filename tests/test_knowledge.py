import json
from champions_rl.engine import ROOT, Engine
from champions_rl.knowledge import knowledge
from champions_rl.teams import TeamPool


def test_format_and_restricted_data():
    k=knowledge()
    assert k.data["format"]=="gen9championsbssregmc"
    assert k.data["rules"]["pickedTeamSize"]==3
    assert not k.data["rules"]["terastallization"]
    assert len({s["num"] for s in k.species.values() if not s["battleOnly"]})==231
    assert "miraidon" not in k.species
    assert k.effectiveness("Electric",["Water","Flying"])==4
    assert k.effectiveness("Ground",["Flying"])==0
    assert "recover" in k.species["gholdengo"]["learnset"]


def test_all_sets_and_all_teams_are_engine_legal():
    sets=TeamPool().data
    assert sets["uncovered"]==[]
    with Engine() as engine:
        assert not any(engine.request("validate",sets=[r["set"] for r in sets["sets"]]))
        hashes=[]
        for split in ("train","validation","test"):
            data=json.loads((ROOT/f"data/teams_{split}.json").read_text())
            assert not any(engine.request("validate",teams=[r["team"] for r in data["teams"]]))
            group={r["id"] for r in data["teams"]}
            assert all(not group.intersection(previous) for previous in hashes)
            hashes.append(group)


def test_reference_is_namespaced_and_has_no_device_modules():
    folder=ROOT/"opponents/autoplay_reference/bot"
    assert not (folder/"perception").exists()
    assert not (folder/"tablet").exists()
    assert not (folder/"runtime/loop.py").exists()
    from opponents.autoplay_reference.bot.decision.evaluation import TRIALS
    assert TRIALS==32


def test_curated_weather_teams_and_complete_sets_are_coherent():
    import random
    from champions_rl.curated import repair_synthetic
    from champions_rl.teams import fingerprint
    pool=TeamPool()
    assert sum(bool(r.get('curated')) for r in pool.rows)==50
    for mode,ability in [('rain','Swift Swim'),('sun','Chlorophyll'),('sand','Sand Rush')]:
        team=pool.sample(random.Random(1),mode)
        assert any(m['ability']==ability for m in team)
        assert len({m['item'] for m in team})==6
        assert fingerprint(team)==fingerprint(list(reversed(team)))
    bad={'species':'Blaziken','item':'Choice Scarf','ability':'Speed Boost','nature':'Modest',
         'evs':{'hp':2,'atk':0,'def':0,'spa':32,'spd':0,'spe':32},
         'moves':['Close Combat','Protect','Swords Dance','Baton Pass']}
    good=repair_synthetic(bad)
    assert good['evs']['atk']==32 and good['evs']['spa']==0
    assert all(knowledge().moves[m.lower().replace(' ','')]['category']!='Status' for m in good['moves'])
