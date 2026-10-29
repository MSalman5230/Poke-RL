import random
import numpy as np
from champions_rl.engine import Engine
from champions_rl.teams import TeamPool,fingerprint
from champions_rl.knowledge import canonical


def test_teammate_sampler_keeps_complete_sets_clauses_and_weather():
    pool=TeamPool();rng=random.Random(912381)
    teams=[pool.sample(rng,'teammates') for _ in range(128)]
    assert len({fingerprint(t) for t in teams})==128
    assert len({s['species'] for t in teams for s in t})>=70
    assert fingerprint(teams[0])==fingerprint(pool.sample(random.Random(912381),'teammates'))
    sampler=pool._teammate_sampler
    for team in teams:
        items=[s['item'] for s in team if s['item']]
        assert len(items)==len(set(items))
        options=[sampler.describe({'set':s}) for s in team]
        assert sum(x['mega'] for x in options)<=1
        climates={x['weather'] for x in options}-{None}
        assert len(climates)<=1
        assert all(not x['dependent'] or x['dependent'] in climates for x in options)
        assert any(x['physical'] for x in options) and any(x['special'] for x in options)
        for mon in team:
            assert any(fingerprint(mon)==fingerprint(row['set']) for row in pool.by_species[mon['species']])
    with Engine() as engine:assert not any(engine.request('validate',teams=teams))
    assert sampler.conditional['Pelipper']['Archaludon']>sampler.prior['Archaludon']
    assert 'Swampert' in sampler.conditional['Pelipper'] # Mega usage resolves to a selectable base


def test_saved_team_vs_varied_evaluation_keeps_our_team_in_both_seats(tmp_path):
    from champions_rl.evaluation import evaluate
    import json
    result=evaluate(opponents=['balanced'],games=4,output=tmp_path,roster='saved_vs_varied')
    rows=json.loads((tmp_path/'balanced_games.json').read_text())
    saved=fingerprint(TeamPool().data['reference_team'])
    for row in rows:
        assert row['team_ids'][row['seat']]==saved
        assert row['team_ids'][1-row['seat']]!=saved
    assert rows[0]['team_ids']==rows[1]['team_ids'][::-1]
    assert rows[2]['team_ids']==rows[3]['team_ids'][::-1]
    assert result['results']['balanced']['errors']==0
