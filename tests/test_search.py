import copy
import random
import numpy as np
import pytest
import torch
from champions_rl.engine import Engine
from champions_rl.features import Encoder
from champions_rl.model import BattleModel,PrivilegedCritic
from champions_rl.search import SearchBot,hypothetical_world
from champions_rl.teams import TeamPool


def test_lookahead_uses_only_public_worlds_and_does_not_mutate_live_game():
    torch.set_num_threads(1)
    team=TeamPool().data['reference_team']
    with Engine() as engine:
        engine.request('new',game='live',teams=[team,team],seed=[1,2,3,4])
        observation=engine.request('step',game='live',actions=[14,14])['views'][0]
        original=copy.deepcopy(observation)
        trace=engine.request('trace',game='live')
        encoder=Encoder(engine=engine)
        model=BattleModel(width=32,layers=1).eval()
        critic=PrivilegedCritic(width=32,layers=1).eval()
        bot=SearchBot(model,encoder,worlds=2,replies=2,candidates=3,seed=2,confidence=1.,critic=critic)
        action=bot.choose(observation)
        assert action in [a['id'] for a in observation['legal']]
        assert bot.last_search['used'],bot.last_search
        assert bot.last_search['groups']==4
        assert not bot.last_search['errors']
        assert observation==original and engine.request('trace',game='live')==trace
        forbidden=copy.deepcopy(observation)
        forbidden['opponent'][0]['set']=team[0]
        with pytest.raises(ValueError,match='privileged'):
            hypothetical_world(forbidden,encoder.beliefs,random.Random(2),2)


def test_hypotheses_preserve_revealed_moves_and_pp_and_search_resolves_turn():
    team=TeamPool().data['reference_team']
    with Engine() as engine:
        engine.request('new',game='turn',teams=[team,team],seed=[4,3,2,1])
        obs=engine.request('step',game='turn',actions=[14,14])['views'][0]
        foe=next(m for m in obs['opponent'] if m['active'])
        foe['moves']=['nastyplot'];foe['pp_used']={'nastyplot':2}
        encoder=Encoder(engine=engine)
        world=hypothetical_world(obs,encoder.beliefs,random.Random(3),1)
        enemy=world['input']['sides'][1]['pokemon'][0]
        assert 'nastyplot' in [m['id'] for m in enemy['moves']]
        assert len(enemy['moves'])==4
        candidates=[a['id'] for a in obs['legal'] if a['kind']=='move'][:2]
        result=engine.request('lookahead',observation=obs,worlds=[world],candidates=candidates,replies=1,
                              reply_probabilities={'move:nastyplot':1.})
        assert not result['errors'],result['errors']
        assert len(result['rows'])==len(candidates)
        for row in result['rows']:
            assert row['ended'] or row['observation']['turn']>obs['turn']
            assert all('set' not in m and 'stats' not in m for m in row['observation']['opponent'])
            assert all('set' in m and 'stats' in m for m in row['critic_observation']['opponent'] if m.get('selected'))


def test_search_margin_requires_enough_worlds_and_paired_evidence():
    from champions_rl.search import paired_search_margin
    rng=np.random.default_rng(7)
    assert paired_search_margin({0:np.zeros(8),1:np.ones(8)},0,rng)[0]==0
    assert paired_search_margin({0:np.zeros(16),1:np.full(16,.01)},0,rng)[0]==0
    assert paired_search_margin({0:np.zeros(16),1:np.full(16,.2)},0,rng)[0]==1
    assert paired_search_margin({0:np.zeros(16),1:np.array([-1,1]*8)},0,rng)[0]==0


def test_search_demonstrations_require_battle_evidence(tmp_path):
    from champions_rl.expert_data import collect
    with pytest.raises(ValueError,match='paired battle validation'):
        collect(tmp_path,teacher='search',games=1)
