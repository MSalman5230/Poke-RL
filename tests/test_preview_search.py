import copy
import pytest
import torch
from champions_rl.engine import Engine
from champions_rl.features import Encoder
from champions_rl.model import BattleModel
from champions_rl.preview_search import TerminalPreviewBot
from champions_rl.teams import TeamPool


def test_preview_search_requires_public_input_and_distinct_trio_leads():
    torch.set_num_threads(1)
    team=TeamPool().data['reference_team'];model=BattleModel(width=32,layers=1).eval()
    with Engine() as engine:
        state=engine.request('new',game='live_preview',teams=[team,team],seed=[1,2,3,4],training=True)
        obs=state['views'][0];before=copy.deepcopy(obs);trace=engine.request('trace',game='live_preview')
        encoder=Encoder(native=False)
        bot=TerminalPreviewBot(model,encoder,engine=engine,max_turns=1)
        action=bot.choose(obs)
        assert action in [a['id'] for a in obs['legal']]
        # Caps must retain the policy, never become value estimates or wins.
        assert not bot.last_search['used'] and 'unresolved' in bot.last_search['error']
        assert bot.last_search['chosen']==bot.last_search['policy'] and not bot.last_search['value_head_used']
        assert obs==before and engine.request('trace',game='live_preview')==trace
        with pytest.raises(ValueError,match='public opponent'):
            bot.choose(state['training']['critic_views'][0])
        options=bot.options(obs,[1.]*134)
        picks=[next(a['picks'] for a in obs['legal'] if a['id']==i) for i in options]
        assert len({(p[0],*sorted(p[1:])) for p in picks})==3
        with pytest.raises(ValueError,match='16 worlds'):TerminalPreviewBot(model,worlds=8)
