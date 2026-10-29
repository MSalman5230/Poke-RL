import copy
import numpy as np
import torch
from champions_rl.engine import Engine
from champions_rl.features import Encoder
from champions_rl.model import BattleModel,PrivilegedCritic,tensor_batch
from champions_rl.teams import TeamPool


def test_private_teams_change_only_training_critic_inputs():
    torch.set_num_threads(1)
    team=copy.deepcopy(TeamPool().data['reference_team'])
    team[0]['item']='Sitrus Berry'
    changed=copy.deepcopy(team)
    changed[0]['item']='Lum Berry';changed[0]['moves'][3]='Thunderbolt'
    encoder=Encoder(native=False)
    with Engine() as engine:
        ordinary=engine.request('new',game='ordinary',teams=[team,team],seed=[1,2,3,4])
        a=engine.request('new',game='a',teams=[team,team],seed=[1,2,3,4],training=True)
        b=engine.request('new',game='b',teams=[team,changed],seed=[1,2,3,4],training=True)
        assert 'training' not in ordinary
        assert a['views'][0]==b['views'][0]==ordinary['views'][0]
        x=encoder.encode(a['views'][0]);y=encoder.encode(b['views'][0])
        assert all(np.array_equal(x[k],y[k]) for k in x)
        private_a=encoder.encode(a['training']['critic_views'][0],entities_only=True)
        private_b=encoder.encode(b['training']['critic_views'][0],entities_only=True)
        assert not np.array_equal(private_a['moves'],private_b['moves'])
        assert not np.array_equal(private_a['items_prior'],private_b['items_prior'])
        model=BattleModel(width=32,layers=1).eval()
        critic=PrivilegedCritic(width=32,layers=1).eval()
        with torch.no_grad():
            assert torch.equal(model(tensor_batch([x]))[0],model(tensor_batch([y]))[0])
            assert not torch.equal(critic(tensor_batch([private_a])),critic(tensor_batch([private_b])))
        assert not ({id(p) for p in model.parameters()} & {id(p) for p in critic.parameters()})


def test_semantic_reply_labels_do_not_depend_on_hidden_move_order():
    from champions_rl.opponent_labels import action_label,switch_offset
    a={'legal':[{'id':0,'kind':'move','move':'earthquake'},{'id':10,'kind':'switch','slot':2}]}
    b={'legal':[{'id':3,'kind':'move','move':'earthquake'}]}
    assert action_label(a,0)==action_label(b,3)
    assert action_label(a,10)==switch_offset()+2
    assert action_label(a,None)==-100


def test_curriculum_retains_strong_old_opponents_and_targets_weaknesses():
    from champions_rl.curriculum import retain_league,style_weights
    from champions_rl.arena import STYLES
    names=[str(i) for i in range(20)]
    kept=retain_league(names,{'1':.9,'2':.8})
    assert len(kept)==12 and {'0','1','2','16','17','18','19'}<=set(kept)
    outcomes=[{'opponent':'defensive','win':False}]*40+[{'opponent':'damage','win':True}]*40
    weights=style_weights(outcomes)
    assert np.isclose(sum(weights),1.) and min(weights)>0
    assert weights[STYLES.index('defensive')]>weights[STYLES.index('damage')]


def test_reference_fallen_marker_preserves_the_public_faint_count():
    from champions_rl.reference import ReferenceBot
    team=TeamPool().data['reference_team']
    with Engine() as engine:
        engine.request('new',game='ref',teams=[team,team],seed=[1,2,3,4])
        obs=engine.request('step',game='ref',actions=[14,14])['views'][0]
    foe=next(p for p in obs['opponent'] if p['active'])
    foe['volatiles']['fallen1']={'turn':1}
    obs['faints'][1]=[{'species':'Volcarona','turn':0}]
    reference=ReferenceBot()
    _,battle=reference._state(obs,reference._roster(obs))
    assert 'fallen1' not in battle['vol']['opp']['effects']
    assert len(battle['faint_history']['opp'])==1
