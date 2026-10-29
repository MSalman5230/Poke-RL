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
        model.zero_grad(set_to_none=True)
        model(tensor_batch([x]),detach_value=True)[1].sum().backward()
        assert model.value[-1].weight.grad is not None
        assert model.mon.weight.grad is None and model.species_embedding.weight.grad is None


def test_semantic_reply_labels_do_not_depend_on_hidden_move_order():
    from champions_rl.opponent_labels import action_label,switch_offset
    a={'legal':[{'id':0,'kind':'move','move':'earthquake'},{'id':10,'kind':'switch','slot':2}]}
    b={'legal':[{'id':3,'kind':'move','move':'earthquake'}]}
    assert action_label(a,0)==action_label(b,3)
    assert action_label(a,10)==switch_offset()+2
    assert action_label(a,None)==-100


def test_curriculum_retains_strong_old_opponents_and_targets_weaknesses():
    from champions_rl.curriculum import retain_league,style_weights
    from champions_rl.policies import TRAINING_STYLES
    names=[str(i) for i in range(20)]
    kept=retain_league(names,{'1':.9,'2':.8})
    assert len(kept)==12 and {'0','1','2','16','17','18','19'}<=set(kept)
    outcomes=[{'opponent':'attrition','win':False}]*40+[{'opponent':'hyperoffense','win':True}]*40
    weights=style_weights(outcomes)
    assert np.isclose(sum(weights.values()),1.) and min(weights.values())>0 and set(weights)==set(TRAINING_STYLES)
    assert weights['attrition']>weights['hyperoffense']


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


def test_actor_kl_stop_does_not_discard_critic_samples():
    from champions_rl.parallel_training import ppo_update
    torch.set_num_threads(1);torch.manual_seed(17)
    team=TeamPool().data['reference_team']
    with Engine() as engine:
        state=engine.request('new',game='ppo',teams=[team,team],seed=[1,2,3,4],training=True)
    encoder=Encoder(native=False)
    public=encoder.encode(state['views'][0])
    private=encoder.encode(state['training']['critic_views'][0],entities_only=True)
    actor=BattleModel(width=32,layers=1)
    critic=PrivilegedCritic(width=32,layers=1)
    with torch.no_grad():actions,logp,_=actor.act(tensor_batch([public]*64))
    records=[{'obs':public,'critic_obs':private,'policy_active':True,'action':int(a),'logp':float(p),'enemy':-100}
             for a,p in zip(actions,logp)]
    before={k:v.clone() for k,v in critic.state_dict().items()}
    result=ppo_update(actor,torch.optim.AdamW(actor.parameters(),lr=.01),records,
        torch.linspace(-1,1,64),torch.ones(64),'cpu',batch_size=8,epochs=2,critic=critic,
        critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=.001),target_kl=1e-9,kl_sample_size=64)
    assert result['kl_early_stop'] and result['kl_trigger']>1e-9 and result['kl_probe_samples']==64
    assert result['critic_unique_samples']==64 and result['critic_sample_coverage']==1
    assert result['critic_completed_epochs']==2 and result['critic_optimized_samples']==128
    assert any(not torch.equal(before[k],v) for k,v in critic.state_dict().items())
