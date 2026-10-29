import copy
import json
import subprocess
import numpy as np
import pytest
import torch
from champions_rl.engine import Engine,ROOT
from champions_rl.features import Encoder
from champions_rl.model import BattleModel,PrivilegedCritic,tensor_batch
from champions_rl.teams import TeamPool
from champions_rl.opponent_labels import switch_offset
from champions_rl.augmentation import identity_dropout,permute_encoded_moves


@pytest.fixture(scope='module')
def states():
    team=TeamPool().data['reference_team']
    with Engine() as engine:
        preview=engine.request('new',game='representation',teams=[team,team],seed=[3,4,5,6],training=True)
        battle=engine.request('step',game='representation',actions=[14,14])
    return preview,battle


def test_moves_are_a_set_and_action_labels_follow_permutations(states):
    torch.set_num_threads(1);torch.manual_seed(231)
    model=BattleModel(width=32,layers=1).eval()
    encoder=Encoder(native=False);rng=np.random.default_rng(112)
    for state in states:
        original=encoder.encode(state['views'][0])
        label=int(np.flatnonzero(original['mask'])[0])
        for _ in range(8):
            orders=np.stack([rng.permutation(4) for _ in range(12)])
            changed,new_label=permute_encoded_moves(original,orders,label)
            with torch.no_grad():
                a,va,ea=model(tensor_batch([original]));b,vb,eb=model(tensor_batch([changed]))
            active=int(original['pokemon'][:6,3].argmax())
            mapping=np.r_[orders[active],orders[active]+4,np.arange(8,134)]
            assert changed['mask'][new_label]
            assert np.array_equal(changed['actions'][new_label],original['actions'][label])
            torch.testing.assert_close(b.softmax(-1),a[:,mapping].softmax(-1),atol=2e-7,rtol=2e-6)
            torch.testing.assert_close(vb,va,atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(eb,ea,atol=2e-6,rtol=2e-6)


def test_pp_stays_with_move_and_conversion_preserves_real_first_slot(states):
    encoder=Encoder(native=False)
    obs=copy.deepcopy(states[0]['views'][0]);mon=obs['own'][0]
    mon['moves']=[{'id':'conversion','pp':3,'maxpp':8}, {'id':'thunderbolt','pp':8,'maxpp':16},
                  {'id':'icebeam','pp':1,'maxpp':12}, {'id':'recover','pp':2,'maxpp':8}]
    mon['set']['moves']=['Conversion','Thunderbolt','Ice Beam','Recover']
    before=encoder.encode(obs)
    mon['moves'][0],mon['moves'][1]=mon['moves'][1],mon['moves'][0]
    after=encoder.encode(obs)
    pp=encoder.move_names.index('pp_fraction');first=encoder.move_names.index('conversion_first')
    assert before['move_features'][0,0,pp]==after['move_features'][0,1,pp]
    assert before['move_features'][0,0,first]==after['move_features'][0,0,first]==1
    assert before['moves'][0,0]!=after['moves'][0,0] # real Conversion type changed
    # Public reveal order must never claim to be the hidden opponent's slot order.
    mon.pop('set');assert not np.asarray(encoder.move_features(mon))[:,first].any()


def test_opponent_switch_scores_follow_destination_and_public_masks(states):
    torch.manual_seed(82)
    model=BattleModel(width=32,layers=1).eval();encoder=Encoder(native=False)
    obs=copy.deepcopy(states[1]['views'][0]);obs['history']=[]
    a=encoder.encode(obs);obs['opponent'][1],obs['opponent'][2]=obs['opponent'][2],obs['opponent'][1]
    b=encoder.encode(obs)
    with torch.no_grad():
        x=model(tensor_batch([a]))[2];y=model(tensor_batch([b]))[2]
    offset=switch_offset();mapping=np.arange(offset+6);mapping[offset+1:offset+3]=[offset+2,offset+1]
    torch.testing.assert_close(y,x[:,mapping],atol=2e-6,rtol=2e-6)
    assert not a['opponent_switch_mask'][0] and a['opponent_switch_mask'][1:].all()
    obs['opponent'][1]['fainted']=True
    assert not encoder.opponent_switch_mask(obs)[1]
    obs['opponent'][0]['species']='Pawmot';obs['opponent'][0]['moves']=['revivalblessing']
    assert encoder.opponent_switch_mask(obs)[1]


def test_critic_sees_rage_fist_and_revival_counts_without_policy_leak(states):
    encoder=Encoder(native=False);obs=copy.deepcopy(states[1]['training']['critic_views'][0])
    before=encoder.encode(obs,entities_only=True)
    obs['opponent'][0]['times_attacked']=5
    obs['faints'][1]=[{'species':'Pawmot','turn':0}]*4
    after=encoder.encode(obs,entities_only=True)
    assert before['pokemon'][6,encoder.mon_names.index('hits_received')]==0
    assert after['pokemon'][6,encoder.mon_names.index('hits_received')]==pytest.approx(5/6)
    assert after['field'][encoder.field_names.index('enemy_total_faints')]==pytest.approx(.04)
    assert after['field'][encoder.field_names.index('enemy_fainted')]==0
    critic=PrivilegedCritic(width=32,layers=1).eval()
    with torch.no_grad():assert not torch.allclose(critic(tensor_batch([before])),critic(tensor_batch([after])))


def test_public_protocol_replaces_stockpile_and_perish_counters():
    code="""const {PublicState}=require('./simulator/worker.cjs');const p=new PublicState(0);
    p.feed(['|poke|p1|Gholdengo, L50|','|switch|p1a: Gold|Gholdengo, L50|100/100',
      '|-start|p1a: Gold|stockpile1','|-start|p1a: Gold|stockpile2',
      '|-start|p1a: Gold|perish3','|-start|p1a: Gold|perish2']);
    const before=Object.keys(p.teams[0][0].volatiles);
    p.feed(['|-end|p1a: Gold|Stockpile']);console.log(JSON.stringify([before,Object.keys(p.teams[0][0].volatiles)]));"""
    before,after=json.loads(subprocess.check_output(['node','-e',code],cwd=ROOT,text=True))
    assert set(before)=={'stockpile2','perish2'} and after==['perish2']


def test_identity_dropout_is_consistent_and_ppo_old_logp_is_replayable(states):
    encoder=Encoder(native=False)
    public=encoder.encode(states[1]['views'][0]);private=encoder.encode(states[1]['training']['critic_views'][0],entities_only=True)
    torch.manual_seed(110);model=BattleModel(width=32,layers=1)
    source=tensor_batch([public]*64);changed,masks=identity_dropout(source,.25)
    truth,_=identity_dropout(tensor_batch([private]*64),.25,masks)
    assert any((source[k]!=changed[k]).any() for k in ('species','moves','items_prior','abilities_prior'))
    for key in ('pokemon','field','move_mask','move_features','mask','opponent_switch_mask'):
        assert torch.equal(changed[key],source[key])
    assert torch.equal(changed['species'][:,:6],truth['species'][:,:6])
    for i in range(64):
        for token in source['moves'][i].unique():
            if token<=1:continue
            assert changed['moves'][i][source['moves'][i]==token].unique().numel()==1
    with torch.no_grad():
        actions,old_logp,_=model.act(changed)
        model.train() # forward itself adds no stochastic augmentation
        new_logp=torch.distributions.Categorical(logits=model(changed)[0]).log_prob(actions)
    torch.testing.assert_close(old_logp,new_logp,atol=1e-6,rtol=1e-6)
