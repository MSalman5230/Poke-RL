import copy
import numpy as np
import pytest
import torch
from champions_rl.engine import Engine
from champions_rl.features import Encoder
from champions_rl.model import BattleModel,tensor_batch
from champions_rl.opponent_labels import action_label,opponent_loss,switch_offset
from champions_rl.preview_search import TerminalPreviewBot
from champions_rl.teams import TeamPool


def illusion_team(zoroark_first=False):
    pool=TeamPool()
    def mon(species,moves,ability=None):
        result=copy.deepcopy(pool.by_species[species][0]['set'])
        result.update(moves=moves.split(','),item='')
        if ability:result['ability']=ability
        return result
    corv=mon('Corviknight','Roost,Iron Defense,Body Press,U-turn','Pressure')
    zoroark=mon('Zoroark','Nasty Plot,Protect,Dark Pulse,Flamethrower','Illusion')
    tyranitar=mon('Tyranitar','Protect')
    result=([zoroark,corv] if zoroark_first else [corv,zoroark])+[tyranitar]
    for source in pool.data['reference_team'][:3]:
        source=copy.deepcopy(source);source['item']='';result.append(source)
    return result


def test_previously_revealed_pressure_cannot_identify_a_later_disguise():
    torch.set_num_threads(1);torch.manual_seed(188)
    own=illusion_team();foe=copy.deepcopy(TeamPool().data['reference_team'])
    foe[0]['moves']=['Protect']
    encoder=Encoder(native=False);model=BattleModel(width=32,layers=1).eval()
    with Engine() as engine:
        state=engine.request('new',game='illusion_mask',teams=[own,foe],seed=[8,13,21,34])
        state=engine.request('step',game='illusion_mask',actions=[14,14])
        view=state['views'][1]
        assert view['opponent'][0]['ability']=='pressure'
        assert not view['opponent'][0]['identity_uncertain']
        assert not encoder.opponent_switch_mask(view)[0]
        # Corviknight -> Tyranitar -> Zoroark. Party swapping puts Corviknight
        # last, so Zoroark now borrows the previously observed Corviknight face.
        state=engine.request('step',game='illusion_mask',actions=[10,0])
        state=engine.request('step',game='illusion_mask',actions=[9,0])
        view=state['views'][1];apparent=view['opponent'][0]
        assert apparent['active'] and apparent['species']=='Corviknight'
        assert apparent['ability']=='pressure' and apparent['identity_uncertain']
        assert state['views'][0]['own'][1]['active']
        label=action_label(state['views'][0],8) # actual legal switch to Corviknight
        assert label==switch_offset()
        encoded=encoder.encode(view)
        assert encoded['opponent_switch_mask'][0]
        logits=model(tensor_batch([encoded]))[2]
        loss=opponent_loss(logits,torch.tensor([label]))
        assert torch.isfinite(loss) and loss<100
        poisoned=logits.clone();poisoned[0,label]=-1e9
        with pytest.raises(ValueError,match='Masked/nonfinite opponent training target'):
            opponent_loss(poisoned,torch.tensor([label]))
        # Old serialized observations must use a conservative fallback too.
        apparent.pop('identity_uncertain')
        assert encoder.opponent_switch_mask(view)[0]
        state=engine.request('step',game='illusion_mask',actions=[8,0])
        assert not encoder.opponent_switch_mask(state['views'][1])[0]


def test_preview_preserves_simulator_distinct_illusion_targets():
    torch.set_num_threads(1);torch.manual_seed(377)
    own=illusion_team(zoroark_first=True);foe=TeamPool().data['reference_team']
    encoder=Encoder(native=False);model=BattleModel(width=32,layers=1).eval()
    with Engine() as engine:
        a=engine.request('new',game='preview_order_a',teams=[own,foe],seed=[3,5,7,11])
        b=engine.request('new',game='preview_order_b',teams=[own,foe],seed=[3,5,7,11])
        obs=a['views'][0]
        first=next(x['id'] for x in obs['legal'] if x['picks']==[0,1,2])
        second=next(x['id'] for x in obs['legal'] if x['picks']==[0,2,1])
        a=engine.request('step',game='preview_order_a',actions=[first,14])
        b=engine.request('step',game='preview_order_b',actions=[second,14])
        visible=lambda state:next(m['species'] for m in state['views'][1]['opponent'] if m['active'])
        assert visible(a)=='Tyranitar' and visible(b)=='Corviknight'
        encoded=encoder.encode(obs)
        assert encoded['preview_disguise'][first,2]==encoded['preview_disguise'][second,1]==1
        with torch.no_grad():scores=model(tensor_batch([encoded]))[0][0]
        assert abs(float(scores[first]-scores[second]))>1e-7
        probabilities=np.zeros(134);probabilities[first]=.6;probabilities[second]=.4
        search=TerminalPreviewBot(model,encoder,candidates=2)
        assert search.options(obs,probabilities)==[first,second]
