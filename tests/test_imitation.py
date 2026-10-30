import numpy as np
import torch
from champions_rl.training import imitation_validation


def test_validation_balances_teacher_groups_despite_unequal_sample_counts():
    class Fixed(torch.nn.Module):
        def forward(self,obs):return obs['given_logits'],None,None
    logits=np.array([[3.,0.]]*9+[[0.,3.]],dtype=np.float32)
    data={'given_logits':logits,'labels':np.zeros(10,dtype=np.int64),
          'expert':np.array([False]*9+[True]),'teacher':np.array(['balanced']*9+['reference'])}
    metrics=imitation_validation(Fixed(),data,np.arange(10),4,'cpu')
    ref=metrics['validation_by_teacher']['reference'];other=metrics['validation_by_teacher']['balanced']
    assert ref['samples']==1 and other['samples']==9
    assert ref['agreement']==0 and other['agreement']==1
    assert np.isclose(metrics['balanced_validation_loss'],(ref['loss']+other['loss'])/2)
    assert metrics['balanced_validation_loss']>metrics['validation_loss']*2


def test_correction_mass_matches_selection_metric_without_dropping_original_teachers():
    from champions_rl.training import imitation_groups
    data={'expert':np.array([True]*10+[False]*90+[False]*5),
          'correction':np.array([False]*100+[True]*5)}
    groups=imitation_groups(data,np.arange(105),.25)
    assert {name:mass for name,(_,mass) in groups.items()}=={'correction':.25,'reference':.375,'other':.375}
    weights=np.zeros(105)
    for mask,mass in groups.values():weights[mask]=mass/mask.sum()
    assert np.isclose(weights.sum(),1) and np.isclose(weights[-5:].sum(),.25)


def test_step_zero_is_a_checkpoint_candidate(tmp_path):
    import hashlib
    from champions_rl.engine import Engine
    from champions_rl.features import Encoder,FEATURE_VERSION
    from champions_rl.model import BattleModel
    from champions_rl.provenance import feature_signature
    from champions_rl.knowledge import knowledge
    from champions_rl.teams import TeamPool
    from champions_rl.persistence import atomic_json,durable_npz
    from champions_rl.training import behavior_clone
    with Engine() as engine:
        team=TeamPool().data['reference_team']
        obs=engine.request('new',game='bc',teams=[team,team],seed=[1,2,3,4])['views'][0]
    encoded=Encoder(native=False).encode(obs)
    data=tmp_path/'data';data.mkdir()
    for heldout in (False,True):
        stem=next('game'+str(i) for i in range(1000) if (int(hashlib.sha256(('game'+str(i)).encode()).hexdigest()[:8],16)%10==0)==heldout)
        payload={k:np.stack([v]) for k,v in encoded.items()};payload.update(labels=np.array([14]),outcomes=np.array([1.],dtype=np.float32))
        durable_npz(data/(stem+'.npz'),**payload)
        atomic_json(data/(stem+'.json'),{'teacher':'reference','seed':1,'samples':1,'feature_version':FEATURE_VERSION,
                    'feature_signature':feature_signature(),'knowledge_sha256':knowledge().sha256})
    initial=tmp_path/'initial.pt';output=tmp_path/'candidate.pt'
    BattleModel(width=32,layers=1).save(initial)
    behavior_clone(data,output,steps=0,device='cpu',initial=initial)
    _,metadata=BattleModel.load(output)
    assert metadata['updates']==0 and 'balanced_validation_loss' in metadata and 'promotion' in metadata
    from champions_rl.training import load_demonstrations
    import pytest
    assert len(load_demonstrations([data])['labels'])==2
    with pytest.raises(ValueError,match='Duplicate demonstration'):
        load_demonstrations([data,data])
    # Explicit correction partitions apply to every turn of their game, even
    # if a tiny corpus would receive no validation games from the default hash.
    import json
    for path in data.glob('*.json'):
        meta=json.loads(path.read_text());meta['heldout']=True;atomic_json(path,meta)
    assert load_demonstrations(data)['validation'].all()
