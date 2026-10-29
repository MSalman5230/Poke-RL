import copy
import numpy as np
import torch
from champions_rl.engine import Engine
from champions_rl.features import Encoder
from champions_rl.model import BattleModel,tensor_batch
from champions_rl.rewards import battle_reward,potential
from champions_rl.teams import TeamPool
from champions_rl.training import advantages
from champions_rl.evaluation import acceptance,wilson


def test_masked_network_checkpoint_and_auxiliary_gradients(tmp_path):
    torch.set_num_threads(2)
    with Engine() as e:
        t=TeamPool().data["reference_team"]
        obs=e.request("new",game="x",teams=[t,t],seed=[1,2,3,4])["views"][0]
    encoded=Encoder().encode(obs)
    model=BattleModel()
    batch=tensor_batch([encoded])
    logits,value,opponent=model(batch)
    assert torch.isfinite(value).all()
    assert (logits[~batch["mask"]] == -1e9).all()
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([14]))+value.square().mean()+opponent.square().mean()
    loss.backward()
    assert torch.isfinite(model.action[-1].weight.grad).all()
    model.save(tmp_path/"model.pt",test=True)
    restored,metadata=BattleModel.load(tmp_path/"model.pt")
    assert metadata["test"]
    assert torch.allclose(restored(batch)[0],logits)


def test_terminal_rewards_telescope_and_truncation_bootstraps():
    start={"phase":"preview","ended":False,"side":0}
    middle={"phase":"move","ended":False,"side":0,"own":[{"hp":.5,"selected":True,"fainted":False}]*3,
            "opponent":[{"hp":.2,"seen":True,"fainted":False}]+[{"hp":1,"seen":False,"fainted":False}]*5}
    end={**middle,"ended":True}
    assert np.isclose(battle_reward(start,middle)+battle_reward(middle,end,0),1.)
    r=torch.tensor([[.1],[.2]])
    v=torch.zeros_like(r)
    nv=torch.ones_like(r)*.5
    term=torch.tensor([[False],[True]])
    done=torch.tensor([[True],[True]])  # time cap at t=0; bootstrap, stop recurrence
    adv,ret=advantages(r,v,nv,term,done)
    assert torch.allclose(adv,torch.tensor([[.6],[.2]]))


def test_success_is_never_inferred_from_small_or_degraded_samples():
    assert wilson(60,100)[0]<.6
    row={"games":200,"wins":150,"win_rate":.75,"confidence":wilson(150,200),"fallbacks":0,"errors":0,"truncations":0}
    report={name:dict(row) for name in ("reference","damage","balanced","setup","defensive")}
    assert acceptance(report,"test")["passed"]
    assert not acceptance(report,"validation")["passed"]
    report["reference"]["fallbacks"]=1
    assert not acceptance(report,"test")["passed"]


def test_evaluation_pairs_cancel_team_advantage(tmp_path):
    from champions_rl.evaluation import evaluate
    result=evaluate(opponents=["balanced"],games=10,output=tmp_path,learner_style="balanced")
    row=result["results"]["balanced"]
    # Identical deterministic policies replay the same battle; each controls
    # each team once. Team quality alone cannot score above or below 50%.
    assert row["wins"]==5 and row["losses"]==5 and row["errors"]==0


def test_reference_integrity_rejects_modified_strategy(tmp_path):
    import json,hashlib,pytest
    from champions_rl.integrity import verify_reference
    path=tmp_path/"policy.py"
    path.write_text("original")
    (tmp_path/"SOURCE_MANIFEST.json").write_text(json.dumps({"files":{"policy.py":{"snapshot_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}}}))
    verify_reference(tmp_path)
    path.write_text("different strategy")
    with pytest.raises(RuntimeError,match="Frozen reference changed"):
        verify_reference(tmp_path)


def test_residual_policy_training_resume_preserves_league_and_step_count(tmp_path):
    from champions_rl.training import train_ppo
    first=tmp_path/"first"
    train_ppo(first,steps=8,n_envs=2,horizon=4,epochs=1,batch_size=8,device="cpu",heuristic_weight=2.)
    resumed=tmp_path/"resumed"
    train_ppo(resumed,steps=8,n_envs=2,horizon=4,epochs=1,batch_size=8,device="cpu",resume=first/"trainer_state.pt")
    state=torch.load(resumed/"trainer_state.pt",map_location="cpu",weights_only=True)
    assert state["steps"]==16
    assert state["config"]["heuristic_weight"]==2.
    assert len(state["league"])>=2 and state["arena_rng"]


def test_history_attention_can_distinguish_move_order():
    torch.manual_seed(12)
    with Engine() as engine:
        t=TeamPool().data["reference_team"]
        obs=engine.request("new",game="history",teams=[t,t],seed=[1,2,3,4])["views"][0]
    obs["history"]=[{"side":0,"turn":0,"move":"shadowball"},{"side":1,"turn":0,"move":"earthquake"}]
    first=Encoder().encode(obs)
    obs["history"].reverse()
    second=Encoder().encode(obs)
    batch=tensor_batch([first,second])
    model=BattleModel().eval()
    with torch.no_grad():
        value=model(batch)[1]
        assert not torch.isclose(value[0],value[1],atol=1e-7,rtol=0)
        model.config["history_position"]=False
        unordered=model(batch)[1]
        assert torch.isclose(unordered[0],unordered[1],atol=1e-6,rtol=0)
