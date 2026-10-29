from collections import defaultdict
import numpy as np
import torch
from champions_rl.model import BattleModel
from champions_rl.parallel_rollouts import ParallelRollouts,trajectory_targets


def test_parallel_selfplay_learns_both_seats_and_complete_rewards():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model=BattleModel(width=32,layers=1).eval()
    with ParallelRollouts(workers=2,envs=4,seed=7,selfplay_fraction=1.,snapshot_fraction=0.,max_turns=30) as actors:
        rows,results,stats=actors.collect({"current":model},horizon=40,device="cpu")
        assert {r["side"] for r in rows}=={0,1}
        assert len(rows)==stats["simulation_steps"]*2
        assert stats["policy_decisions"]>stats["simulation_steps"]
        assert stats["mean_current_batch"]>1
        assert results and all(r["win"] is None for r in results)
        reward=defaultdict(float)
        for r in rows:
            reward[r["trajectory"]]+=r["reward"]
            if r["terminal"]:
                assert np.isclose(abs(reward[r["trajectory"]]),1.) or np.isclose(reward[r["trajectory"]],0.)
        advantage,returns=trajectory_targets(rows,model,device="cpu")
        assert torch.isfinite(advantage).all() and torch.isfinite(returns).all()
    assert all(not p.is_alive() for p in actors.processes)


def test_gae_separates_opposing_players_and_waiting_states():
    rows=[]
    for t in range(2):
        for side in (0,1):
            rows.append({"trajectory":(0,1,side),"terminal":t==1,"truncated":False,"value":0.,
                         "reward":(1. if side==0 else -1.) if t==1 else 0.})
    model=BattleModel(width=32,layers=1)
    advantage,returns=trajectory_targets(rows,model,device="cpu",lam=.95)
    assert torch.allclose(returns,torch.tensor([.95,-.95,1.,-1.]))


def test_self_imitation_uses_only_better_than_expected_returns():
    from champions_rl.reference_replay import self_imitation_loss
    logits=torch.zeros(2,4,requires_grad=True)
    values=torch.zeros(2,requires_grad=True)
    loss=self_imitation_loss(logits,values,torch.tensor([1,2]),torch.tensor([1.,-1.]))
    loss.backward()
    assert logits.grad[0,1]<0
    assert torch.equal(logits.grad[1],torch.zeros(4))
    assert values.grad[0]<0 and values.grad[1]==0


def test_private_value_targets_use_finished_matches_and_preserve_truncation_bootstrap():
    from champions_rl.parallel_rollouts import terminal_critic_targets
    rows=[{'trajectory':0,'reward':.1,'terminal':False,'truncated':False},
          {'trajectory':1,'reward':.2,'terminal':False,'truncated':True},
          {'trajectory':0,'reward':.9,'terminal':True,'truncated':False}]
    targets,n=terminal_critic_targets(rows,torch.tensor([.3,.6,.9]))
    assert torch.allclose(targets,torch.tensor([1.,.6,.9])) and n==2


def test_parallel_training_checkpoint_resume(tmp_path):
    from champions_rl.parallel_training import train_parallel
    from champions_rl.model import BattleModel
    initial=tmp_path/"initial.pt"
    BattleModel(width=32,layers=1).save(initial)
    first=tmp_path/"first"
    train_parallel(first,initial=initial,steps=8,workers=2,envs=4,horizon=2,device="cpu",epochs=1,
                   selfplay_fraction=1.,snapshot_fraction=0.,reference_workers=0)
    first_state=torch.load(first/"trainer_state.pt",map_location="cpu",weights_only=True)
    second=tmp_path/"second"
    train_parallel(second,resume=first/"trainer_state.pt",steps=8,workers=2,envs=4,horizon=2,device="cpu",epochs=1,
                   selfplay_fraction=1.,snapshot_fraction=0.,reference_workers=0)
    state=torch.load(second/"trainer_state.pt",map_location="cpu",weights_only=True)
    assert state["steps"]>=first_state["steps"]+8
    assert len(state["worker_states"])==2 and state["league"]
    assert state['critic'] and state['feature_signature']
    saved=torch.load(second/'latest.pt',map_location='cpu',weights_only=True)
    assert 'critic' not in saved
