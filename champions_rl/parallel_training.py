"""Parallel PPO, batched inference, and nonblocking reference self-imitation."""
from collections import Counter
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .engine import ROOT
from .features import FEATURE_VERSION
from .knowledge import knowledge
from .model import BattleModel,PrivilegedCritic,tensor_batch
from .parallel_rollouts import ParallelRollouts,trajectory_targets
from .reference_replay import ReferencePool,ReferenceReplay,self_imitation_loss
from .training import emit,seed_all
from .provenance import feature_signature
from .persistence import durable_torch_save,atomic_json,append_json
from .curriculum import style_weights,retain_league


def cpu_weights(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def ppo_update(model,optimizer,records,advantage,returns,device,batch_size=128,epochs=4,critic=None):
    data=tensor_batch([r["obs"] for r in records],device)
    private=tensor_batch([r['critic_obs'] for r in records],device) if critic is not None else None
    active=torch.tensor([r["policy_active"] for r in records],device=device,dtype=torch.bool)
    selected=advantage[active]
    if not len(selected):
        raise RuntimeError("Rollout contained no learning-player decisions")
    advantage=(advantage-selected.mean())/(selected.std(unbiased=False)+1e-8)
    actions=torch.tensor([r["action"] for r in records],device=device)
    old_logp=torch.tensor([r["logp"] for r in records],device=device)
    enemies=torch.tensor([r["enemy"] for r in records],device=device)
    model.train()
    if critic is not None:critic.train()
    losses,entropies,kls,private_errors=[],[],[],[]
    optimized_samples=0
    early=False
    for _ in range(epochs):
        for index in torch.randperm(len(records),device=device).split(batch_size):
            obs={k:v[index] for k,v in data.items()}
            logits,value,enemy=model(obs,detach_value=critic is not None)
            distribution=Categorical(logits=logits)
            logratio=distribution.log_prob(actions[index])-old_logp[index]
            ratio=logratio.exp()
            learn=active[index]
            policy=torch.maximum(-advantage[index]*ratio,-advantage[index]*ratio.clamp(.8,1.2))
            policy=policy[learn].mean() if learn.any() else policy.sum()*0
            entropy=distribution.entropy()[learn].mean() if learn.any() else value.sum()*0
            mask=obs["mask"]
            uniform=-(logits.log_softmax(-1).masked_fill(~mask,0).sum(-1)/mask.sum(-1).clamp(min=1))
            uniform=uniform[learn].mean() if learn.any() else value.sum()*0
            valid=enemies[index]>=0
            auxiliary=nn.functional.cross_entropy(enemy[valid],enemies[index][valid]) if valid.any() else value.sum()*0
            # The private critic supplies PPO's baseline. Keep a public value
            # head for experimental search, but its noisy partial-view targets
            # must not push the policy encoder through a second critic loss.
            value_weight=.1 if critic is not None else .5
            loss=policy+value_weight*nn.functional.mse_loss(value,returns[index])-.01*entropy+.001*uniform+.03*auxiliary
            if critic is not None:
                private_error=nn.functional.mse_loss(critic({k:v[index] for k,v in private.items()}),returns[index])
                private_errors.append(float(private_error.detach()))
                loss+=.5*private_error
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),.5)
            if critic is not None:nn.utils.clip_grad_norm_(critic.parameters(),.5)
            optimizer.step()
            optimized_samples+=len(index)
            kl=((ratio-1)-logratio)[learn].mean().detach() if learn.any() else torch.tensor(0.)
            losses.append(float(loss.detach()))
            entropies.append(float(entropy.detach()))
            kls.append(float(kl))
            if float(kl)>.04:
                early=True
                break
        if early:
            break
    return {"loss":float(np.mean(losses)),"entropy":float(np.mean(entropies)),"approx_kl":float(np.mean(kls)),
            'private_value_mse':float(np.mean(private_errors)) if private_errors else None,
            'optimized_samples':optimized_samples,'kl_early_stop':early}


def train_parallel(output,initial=None,resume=None,steps=200000,workers=4,envs=64,horizon=32,
                   device="cuda",seed=101,selfplay_fraction=.1,snapshot_fraction=.1,reference_workers=0,
                   reference_weight=0.,learning_rate=1e-4,batch_size=128,epochs=4,checkpoint_every=1,
                   stop_file=None,history_position=None,heuristic_weight=None,rollout_decisions=32768,privileged=True):
    if min(steps,workers,envs,horizon,batch_size,epochs,checkpoint_every,rollout_decisions)<1:
        raise ValueError("Training sizes and intervals must be positive")
    if reference_workers<0 or reference_weight<0:
        raise ValueError("Reference worker count and loss weight must be nonnegative")
    seed_all(seed)
    torch.set_num_threads(2)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    restored=torch.load(resume,map_location="cpu",weights_only=True) if resume else None
    if restored and (restored["knowledge_sha256"]!=knowledge().sha256 or restored.get("feature_version")!=FEATURE_VERSION or restored.get('feature_signature')!=feature_signature()):
        raise ValueError("Incompatible trainer state")
    if restored:
        config=dict(restored["config"])
        config.setdefault("action_context",False)
        config.setdefault("history_position",False)
        model=BattleModel(**config).to(device)
        model.load_state_dict(restored["model"])
    elif initial:
        model,_=BattleModel.load(initial,device)
    else:
        model=BattleModel().to(device)
    if history_position is not None:
        model.config["history_position"]=history_position
    if heuristic_weight is not None:
        if heuristic_weight<0:
            raise ValueError("heuristic_weight must be nonnegative")
        model.config["heuristic_weight"]=heuristic_weight
    critic=PrivilegedCritic(model.config['width'],model.config['layers'],model.config['heads']).to(device) if privileged else None
    if critic is not None and not restored:
        critic.load_state_dict({key:model.state_dict()[key] for key in critic.state_dict()})
    if restored:
        if bool(restored.get('critic'))!=privileged:raise ValueError('Resume must keep the critic configuration')
        if critic is not None:critic.load_state_dict(restored['critic'])
    parameters=list(model.parameters())+(list(critic.parameters()) if critic is not None else [])
    optimizer=torch.optim.AdamW(parameters,lr=learning_rate,eps=1e-5,weight_decay=1e-4)
    if restored:
        optimizer.load_state_dict(restored["optimizer"])
        for group in optimizer.param_groups:
            group["lr"]=learning_rate
        torch.set_rng_state(restored["torch_rng"].cpu())
        if str(device).startswith("cuda") and restored.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(restored["cuda_rng"].cpu())
    rng=random.Random(seed)
    if restored and restored.get("snapshot_rng"):
        rng.setstate(restored["snapshot_rng"])
    models={"current":model}
    eligible=[]
    for row in restored.get("league",[]) if restored else []:
        config=dict(row["config"])
        config.setdefault("history_position",False)
        frozen=BattleModel(**config).to(device)
        frozen.load_state_dict(row["model"])
        frozen.eval()
        models[row["name"]]=frozen
        eligible.append(row["name"])
    snapshot_counter=restored.get("snapshot_counter",0) if restored else 0
    league_ratings=restored.get('league_ratings',{}) if restored else {}
    script_weights=restored.get('style_weights') if restored else None

    def snapshot():
        nonlocal snapshot_counter,eligible
        name=f"snapshot_fast_{snapshot_counter:08d}"
        snapshot_counter+=1
        frozen=BattleModel(**model.config).to(device)
        frozen.load_state_dict(model.state_dict())
        frozen.eval()
        models[name]=frozen
        league_ratings.setdefault(name,.5)
        eligible=retain_league(eligible+[name],league_ratings)

    if not eligible:
        snapshot()
    starting_steps=restored["steps"] if restored else 0
    total=starting_steps
    update=restored.get("updates",0) if restored else 0
    replay_folder=output/"reference_replay"
    replay=ReferenceReplay(model.config,seed=seed)
    if restored and restored.get("replay_rng"):
        replay.rng.bit_generator.state=restored["replay_rng"]
    for path in sorted(replay_folder.glob("*.npz"))[-400:]:
        if path.with_suffix(".json").exists():
            replay.add(path)
    settings={"trainer":"parallel_ppo_sil","initial":str(initial) if initial else None,"resumed_from":str(resume) if resume else None,
              "steps_requested":steps,"starting_steps":starting_steps,"workers":workers,"envs":envs,"horizon":horizon,
              "selfplay_fraction":selfplay_fraction,"snapshot_fraction":snapshot_fraction,
              "reference_workers":reference_workers,"reference_weight":reference_weight,
              "rollout_decisions":rollout_decisions,"privileged_critic":privileged,
              "learning_rate":learning_rate,"device":device,"seed":seed,"model_config":model.config}
    atomic_json(output/f"settings_{starting_steps:09d}.json",settings)
    emit("parallel_training_started",**settings)
    model.save(output/"initial.pt",phase="parallel_ppo_initial",seed=seed)
    startup=time.monotonic()
    actors=None
    references=None
    outcomes=[]
    reference_counts=Counter()
    stopped=False
    row={}

    def checkpoint():
        model.save(output/"latest.pt",phase="parallel_ppo_sil",steps=total,updates=update,seed=seed)
        state={"trainer":"parallel_ppo_sil","model":cpu_weights(model),"optimizer":optimizer.state_dict(),
               "config":model.config,"steps":total,"updates":update,
               "knowledge_sha256":knowledge().sha256,"feature_version":FEATURE_VERSION,
               "feature_signature":feature_signature(),'critic':cpu_weights(critic) if critic is not None else None,
               'league_ratings':league_ratings,'style_weights':script_weights,
               "worker_states":actors.states,"snapshot_counter":snapshot_counter,"snapshot_rng":rng.getstate(),
               "league":[{"name":name,"config":models[name].config,"model":cpu_weights(models[name])} for name in eligible],
               "replay_rng":replay.rng.bit_generator.state,"torch_rng":torch.get_rng_state(),
               "cuda_rng":torch.cuda.get_rng_state() if str(device).startswith("cuda") else None}
        durable_torch_save(state,output/'trainer_state.pt')

    try:
        actors=ParallelRollouts(workers,envs,seed,selfplay_fraction,snapshot_fraction,
                               privileged=privileged,
                               restored=restored.get("worker_states",()) if restored else ())
        references=ReferencePool(reference_workers,replay_folder,model,seed+391)
        started=time.monotonic()
        emit("parallel_workers_ready",startup_seconds=round(started-startup,2))
        while total-starting_steps<steps:
            if stop_file and Path(stop_file).exists():
                stopped=True
                break
            update+=1
            # One historical snapshot per rollout produces useful inference
            # batches; different rollouts sample the full retained league.
            opponents=[rng.choice(eligible)] if eligible else []
            records=[];finished=[];timing=Counter()
            target=min(rollout_decisions,steps-(total-starting_steps))
            while timing['policy_decisions']<target:
                fragment,games,stats=actors.collect(models,horizon,opponents,device,critic,script_weights)
                records.extend(fragment);finished.extend(games)
                for key in ('neural_batches','neural_positions','current_batches','current_positions','simulation_steps',
                            'rollout_seconds','policy_decisions','value_samples'):timing[key]+=stats[key]
                emit('rollout_progress',update=update,decisions=timing['policy_decisions'],target=target)
            timing['mean_inference_batch']=timing['neural_positions']/max(1,timing['neural_batches'])
            timing['mean_current_batch']=timing['current_positions']/max(1,timing['current_batches'])
            advantage,returns=trajectory_targets(records,model,device,critic=critic)
            optimize_started=time.monotonic()
            losses=ppo_update(model,optimizer,records,advantage,returns,device,batch_size,epochs,critic)
            for message in references.poll():
                reference_counts[message["kind"]]+=1
                if message["kind"]=="game":
                    replay.add(message["path"])
                    reference_counts["wins"]+=message["outcome"]==1
                emit("background_reference",**{k:v for k,v in message.items() if k!="kind"},kind=message["kind"])
            replay_batch=replay.sample(batch_size,device)
            sil_loss=0.
            if replay_batch is not None and reference_weight:
                obs,actions,replay_returns=replay_batch
                logits,values,_=model(obs)
                loss=self_imitation_loss(logits,values,actions,replay_returns)*reference_weight
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),.5)
                optimizer.step()
                sil_loss=float(loss.detach())
            model.eval()
            references.publish(model,update)
            total+=timing["policy_decisions"]
            outcomes.extend(finished)
            recent=outcomes[-500:]
            script_weights=style_weights(recent)
            for name in eligible:
                matches=[r for r in finished if r['opponent']==name and not r['draw']]
                for result in matches:league_ratings[name]=.98*league_ratings.get(name,.5)+.02*(not result['win'])
            counts=Counter(r["opponent"] for r in recent if r["win"] is not None)
            win_rates={name:sum(r["win"] for r in recent if r["opponent"]==name)/n for name,n in counts.items()}
            row={"update":update,"steps":total,"games":len(outcomes),"selfplay_games":sum(r["opponent"]=="current_selfplay" for r in outcomes),
                 "win_rates":win_rates,"reference_games":reference_counts["game"],"reference_errors":reference_counts["error"],
                 'script_sampling_weights':script_weights,'league_ratings':{n:league_ratings[n] for n in eligible},
                 "reference_training_win_rate":reference_counts["wins"]/max(1,reference_counts["game"]),
                 "replay_samples":len(replay.rows),"self_imitation_loss":sil_loss,
                 "optimization_seconds":round(time.monotonic()-optimize_started,3),
                 "seconds":round(time.monotonic()-started,2),
                 "decisions_per_second":round((total-starting_steps)/(time.monotonic()-started),2),**losses,**timing}
            append_json(output/'metrics.jsonl',row)
            emit("parallel_ppo_update",**row)
            if update%checkpoint_every==0 or total-starting_steps>=steps:
                snapshot()
                model.save(output/f"step_{total:09d}.pt",phase="parallel_ppo_sil",**row,seed=seed)
                checkpoint()
                retained={"current",*eligible,*set().union(*actors.active_models)}
                for name in list(models):
                    if name not in retained:
                        del models[name]
        checkpoint()
        summary={"stopped":stopped,"steps":total,"additional_policy_decisions":total-starting_steps,
                 "startup_seconds":started-startup,"elapsed_training_seconds":time.monotonic()-started,
                 "last_update":row,"settings":settings}
        atomic_json(output/'summary.json',summary)
        return output/"latest.pt"
    finally:
        if references:
            references.close()
        if actors:
            actors.close()
