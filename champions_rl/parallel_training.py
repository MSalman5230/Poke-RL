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
from .parallel_rollouts import ParallelRollouts,trajectory_targets,terminal_critic_targets,pre_update_value_metrics
from .reference_replay import ReferencePool,ReferenceReplay,self_imitation_loss
from .training import emit,seed_all
from .provenance import feature_signature
from .persistence import durable_torch_save,atomic_json,append_json
from .curriculum import style_weights,retain_league
from .power import keep_awake,awake_seconds


def cpu_weights(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def ppo_update(model,optimizer,records,advantage,returns,device,batch_size=512,epochs=4,critic=None,
               critic_optimizer=None,target_kl=.02,kl_sample_size=4096,critic_passes=None):
    critic_passes=epochs if critic_passes is None else critic_passes
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
    actor_ids=active.nonzero().squeeze(-1)
    probe=actor_ids[torch.randperm(len(actor_ids),device=device)[:kl_sample_size]]
    actor_seen=torch.zeros(len(records),dtype=torch.bool,device=device)
    critic_seen=torch.zeros_like(actor_seen)
    optimized_samples=critic_samples=actor_epochs=critic_epochs=0
    early=False;trigger_kl=None;updates=0

    @torch.no_grad()
    def representative_kl():
        total=0.
        for index in probe.split(batch_size):
            logits=model({k:v[index] for k,v in data.items()},detach_value=critic is not None)[0]
            difference=Categorical(logits=logits).log_prob(actions[index])-old_logp[index]
            total+=float(((difference.exp()-1)-difference).sum())
        return total/len(probe)

    for epoch in range(epochs):
        batches=list(actor_ids[torch.randperm(len(actor_ids),device=device)].split(batch_size))
        for batch_index,index in enumerate(batches):
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
            # Keep the public value head usable as a baseline, without a second
            # value gradient competing with the policy encoder when the
            # independent private critic is trained alongside it.
            value_weight=.1 if critic is not None else .5
            loss=policy+value_weight*nn.functional.mse_loss(value,returns[index])-.01*entropy+.001*uniform+.03*auxiliary
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),.5)
            optimizer.step()
            actor_seen[index]=True
            optimized_samples+=len(index)
            updates+=1
            kl=((ratio-1)-logratio)[learn].mean().detach() if learn.any() else torch.tensor(0.)
            losses.append(float(loss.detach()))
            entropies.append(float(entropy.detach()))
            # A noisy 128-position minibatch used to stop both networks. Audit
            # a fixed representative sample after small groups of actor steps.
            if updates%8==0 or batch_index==len(batches)-1 or float(kl)>target_kl*2:
                measured=representative_kl();kls.append(measured)
                if measured>target_kl:
                    early=True;trigger_kl=measured
                    break
        if batch_index==len(batches)-1:actor_epochs+=1
        if early:
            break
    # The independent critic sees every rollout record, including wait states,
    # for its separately requested passes even if the actor hits its KL cap.
    if critic is not None:
        if critic_optimizer is None:raise ValueError('A separate critic optimizer is required')
        for epoch in range(critic_passes):
            for index in torch.randperm(len(records),device=device).split(batch_size):
                private_error=nn.functional.mse_loss(critic({k:v[index] for k,v in private.items()}),returns[index])
                critic_optimizer.zero_grad(set_to_none=True)
                private_error.backward()
                nn.utils.clip_grad_norm_(critic.parameters(),.5)
                critic_optimizer.step()
                critic_seen[index]=True;critic_samples+=len(index)
                private_errors.append(float(private_error.detach()))
            critic_epochs+=1
    return {"loss":float(np.mean(losses)),"entropy":float(np.mean(entropies)),"approx_kl":float(np.mean(kls)),
            'private_value_mse':float(np.mean(private_errors)) if private_errors else None,
            'optimized_samples':optimized_samples,'kl_early_stop':early,'kl_trigger':trigger_kl,
            'kl_probe_samples':len(probe),'actor_unique_samples':int(actor_seen.sum()),
            'actor_sample_coverage':float(actor_seen.sum())/len(actor_ids),'actor_completed_epochs':actor_epochs,
            'critic_optimized_samples':critic_samples,'critic_unique_samples':int(critic_seen.sum()),
            'critic_sample_coverage':float(critic_seen.sum())/len(records) if critic is not None else None,
            'critic_completed_epochs':critic_epochs}


@keep_awake()
def train_parallel(output,initial=None,resume=None,steps=200000,workers=4,envs=64,horizon=32,
                   device="cuda",seed=101,selfplay_fraction=.1,snapshot_fraction=.1,reference_workers=0,
                   reference_weight=0.,learning_rate=3e-5,batch_size=512,epochs=4,checkpoint_every=1,
                   stop_file=None,history_position=None,heuristic_weight=None,rollout_decisions=32768,privileged=True,
                   critic_learning_rate=1e-4,league_checkpoints=(),validation_every=10,validation_workers=4,validation_games=256,
                   critic_epochs=2,value_baseline='public',private_blend=.5,critic_targets='bootstrapped',identity_dropout=None):
    if min(steps,workers,envs,horizon,batch_size,epochs,checkpoint_every,rollout_decisions)<1:
        raise ValueError("Training sizes and intervals must be positive")
    if reference_workers<0 or reference_weight<0:
        raise ValueError("Reference worker count and loss weight must be nonnegative")
    if value_baseline not in ('public','private','blend') or not 0<=private_blend<=1:
        raise ValueError('Invalid PPO baseline or blend')
    if critic_epochs<1 or critic_targets not in ('bootstrapped','terminal'):
        raise ValueError('Invalid critic training settings')
    if not privileged and value_baseline!='public':raise ValueError('Private/blended baseline requires a private critic')
    seed_all(seed)
    torch.set_num_threads(2)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics.jsonl').exists() and not resume:raise ValueError('Existing training run: use --resume or a new output folder')
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
    if identity_dropout is not None:
        if not 0<=identity_dropout<1:raise ValueError('Identity dropout must be in [0,1)')
        model.config['identity_dropout']=identity_dropout
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
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,eps=1e-5,weight_decay=1e-4)
    critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=critic_learning_rate,eps=1e-5,weight_decay=1e-4) if critic is not None else None
    if restored:
        optimizer.load_state_dict(restored["optimizer"])
        if critic is not None:critic_optimizer.load_state_dict(restored['critic_optimizer'])
        for group in optimizer.param_groups:
            group["lr"]=learning_rate
        if critic is not None:
            for group in critic_optimizer.param_groups:group['lr']=critic_learning_rate
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
    for index,path in enumerate(league_checkpoints):
        name=f'seeded_{index:02d}_{Path(path).stem}'
        if name not in models:
            frozen,_=BattleModel.load(path,device);models[name]=frozen
            eligible.append(name);league_ratings[name]=.5

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
              "learning_rate":learning_rate,'critic_learning_rate':critic_learning_rate,'batch_size':batch_size,'epochs':epochs,
              'critic_epochs':critic_epochs,'value_baseline':value_baseline,'private_blend':private_blend,
              'critic_targets':critic_targets,'identity_dropout':model.config.get('identity_dropout',0.),
              'league_checkpoints':list(map(str,league_checkpoints)),
              'validation_every':validation_every,'validation_workers':validation_workers,'validation_games':validation_games,
              "device":device,"seed":seed,"model_config":model.config}
    atomic_json(output/f"settings_{starting_steps:09d}.json",settings)
    emit("parallel_training_started",**settings)
    initial_checkpoint=output/'initial.pt'
    if not initial_checkpoint.exists():
        if restored:
            import shutil
            origin=Path(restored.get('initial_checkpoint',Path(resume).parent/'initial.pt'))
            shutil.copyfile(origin,initial_checkpoint)
        else:model.save(initial_checkpoint,phase='parallel_ppo_initial',seed=seed)
    from .validation_queue import ValidationQueue
    validation=ValidationQueue(output/'validation',output/'initial.pt',validation_workers,validation_games) if validation_every else None
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
               'initial_checkpoint':str(initial_checkpoint.resolve()),
               'critic_optimizer':critic_optimizer.state_dict() if critic_optimizer is not None else None,
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
        awake_started=awake_seconds()
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
                fragment,games,stats=actors.collect(models,horizon,opponents,device,critic,script_weights,value_baseline,private_blend)
                records.extend(fragment);finished.extend(games)
                for key in ('neural_batches','neural_positions','current_batches','current_positions','simulation_steps',
                            'rollout_seconds','policy_decisions','value_samples'):timing[key]+=stats[key]
                emit('rollout_progress',update=update,decisions=timing['policy_decisions'],target=target)
            timing['mean_inference_batch']=timing['neural_positions']/max(1,timing['neural_batches'])
            timing['mean_current_batch']=timing['current_positions']/max(1,timing['current_batches'])
            advantage,returns=trajectory_targets(records,model,device,critic=critic,value_baseline=value_baseline,private_blend=private_blend)
            diagnostics=pre_update_value_metrics(records,returns)
            terminal_returns,terminal_samples=terminal_critic_targets(records,returns)
            if critic_targets=='terminal':returns=terminal_returns
            optimize_started=time.monotonic()
            losses=ppo_update(model,optimizer,records,advantage,returns,device,batch_size,epochs,critic,critic_optimizer,critic_passes=critic_epochs)
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
            awake_elapsed=awake_seconds()-awake_started
            row={"update":update,"steps":total,'phase_start_steps':starting_steps,'additional_policy_decisions':total-starting_steps,
                 'awake_seconds':round(awake_elapsed,3),
                 'suspended_seconds':round(max(0.,time.monotonic()-started-awake_elapsed),3),
                 'decisions_per_awake_second':round((total-starting_steps)/max(.001,awake_elapsed),2),
                 "games":len(outcomes),"selfplay_games":sum(r["opponent"]=="current_selfplay" for r in outcomes),
                 "win_rates":win_rates,"reference_games":reference_counts["game"],"reference_errors":reference_counts["error"],
                 'script_sampling_weights':script_weights,'league_ratings':{n:league_ratings[n] for n in eligible},
                 "reference_training_win_rate":reference_counts["wins"]/max(1,reference_counts["game"]),
                 "replay_samples":len(replay.rows),"self_imitation_loss":sil_loss,
                 "optimization_seconds":round(time.monotonic()-optimize_started,3),
                 'critic_terminal_samples':terminal_samples,'critic_terminal_fraction':terminal_samples/len(records),
                 "seconds":round(time.monotonic()-started,2),
                 "decisions_per_second":round((total-starting_steps)/(time.monotonic()-started),2),**diagnostics,**losses,**timing}
            append_json(output/'metrics.jsonl',row)
            emit("parallel_ppo_update",**row)
            if validation:validation.poll()
            if update%checkpoint_every==0 or total-starting_steps>=steps or validation and update%validation_every==0:
                snapshot()
                model.save(output/f"step_{total:09d}.pt",phase="parallel_ppo_sil",**row,seed=seed)
                checkpoint()
                if critic is not None:
                    durable_torch_save({'critic':cpu_weights(critic),'config':model.config,
                                        'steps':total,'updates':update,'feature_version':FEATURE_VERSION,
                                        'feature_signature':feature_signature(),'knowledge_sha256':knowledge().sha256},
                                       output/f'critic_{total:09d}.pt')
                if validation and update%validation_every==0:validation.schedule(output/f'step_{total:09d}.pt',total)
                retained={"current",*eligible,*set().union(*actors.active_models)}
                for name in list(models):
                    if name not in retained:
                        del models[name]
        checkpoint()
        training_seconds=time.monotonic()-started
        if validation:
            validation.close()
        summary={"stopped":stopped,"steps":total,"additional_policy_decisions":total-starting_steps,
                 "startup_seconds":started-startup,"elapsed_training_seconds":training_seconds,
                 'validation_results':validation.finished if validation else [],
                 "last_update":row,"settings":settings}
        atomic_json(output/'summary.json',summary)
        return output/"latest.pt"
    finally:
        if references:
            references.close()
        if actors:
            actors.close()
