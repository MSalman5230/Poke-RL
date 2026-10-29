"""Domain pretraining, imitation, and masked PPO with an opponent league."""
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .arena import VectorArena, STYLES, load_teams, play_match
from .engine import Engine, ROOT
from .features import Encoder, FEATURE_VERSION
from .knowledge import knowledge
from .model import BattleModel, NeuralBot, tensor_batch, encode_for_model
from .policies import ScriptedBot
from .reference import ReferenceBot
from .teams import TeamPool
from .persistence import durable_torch_save
from .provenance import feature_signature


def emit(event, **fields):
    print(json.dumps({"event":event,**fields}),flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")


def collect_demonstrations(output, games=100, teacher="balanced", seed=41):
    from .expert_data import collect
    return collect(output,games,teacher,workers=1,seed=seed,
                   teams='saved' if teacher=='reference' else 'train')['samples']


def load_demonstrations(folder,with_manifest=False):
    buckets=defaultdict(list)
    manifest=[]
    folders=folder if isinstance(folder,(list,tuple)) else [folder]
    paths=sorted(p for directory in folders for p in Path(directory).glob('*.npz'))
    if len({p.stem for p in paths})!=len(paths):
        raise ValueError('Duplicate demonstration game identifiers across data folders')
    for path in paths:
        if not path.with_suffix(".json").exists():
            continue  # collection commits metadata after finishing a shard
        meta=json.loads(path.with_suffix(".json").read_text())
        if meta["knowledge_sha256"]!=knowledge().sha256 or meta.get("feature_version")!=FEATURE_VERSION or meta.get("reference_fallbacks",0) or meta.get('feature_signature')!=feature_signature():
            raise ValueError(f"Incompatible or degraded teacher shard {path}")
        manifest.append({'path':str(path.resolve()),'samples':meta['samples'],'teacher':meta['teacher'],'seed':meta['seed']})
        with np.load(path,allow_pickle=False) as data:
            for key in data.files:
                buckets[key].append(data[key])
            buckets['expert'].append(np.full(len(data['labels']),meta['teacher']=='reference'))
            buckets['teacher'].append(np.full(len(data['labels']),meta['teacher']))
            buckets['correction'].append(np.full(len(data['labels']),bool(meta.get('student_sha256'))))
            import hashlib
            # Hold out entire games, not correlated turns from the same game.
            heldout=int(hashlib.sha256(path.stem.encode()).hexdigest()[:8],16)%10==0
            buckets['validation'].append(np.full(len(data['labels']),heldout))
    if not buckets:
        raise ValueError(f"No demonstration shards in {folder}")
    data={k:np.concatenate(v) for k,v in buckets.items()}
    return (data,manifest) if with_manifest else data


def pretrain_knowledge(model,steps=300,device="cuda"):
    k=knowledge()
    species=torch.tensor([k.token("species",n) for n in k.species],device=device)
    types=torch.tensor([[float(t in row["types"]) for t in k.type_names] for row in k.species.values()],device=device)
    moves=torch.tensor([k.token("moves",n) for n in k.moves],device=device)
    properties=torch.tensor([[float(t==m["type"]) for t in k.type_names]+[
        m["basePower"]/150,m["accuracy"]/100,float(m["category"]=="Physical"),float(m["category"]=="Special"),float(m["category"]=="Status")]
        for m in k.moves.values()],device=device)
    optimizer=torch.optim.AdamW([*model.species_embedding.parameters(),*model.move_embedding.parameters(),
                                *model.species_type.parameters(),*model.move_properties.parameters()],lr=.003)
    for _ in range(steps):
        loss=nn.functional.binary_cross_entropy_with_logits(model.species_type(model.species_embedding(species)),types)
        loss+=nn.functional.mse_loss(model.move_properties(model.move_embedding(moves)),properties)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    emit("domain_pretraining",steps=steps,loss=float(loss.detach()))


def imitation_groups(data,ids,correction_mass=0.):
    expert=data['expert'][ids]
    correction=data.get('correction',np.zeros(len(data['expert']),dtype=bool))[ids]
    if correction_mass:
        if not correction.any():raise ValueError('Correction weighting requires learner-state labels in both train and validation')
        other={'reference':expert&~correction,'other':~expert&~correction}
        other={name:mask for name,mask in other.items() if mask.any()}
        if not other:raise ValueError('Retain original demonstrations alongside corrections')
        return {'correction':(correction,correction_mass),**{name:(mask,(1-correction_mass)/len(other)) for name,mask in other.items()}}
    groups={name:mask for name,mask in {'reference':expert,'other':~expert}.items() if mask.any()}
    return {name:(mask,1/len(groups)) for name,mask in groups.items()}


def imitation_validation(model,data,valid_ids,batch_size,device,correction_mass=0.):
    excluded={'labels','outcomes','expert','validation','teacher','student_actions','enemy_labels','correction'}
    losses=[];agreements=[]
    model.eval()
    with torch.no_grad():
        for ids in np.array_split(valid_ids,max(1,math.ceil(len(valid_ids)/batch_size))):
            obs={k:torch.as_tensor(v[ids],device=device) for k,v in data.items() if k not in excluded}
            predictions=model(obs)[0]
            targets=torch.as_tensor(data['labels'][ids],device=device)
            losses.extend(nn.functional.cross_entropy(predictions,targets,reduction='none').cpu().tolist())
            agreements.extend((predictions.argmax(-1)==targets).cpu().tolist())
    losses=np.asarray(losses);agreements=np.asarray(agreements)
    groups=imitation_groups(data,valid_ids,correction_mass)
    balanced=float(sum(mass*losses[mask].mean() for mask,mass in groups.values()))
    teachers={str(name):{'samples':int((data['teacher'][valid_ids]==name).sum()),
                        'loss':float(losses[data['teacher'][valid_ids]==name].mean()),
                        'agreement':float(agreements[data['teacher'][valid_ids]==name].mean())}
              for name in np.unique(data['teacher'][valid_ids])}
    return {'validation_loss':float(losses.mean()),'validation_accuracy':float(agreements.mean()),
            'balanced_validation_loss':balanced,'validation_by_teacher':teachers,
            'validation_by_source':{name:{'samples':int(mask.sum()),'mass':mass,'loss':float(losses[mask].mean()),
                                         'agreement':float(agreements[mask].mean())} for name,(mask,mass) in groups.items()}}


def behavior_clone(folder,output,steps=1500,batch_size=128,device="cuda",seed=43,width=128,layers=2,initial=None,learning_rate=1e-4,correction_mass=0.):
    if not 0<=correction_mass<1:raise ValueError('correction_mass must be in [0,1)')
    seed_all(seed)
    model=BattleModel.load(initial,device)[0] if initial else BattleModel(width=width,layers=layers).to(device)
    if not initial:pretrain_knowledge(model,device=device)
    data,manifest=load_demonstrations(folder,with_manifest=True)
    from .persistence import atomic_json
    atomic_json(Path(output).with_suffix('.data.json'),{'shards':manifest,'feature_signature':feature_signature(),
                'validation':'10% of whole games by SHA256 shard name','correction_mass':correction_mass,
                'sampling':'Requested correction mass, then equal reference/other mass among original demonstrations; zero uses the original 50/50 teacher mixture'})
    n=len(data["labels"])
    rng=np.random.default_rng(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    excluded={'labels','outcomes','expert','validation','teacher','student_actions','enemy_labels','correction'}
    weights=np.zeros(n)
    train_ids=np.flatnonzero(~data['validation'])
    for mask,mass in imitation_groups(data,train_ids,correction_mass).values():
        weights[train_ids[mask]]=mass/mask.sum()
    if not weights.any():raise ValueError('No training games outside validation split')
    weights/=weights.sum()
    valid_ids=np.flatnonzero(data['validation'])
    if not len(valid_ids):raise ValueError('Collect more games to form a game-disjoint validation split')
    baseline=imitation_validation(model,data,valid_ids,batch_size,device,correction_mass)
    best=baseline['balanced_validation_loss']
    model.save(output,phase='behavior_cloning_candidate',samples=n,updates=0,seed=seed,**baseline,
               expert_samples=int(data['expert'].sum()),initial=str(initial) if initial else None,correction_mass=correction_mass,
               promotion='Requires paired battle validation against the starting checkpoint')
    from .persistence import append_json
    metrics=Path(output).with_suffix('.metrics.jsonl')
    append_json(metrics,{'step':0,**baseline})
    emit('behavior_cloning_validation',step=0,**baseline)
    emit('demonstrations_loaded',samples=n,expert_samples=int(data['expert'].sum()),validation_samples=len(valid_ids),
         expert_sampling_mass=float(weights[data['expert']].sum()),history_position=model.config['history_position'])
    model.train()
    started=time.monotonic()
    for step in range(steps):
        index=rng.choice(n,size=batch_size,p=weights)
        obs={k:torch.as_tensor(v[index],device=device) for k,v in data.items() if k not in excluded}
        label=torch.as_tensor(data["labels"][index],device=device)
        target=torch.as_tensor(data["outcomes"][index],device=device)
        logits,value,enemy=model(obs)
        loss=nn.functional.cross_entropy(logits,label)+.1*nn.functional.mse_loss(value,target)
        if 'enemy_labels' in data:
            enemies=torch.as_tensor(data['enemy_labels'][index],device=device)
            mask=enemies>=0
            if mask.any():loss+=.05*nn.functional.cross_entropy(enemy[mask],enemies[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step%100==0 or step==steps-1:
            validation=imitation_validation(model,data,valid_ids,batch_size,device,correction_mass)
            if validation['balanced_validation_loss']<best:
                best=validation['balanced_validation_loss']
                model.save(output,phase='behavior_cloning_candidate',samples=n,updates=step+1,seed=seed,**validation,
                           expert_samples=int(data['expert'].sum()),initial=str(initial) if initial else None,correction_mass=correction_mass,
                           promotion='Requires paired battle validation against the starting checkpoint')
            append_json(metrics,{'step':step+1,**validation})
            emit("behavior_cloning",step=step+1,steps=steps,loss=float(loss.detach()),
                 **validation,
                 batch_accuracy=float((logits.argmax(-1)==label).float().mean()),samples=n,seconds=round(time.monotonic()-started,1))
            model.train()
    model.eval()
    model.save(Path(output).with_suffix('.last.pt'),phase="behavior_cloning_last",samples=n,updates=steps,seed=seed)
    return BattleModel.load(output,device)[0]


def advantages(rewards,values,next_values,terminated,done,lam=.95):
    """GAE with gamma=1; truncate bootstraps but does not join the next episode."""
    result=torch.zeros_like(rewards)
    carry=torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+(~terminated[t])*next_values[t]-values[t]
        carry=delta+lam*(~done[t])*carry
        result[t]=carry
    return result,result+values


def train_ppo(output,initial=None,steps=200_000,n_envs=16,horizon=64,epochs=4,batch_size=128,
              device="cuda",seed=47,reference_fraction=0.,snapshot_fraction=.3,checkpoint_every=10,stop_file=None,resume=None,
              procedural_teams=True,learning_rate=2e-4,heuristic_weight=None,history_position=None):
    if steps<1 or n_envs<1 or horizon<1 or checkpoint_every<1:
        raise ValueError("steps, envs, horizon, and checkpoint interval must be positive")
    if reference_fraction<0 or snapshot_fraction<0 or reference_fraction+snapshot_fraction>1:
        raise ValueError("Opponent fractions must be nonnegative and sum to at most one")
    seed_all(seed)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    restored=torch.load(resume,map_location=device,weights_only=True) if resume else None
    if restored and (restored["knowledge_sha256"]!=knowledge().sha256 or restored.get("feature_version")!=FEATURE_VERSION or restored.get('feature_signature')!=feature_signature()):
        raise ValueError("Resume state is incompatible with the current knowledge/features")
    if restored:
        restored["config"].setdefault("action_context",False)
        restored["config"].setdefault("history_position",False)
    model,metadata=BattleModel.load(initial,device) if initial else (BattleModel(**(restored["config"] if restored else {})).to(device),{})
    if restored:
        model.load_state_dict(restored["model"])
    if heuristic_weight is not None:
        if heuristic_weight<0:
            raise ValueError("heuristic_weight must be nonnegative")
        model.config["heuristic_weight"]=heuristic_weight
    if history_position is not None:
        model.config["history_position"]=history_position
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,eps=1e-5,weight_decay=1e-4)
    if restored:
        optimizer.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["torch_rng"].cpu())
        if device.startswith("cuda") and restored.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(restored["cuda_rng"].cpu())
        for group in optimizer.param_groups:
            group["lr"]=learning_rate
    arena=VectorArena(n_envs,seed,reference_fraction=reference_fraction,snapshot_fraction=snapshot_fraction,procedural_teams=procedural_teams,
                      rng_state=restored.get("arena_rng") if restored else None,
                      league_state=restored.get("league",()) if restored else (),
                      games_started=restored.get("games_started",0) if restored else 0)
    encoder=arena.encoder
    if not arena.snapshots:
        arena.add_snapshot(model)
    starting_steps=restored["steps"] if restored else 0
    total=starting_steps
    updates=math.ceil(steps/(n_envs*horizon))
    started=time.monotonic()
    metrics_path=output/"metrics.jsonl"
    settings={"steps_requested":steps,"envs":n_envs,"horizon":horizon,"seed":seed,"device":device,
              "reference_fraction":reference_fraction,"snapshot_fraction":snapshot_fraction,
              "procedural_teams":procedural_teams,"learning_rate":learning_rate,"resumed_from":str(resume) if resume else None,
              "heuristic_weight":model.config.get("heuristic_weight",0.),
              "history_position":model.config["history_position"],
              "initial":str(initial) if initial else None,"starting_steps":starting_steps}
    (output/f"settings_{starting_steps:09d}.json").write_text(json.dumps(settings,indent=2)+"\n")
    emit("training_started",**settings)
    model.save(output/"initial.pt",phase="ppo_initial",seed=seed)
    try:
        encoded=[encode_for_model(model,encoder,o) for o in arena.observations()]
        for update in range(updates):
            if stop_file and Path(stop_file).exists():
                emit("stopped",reason="STOP file",steps=total)
                break
            buffer=defaultdict(list)
            obs_buffer=[]
            model.eval()
            for t in range(horizon):
                obs=tensor_batch(encoded,device)
                with torch.no_grad():
                    action,logp,value=model.act(obs)
                after,reward,terminal,truncated,enemy=arena.step(action.cpu().tolist())
                next_encoded=[encode_for_model(model,encoder,o) for o in after]
                with torch.no_grad():
                    _,next_value,_=model(tensor_batch(next_encoded,device))
                obs_buffer.append(encoded)
                # Ongoing states are exactly the next decisions. Only games
                # reset by the arena need encoding again; retain terminal
                # observations above for correct truncation bootstrapping.
                current=arena.observations()
                encoded=[encode_for_model(model,encoder,current[i]) if terminal[i] or truncated[i] else next_encoded[i] for i in range(n_envs)]
                for key,v in (("action",action),("logp",logp),("value",value),("next_value",next_value)):
                    buffer[key].append(v.detach())
                buffer["reward"].append(torch.tensor(reward,device=device))
                buffer["terminal"].append(torch.tensor(terminal,device=device,dtype=torch.bool))
                buffer["done"].append(torch.tensor([a or b for a,b in zip(terminal,truncated)],device=device,dtype=torch.bool))
                buffer["enemy"].append(torch.tensor(enemy,device=device))
                total+=n_envs
                if t%16==15:
                    emit("rollout_progress",update=update+1,steps=total,episodes=len(arena.results),seconds=round(time.monotonic()-started,1))
            stacked={k:torch.stack(v) for k,v in buffer.items()}
            advantage,returns=advantages(stacked["reward"],stacked["value"],stacked["next_value"],stacked["terminal"],stacked["done"])
            advantage=advantage.flatten()
            advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
            returns=returns.flatten()
            data=tensor_batch([o for row in obs_buffer for o in row],device)
            old_logp=stacked["logp"].flatten()
            action=stacked["action"].flatten()
            enemies=stacked["enemy"].flatten()
            size=len(action)
            model.train()
            losses=[]
            early=False
            for epoch in range(epochs):
                for index in torch.randperm(size,device=device).split(batch_size):
                    current={k:v[index] for k,v in data.items()}
                    logits,value,opponent=model(current)
                    distribution=Categorical(logits=logits)
                    logp=distribution.log_prob(action[index])
                    logratio=logp-old_logp[index]
                    ratio=logratio.exp()
                    policy=torch.maximum(-advantage[index]*ratio,-advantage[index]*ratio.clamp(.8,1.2)).mean()
                    value_loss=.5*nn.functional.mse_loss(value,returns[index])
                    entropy=distribution.entropy().mean()
                    mask=current["mask"]
                    log_probs=logits.log_softmax(-1)
                    # Uniform-to-policy KL prevents legal situational actions
                    # disappearing permanently; illegal logits never contribute.
                    explore=-(log_probs.masked_fill(~mask,0).sum(-1)/mask.sum(-1).clamp(min=1)).mean()
                    target=enemies[index]
                    valid=target>=0
                    opponent_loss=nn.functional.cross_entropy(opponent[valid],target[valid]) if valid.any() else value.sum()*0
                    loss=policy+value_loss-.01*entropy+.001*explore+.03*opponent_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),.5)
                    optimizer.step()
                    kl=((ratio-1)-logratio).mean().detach()
                    losses.append(float(loss.detach()))
                    if float(kl)>.04:
                        early=True
                        break
                if early:
                    break
            recent=arena.results[-100:]
            row={"update":update+1,"steps":total,"episodes":len(arena.results),"loss":float(np.mean(losses)),
                 "recent_win_rate":sum(r["win"] for r in recent)/max(1,len(recent)),
                 "recent_truncations":sum(r["truncated"] for r in recent),
                 "seconds":round(time.monotonic()-started,1),"decisions_per_second":round((total-starting_steps)/(time.monotonic()-started),2)}
            with metrics_path.open("a",encoding="utf-8") as handle:
                handle.write(json.dumps(row)+"\n")
            emit("ppo_update",**row)
            model.eval()
            if (update+1)%checkpoint_every==0 or update==updates-1:
                model.save(output/f"step_{total:09d}.pt",phase="ppo",**row,seed=seed)
                model.save(output/"latest.pt",phase="ppo",**row,seed=seed)
                # Optimizer/RNG state is separate from inference-only weights.
                resume={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":model.config,
                        "steps":total,"knowledge_sha256":knowledge().sha256,"feature_version":FEATURE_VERSION,
                        'feature_signature':feature_signature(),
                        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state() if device.startswith("cuda") else None}
                resume.update(arena.checkpoint_state())
                durable_torch_save(resume,output/'trainer_state.pt')
                arena.add_snapshot(model)
        model.save(output/"latest.pt",phase="ppo",steps=total,seed=seed)
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":model.config,
               "steps":total,"knowledge_sha256":knowledge().sha256,"feature_version":FEATURE_VERSION,
               'feature_signature':feature_signature(),
               "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state() if device.startswith("cuda") else None}
        state.update(arena.checkpoint_state())
        durable_torch_save(state,output/'trainer_state.pt')
    finally:
        arena.close()
    return output/"latest.pt"
