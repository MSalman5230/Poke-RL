"""Parallel CPU rollouts with one GPU inference broker and on-policy batches."""
from collections import defaultdict
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import subprocess
import time
import numpy as np
import torch
from .model import tensor_batch
from .rollout_workers import fast_worker


def stop_process(process,timeout=5):
    process.join(timeout)
    if process.is_alive():
        # Only terminate the tree of a process started by this collector.
        if os.name=="nt":
            subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            process.terminate()
        process.join(5)


class ParallelRollouts:
    def __init__(self,workers=4,envs=32,seed=47,selfplay_fraction=.3,snapshot_fraction=.3,
                 max_turns=200,saved_roster_fraction=.15,restored=(),timeout=120,privileged=False):
        if not 1<=workers<=envs:
            raise ValueError("workers must be between 1 and envs")
        if min(selfplay_fraction,snapshot_fraction)<0 or selfplay_fraction+snapshot_fraction>1:
            raise ValueError("Opponent fractions must be nonnegative and sum to at most one")
        self.context=mp.get_context("spawn")
        self.connections=[]
        self.processes=[]
        self.states=[None]*workers
        self.active_models=[set() for _ in range(workers)]
        self.timeout=timeout
        try:
            for worker in range(workers):
                parent,child=self.context.Pipe()
                settings={"worker_id":worker,"count":envs//workers+int(worker<envs%workers),
                          "seed":seed+104729*worker,"selfplay_fraction":selfplay_fraction,
                          "snapshot_fraction":snapshot_fraction,"max_turns":max_turns,
                          "saved_roster_fraction":saved_roster_fraction,
                          "privileged":privileged,
                          "restored":restored[worker] if len(restored)==workers else None}
                process=self.context.Process(target=fast_worker,args=(child,settings),name=f"battle-worker-{worker}")
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                if not connection.poll(timeout):
                    raise TimeoutError("CPU battle worker did not initialize")
                reply=connection.recv()
                if reply["kind"]!="ready":
                    raise RuntimeError(reply.get("error",str(reply)))
        except BaseException:
            self.close()
            raise

    def collect(self,models,horizon=32,snapshot_ids=(),device="cuda",critic=None,style_weights=None):
        """Models stay frozen for this whole collection; PPO updates afterward.

        A shared current policy acts for both sides of self-play. Historical
        opponents are grouped by model ID rather than inferred one at a time.
        No reference-bot work enters this synchronization path.
        """
        for model in models.values():
            model.eval()
        if critic is not None:critic.eval()
        configs={name:model.config for name,model in models.items()}
        for connection in self.connections:
            connection.send({"kind":"collect","horizon":horizon,"models":configs,"snapshot_ids":list(snapshot_ids),'style_weights':style_weights})
        live=set(range(len(self.connections)))
        lookup={connection:i for i,connection in enumerate(self.connections)}
        stash={}
        records=[]
        outcomes=[]
        stats={"neural_batches":0,"neural_positions":0,"current_batches":0,"current_positions":0,"simulation_steps":0}
        last_message=time.monotonic()
        started=last_message

        def finish(worker,completed):
            for transition in completed:
                key=(worker,transition.pop("id"))
                if key not in stash:
                    raise RuntimeError(f"Unmatched rollout transition {key}")
                row=stash.pop(key)
                row.update(transition)
                row["trajectory"]=(worker,row["episode"],row["side"])
                records.append(row)

        while live:
            waiting=[self.connections[i] for i in live]
            ready=wait(waiting,timeout=.05)
            if not ready:
                for i in live:
                    if not self.processes[i].is_alive():
                        raise RuntimeError(f"Battle worker {i} exited unexpectedly")
                if time.monotonic()-last_message>self.timeout:
                    raise TimeoutError("No progress from CPU battle workers")
                continue
            # Briefly coalesce independently ready actors into larger GPU calls.
            deadline=time.monotonic()+.002
            messages=[]
            seen=set()
            while ready:
                for connection in ready:
                    if connection in seen:
                        continue
                    seen.add(connection)
                    messages.append((lookup[connection],connection.recv()))
                remaining=[c for c in waiting if c not in seen]
                delay=max(0.,deadline-time.monotonic())
                ready=wait(remaining,timeout=delay) if remaining and delay else []
            last_message=time.monotonic()
            groups=defaultdict(list)
            responses={}
            for worker,message in messages:
                if message["kind"]=="error":
                    raise RuntimeError(message["error"])
                finish(worker,message.get("completed",[]))
                if message["kind"]=="done":
                    live.remove(worker)
                    outcomes.extend(message["outcomes"])
                    stats["simulation_steps"]+=message["simulation_steps"]
                    self.states[worker]=message["state"]
                    self.active_models[worker]=set(message["active_models"])
                elif message["kind"]=="infer":
                    responses[worker]=[None]*len(message["requests"])
                    for index,request in enumerate(message["requests"]):
                        groups[request["model"]].append((worker,index,request))
                else:
                    raise RuntimeError(f"Unexpected actor message: {message['kind']}")
            with torch.no_grad():
                for model_id,requests in groups.items():
                    model=models[model_id]
                    batch=tensor_batch([r["obs"] for _,_,r in requests],device)
                    action,logp,value=model.act(batch)
                    if model_id=='current' and critic is not None:
                        value=critic(tensor_batch([r['critic_obs'] for _,_,r in requests],device))
                    action,logp,value=action.cpu().tolist(),logp.cpu().tolist(),value.cpu().tolist()
                    stats["neural_batches"]+=1
                    stats["neural_positions"]+=len(requests)
                    if model_id=="current":
                        stats["current_batches"]+=1
                        stats["current_positions"]+=len(requests)
                    for (worker,index,request),a,p,v in zip(requests,action,logp,value):
                        responses[worker][index]=a
                        if request["learn"]:
                            stash[worker,request["id"]]={"obs":request["obs"],"action":a,"logp":p,"value":v,
                                "policy_active":request["active"]}
                            if critic is not None:
                                stash[worker,request['id']]['critic_obs']=request['critic_obs']
            for worker,actions in responses.items():
                self.connections[worker].send({"kind":"actions","actions":actions})
        if stash:
            raise RuntimeError("Unfinished policy transitions after rollout collection")
        stats.update(rollout_seconds=time.monotonic()-started,
                     policy_decisions=sum(r["policy_active"] for r in records),value_samples=len(records),
                     mean_inference_batch=stats["neural_positions"]/max(1,stats["neural_batches"]),
                     mean_current_batch=stats["current_positions"]/max(1,stats["current_batches"]))
        return records,outcomes,stats

    def close(self):
        for connection in self.connections:
            try:
                connection.send({"kind":"close"})
            except (BrokenPipeError,EOFError,OSError):
                pass
        for process in self.processes:
            stop_process(process)
        for connection in self.connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()


def trajectory_targets(records,model,device="cuda",lam=.95,critic=None):
    """GAE per player/episode, including passive turns and truncation bootstrap."""
    bootstraps=[i for i,r in enumerate(records) if "bootstrap" in r]
    next_values={}
    with torch.no_grad():
        for chunk in range(0,len(bootstraps),256):
            ids=bootstraps[chunk:chunk+256]
            values=(critic(tensor_batch([records[i]['bootstrap_critic'] for i in ids],device)) if critic is not None else
                    model(tensor_batch([records[i]["bootstrap"] for i in ids],device))[1]).cpu().tolist()
            next_values.update(zip(ids,values))
    trajectories=defaultdict(list)
    for i,row in enumerate(records):
        trajectories[row["trajectory"]].append(i)
    adv=np.zeros(len(records),dtype=np.float32)
    returns=np.zeros(len(records),dtype=np.float32)
    for indices in trajectories.values():
        carry=0.
        following=None
        for i in reversed(indices):
            row=records[i]
            value_after=0. if row["terminal"] else records[following]["value"] if following is not None else next_values[i]
            delta=row["reward"]+value_after-row["value"]
            carry=delta+(0. if row["terminal"] or row["truncated"] else lam*carry)
            adv[i]=carry
            returns[i]=carry+row["value"]
            following=i
    return torch.as_tensor(adv,device=device),torch.as_tensor(returns,device=device)


def terminal_critic_targets(records,bootstrap_returns):
    """Observed match returns where available; bootstrap only unfinished games."""
    trajectories=defaultdict(list)
    for i,row in enumerate(records):trajectories[row['trajectory']].append(i)
    targets=bootstrap_returns.detach().clone();observed=0
    for indices in trajectories.values():
        complete=False;total=0.
        for i in reversed(indices):
            row=records[i]
            if row['truncated']:complete=False
            if row['terminal']:complete=True;total=0.
            if complete:
                total+=row['reward'];targets[i]=total;observed+=1
    return targets,observed
