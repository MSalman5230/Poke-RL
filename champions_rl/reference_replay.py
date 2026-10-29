"""Asynchronous reference games and a bounded self-imitation replay buffer.

The original 32-trial bot runs only in these separate processes. Completed
episodes supply Monte Carlo returns to an explicitly off-policy SIL loss;
they are never mislabeled as fresh on-policy PPO rollouts.
"""
from collections import deque
import json
import multiprocessing as mp
from pathlib import Path
import queue
import traceback
import numpy as np
from .engine import ROOT
from .features import FEATURE_VERSION
from .knowledge import knowledge


class ReferenceStopped(Exception):
    pass


def policy_payload(model,version):
    return {"version":version,"config":dict(model.config),
            "model":{k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}}


def reference_worker(index,folder,seed,policy_queue,result_queue,stop_event):
    import torch
    from .arena import play_match
    from .engine import Engine
    from .features import Encoder
    from .model import BattleModel,NeuralBot,encode_for_model
    from .reference import ReferenceBot
    from .rewards import potential
    from .teams import TeamPool
    torch.set_num_threads(1)
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    prior=list(folder.glob(f"worker_{index}_*.json"))
    counter=max([int(p.stem.split('_')[-1]) for p in prior if p.stem.split('_')[-1].isdigit()],default=-1)+1
    payload=None
    model=None
    encoder=Encoder()
    team=TeamPool().data["reference_team"]
    consecutive_errors=0
    with Engine() as engine:
        while not stop_event.is_set():
            if payload is None:
                try:
                    payload=policy_queue.get(timeout=.2)
                except queue.Empty:
                    continue
            try:
                while True:
                    payload=policy_queue.get_nowait()
            except queue.Empty:
                pass
            if payload is None:
                break
            if model is None or model.config!=payload["config"]:
                model=BattleModel(**payload["config"]).cpu()
            model.load_state_dict({k:torch.from_numpy(v) for k,v in payload["model"].items()})
            model.eval()
            game_seed=seed+index*1000003+counter
            torch.manual_seed(game_seed)
            seat=counter%2
            reference=ReferenceBot(game_seed,strict=True)
            bots=[reference,reference]
            bots[seat]=NeuralBot(model,encoder)
            captured=[]

            def capture(side,obs,action):
                if stop_event.is_set():
                    raise ReferenceStopped()
                if side==seat:
                    captured.append((encode_for_model(model,encoder,obs),action,potential(obs)))

            def progress(_):
                if stop_event.is_set():
                    raise ReferenceStopped()

            path=folder/f"worker_{index}_{counter:08d}"
            try:
                result=play_match(engine,[team,team],bots,game_seed,capture=capture,progress=progress)
                if result["truncated"]:
                    result_queue.put({"kind":"truncated","worker":index,"game":counter})
                    counter+=1
                    continue  # Monte Carlo terminal returns would be invalid
                outcome=0. if result["winner"] is None else 1. if result["winner"]==seat else -1.
                arrays={k:np.stack([o[k] for o,_,_ in captured]) for k in captured[0][0]}
                arrays["labels"]=np.asarray([a for _,a,_ in captured],dtype=np.int64)
                # Gamma=1 and terminal potential=0 make this the exact return
                # from each decision, including intervening waiting phases.
                arrays["returns"]=np.asarray([outcome-.15*phi for _,_,phi in captured],dtype=np.float32)
                with path.with_suffix(".npz.tmp").open("wb") as handle:
                    np.savez_compressed(handle,**arrays)
                path.with_suffix(".npz.tmp").replace(path.with_suffix(".npz"))
                metadata={"knowledge_sha256":knowledge().sha256,"feature_version":FEATURE_VERSION,
                          "config":model.config,"policy_version":payload["version"],"seed":game_seed,
                          "outcome":outcome,"samples":len(captured),"seconds":result["seconds"],
                          "reference_manifest_sha256":reference.snapshot_sha256,
                          "reference_decisions":reference.audit,"reference_fallbacks":reference.fallbacks}
                path.with_suffix(".json.tmp").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
                path.with_suffix(".json.tmp").replace(path.with_suffix(".json"))
                result_queue.put({"kind":"game","worker":index,"path":str(path.with_suffix(".npz")),
                                  "outcome":outcome,"samples":len(captured),"policy_version":payload["version"]})
                consecutive_errors=0
            except ReferenceStopped:
                break
            except Exception:
                consecutive_errors+=1
                error={"kind":"error","worker":index,"game":counter,"error":traceback.format_exc(),
                       "diagnostics":reference.last_diagnostics}
                path.with_suffix(".error.json").write_text(json.dumps(error,indent=2)+"\n",encoding="utf-8")
                result_queue.put(error)
                if consecutive_errors>=3:
                    raise RuntimeError("Three consecutive reference games failed; inspect reference replay errors")
            counter+=1


class ReferencePool:
    def __init__(self,workers,folder,model,seed=71):
        self.context=mp.get_context("spawn")
        self.stop_event=self.context.Event()
        self.results=self.context.Queue(maxsize=64)
        self.policies=[]
        self.processes=[]
        for i in range(workers):
            policy=self.context.Queue(maxsize=1)
            process=self.context.Process(target=reference_worker,
                args=(i,str(folder),seed,policy,self.results,self.stop_event),name=f"reference-worker-{i}")
            process.start()
            self.policies.append(policy)
            self.processes.append(process)
        self.publish(model,0)

    def publish(self,model,version):
        if not self.policies:
            return
        payload=policy_payload(model,version)
        for policy in self.policies:
            try:
                policy.get_nowait()  # replace an unused update with the newest
            except queue.Empty:
                pass
            try:
                policy.put_nowait(payload)
            except queue.Full:
                pass  # an in-flight older update is sufficient until next time

    def poll(self,limit=8):
        messages=[]
        for _ in range(limit):
            try:
                messages.append(self.results.get_nowait())
            except queue.Empty:
                break
        for process in self.processes:
            if process.exitcode is not None and not self.stop_event.is_set():
                raise RuntimeError(f"Reference actor exited with code {process.exitcode}")
        return messages

    def close(self):
        from .parallel_rollouts import stop_process
        self.stop_event.set()
        for process in self.processes:
            stop_process(process)
        for channel in [*self.policies,self.results]:
            channel.cancel_join_thread()
            channel.close()


class ReferenceReplay:
    def __init__(self,config,capacity=4096,seed=71):
        self.config=config
        self.rows=deque(maxlen=capacity)
        self.loaded=set()
        self.rng=np.random.default_rng(seed)

    def add(self,path):
        path=Path(path)
        key=str(path.resolve())
        if key in self.loaded:
            return 0
        metadata=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata["knowledge_sha256"]!=knowledge().sha256 or metadata["feature_version"]!=FEATURE_VERSION:
            raise ValueError("Incompatible reference replay knowledge/features")
        if metadata.get("reference_fallbacks"):
            raise ValueError("Reference replay contains a degraded reference policy")
        if metadata["config"].get("heuristic_weight",0)!=self.config.get("heuristic_weight",0):
            return 0
        with np.load(path,allow_pickle=False) as data:
            labels=data["labels"]
            if not data["mask"][np.arange(len(labels)),labels].all():
                raise ValueError("Reference replay contains illegal labels")
            for i,label in enumerate(labels):
                obs={k:data[k][i].copy() for k in data.files if k not in ("labels","returns")}
                self.rows.append((obs,int(label),float(data["returns"][i])))
        self.loaded.add(key)
        return len(labels)

    def sample(self,count=128,device="cuda"):
        if len(self.rows)<32:
            return None
        import torch
        from .model import tensor_batch
        items=[self.rows[i] for i in self.rng.integers(len(self.rows),size=min(count,len(self.rows)))]
        return (tensor_batch([r[0] for r in items],device),
                torch.tensor([r[1] for r in items],device=device),
                torch.tensor([r[2] for r in items],device=device))


def self_imitation_loss(logits,values,actions,returns):
    """SIL objective (Oh et al., ICML 2018); replay is deliberately off-policy."""
    from torch.distributions import Categorical
    advantage=(returns-values).clamp(min=0)
    policy=-(Categorical(logits=logits).log_prob(actions)*advantage.detach()).mean()
    return policy+.5*advantage.square().mean()
