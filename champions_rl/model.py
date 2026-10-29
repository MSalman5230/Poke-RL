"""Small entity transformer with policy, value, and opponent-action heads."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .features import Encoder, N_ACTIONS, FEATURE_VERSION
from .knowledge import knowledge


def tensor_batch(observations, device="cpu"):
    return {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=device) for k in observations[0]}


def encode_for_model(model, encoder, observation):
    encoded=encoder.encode(observation)
    weight=model.config.get("heuristic_weight",0.)
    if weight:
        from .policies import ScriptedBot
        scores=ScriptedBot("balanced",encoder=encoder).scores(observation,encoded) if observation["legal"] else np.zeros(N_ACTIONS)
        encoded["guide"]=np.where(encoded["mask"],np.clip(scores,-5,5)*weight,0).astype(np.float32)
    return encoded


class BattleModel(nn.Module):
    def __init__(self, width=128, layers=2, heads=4, action_context=True, heuristic_weight=0.,history_position=True):
        super().__init__()
        encoder = Encoder()
        self.config = {"width":width,"layers":layers,"heads":heads,"action_context":action_context,"heuristic_weight":heuristic_weight,
                       "history_position":history_position}
        k = knowledge()
        self.species_embedding = nn.Embedding(len(k.vocab["species"])+2,32)
        self.move_embedding = nn.Embedding(len(k.vocab["moves"])+2,24)
        self.item_embedding = nn.Embedding(len(k.vocab["items"])+2,16)
        self.ability_embedding = nn.Embedding(len(k.vocab["abilities"])+2,16)
        self.mon = nn.Linear(len(encoder.mon_names)+32+4*24+16+16,width)
        self.field = nn.Linear(len(encoder.field_names),width)
        self.history = nn.Linear(27,width)
        self.segment = nn.Embedding(4,width)
        layer = nn.TransformerEncoderLayer(width,heads,dim_feedforward=width*3,dropout=0.,activation="gelu",batch_first=True,norm_first=True)
        self.transformer = nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        # Bind each candidate to the contextual representation of its Pokémon.
        # Global pooling alone cannot associate a switch/preview choice with a
        # particular roster member, even if the transformer understands it.
        self.action = nn.Sequential(nn.Linear(len(encoder.action_names)+24+width*(4 if action_context else 2),width),nn.GELU(),nn.Linear(width,1))
        self.value = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.opponent_action = nn.Linear(width,N_ACTIONS)
        self.species_type = nn.Linear(32,len(k.type_names))
        self.move_properties = nn.Linear(24,len(k.type_names)+5)
        self.register_buffer("segments",torch.tensor([0]+[1]*6+[2]*6+[3]*12,dtype=torch.long))
        position=torch.arange(12).unsqueeze(1)
        frequency=torch.exp(torch.arange(0,width,2)*(-np.log(10000.)/width))
        order=torch.zeros(12,width)
        order[:,0::2]=torch.sin(position*frequency)
        order[:,1::2]=torch.cos(position*frequency)
        self.register_buffer("history_order",order*.1,persistent=False)
        if action_context:
            from itertools import permutations
            self.register_buffer("preview_picks",torch.tensor(list(permutations(range(6),3)),dtype=torch.long))
        nn.init.orthogonal_(self.action[-1].weight,gain=.01)
        nn.init.zeros_(self.action[-1].bias)

    def forward(self, obs):
        mons = self.mon(torch.cat([obs["pokemon"],self.species_embedding(obs["species"]),
            self.move_embedding(obs["moves"]).flatten(-2),self.item_embedding(obs["items"]),self.ability_embedding(obs["abilities"])],-1))
        tokens = torch.cat([self.field(obs["field"]).unsqueeze(1),mons,
                            self.history(torch.cat([self.move_embedding(obs["history_moves"]),obs["history"]],-1))],1)
        if self.config["history_position"]:
            # Without order information, attention treats simultaneous move
            # histories as a bag and cannot learn observed speed order.
            tokens[:,13:]=tokens[:,13:]+self.history_order.unsqueeze(0)
        context = self.norm(self.transformer(tokens + self.segment(self.segments).unsqueeze(0)))
        pooled = context[:,0]
        if self.config["action_context"]:
            own = context[:,1:7]
            active_index = obs["pokemon"][:,:6,3].argmax(-1)
            active = own[torch.arange(len(own),device=own.device),active_index]
            actors = torch.cat([active.unsqueeze(1).expand(-1,8,-1),own,own[:,self.preview_picks[:,0]]],1)
            reserve = own[:,self.preview_picks[:,1:]].mean(2)
            reserve = torch.cat([torch.zeros_like(actors[:,:14]),reserve],1)
            opposition = context[:,7:13].mean(1)
            global_state = torch.cat([pooled.unsqueeze(1).expand(-1,N_ACTIONS,-1),actors,reserve,
                                      opposition.unsqueeze(1).expand(-1,N_ACTIONS,-1)],-1)
        else:
            own = context[:,1:7].mean(1)
            global_state = torch.cat([pooled,own],-1).unsqueeze(1).expand(-1,N_ACTIONS,-1)
        logits = self.action(torch.cat([obs["actions"],self.move_embedding(obs["action_moves"]),global_state],-1)).squeeze(-1)
        if self.config["heuristic_weight"]:
            logits=logits+obs["guide"]
        mask = obs["mask"].clone()
        # Terminal/no-choice states are evaluated only for value bootstrapping.
        mask[~mask.any(-1),0] = True
        logits = logits.masked_fill(~mask,-1e9)
        return logits,self.value(pooled).squeeze(-1),self.opponent_action(pooled)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits,value,_ = self(obs)
        distribution = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else distribution.sample()
        return action,distribution.log_prob(action),value

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_suffix(path.suffix+".tmp")
        torch.save({"schema":1,"feature_version":FEATURE_VERSION,"knowledge_sha256":knowledge().sha256,"config":self.config,
                    "model":self.state_dict(),"metadata":metadata},temp)
        temp.replace(path)

    @classmethod
    def load(cls,path,device="cpu"):
        data=torch.load(path,map_location=device,weights_only=True)
        if data["knowledge_sha256"] != knowledge().sha256 or data.get("feature_version")!=FEATURE_VERSION:
            raise ValueError("Checkpoint and knowledge snapshot differ")
        config=dict(data["config"])
        config.setdefault("action_context",False)  # first local experiment
        config.setdefault("history_position",False)
        model=cls(**config).to(device)
        model.load_state_dict(data["model"])
        model.eval()
        return model,data.get("metadata",{})


class NeuralBot:
    def __init__(self, model, encoder=None, deterministic=False, name="neural"):
        self.model=model
        self.encoder=encoder or Encoder()
        self.device=next(model.parameters()).device
        self.deterministic=deterministic
        self.name=name

    def choose(self,obs):
        with torch.no_grad():
            action,_,_=self.model.act(tensor_batch([encode_for_model(self.model,self.encoder,obs)],self.device),self.deterministic)
        return int(action.item())
