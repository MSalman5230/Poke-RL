"""Gymnasium interface for another RL library or a custom training loop."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .arena import load_teams
from .engine import Engine
from .features import Encoder, N_ACTIONS
from .policies import ScriptedBot
from .reference import ReferenceBot
from .rewards import battle_reward
from .teams import TeamPool


class ChampionsEnv(gym.Env):
    """One learning player, with simultaneous opponent choices kept private.

    Reset/step observations contain only numerical player-visible features.
    `info['action_mask']` identifies legal actions. Time caps truncate instead
    of rewarding a fabricated win. No render window or internet is required.
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent="balanced", split="train", max_turns=200):
        super().__init__()
        self.encoder = Encoder()
        self.teams = load_teams(split)
        self.style = opponent
        self.max_turns = max_turns
        self.engine = None
        self.state = None
        k = self.encoder.k
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict({
            "pokemon": spaces.Box(-np.inf,np.inf,(12,len(self.encoder.mon_names)),np.float32),
            "species": spaces.Box(0,len(k.vocab["species"])+1,(12,),np.int64),
            "items": spaces.Box(0,len(k.vocab["items"])+1,(12,),np.int64),
            "abilities": spaces.Box(0,len(k.vocab["abilities"])+1,(12,),np.int64),
            "moves": spaces.Box(0,len(k.vocab["moves"])+1,(12,4),np.int64),
            "move_features": spaces.Box(-np.inf,np.inf,(12,4,len(self.encoder.move_names)),np.float32),
            "move_mask": spaces.MultiBinary((12,4)),
            "last_moves": spaces.Box(0,len(k.vocab['moves'])+1,(12,),np.int64),
            "opponent_switch_mask": spaces.MultiBinary(6),
            "field": spaces.Box(-np.inf,np.inf,(len(self.encoder.field_names),),np.float32),
            "history_moves": spaces.Box(0,len(k.vocab["moves"])+1,(12,),np.int64),
            "history": spaces.Box(0,1,(12,len(self.encoder.history_names)),np.float32),
            "history_species": spaces.Box(0,len(k.vocab['species'])+1,(12,),np.int64),
            "actions": spaces.Box(-np.inf,np.inf,(N_ACTIONS,len(self.encoder.action_names)),np.float32),
            "action_moves": spaces.Box(0,len(k.vocab["moves"])+1,(N_ACTIONS,),np.int64),
            "preview_disguise": spaces.Box(0,3,(N_ACTIONS,6),np.float32),
            "mask": spaces.MultiBinary(N_ACTIONS),
        })
        for kind in ('items','abilities'):
            self.observation_space[kind+'_prior']=spaces.Box(0,len(k.vocab[kind])+1,(12,3),np.int64)
            self.observation_space[kind+'_weights']=spaces.Box(0,1,(12,3),np.float32)

    def _observation(self):
        encoded = self.encoder.encode(self.state["views"][self.seat])
        info = {"action_mask":encoded["mask"].copy(),"turn":self.state["views"][self.seat]["turn"]}
        if self.state["ended"] or self.state["truncated"]:
            info.update(winner=self.state["winner"],learner_seat=self.seat,
                        trace=self.engine.request("trace",game="gym"))
        return encoded,info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self.engine is None:
            self.engine = Engine()
        self.seat = int(options.get("seat",self.np_random.integers(2)))
        if self.seat not in (0,1):
            raise ValueError("seat must be 0 or 1")
        opponent_seed = int(self.np_random.integers(2**31))
        self.opponent = ReferenceBot(opponent_seed) if self.style=="reference" else ScriptedBot(self.style,opponent_seed,self.encoder)
        teams = options.get("teams")
        if teams is None:
            teams = [self.teams[int(self.np_random.integers(len(self.teams)))] for _ in range(2)]
            if self.style=="reference":
                teams = [TeamPool().data["reference_team"]]*2
        self.state = self.engine.request("new",game="gym",teams=teams,
            seed=self.np_random.integers(65536,size=4).tolist(),max_turns=self.max_turns)
        return self._observation()

    def step(self, action):
        if self.state is None or self.state["ended"] or self.state["truncated"]:
            raise RuntimeError("reset() is required before stepping this episode")
        before = self.state["views"][self.seat]
        if action not in [a["id"] for a in before["legal"]]:
            raise ValueError(f"Illegal action {action}; use info['action_mask']")
        for iteration in range(24):
            other = self.state["views"][1-self.seat]
            actions = [None,None]
            actions[self.seat] = int(action) if iteration==0 else None
            actions[1-self.seat] = self.opponent.choose(other) if other["legal"] else None
            self.state = self.engine.request("step",game="gym",actions=actions)
            if self.state["ended"] or self.state["truncated"] or self.state["views"][self.seat]["legal"]:
                break
        else:
            raise RuntimeError("Unbounded forced replacement sequence")
        after = self.state["views"][self.seat]
        reward = battle_reward(before,after,self.state["winner"])
        encoded,info = self._observation()
        return encoded,reward,self.state["ended"],self.state["truncated"],info

    def close(self):
        if self.engine:
            self.engine.close()
            self.engine = None
        self.state = None
