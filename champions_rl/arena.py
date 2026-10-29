"""Parallel headless matches, opponent sampling, and terminal accounting."""
import json
import random
import time
from pathlib import Path
from .engine import Engine, ROOT
from .features import Encoder
from .policies import ScriptedBot
from .reference import ReferenceBot
from .rewards import battle_reward
from .teams import TeamPool, fingerprint

STYLES = ("random", "damage", "balanced", "setup", "defensive", "stochastic")


def load_teams(split):
    return [r["team"] for r in json.loads((ROOT / f"data/teams_{split}.json").read_text(encoding="utf-8"))["teams"]]


def play_match(engine, teams, bots, seed, game="evaluation", max_turns=200, capture=None, progress=None):
    rng = random.Random(seed)
    result = engine.request("new", game=game, teams=teams, seed=[rng.randrange(65536) for _ in range(4)], max_turns=max_turns)
    count = 0
    started = time.monotonic()
    while not (result["ended"] or result["truncated"]):
        actions = []
        for i, view in enumerate(result["views"]):
            action = bots[i].choose(view) if view["legal"] else None
            actions.append(action)
            if capture and action is not None:
                capture(i, view, action)
            if progress:
                progress({"event": "decision", "turn": view["turn"], "side": i,
                          "policy": bots[i].name, "seconds": round(time.monotonic()-started, 2)})
        result = engine.request("step", game=game, actions=actions)
        count += 1
        if count > max_turns*12 + 120:
            raise RuntimeError("Decision limit reached without engine termination")
    trace = engine.request("trace", game=game)
    engine.request("close", game=game)
    return {"winner": result["winner"], "truncated": result["truncated"], "turns": result["views"][0]["turn"],
            "decisions": count, "seconds": time.monotonic()-started, "trace": trace}


class VectorArena:
    def __init__(self, n=16, seed=0, split="train", reference_fraction=.0, snapshot_fraction=.25, max_turns=200, procedural_teams=True,
                 rng_state=None,league_state=(),games_started=0):
        self.engine = Engine()
        self.rng = random.Random(seed)
        if rng_state is not None:
            self.rng.setstate(rng_state)
        self.teams = load_teams(split)
        self.n = n
        self.max_turns = max_turns
        self.reference_fraction = reference_fraction
        self.snapshot_fraction = snapshot_fraction
        self.team_pool = TeamPool()
        self.reference_team = self.team_pool.data["reference_team"]
        self.procedural_teams = procedural_teams and split=="train"
        self.held_out = {fingerprint(t) for s in ("validation","test","validation_competitive") for t in load_teams(s)} if self.procedural_teams else set()
        self.encoder = Encoder()
        self.snapshots = []
        self.opponents = [None]*n
        self.seats = [0]*n
        self.states = [None]*n
        self.results = []
        self.games = games_started
        if league_state:
            from .model import BattleModel,NeuralBot
            for row in league_state:
                config=dict(row["config"])
                config.setdefault("history_position",False)
                frozen=BattleModel(**config).cpu()
                frozen.load_state_dict(row["model"])
                frozen.eval()
                self.snapshots.append(NeuralBot(frozen,self.encoder,name=row["name"]))
        for i in range(n):
            self.reset(i)

    def add_snapshot(self, model):
        from .model import BattleModel, NeuralBot
        frozen = BattleModel(**model.config).cpu()
        frozen.load_state_dict({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        frozen.eval()
        self.snapshots.append(NeuralBot(frozen,self.encoder,name=f"snapshot_{self.games}"))
        self.snapshots = self.snapshots[-12:]

    def checkpoint_state(self):
        return {"arena_rng":self.rng.getstate(),"games_started":self.games,
                "league":[{"name":bot.name,"config":bot.model.config,
                           "model":{k:v.detach().cpu().clone() for k,v in bot.model.state_dict().items()}}
                          for bot in self.snapshots]}

    def reset(self, i):
        seat = self.rng.randrange(2)
        self.seats[i] = seat
        value = self.rng.random()
        if value < self.reference_fraction:
            opponent = ReferenceBot(self.rng.randrange(2**31),strict=True)
        elif value < self.reference_fraction + self.snapshot_fraction and self.snapshots:
            opponent = self.rng.choice(self.snapshots)
        else:
            opponent = ScriptedBot(self.rng.choices(STYLES,[.05,.2,.3,.15,.15,.15])[0],self.rng.randrange(2**31),self.encoder)
        self.opponents[i] = opponent
        teams = [self.sample_team(), self.sample_team()]
        if opponent.name == "autoplay_reference":
            teams = [self.reference_team,self.reference_team]
        self.states[i] = self.engine.request("new",game=str(i),teams=teams,
            seed=[self.rng.randrange(65536) for _ in range(4)],max_turns=self.max_turns)
        self.games += 1

    def sample_team(self):
        from .holdouts import held_out_core
        if not self.procedural_teams or self.rng.random()<.2:
            return self.rng.choice([t for t in self.teams if not held_out_core(t)])
        while True:
            team=self.team_pool.sample(self.rng)
            if fingerprint(team) not in self.held_out and not held_out_core(team):
                return team

    def observations(self):
        return [s["views"][seat] for s, seat in zip(self.states,self.seats)]

    def step(self, actions):
        before = self.observations()
        requests, enemy_actions = [], []
        for i, (state, seat, action) in enumerate(zip(self.states,self.seats,actions)):
            other = state["views"][1-seat]
            enemy = self.opponents[i].choose(other) if other["legal"] else None
            from .opponent_labels import action_label
            enemy_actions.append(action_label(other,enemy))
            choices = [None,None]
            choices[seat] = int(action)
            choices[1-seat] = enemy
            requests.append({"op":"step","game":str(i),"actions":choices})
        self.states = self.engine.batch(requests)
        # Replacement/pivot phases can ask only the opponent to act.
        for _ in range(24):
            pending = [i for i,s in enumerate(self.states) if not(s["ended"] or s["truncated"]) and not s["views"][self.seats[i]]["legal"]]
            if not pending:
                break
            requests = []
            for i in pending:
                seat = self.seats[i]
                other = self.states[i]["views"][1-seat]
                if not other["legal"]:
                    raise RuntimeError("Both players waiting in a nonterminal match")
                choices = [None,None]
                choices[1-seat] = self.opponents[i].choose(other)
                requests.append({"op":"step","game":str(i),"actions":choices})
            for i,result in zip(pending,self.engine.batch(requests)):
                self.states[i] = result
        else:
            raise RuntimeError("Unbounded replacement sequence")
        after = self.observations()
        rewards = [battle_reward(a,b,s["winner"]) for a,b,s in zip(before,after,self.states)]
        terminated = [s["ended"] for s in self.states]
        truncated = [s["truncated"] for s in self.states]
        for i, state in enumerate(self.states):
            if state["ended"] or state["truncated"]:
                self.results.append({"opponent":self.opponents[i].name,"win":state["winner"]==self.seats[i],
                    "draw":state["winner"] is None,"truncated":state["truncated"],"turns":after[i]["turn"]})
                self.reset(i)
        return after,rewards,terminated,truncated,enemy_actions

    def close(self):
        self.engine.close()
