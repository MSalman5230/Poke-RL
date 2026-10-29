"""CPU battle actors. Neural inference belongs to the central learner.

Fast actors never instantiate the expensive reference bot or a Torch network.
Both current-policy seats produce trajectories, including value-only waiting
states so forced switches do not leave gaps in rewards or credit assignment.
"""
import random
import traceback
from .engine import Engine
from .features import Encoder,encode_observation
from .policies import ScriptedBot
from .rewards import battle_reward
from .teams import TeamPool,fingerprint
from .opponent_labels import action_label


class FastGames:
    def __init__(self, worker_id, count, seed, selfplay_fraction=.3, snapshot_fraction=.3,
                 max_turns=200, saved_roster_fraction=.15, restored=None,privileged=False):
        from .arena import load_teams
        self.worker_id=worker_id
        self.count=count
        self.rng=random.Random(seed)
        if restored:
            self.rng.setstate(restored["rng"])
        self.counter=restored.get("games_started",0) if restored else 0
        self.sequence=0
        self.selfplay_fraction=selfplay_fraction
        self.snapshot_fraction=snapshot_fraction
        self.saved_roster_fraction=saved_roster_fraction
        self.max_turns=max_turns
        self.privileged=privileged
        self.style_weights=[.025,.2,.25,.15,.275,.1]
        self.pool=TeamPool()
        self.fixed=load_teams("train")
        self.held_out={fingerprint(t) for split in ("validation","test","validation_competitive") for t in load_teams(split)}
        self.engine=Engine()
        self.encoder=Encoder(engine=self.engine)
        self.games=[None]*count

    def sample_team(self):
        if self.rng.random()<.2:
            return self.rng.choice(self.fixed)
        while True:
            team=self.pool.sample(self.rng)
            if fingerprint(team) not in self.held_out:
                return team

    def reset(self,slot,models,snapshot_ids):
        value=self.rng.random()
        seat=self.rng.randrange(2)
        if value<self.selfplay_fraction:
            opponent="current"
        elif value<self.selfplay_fraction+self.snapshot_fraction and snapshot_ids:
            opponent=self.rng.choice(snapshot_ids)
        else:
            opponent=self.rng.choices(["random","damage","balanced","setup","defensive","stochastic"],
                                     self.style_weights)[0]
        parties=[self.sample_team(),self.sample_team()]
        # Familiar rosters also face unrelated styles, so this roster does not
        # uniquely identify the reference policy to the learner.
        if self.rng.random()<self.saved_roster_fraction:
            parties=[self.pool.data["reference_team"]]*2
        state=self.engine.request("new",game=str(slot),teams=parties,
            seed=[self.rng.randrange(65536) for _ in range(4)],max_turns=self.max_turns,training=self.privileged)
        self.counter+=1
        self.games[slot]={"state":state,"seat":seat,"opponent":opponent,"episode":self.counter,
                          "bot":None if opponent in models else ScriptedBot(opponent,self.rng.randrange(2**31),self.encoder),
                          "cache":{}}

    def collect(self,connection,horizon,models,snapshot_ids):
        for slot,g in enumerate(self.games):
            if g is None:
                self.reset(slot,models,snapshot_ids)
        completed=[]
        outcomes=[]
        simulation_steps=0
        for t in range(horizon):
            requests=[]
            choices=[[None,None] for _ in self.games]
            learners=[]
            for slot,g in enumerate(self.games):
                for side,view in enumerate(g["state"]["views"]):
                    learned=side==g["seat"] or g["opponent"]=="current"
                    model_id="current" if learned else g["opponent"]
                    if learned or (model_id in models and view["legal"]):
                        encoded=g["cache"].get(side)
                        if encoded is None:
                            encoded=encode_observation(self.encoder,models[model_id],view)
                        self.sequence+=1
                        request_id=self.sequence
                        requests.append({"id":request_id,"model":model_id,"obs":encoded,"learn":learned,
                                         "active":bool(view["legal"]),"slot":slot,"side":side})
                        if learned and self.privileged:
                            requests[-1]['critic_obs']=self.encoder.encode(g['state']['training']['critic_views'][side],entities_only=True)
                        if learned:
                            learners.append((slot,side,request_id,view,g["episode"],bool(view["legal"])))
                    elif view["legal"]:
                        choices[slot][side]=g["bot"].choose(view)
            connection.send({"kind":"infer","requests":requests,"completed":completed})
            completed=[]
            response=connection.recv()
            if response["kind"]=="close":
                return False
            if response["kind"]!="actions":
                raise RuntimeError("Expected a central inference response")
            for request,action in zip(requests,response["actions"]):
                if request["active"]:
                    choices[request["slot"]][request["side"]]=action
            after=self.engine.batch([{"op":"step","game":str(slot),"actions":actions} for slot,actions in enumerate(choices)])
            simulation_steps+=len(after)
            for slot,side,request_id,before,episode,active in learners:
                state=after[slot]
                view=state["views"][side]
                row={"id":request_id,"episode":episode,"side":side,
                     "reward":battle_reward(before,view,state["winner"]),"terminal":state["ended"],
                     "truncated":state["truncated"],"enemy":action_label(self.games[slot]['state']['views'][1-side],choices[slot][1-side])}
                encoded=encode_observation(self.encoder,models["current"],view)
                self.games[slot]["cache"][side]=encoded
                if not state["ended"] and (state["truncated"] or t==horizon-1):
                    row["bootstrap"]=encoded
                    if self.privileged:
                        row['bootstrap_critic']=self.encoder.encode(state['training']['critic_views'][side],entities_only=True)
                completed.append(row)
            for slot,state in enumerate(after):
                g=self.games[slot]
                g["state"]=state
                # Frozen opponents' next positions are encoded only when they
                # actually need a decision; never reuse their previous state.
                if g["opponent"]!="current":
                    g["cache"].pop(1-g["seat"],None)
                if state["ended"] or state["truncated"]:
                    outcomes.append({"opponent":"current_selfplay" if g["opponent"]=="current" else g["opponent"],
                        "win":None if g["opponent"]=="current" else state["winner"]==g["seat"],
                        "draw":state["winner"] is None,"truncated":state["truncated"],"turns":state["views"][0]["turn"]})
                    self.reset(slot,models,snapshot_ids)
        connection.send({"kind":"done","completed":completed,"outcomes":outcomes,
                         "simulation_steps":simulation_steps,
                         "active_models":[g["opponent"] for g in self.games if g["opponent"] in models],
                         "state":{"rng":self.rng.getstate(),"games_started":self.counter}})
        return True

    def close(self):
        self.engine.close()


def fast_worker(connection,settings):
    actor=None
    try:
        actor=FastGames(**settings)
        connection.send({"kind":"ready"})
        while True:
            message=connection.recv()
            if message["kind"]=="close":
                break
            if message["kind"]!="collect":
                raise RuntimeError("Unknown worker command")
            if message.get('style_weights') is not None:actor.style_weights=message['style_weights']
            if not actor.collect(connection,message["horizon"],message["models"],message["snapshot_ids"]):
                break
    except (EOFError,BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send({"kind":"error","error":traceback.format_exc()})
        except (EOFError,BrokenPipeError):
            pass
    finally:
        if actor:
            actor.close()
        connection.close()
