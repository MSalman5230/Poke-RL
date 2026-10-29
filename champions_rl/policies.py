"""Different opponent styles. All receive the same player-visible observation."""
import random
import numpy as np
from .features import Encoder

TRAINING_STYLES=('random','balanced','hyperoffense','setup_sweeper','attrition','pivot')


class ScriptedBot:
    def __init__(self, style="balanced", seed=0, encoder=None):
        if style not in (*TRAINING_STYLES,'damage','setup','defensive','stochastic'):
            raise ValueError(f"Unknown opponent style: {style}")
        self.name = style
        self.rng = random.Random(seed)
        self.encoder = encoder or Encoder()

    def scores(self, obs, encoded=None):
        encoded = encoded or self.encoder.encode(obs)
        f = encoded["actions"]
        ix = {name: i for i, name in enumerate(self.encoder.action_names)}
        get = lambda name: f[:, ix[name]]
        score = np.full(len(f), -1e9, dtype=np.float32)
        active = next((p for p in obs["own"] if p["active"]), obs["own"][0])
        foe = next((p for p in obs["opponent"] if p["active"]), obs["opponent"][0])
        incoming = self.encoder.best_damage(foe, active, obs["weather"])
        for action in obs["legal"]:
            a = action["id"]
            if action["kind"] == "preview":
                score[a] = 2.5*get("coverage_worst")[a]+get("coverage_mean")[a]+.5*get("coverage_backup")[a]+.6*get("lead_matchup")[a]
                if self.name=='hyperoffense':score[a]+=1.8*get('lead_matchup')[a]
                if self.name=='attrition':
                    score[a]+=.002*sum(sum(self.encoder.k.species_info(obs['own'][i]['species'])['baseStats'][s] for s in ('hp','def','spd')) for i in action['picks'])
                if self.name=='pivot':score[a]+=1.5*get('coverage_backup')[a]
                continue
            if action["kind"] == "switch":
                if action.get("revive"):
                    score[a] = get("switch_offense")[a] - .4*get("incoming_damage")[a]
                    continue
                score[a] = .45*get("switch_offense")[a] - .8*get("incoming_damage")[a] - get("hazard_cost")[a] - .2
                if obs["phase"] == "switch":
                    score[a] += get("actor_hp")[a]
                elif incoming >= active.get("hp", 1):
                    score[a] += .35
                if self.name == "damage":
                    score[a] -= 2
                if self.name == "defensive":
                    score[a] += .4*(incoming-get("incoming_damage")[a])
                if self.name=='hyperoffense':score[a]-=3
                if self.name=='setup_sweeper':score[a]-=.6
                if self.name=='attrition':score[a]+=.8*get('survival_probability')[a]-.4*get('incoming_damage')[a]
                if self.name=='pivot':score[a]+=1.5*get('switch_offense')[a]-.6*get('incoming_damage')[a]-.3
                continue
            damage = get("estimated_damage")[a]
            score[a] = min(damage, foe["hp"]+.1)*get("accuracy")[a] + .15*get("mega")[a]
            if damage >= foe["hp"]:
                score[a] += .6 + .15*(get("priority")[a]>0)
            if self.name != "damage":
                score[a] += min(get("heal")[a], max(0,1-active.get("hp",1)))*(.9 if self.name != "defensive" else 1.5)
                score[a] += .22*get("inflict_status")[a]*(not foe.get("status"))
                score[a] += .16*get("hazard")[a]*(obs["turn"]<5)
                score[a] += .08*get("pivot")[a]
                if incoming < active.get("hp",1)*.6 and obs["turn"] < 30:
                    score[a] += get("setup")[a]*(.7 if self.name == "setup" else .35)
                score[a] -= get("recoil")[a]*damage*.2
                if active.get("boosts",{}).get("atk",0)>=4 or active.get("boosts",{}).get("spa",0)>=4:
                    score[a] -= get("setup")[a]
            if self.name=='hyperoffense':score[a]=1.4*min(damage,foe['hp']+.1)*get('accuracy')[a]+.8*get('ko_probability')[a]+.2*get('mega')[a]
            if self.name=='setup_sweeper':
                boosted=max(active.get('boosts',{}).get('atk',0),active.get('boosts',{}).get('spa',0))
                if boosted<4 and active['hp']>.35 and get('survival_probability')[a]>.5:
                    score[a]+=1.3*get('setup')[a]
            if self.name=='attrition':
                score[a]=.5*score[a]+2.4*min(get('heal')[a],max(0,1-active['hp']))
                score[a]+=1.1*get('inflict_status')[a]*(not foe.get('status'))+.45*get('hazard')[a]*(obs['turn']<4)
                score[a]-=.3*get('recoil')[a]
            if self.name=='pivot':score[a]+=.8*get('pivot')[a]+.25*get('force_switch')[a]
        return score

    def choose(self, obs):
        if not obs["legal"]:
            return None
        if self.name == "random":
            return self.rng.choice(obs["legal"])["id"]
        score = self.scores(obs)
        if self.name == "stochastic" and self.rng.random()<.15:
            return self.rng.choice(obs["legal"])["id"]
        return int(np.argmax(score))
