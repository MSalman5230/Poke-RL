"""Auditable feature engineering over a single player's information only."""
import numpy as np
from .knowledge import knowledge, canonical, STATS

STATUSES = ("", "brn", "par", "psn", "tox", "slp", "frz")
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
VOLATILES = ("substitute", "leechseed", "confusion", "taunt", "encore", "disable", "protect", "destinybond",
             "yawn", "partiallytrapped", "trapped", "perishsong", "aquaring", "ingrain", "focusenergy", "transform")
CONDITIONS = ("stealthrock", "spikes", "toxicspikes", "stickyweb", "reflect", "lightscreen", "auroraveil", "tailwind", "safeguard")
WEATHER = ("", "sunnyday", "raindance", "sandstorm", "snow", "desolateland", "primordialsea")
TERRAIN = ("", "electricterrain", "grassyterrain", "psychicterrain", "mistyterrain")
N_ACTIONS = 134
FEATURE_VERSION = 2
ACTION_NAMES = ["move", "switch", "preview", "mega", "power", "accuracy", "priority", "pp",
                "physical", "special", "status", "estimated_damage", "ko_margin", "effectiveness", "stab",
                "actor_hp", "target_hp", "speed_ratio", "heal", "recoil", "drain", "setup", "protect",
                "inflict_status", "hazard", "pivot", "force_switch", "contact", "sound", "boost_atk", "boost_spa", "boost_spe",
                "boost_def", "boost_spd", "incoming_damage", "switch_offense", "hazard_cost",
                "coverage_mean", "coverage_worst", "coverage_backup", "lead_matchup", "team_hp",
                "target_item_known", "target_ability_known", "force_replacement", "revive"]


def move_ids(mon):
    values = mon.get("moves", []) or mon.get("set", {}).get("moves", [])
    return [canonical(m["id"] if isinstance(m, dict) else m) for m in values][:4]


class Encoder:
    def __init__(self):
        self.k = knowledge()
        self.priors = {}
        from .teams import TeamPool
        for name, rows in TeamPool().by_species.items():
            self.priors[name] = [canonical(m) for m in rows[0]["set"]["moves"]]
        self.mon_names = ["hp", "hp_known", "seen", "active", "fainted", "selected", "selection_known", "own", "item_known", "ability_known"]
        self.mon_names += [f"boost_{s}" for s in BOOSTS] + [f"base_{s}" for s in STATS] + [f"stat_{s}" for s in STATS]
        self.mon_names += [f"type_{t}" for t in self.k.type_names] + [f"status_{s}" for s in STATUSES]
        self.mon_names += [f"volatile_{v}" for v in VOLATILES] + [f"pp_{i}" for i in range(4)]
        self.action_names = ACTION_NAMES + [f"type_{t}" for t in self.k.type_names]
        self.field_names = ["preview", "replacement", "turn", "our_mega", "enemy_mega", "trickroom", "gravity"]
        self.field_names += [f"weather_{w}" for w in WEATHER] + [f"terrain_{t}" for t in TERRAIN]
        self.field_names += [f"side_{s}_{c}" for s in range(2) for c in CONDITIONS]
        self.field_names += ["our_fainted", "enemy_fainted"]

    def mon_features(self, mon, ours):
        k = self.k
        stats = k.mon_stats(mon)
        base = k.species_info(mon.get("species")).get("baseStats", dict.fromkeys(STATS, 0))
        pp = [m["pp"] / max(1, m["maxpp"]) for m in mon.get("moves", []) if isinstance(m, dict)]
        pp = (pp + [0.] * 4)[:4]
        selected = mon.get("selected") if ours else True if mon.get("seen") else None
        values = [mon.get("hp", 1), ours or mon.get("seen", False), mon.get("seen", False), mon.get("active", False),
                  mon.get("fainted", False), bool(selected), selected is not None, ours,
                  mon.get("item") is not None, mon.get("ability") is not None]
        values += [mon.get("boosts", {}).get(s, 0) / 6 for s in BOOSTS]
        values += [base[s] / 200 for s in STATS] + [stats[s] / 300 for s in STATS]
        values += [t in mon.get("types", []) for t in k.type_names]
        values += [mon.get("status", "") == s for s in STATUSES]
        volatiles = set(mon.get("volatiles", {})) | set(mon.get("volatile_ids", []))
        values += [v in volatiles for v in VOLATILES] + pp
        return values

    def attacks(self, mon):
        return move_ids(mon) or self.priors.get(mon.get("base_species", mon.get("species")), [])

    def best_damage(self, a, b, weather=""):
        return max((self.k.damage(a, b, m, weather) for m in self.attacks(a)), default=0.)

    def hazard_cost(self, mon, conditions):
        types = mon.get("types", [])
        cost = .125 * self.k.effectiveness("Rock", types) if "stealthrock" in conditions else 0.
        if "Flying" not in types and mon.get("ability") != "levitate":
            layers = conditions.get("spikes", {}).get("layers", 0)
            cost += [0., .125, 1/6, .25][min(3, layers)]
        return 0. if mon.get("ability") == "magicguard" or mon.get("item") == "heavydutyboots" else cost

    def action_features(self, obs, action, matchup=None):
        k = self.k
        values = dict.fromkeys(self.action_names, 0.)
        kind = action["kind"]
        values[kind] = 1.
        active = next((p for p in obs["own"] if p["active"]), obs["own"][0])
        target = next((p for p in obs["opponent"] if p["active"]), obs["opponent"][0])
        actor = obs["own"][action["slot"]] if kind == "switch" else active
        values.update(actor_hp=actor.get("hp", 1), target_hp=target["hp"],
                      target_item_known=target.get("item") is not None,
                      target_ability_known=target.get("ability") is not None,
                      force_replacement=obs["phase"] == "switch")
        values["speed_ratio"] = min(3., k.mon_stats(actor)["spe"] / max(1, k.mon_stats(target)["spe"])) / 3
        if kind == "move":
            move = k.moves.get(action["move"], {})
            values.update(mega=action["mega"], power=move.get("basePower", 0) / 150,
                          accuracy=move.get("accuracy", 100)/100, priority=move.get("priority", 0)/6,
                          physical=move.get("category") == "Physical", special=move.get("category") == "Special",
                          status=move.get("category") == "Status")
            values["pp"] = next((m["pp"]/max(1,m["maxpp"]) for m in actor.get("moves", []) if isinstance(m, dict) and m["id"] == action["move"]), 1.)
            damage = k.damage(actor, target, action["move"], obs["weather"], action["mega"])
            values.update(estimated_damage=min(2., damage), ko_margin=max(-1., min(1., damage-target["hp"])),
                          effectiveness=k.effectiveness(move.get("type"), target["types"])/4,
                          stab=move.get("type") in actor["types"])
            for attr in ("heal", "drain", "recoil"):
                fraction = move.get(attr)
                values[attr] = fraction[0]/fraction[1] if fraction else 0.
            boost = move.get("self", {}).get("boosts", {}) if move.get("self") else {}
            if move.get("target") == "self":
                boost = move.get("boosts") or boost
            for s in ("atk", "spa", "spe", "def", "spd"):
                values[f"boost_{s}"] = boost.get(s, 0)/3
            values.update(setup=bool(boost), protect=action["move"] in ("protect", "detect", "banefulbunker", "kingsshield", "spikyshield", "burningbulwark"),
                          inflict_status=bool(move.get("status")), hazard=bool(move.get("sideCondition")),
                          pivot=bool(move.get("selfSwitch")), force_switch=move.get("forceSwitch", False),
                          contact=move.get("flags", {}).get("contact", False), sound=move.get("flags", {}).get("sound", False))
            values[f"type_{move.get('type')}"] = 1.
        if kind == "switch":
            values["revive"] = bool(action.get("revive"))
            values["incoming_damage"] = min(2., self.best_damage(target, actor, obs["weather"]))
            values["switch_offense"] = min(2., self.best_damage(actor, target, obs["weather"]))
            values["hazard_cost"] = self.hazard_cost(actor, obs["conditions"][0])
        if kind == "preview":
            # All 120 permutations are legal. Scores depend only on visible species
            # and our known sets; an opponent's moves here are usage priors.
            matrix = matchup[list(action["picks"])]
            ordered = np.sort(matrix, axis=0)
            values.update(coverage_mean=float(ordered[-1].mean()), coverage_worst=float(ordered[-1].min()),
                          coverage_backup=float(ordered[-2].mean()), lead_matchup=float(matrix[0].mean()),
                          team_hp=sum(obs["own"][i]["hp"] for i in action["picks"])/3)
        return [float(values[n]) for n in self.action_names]

    def encode(self, obs):
        mons = obs["own"] + obs["opponent"]
        if len(mons) != 12:
            raise ValueError("Expected six public preview species per team")
        result = {"pokemon": np.asarray([self.mon_features(m, i < 6) for i, m in enumerate(mons)], dtype=np.float32),
                  "species": np.array([self.k.token("species", m["species"]) for m in mons], dtype=np.int64),
                  "items": np.array([self.k.token("items", m.get("item")) for m in mons], dtype=np.int64),
                  "abilities": np.array([self.k.token("abilities", m.get("ability")) for m in mons], dtype=np.int64),
                  "moves": np.asarray([(list(map(lambda x:self.k.token("moves", x), move_ids(m))) + [0]*4)[:4] for m in mons], dtype=np.int64)}
        field = [obs["phase"] == "preview", obs["phase"] == "switch", min(obs["turn"],200)/200,
                 *obs["mega"], "trickroom" in obs["pseudo"], "gravity" in obs["pseudo"]]
        field += [obs["weather"] == w for w in WEATHER] + [obs["terrain"] == t for t in TERRAIN]
        field += [obs["conditions"][s].get(c, {}).get("layers", 0)/3 for s in range(2) for c in CONDITIONS]
        field += [sum(m["fainted"] for m in team)/3 for team in (obs["own"], obs["opponent"])]
        result["field"] = np.array(field, dtype=np.float32)
        result["history_moves"] = np.zeros(12, dtype=np.int64)
        result["history"] = np.zeros((12, 3), dtype=np.float32)
        for i,e in enumerate(obs["history"][-12:]):
            result["history_moves"][i] = self.k.token("moves", e.get("move"))
            result["history"][i] = [e["side"]==0, e["side"]==1, min(20,obs["turn"]-e["turn"])/20]
        result["actions"] = np.zeros((N_ACTIONS,len(self.action_names)),dtype=np.float32)
        result["action_moves"] = np.zeros(N_ACTIONS,dtype=np.int64)
        result["mask"] = np.zeros(N_ACTIONS,dtype=bool)
        matchup = None
        if obs["phase"] == "preview":
            matchup = np.array([[np.clip(self.best_damage(a,b)-self.best_damage(b,a),-2,2) for b in obs["opponent"]] for a in obs["own"]])
            # Vectorize the 120 combinations. Everything except team coverage
            # is shared at preview; avoid redoing static stats 120 times.
            legal=obs["legal"]
            indices=np.asarray([a["id"] for a in legal])
            picks=np.asarray([a["picks"] for a in legal])
            result["mask"][indices]=True
            shared=np.asarray(self.action_features(obs,legal[0],matchup),dtype=np.float32)
            result["actions"][indices]=shared
            result["action_moves"][indices]=self.k.token("moves",None)
            matrix=matchup[picks]
            ordered=np.sort(matrix,axis=1)
            columns={"coverage_mean":ordered[:,-1].mean(1),"coverage_worst":ordered[:,-1].min(1),
                     "coverage_backup":ordered[:,-2].mean(1),"lead_matchup":matrix[:,0].mean(1),
                     "team_hp":np.asarray([p["hp"] for p in obs["own"]])[picks].sum(1)/3}
            for name,values in columns.items():
                result["actions"][indices,self.action_names.index(name)]=values
            return result
        for action in obs["legal"]:
            index = action["id"]
            result["mask"][index] = True
            result["actions"][index] = self.action_features(obs,action,matchup)
            result["action_moves"][index] = self.k.token("moves", action.get("move"))
        if not result["mask"].any() and not (obs["ended"] or obs["truncated"] or obs["phase"] == "wait"):
            raise ValueError("Nonterminal decision has no legal actions")
        return result

    def schema(self):
        return {"version": FEATURE_VERSION, "knowledge_sha256": self.k.sha256,
                "pokemon": self.mon_names, "actions": self.action_names, "field": self.field_names,
                "history": ["ours", "opponent", "turn_age_div_20"],
                "action_ids": {"0-3": "move slots", "4-7": "Mega + move", "8-13": "switch to original roster slot",
                               "14-133": "ordered 3-of-6 team preview permutations"},
                "unknown_id": 0, "known_none_id": 1,
                "estimates": "Damage features are approximations; unknown opponent sets use priors/base stats, never engine secrets."}
