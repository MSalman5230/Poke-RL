"""Translate player-visible engine observations into the frozen user's policy.

No actual opposing set, selected trio, private HP, or chosen action is supplied.
The original prediction, action evaluation, 32-trial rollouts and replacement
rules remain in opponents/autoplay_reference; failures are counted explicitly.
"""
import copy
import time
from .engine import ROOT, reference_environment
from .knowledge import canonical


class ReferenceBot:
    name = "autoplay_reference"

    def __init__(self, seed=0, budget=25, strict=True):
        from .integrity import verify_reference
        self.snapshot_sha256=verify_reference()
        reference_environment()
        from opponents.autoplay_reference.bot.domain import catalog
        self.c = catalog.get()
        self.seed = seed
        self.budget = budget
        self.strict = strict
        self.decisions = 0
        self.fallbacks = 0
        self.errors = []
        self.last_diagnostics = {}
        self.audit = []

    def _roster(self, obs):
        from opponents.autoplay_reference.bot.domain.roster import Roster
        members = []
        for p in obs["own"]:
            s = p["set"]
            member = {"species": s["species"], "nature": s["nature"], "sp": s["evs"],
                      "ability": s["ability"], "item": s["item"],
                      "moves": [{"name": self.c.display("moves", m), "pp": self.c.move(m)["max_pp"]} for m in s["moves"]]}
            member["stats"] = self.c.stats(member)
            members.append(member)
        return Roster({"team": members})

    def _state(self, obs, roster):
        from opponents.autoplay_reference.bot.runtime.context import BattleContext
        context = BattleContext(roster=roster)
        ours = [p for p in obs["own"] if p["selected"]]
        me = next(p for p in ours if p["active"])
        foe = next(p for p in obs["opponent"] if p["active"])
        names = [p["base_species"] for p in ours]
        flags = {"mega_used": obs["mega"][0], "mega_species": next((p["base_species"] for p in ours if "-Mega" in p["species"]), None),
                 "hero": any(p["species"] == "Palafin-Hero" for p in ours), "fresh": me.get("entry_turn") == obs["turn"]}
        for p in obs["opponent"]:
            belief = context.belief(p["base_species"])
            for mid in p["moves"]:
                belief.reveal_move(mid)
            if p["item"] is not None:
                belief.reveal_item(p["item"])
            if p["ability"] is not None:
                belief.reveal_ability(p["ability"])
            if p["species"] != p["base_species"]:
                belief.set_form(p["species"])
            belief.moves_this_stint = [self.c.display("moves", mid) for mid in p.get("moves_this_stint",[])]
            belief.pp_used = {self.c.display("moves",mid):cost for mid,cost in p.get("pp_used",{}).items()}
            first=p.get("first_native_move")
            if first and first["without_mega"] and belief.stone_form and "-Mega" not in p["species"]:
                context.observe_mega_prompt(p["base_species"],first["turn"])
                context.note_mega_opportunity(p["base_species"],first["turn"])
                context.finish_mega_opportunities()
            belief.hero_ready = p["species"] == "Palafin-Hero"
            belief.disguise_busted = p["species"] == "Mimikyu-Busted"
            if "-Mega" in p["species"]:
                context.opp_mega_used(p["base_species"])
        field = context.field
        field.weather = {"sunnyday": "Sun", "raindance": "Rain", "sandstorm": "Sand", "snow": "Snow",
                         "desolateland": "Harsh Sunshine", "primordialsea": "Heavy Rain"}.get(obs["weather"])
        field.weather_turns = 5 if field.weather else 0
        field.terrain = obs["terrain"].removesuffix("terrain").title() or None
        field.terrain_turns = 5 if field.terrain else 0
        field.trick_room = max(1, 5 - (obs["turn"] - obs["pseudo"].get("trickroom", {}).get("turn", obs["turn"]))) if "trickroom" in obs["pseudo"] else 0
        field.unrepresented = {k: {"remaining": max(1, 5 - (obs["turn"] - v["turn"]))} for k, v in obs["pseudo"].items() if k != "trickroom"}
        aliases = {"stealthrock": "sr", "toxicspikes": "tspikes", "stickyweb": "web", "lightscreen": "light_screen", "auroraveil": "aurora_veil"}
        for side, conditions in zip(("me", "opp"), obs["conditions"]):
            for effect, row in conditions.items():
                field.sides[side][aliases.get(effect, effect)] = row["layers"] if effect in ("stealthrock", "spikes", "toxicspikes", "stickyweb") else max(1, 5 - obs["turn"] + row["turn"])
        def vol(p):
            effects = {key: {"elapsed": max(0, obs["turn"] - row["turn"]),
                             **({"move": row["move"]} if row.get("move") else {})}
                       for key, row in p.get("volatiles", {}).items() if not row.get("single") or row["turn"] >= obs["turn"] - 1}
            result = {"effects": effects, "times_attacked": p.get("times_attacked", 0),
                      "last_move": self.c.display("moves", p.get("last_move")), "entry_round": p.get("entry_turn", 0)}
            for name in ("substitute", "leechseed", "confusion", "partiallytrapped"):
                if name in effects:
                    result[name] = effects.pop(name) or True
            return result
        status = {"me": {p["base_species"]: p["status"] for p in ours},
                  "opp": {p["base_species"]: p["status"] for p in obs["opponent"] if p["seen"]}}
        item_state = {"me": {p["base_species"]: {"certainty": "held" if p["item"] else "empty", "item": self.c.display("items", p["item"])} for p in ours}}
        battle = {"sim": context, "turn": obs["turn"], "picks": names, "flags": flags,
                  "opp": [p["base_species"] for p in obs["opponent"]],
                  "opp_seen": [p["base_species"] for p in obs["opponent"] if p["seen"]],
                  "hp": {p["base_species"]: p["current_hp"] for p in ours},
                  "opp_hp": {p["base_species"]: p["hp"] * 100 for p in obs["opponent"] if p["seen"]},
                  "my_fainted": [p["base_species"] for p in ours if p["fainted"]],
                  "opp_fainted": [p["base_species"] for p in obs["opponent"] if p["fainted"]],
                  "vol": {"me": vol(me), "opp": vol(foe)}, "status": status, "item_state": item_state,
                  "_effect_round": obs["turn"], "active": me["base_species"], "opp_active": foe["base_species"]}
        battle["faint_history"] = {side: [{"species": e["species"], "round": e["turn"]} for e in events]
                                   for side, events in zip(("me", "opp"), obs["faints"])}
        battle["pp_used"]={p["base_species"]:{self.c.display("moves",m["id"]):m["maxpp"]-m["pp"] for m in p["moves"]} for p in ours}
        battle["status_progress"]={side:{p["base_species"]:dict(p.get("status_progress",{})) for p in mons}
                                   for side,mons in (("me",ours),("opp",obs["opponent"]))}
        ctx = {"me": me["base_species"], "opp": foe["base_species"], "my_hp": me["current_hp"], "opp_hp": foe["hp"] * 100,
               "flags": flags, "status": status, "item_state": item_state, "vol": battle["vol"],
               "my_boosts": me["boosts"], "opp_boosts": foe["boosts"], "opp_fainted": battle["opp_fainted"],
               "pp_left": {self.c.display("moves", m["id"]): m["pp"] for m in me["moves"]},
               "confirmed_moves": [{"slot": a["slot"] + 1, "name": self.c.display("moves", a["move"]),
                                    "pp": next((m["pp"] for m in me["moves"] if m["id"] == a["move"]), 1), "verified": True}
                                   for a in obs["legal"] if a["kind"] == "move" and not a["mega"]],
               "bench": [{"species": p["base_species"], "hp": p["current_hp"], "fainted": p["fainted"], "row": j + 1}
                         for j, p in enumerate(ours) if not p["active"]],
               "trapped": obs["phase"] == "move" and not any(a["kind"] == "switch" for a in obs["legal"]),
               "replacement_phase": "faint" if me["fainted"] else "pivot",
               "types_override": {"me": {p["base_species"]: p["types"] for p in ours},
                                  "opp": {p["base_species"]: p["types"] for p in obs["opponent"] if p["seen"]}}}
        battle.update({"types_override": ctx["types_override"]})
        return ctx, battle

    def choose(self, obs):
        if len(obs["legal"]) == 1:
            self.last_diagnostics = {"source": "forced_legal_action", "errors": []}
            return obs["legal"][0]["id"]
        from opponents.autoplay_reference.bot.decision import greedy
        roster = self._roster(obs)
        if obs["phase"] == "preview":
            from opponents.autoplay_reference.bot.team_selection.prepare import prepare
            # The live bot prepares this cache before matchmaking. Preserve that
            # strength rather than benchmarking its missing-cache fallback.
            prepare(roster)
            decision = greedy.decide_preview([p["base_species"] for p in obs["own"]],
                [{"species": p["base_species"]} for p in obs["opponent"]], None, roster, budget=max(60, self.budget))
        else:
            ctx, battle = self._state(obs, roster)
            decision = greedy.decide(ctx, battle, "send" if obs["phase"] == "switch" else "turn",
                                     budget=self.budget, seed=self.seed + self.decisions)
        self.decisions += 1
        d = decision.diagnostics
        self.last_diagnostics = {"source": decision.source, "errors": d.get("errors", []),
            "seconds": d.get("timings", {}).get("total"), "selected": d.get("selected_action"),
            "preview_policy": d.get("team_selection", {}).get("policy"),
            "preview_details": d.get("team_selection", {}).get("details", {})}
        self.audit.append({"turn":obs["turn"],"source":decision.source,"errors":d.get("errors",[]),
                           "trials":len(d.get("rollout_seeds",[])),"deadline_passed":d.get("deadline_passed",False),
                           "preview_policy":self.last_diagnostics["preview_policy"]})
        if self.strict and (d.get("errors") or d.get("deadline_passed")):
            self.errors.extend(d.get("errors",[]))
            raise RuntimeError(f"Reference evaluation incomplete: {self.last_diagnostics}")
        if obs["phase"] == "preview" and self.strict and self.last_diagnostics["preview_policy"] != "damage-speed-coverage-v1":
            raise RuntimeError(f"Reference preview cache incomplete: {self.last_diagnostics}")
        if decision.source in ("heuristic_recovery", "emergency_scored", "unavailable"):
            self.fallbacks += 1
            self.errors.extend(d.get("errors", []))
            if self.strict:
                raise RuntimeError(f"Reference policy did not finish its full evaluation: {self.last_diagnostics}")
        action = decision.action
        if action:
            for candidate in obs["legal"]:
                if action.kind == "preview" and candidate["kind"] == "preview":
                    if tuple(obs["own"][i]["base_species"] for i in candidate["picks"]) == action.picks:
                        return candidate["id"]
                elif action.kind == "move" and candidate["kind"] == "move" and candidate["slot"] + 1 == action.slot and candidate["mega"] == action.mega:
                    return candidate["id"]
                elif action.kind in ("switch", "send") and candidate["kind"] == "switch" and obs["own"][candidate["slot"]]["base_species"] == action.species:
                    return candidate["id"]
        raise RuntimeError(f"Reference action failed programmatic legality mapping: {self.last_diagnostics}")
