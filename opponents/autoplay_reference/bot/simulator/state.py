"""Authoritative observations + explicit build hypotheses -> Showdown hydration input."""
import copy
from opponents.autoplay_reference.bot.domain import actions, catalog
from opponents.autoplay_reference.bot.state_support import tracker

WEATHER = {None: "", "Sun": "sunnyday", "Rain": "raindance", "Sand": "sandstorm", "Snow": "snow",
           "Harsh Sunshine": "desolateland", "Heavy Rain": "primordialsea"}
CONDITIONS = {"sr": "stealthrock", "tspikes": "toxicspikes", "web": "stickyweb",
              "light_screen": "lightscreen", "aurora_veil": "auroraveil", "lucky_chant": "luckychant",
              "healing_wish": "healingwish", "lunar_dance": "lunardance", "crafty_shield": "craftyshield",
              "mat_block": "matblock", "quick_guard": "quickguard", "wide_guard": "wideguard"}


def validate_observed(ctx, battle):
    if ctx.get('hud_reading') and any(ctx.get(k) is None for k in ('my_hp','opp_hp','opp')):
        raise ValueError('Active HUD incomplete; confirmed-action tactical recovery required')
    if battle.get("required_item_recheck"):
        raise ValueError(f"Required own item recheck: {battle['required_item_recheck']}")
    if ctx.get("vol", battle.get("vol", {})).get("me", {}).get("transformed"):
        raise ValueError("Transform: copied stats and move slots are not fully observed")


def _pokemon(name, member, hp, status, progress, types, pp):
    c = catalog.get()
    if hp is None or not 0 <= hp <= member["stats"]["hp"]:
        raise ValueError(f"Required HP unknown/invalid: {name} ({hp})")
    moves = [{"id": catalog.canonical(m["name"]), "pp": int(pp.get(m["name"], m["pp"]))} for m in member["moves"]]
    return {"identity": name, "set": {"name": name, "species": member["species"], "ability": member["ability"],
                                     "item": member["item"], "nature": member["nature"], "evs": member["sp"],
                                     "level": 50, "moves": [m["name"] for m in member["moves"]]},
            "hp": int(hp), "maxhp": member["stats"]["hp"], "stats": member["stats"],
            "status": status or "", "progress": progress or {}, "types": types or c.species(member["species"])["types"],
            "ability": member["ability"], "base_ability": member.get("original_ability") or member["ability"],
            "item": member["item"], "moves": moves}


def build(ctx, battle, kind, predictions):
    context = battle["sim"]
    c, roster = catalog.get(), context.roster
    validate_observed(ctx, battle)
    notes = []
    for side, rows in battle.get("state_uncertainties", {}).items():
        for row in rows:
            notes.append({"rule": "BE-22", "fact": "unread observed event", "side": side,
                          **copy.deepcopy(row), "assumed": "last confirmed values; no inferred change"})
    flags = ctx.get("flags", battle.get("flags", {}))
    active = ctx.get("me") or ctx.get("outgoing") or battle.get("active")
    own_names = list(battle.get("picks") or [])
    if not own_names or active not in own_names:
        raise ValueError("Active identity or selected party is unknown")
    own_names = [active, *(n for n in own_names if n != active)]
    foe_active = ctx.get("opp") or battle.get("opp_active")
    if foe_active not in predictions:
        raise ValueError("Active opponent build is unavailable")
    foe_names = [foe_active, *(n for n in predictions if n != foe_active)]
    if len(own_names) > 3 or len(foe_names) > 3:
        raise ValueError("Ranked singles party exceeds three members")
    mega_species = flags.get("mega_species")
    if flags.get("mega_used") and not mega_species:
        possible = [n for n in own_names if roster.members[n].get("item") in c.stones.get(n, {})]
        if len(possible) != 1:
            raise ValueError("Observed own Mega identity is unresolved")
        mega_species = possible[0]
    sides = []
    members = {}
    for side, names in (("me", own_names), ("opp", foe_names)):
        mons = []
        vol = copy.deepcopy(ctx.get("vol", battle.get("vol", {})).get(side, {}))
        for i, name in enumerate(names):
            if side == "me":
                member = roster.view(name, mega=bool(flags.get("mega_used") and name == mega_species), hero=bool(flags.get("hero")))
                item = actions.current_item({**battle, **ctx}, roster, name)
                if item is None:
                    raise ValueError(f"Required own item unknown: {name}")
                member["item"] = item
                hp = ctx.get("my_hp") if name == active else None
                if hp is None:
                    hp = battle.get("hp", {}).get(name)
                pp = tracker.remaining_pp(battle, name, member['moves'])
                if i == 0:
                    pp.update(ctx.get("pp_left", {}))
                    pp.update({r['name']: r['pp'] for r in ctx.get('confirmed_moves', []) if r.get('verified')})
            else:
                pred = predictions[name]
                if not pred.complete:
                    raise ValueError(f"Incomplete opponent build {name}: {pred.errors}")
                member = copy.deepcopy(pred.member)
                fact = context.belief(name)
                if fact.disguise_busted and name.startswith("Mimikyu"):
                    member["species"] = "Mimikyu-Busted"
                if name in battle.get("opp_item_gone", []):
                    member["item"] = ""
                pct = ctx.get("opp_hp") if i == 0 else battle.get("opp_hp", {}).get(name)
                if pct is None and name in ctx.get("opp_fainted", battle.get("opp_fainted", [])):
                    pct = 0
                if pct is None:
                    if i == 0:
                        raise ValueError("Required active opponent HP unknown")
                    pct = 100
                    notes.append({"fact": "unobserved opponent HP", "species": name, "assumed": "full"})
                member["stats"] = c.stats(member)
                hp = round(member["stats"]["hp"] * pct / 100)
                if pct > 0:
                    hp = max(1, hp)
                pp = {}
            if i == 0 and vol.get("ability_observed") and vol.get("ability"):
                member.setdefault("original_ability", member["ability"])
                member["ability"] = vol["ability"]
            if i == 0 and vol.get("type_ability_used"):
                member["ability"] = vol["type_ability_used"]
            status = ctx.get("status", battle.get("status", {})).get(side, {}).get(name)
            progress = ctx.get("status_progress", battle.get("status_progress", {})).get(side, {}).get(name, {})
            types = ctx.get("types_override", battle.get("types_override", {})).get(side, {}).get(name)
            mon = _pokemon(name, member, hp, status, progress, types, pp)
            if i == 0 and vol.get("transformed"):
                transformed = copy.deepcopy(vol["transformed"])
                if vol.get('ability_observed') and vol.get('ability'):
                    transformed['ability'] = vol['ability']
                required = ("form", "stats", "types", "moves", "ability")
                if any(not transformed.get(k) for k in required):
                    raise ValueError("Observed Transform copy is incomplete")
                mon["transformed"] = transformed
            hits = vol.get("times_attacked", 0) if i == 0 else 0
            if hits is None:
                if hp > 0 and any(catalog.canonical(m["name"]) == "ragefist" for m in mon.get('transformed', {}).get('moves', member["moves"])):
                    raise ValueError(f"Unresolved Rage Fist hit history: {side} {name}")
                notes.append({"fact": "received-hit count", "species": name, "observed": None,
                              "assumed": 0, "reason": "no current Rage Fist move"})
                hits = 0
            if type(hits) is not int or hits < 0:
                raise ValueError(f"Invalid received-hit count: {side} {name}")
            mon["times_attacked"] = hits
            mon["mega_evolved"] = "-Mega" in member["species"]
            mon["hero_ready"] = side == "opp" and context.belief(name).hero_ready or side == "me" and bool(flags.get("hero"))
            mons.append(mon)
            members[(side, name)] = {**member, "item": catalog.canonical(member["item"])}
        effects = [{**vol.get("effects", {}).get(e["id"], {}), **e}
                   for e in tracker.volatiles({**battle, "vol": {side: vol}}, side, battle.get("turn", 1))]
        if vol.get("disabled"):
            effects.append({"id": "disable", **vol["disabled"]})
        last_move = vol.get("last_move")
        if side == "opp" and not last_move and context.belief(names[0]).moves_this_stint:
            last_move = context.belief(names[0]).moves_this_stint[-1]
        locked_move = vol.get('choice_lock') or last_move
        if mons[0]["item"] in ("Choice Band", "Choice Specs", "Choice Scarf") and locked_move and not vol.get("choice_lock_reset"):
            effects.append({"id": "choicelock", "move": locked_move})
        if vol.get("type_ability_used"):
            mons[0]["type_ability_used"] = vol["type_ability_used"]
        if mons[0]["ability"] == "Unburden" and names[0] in battle.get("item_loss_stint", {}).get(side, []):
            effects.append({"id": "unburden"})
        if mons[0]["ability"] == "Slow Start" and not any(e["id"] == "slowstart" for e in effects):
            remaining = 5 - (battle.get("_effect_round", 0) - vol.get("entry_round", 0))
            if remaining > 0:
                effects.append({"id": "slowstart", "remaining": remaining})
        if names[0] in battle.get("drowsy", {}).get(side, []):
            effects.append({"id": "yawn", "remaining": 1})
        boosts = ctx.get("my_boosts" if side == "me" else "opp_boosts", battle.get("boosts", {}).get(side, {}))
        fainted = battle.get("faint_history", {}).get(side, [])
        round_number = battle.get("_effect_round", 0)
        previous_faint = next((r["species"] for r in reversed(fainted) if r["round"] == round_number - 1), None)
        current_faint = next((r["species"] for r in reversed(fainted) if r["round"] == round_number), None)
        if kind == "send" and ctx.get("replacement_phase", "faint" if mons[0]["hp"] == 0 else "pivot") == "faint":
            # Showdown advances the turn after this replacement. The tracker has already closed residuals.
            current_faint, previous_faint = previous_faint or current_faint, None
        sides.append({"id": "p1" if side == "me" else "p2", "names": names, "pokemon": mons,
                      "boosts": copy.deepcopy(boosts), "volatiles": effects, "observation": vol,
                      "last_move": last_move, "mega_used": bool(flags.get("mega_used")) if side == "me" else bool(context.mega_used_by_opp),
                      "fresh": bool(flags.get("fresh")) if side == "me" else
                               vol.get("entry_round", 0) == battle.get("_effect_round", 0),
                      "replacement": copy.deepcopy(ctx.get("replacement", battle.get("replacement", {})).get(side, {})),
                      "total_fainted": max(len(fainted), len(battle.get("my_fainted" if side == "me" else "opp_fainted", []))),
                      "fainted_last_turn": previous_faint, "fainted_this_turn": current_faint})
    field = context.field
    weather_time = field.predicted_duration("weather", field.weather_turns, members, notes)
    terrain_time = field.predicted_duration("terrain", field.terrain_turns, members, notes)
    for index, side in enumerate(("me", "opp")):
        conditions = {}
        for key, value in field.sides[side].items():
            if key in ("reflect", "light_screen", "aurora_veil"):
                value = field.predicted_duration(f"{side}.{key}", value, members, notes)
            if value:
                conditions[CONDITIONS.get(key, key)] = int(value)
        sides[index]["conditions"] = conditions
    root = {"schema_version": 2, "format": "gen9championsbssregmc", "kind": kind,
            "turn": max(1, battle.get("turn", 1)), "sides": sides,
            "field": {"weather": WEATHER[field.weather] if weather_time else "", "weather_turns": weather_time,
                      "terrain": catalog.canonical(field.terrain) + "terrain" if field.terrain and terrain_time else "",
                      "terrain_turns": terrain_time, "trickroom": field.trick_room,
                      "extra": copy.deepcopy(field.unrepresented)},
            "wishes": copy.deepcopy(battle.get("wishes", {})),
            "future_moves": copy.deepcopy(battle.get("future_moves", {})),
            "replacement_phase": ctx.get("replacement_phase", "faint" if sides[0]["pokemon"][0]["hp"] == 0 else "pivot"),
            "pending_opponent_move": ctx.get("pending_opponent_move")}
    return root, notes
