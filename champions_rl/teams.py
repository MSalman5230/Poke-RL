"""Validated sets and diverse, seeded six-member teams with disjoint manifests."""
import copy
import hashlib
import json
import random
from .engine import ROOT, Engine
from .knowledge import knowledge, canonical, STATS


def fingerprint(team):
    if isinstance(team,dict):
        team=[team]
    normalized=[]
    for mon in team:
        row={k:v for k,v in mon.items() if k!='name'}
        row['moves']=sorted(row['moves'],key=canonical)
        normalized.append(row)
    normalized.sort(key=lambda s:canonical(s['species']))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_sets():
    from .curated import complete_sets,repair_synthetic
    k = knowledge()
    source = ROOT / "opponents/autoplay_reference/data"
    usage = json.loads((source / "smogon/2026-09/chaos/gen9championsbssregmc-1500.json").read_text(encoding="utf-8"))["data"]
    rows = {canonical(n): r for n, r in usage.items()}
    candidates = []
    for sid, mon in k.species.items():
        if mon["battleOnly"] or not mon["learnset"]:
            continue
        row = rows.get(sid, {})
        ranked = lambda key: [a for a, v in sorted(row.get(key, {}).items(), key=lambda x: -x[1]) if v > 0]
        ability_ids = [a for a in ranked("Abilities") if a in map(canonical, mon["abilities"].values())]
        ability_ids += [canonical(a) for a in mon["abilities"].values() if canonical(a) not in ability_ids]
        item_ids = [a for a in ranked("Items") if a in k.items]
        stones = [iid for iid, i in k.items.items() if i.get("megaEvolves") == mon["name"] or
                  isinstance(i.get("megaStone"), dict) and mon["name"] in i["megaStone"]]
        item_ids = list(dict.fromkeys([*item_ids[:3], *stones, "sitrusberry", "leftovers", "choicescarf", "focussash", ""]))
        move_ids = [m for m in ranked("Moves") if m in mon["learnset"]]
        def fallback_score(mid):
            m = k.moves[mid]
            return (m["basePower"] * (1.5 if m["type"] in mon["types"] else 1) * m["accuracy"] / 100
                    + (25 if m["category"] == "Status" and (m.get("heal") or m.get("boosts")) else 0))
        move_ids += [m for m in sorted(mon["learnset"], key=fallback_score, reverse=True) if m not in move_ids]
        spreads = ranked("Spreads")[:6]
        primary = "atk" if mon["baseStats"]["atk"] >= mon["baseStats"]["spa"] else "spa"
        default_evs = dict.fromkeys(STATS, 0)
        default_evs.update({primary: 32, "spe": 32, "hp": 2})
        default_nature = "Jolly" if primary == "atk" else "Timid"
        for variant in range(8):
            evs, nature = dict(default_evs), default_nature
            if spreads:
                name, points = spreads[variant % len(spreads)].split(":")
                values = list(map(int, points.split("/")))
                if len(values) == 6 and sum(values) <= 66 and max(values) <= 32:
                    evs, nature = dict(zip(STATS, values)), name
            picked = list(move_ids[:4])
            if variant > 0 and len(move_ids) > 4:
                picked[3] = move_ids[4 + ((variant - 1) % min(4, len(move_ids)-4))]
            if not any(k.moves[m]["category"] != "Status" for m in picked):
                attack = next((m for m in move_ids if k.moves[m]["category"] != "Status"), None)
                if attack:
                    picked[-1] = attack
            item = item_ids[variant % len(item_ids)]
            set_ = {"name": mon["name"], "species": mon["name"], "level": 50,
                    "ability": k.abilities.get(ability_ids[variant % len(ability_ids)], {}).get("name", mon["abilities"]["0"]),
                    "item": k.items.get(item, {}).get("name", ""), "nature": nature,
                    "evs": evs, "moves": [k.moves[m]["name"] for m in picked]}
            candidates.append({"set": repair_synthetic(set_), "usage": row.get("usage", 0), "variant": variant,
                               "source": "2026-09 usage marginals + legal fallbacks; synthetic, not human joint sets"})
    curated=complete_sets()
    with Engine() as engine:
        for row,errors in zip(curated,engine.request('validate',sets=[r['set'] for r in curated])):
            if errors:raise ValueError(f"Curated {row['set']['species']}: {errors}")
        candidates=curated+candidates
        checked = engine.request("validate", sets=[c["set"] for c in candidates])
        valid, rejected = [], []
        seen = set()
        for candidate, errors in zip(candidates, checked):
            if errors:
                rejected.append({"species": candidate["set"]["species"], "variant": candidate["variant"], "errors": errors})
            else:
                sha = fingerprint(candidate["set"])
                if sha not in seen:
                    candidate["id"] = sha
                    valid.append(candidate)
                    seen.add(sha)
        original = json.loads((source / "team.json").read_text(encoding="utf-8"))["team"]
        reference = [{"name": m["species"], "species": m["species"], "level": 50, "item": m["item"],
                      "ability": m["ability"], "nature": m["nature"], "evs": m["sp"],
                      "moves": [x["name"] for x in m["moves"]]} for m in original]
        errors = engine.request("validate", teams=[reference])[0]
        if errors:
            raise ValueError(f"Original user's team invalid: {errors}")
    covered = {canonical(x["set"]["species"]) for x in valid}
    report = {"format": k.data["format"], "knowledge_sha256": k.sha256, "sets": valid,
              "reference_team": reference, "rejected": rejected,
              "uncovered": [s["name"] for s in k.species.values() if not s["battleOnly"] and s["id"] not in covered]}
    (ROOT / "data/sets.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"sets": len(valid), "species": len(covered), "uncovered": report["uncovered"], "rejected": len(rejected)}


class TeamPool:
    def __init__(self):
        self.data = json.loads((ROOT / "data/sets.json").read_text(encoding="utf-8"))
        self.rows = self.data["sets"]
        self.by_species = {}
        self.curated={}
        for row in self.rows:
            self.by_species.setdefault(row["set"]["species"], []).append(row)
            if row.get('curated'):self.curated.setdefault(row['set']['species'],[]).append(row)

    def sample(self, rng, mode="mixed"):
        k = knowledge()
        from .curated import ARCHETYPES
        if self.curated and (mode in ARCHETYPES or mode=='weather' or mode=='mixed' and rng.random()<.65):
            archetype=mode if mode in ARCHETYPES else rng.choice(list(ARCHETYPES)[:4] if mode=='weather' else list(ARCHETYPES))
            team=[copy.deepcopy(rng.choice(self.curated[name])['set']) for name in ARCHETYPES[archetype]]
            rng.shuffle(team)
            return team
        # Mix competitive usage with uniform roster coverage and team archetypes.
        if mode == "mixed":
            mode = rng.choice(["meta", "meta", "coverage", "bulky"])
        names = list(self.by_species)
        weights = []
        for n in names:
            s = k.species_info(n)
            usage = max(r["usage"] for r in self.by_species[n])
            weight = (usage + .00005) ** .6 if mode == "meta" else 1.
            if mode == "offense":
                weight *= (max(s["baseStats"]["atk"], s["baseStats"]["spa"]) / 80) ** 3
            if mode == "bulky":
                weight *= ((s["baseStats"]["hp"] + s["baseStats"]["def"] + s["baseStats"]["spd"]) / 240) ** 4
            weights.append(weight)
        team, numbers, items = [], set(), set()
        if mode == "weather":
            leader = rng.choice([n for n in ("Pelipper", "Torkoal", "Tyranitar", "Ninetales-Alola") if n in self.by_species])
            names = [leader, *[n for n in names if n != leader]]
            weights = [1.] * len(names)
        for slot in range(6):
            allowed = [i for i, name in enumerate(names) if k.species_info(name)["num"] not in numbers]
            index = 0 if mode == "weather" and slot == 0 else rng.choices(allowed, weights=[weights[i] for i in allowed])[0]
            name = names[index]
            options = [r for r in self.by_species[name] if not r["set"]["item"] or r["set"]["item"] not in items]
            chosen = copy.deepcopy(rng.choice(options or self.by_species[name])["set"])
            if chosen["item"] in items:
                chosen["item"] = ""  # Empty slots are legal and do not violate Item Clause.
            numbers.add(k.species_info(name)["num"])
            if chosen["item"]:
                items.add(chosen["item"])
            team.append(chosen)
        return team

    def manifest(self, count=64, seed=1731, split="train", forbidden=()):
        rng = random.Random(seed)
        teams, seen = [], set(forbidden)
        while len(teams) < count:
            team = self.sample(rng)
            sha = fingerprint(team)
            if sha not in seen:
                teams.append({"id": sha, "team": team})
                seen.add(sha)
        with Engine() as engine:
            for errors in engine.request("validate", teams=[t["team"] for t in teams]):
                if errors:
                    raise ValueError(errors)
        result = {"split": split, "seed": seed, "knowledge_sha256": knowledge().sha256, "teams": teams}
        (ROOT / f"data/teams_{split}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
