"""Generation/format-specific facts and explicitly approximate tactical features."""
import hashlib
import json
from functools import lru_cache
import re
from .engine import ROOT

STATS = ("hp", "atk", "def", "spa", "spd", "spe")


@lru_cache(maxsize=16384)
def canonical(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower().replace("é", "e"))


class Knowledge:
    def __init__(self, path=None):
        content = (path or ROOT / "data/knowledge.json").read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        self.data = json.loads(content)
        for k in ("species", "moves", "items", "abilities", "types", "natures"):
            setattr(self, k, self.data[k])
        # 0 is UNKNOWN, 1 is known NONE; every lookup is checkpoint-stable.
        self.vocab = {k: {name: i + 2 for i, name in enumerate(sorted(getattr(self, k)))}
                      for k in ("species", "moves", "items", "abilities")}
        self.type_names = sorted(self.types)

    def token(self, kind, value):
        return 0 if value is None else 1 if value == "" else self.vocab[kind].get(canonical(value), 0)

    def species_info(self, name):
        return self.species.get(canonical(name), {})

    def effectiveness(self, move_type, defender_types):
        result = 1.0
        for typ in defender_types:
            result *= {0: 1., 1: 2., 2: .5, 3: 0.}.get(self.types.get(typ, {}).get(move_type, 0), 1.)
        return result

    def stats(self, species, evs=None, nature="Serious"):
        evs = evs or {}
        return dict(zip(STATS,self._stat_values(canonical(species),nature,tuple(evs.get(s,0) for s in STATS))))

    @lru_cache(maxsize=32768)
    def _stat_values(self, species, nature, points):
        base = self.species_info(species).get("baseStats", dict.fromkeys(STATS, 80))
        stats = {s: base[s] + p + (75 if s == "hp" else 20) for s,p in zip(STATS,points)}
        n = self.natures.get(nature, {})
        for s in STATS[1:]:
            if n.get("plus") == s:
                stats[s] = stats[s] * 110 // 100
            elif n.get("minus") == s:
                stats[s] = stats[s] * 90 // 100
        return tuple(stats[s] for s in STATS)

    def mon_stats(self, mon):
        if "max_hp" in mon and all(s in mon.get("stats",{}) for s in STATS[1:]):
            return {**mon["stats"],"hp":mon["max_hp"]}
        base = self.stats(mon.get("species", ""), mon.get("set", {}).get("evs"), mon.get("set", {}).get("nature", "Serious"))
        return {**base, **mon.get("stats", {}), "hp": mon.get("max_hp", base["hp"])}

    def damage(self, attacker, defender, move_id, weather="", mega=False):
        """Fast approximate fraction of defender HP, not an exact engine damage call.

        Unknown opponent stats use uninvested Champions stats. The policy receives
        uncertainty flags. The reference bot uses its original exact rollouts.
        """
        move = self.moves.get(canonical(move_id), {})
        if move.get("category") == "Status":
            return 0.
        power = move.get("basePower", 0)
        if not power:
            power = {"seismictoss": 70, "nightshade": 70, "lowkick": 80, "grassknot": 80,
                     "heavyslam": 80, "heatcrash": 80, "gyroball": 80, "electroball": 70}.get(canonical(move_id), 0)
        if not power:
            return 0.
        a = self.mon_stats(attacker)
        d = self.mon_stats(defender)
        types = attacker.get("types", self.species_info(attacker.get("species")).get("types", []))
        ability = attacker.get("ability")
        if mega:
            stone = self.items.get(attacker.get("item"), {}).get("megaStone")
            if isinstance(stone, dict):
                stone = stone.get(attacker.get("base_species"))
            if stone:
                s = self.species_info(stone)
                a = self.stats(stone, attacker.get("set", {}).get("evs"), attacker.get("set", {}).get("nature", "Serious"))
                types = s.get("types", types)
                ability = canonical(s.get("abilities", {}).get("0"))
        physical = move.get("category") == "Physical"
        atk, defense = ("atk", "def") if physical else ("spa", "spd")
        def stage(mon, stat):
            v = mon.get("boosts", {}).get(stat, 0)
            return (2 + v) / 2 if v >= 0 else 2 / (2 - v)
        attack = a[atk] * stage(attacker, atk)
        guard = d[defense] * stage(defender, defense)
        if defender.get("ability") == "unaware":
            attack = a[atk]
        if ability == "unaware":
            guard = d[defense]
        if physical and ability in ("hugepower", "purepower"):
            attack *= 2
        if ability == "technician" and power <= 60:
            power *= 1.5
        typ = move.get("type")
        if typ == "Normal" and ability in ("pixilate", "aerilate", "refrigerate", "galvanize"):
            typ = {"pixilate": "Fairy", "aerilate": "Flying", "refrigerate": "Ice", "galvanize": "Electric"}[ability]
            power *= 1.2
        eff = self.effectiveness(typ, defender.get("types", []))
        other_ability = defender.get("ability")
        if (typ == "Ground" and (other_ability == "levitate" or defender.get("item") == "airballoon") or
            typ == "Water" and other_ability in ("waterabsorb", "stormdrain", "dryskin") or
            typ == "Electric" and other_ability in ("voltabsorb", "lightningrod", "motordrive") or
            typ == "Fire" and other_ability == "flashfire" or typ == "Grass" and other_ability == "sapsipper") and ability not in ("moldbreaker", "teravolt", "turboblaze"):
            eff = 0
        stab = (2 if ability == "adaptability" else 1.5) if typ in types else 1
        if attacker.get("status") == "brn" and physical and ability != "guts":
            attack *= .5
        if ability == "guts" and attacker.get("status") and physical:
            attack *= 1.5
        if attacker.get("item") == ("choiceband" if physical else "choicespecs"):
            attack *= 1.5
        modifier = 1.3 if attacker.get("item") == "lifeorb" else 1.
        if weather in ("sunnyday", "desolateland"):
            modifier *= 1.5 if typ == "Fire" else .5 if typ == "Water" else 1
        if weather in ("raindance", "primordialsea"):
            modifier *= 1.5 if typ == "Water" else .5 if typ == "Fire" else 1
        if defender.get("hp", 1) == 1 and other_ability in ("multiscale", "shadowshield"):
            modifier *= .5
        return max(0., ((22 * power * attack / max(1, guard)) / 50 + 2) * stab * eff * .925 * modifier / d["hp"])


@lru_cache(maxsize=1)
def knowledge():
    return Knowledge()
