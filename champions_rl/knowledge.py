"""Generation/format-specific facts and explicitly approximate tactical features."""
import hashlib
import json
from functools import lru_cache
import re
from .engine import ROOT

STATS = ("hp", "atk", "def", "spa", "spd", "spe")


@lru_cache(maxsize=1)
def usage_data():
    path=ROOT/'opponents/autoplay_reference/data/smogon/2026-09/chaos/gen9championsbssregmc-1500.json'
    return {canonical(name):row for name,row in json.loads(path.read_text(encoding='utf-8'))['data'].items()}


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
        if 'set' not in mon:
            options=self.spread_prior(mon.get('species',''))
            base={s:sum(w*self.stats(mon.get('species',''),evs,nature)[s] for nature,evs,w in options) for s in STATS}
        else:
            base = self.stats(mon.get("species", ""), mon.get("set", {}).get("evs"), mon.get("set", {}).get("nature", "Serious"))
        return {**base, **mon.get("stats", {}), "hp": mon.get("max_hp", base["hp"])}

    def usage(self,species):
        return usage_data().get(canonical(species),usage_data().get(canonical(self.species_info(species).get('baseSpecies')),{}))

    def prior(self,species,kind,limit=4):
        row=self.usage(species).get(kind,{})
        entries=sorted(((canonical(n),float(v)) for n,v in row.items() if v>0),key=lambda p:-p[1])[:limit]
        if not entries and kind=='Abilities':
            entries=[(canonical(a),1.) for a in self.species_info(species).get('abilities',{}).values()]
        total=sum(v for _,v in entries)
        return [(n,v/total) for n,v in entries] if total else []

    @lru_cache(maxsize=1024)
    def spread_prior(self,species):
        rows=sorted(self.usage(species).get('Spreads',{}).items(),key=lambda x:-x[1])[:6]
        result=[]
        for spread,weight in rows:
            nature,points=spread.split(':')
            values=list(map(int,points.split('/')))
            if len(values)==6 and min(values)>=0 and max(values)<=32 and sum(values)<=66 and weight>0:
                result.append((nature,dict(zip(STATS,values)),float(weight)))
        if not result:
            base=self.species_info(species).get('baseStats',{})
            stat='atk' if base.get('atk',0)>=base.get('spa',0) else 'spa'
            result=[('Jolly' if stat=='atk' else 'Timid',{'hp':2,stat:32,'spe':32},1.)]
        total=sum(w for _,_,w in result)
        return [(n,e,w/total) for n,e,w in result]

    def effective_speed(self,mon,weather='',conditions=None):
        speed=self.mon_stats(mon)['spe']
        stage=mon.get('boosts',{}).get('spe',0)
        speed=int(speed*((2+stage)/2 if stage>=0 else 2/(2-stage)))
        ability=mon.get('ability')
        item=mon.get('item')
        multiplier=1.
        if item=='choicescarf': multiplier*=1.5
        if item in ('ironball','machobrace'): multiplier*=.5
        if 'tailwind' in (conditions or {}): multiplier*=2
        if (ability=='swiftswim' and weather in ('raindance','primordialsea') or
            ability=='chlorophyll' and weather in ('sunnyday','desolateland') or
            ability=='sandrush' and weather=='sandstorm' or ability=='slushrush' and weather in ('snow','hail')):
            multiplier*=2
        if ability=='quickfeet' and mon.get('status'): multiplier*=1.5
        if ability=='unburden' and 'unburden' in mon.get('volatiles',{}): multiplier*=2
        speed=int(speed*multiplier)
        if mon.get('status')=='par' and ability!='quickfeet': speed//=2
        return max(1,min(10000,speed))

    def damage(self, attacker, defender, move_id, weather="", mega=False):
        """Fast approximate fraction of defender HP, not an exact engine damage call.

        Unknown stats use a usage spread/nature prior. Native tactical probes
        supplement this approximation; it is not an exact damage calculator.
        """
        move = self.moves.get(canonical(move_id), {})
        if move.get("category") == "Status":
            return 0.
        mid=canonical(move_id)
        if mid in ('seismictoss','nightshade','dragonrage','sonicboom','superfang','naturesmadness','ruination'):
            if not self.effectiveness(move.get('type'),defender.get('types',[])): return 0.
            fixed=50 if mid in ('seismictoss','nightshade') else 40 if mid=='dragonrage' else 20 if mid=='sonicboom' else self.mon_stats(defender)['hp']*defender.get('hp',1)/2
            return fixed/self.mon_stats(defender)['hp']
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
        if mid=='bodypress': atk='def'
        attack = (d['atk']*stage(defender,'atk') if mid=='foulplay' else a[atk]*stage(attacker,atk))
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
        hits=move.get('multihit') or 1
        if isinstance(hits,list):
            hits=hits[-1] if ability=='skilllink' else 4.5 if attacker.get('item')=='loadeddice' else 3. if hits==[2,5] else sum(hits)/2
        return max(0., ((22 * power * attack / max(1, guard)) / 50 + 2) * stab * eff * .925 * modifier * hits / d["hp"])


@lru_cache(maxsize=1)
def knowledge():
    return Knowledge()
