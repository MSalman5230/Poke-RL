"""Reader metadata exported from the pinned Showdown Champions Dex."""
import hashlib
import json
from pathlib import Path
import re
from opponents.autoplay_reference.bot.runtime import config

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
NATURES = ["Adamant", "Bashful", "Bold", "Brave", "Calm", "Careful", "Docile", "Gentle", "Hardy",
           "Hasty", "Impish", "Jolly", "Lax", "Lonely", "Mild", "Modest", "Naive", "Naughty",
           "Quiet", "Quirky", "Rash", "Relaxed", "Sassy", "Serious", "Timid"]


def canonical(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower().replace("é", "e"))


class Catalog:
    def __init__(self, path=None):
        path = Path(path or Path(config.DATA) / "catalog.json")
        if not path.is_file():
            raise RuntimeError("Showdown catalog missing; run python tools/setup_showdown.py")
        content = path.read_bytes()
        self.data = json.loads(content)
        self.identity = {"source": self.data["source"], "sha256": hashlib.sha256(content).hexdigest()}
        self.names = json.loads((Path(config.DATA) / "display_names.json").read_text(encoding="utf-8"))
        for kind in ("species", "moves", "items", "abilities"):
            self.names.setdefault(kind, {}).update({key: value["name"] for key, value in self.data[kind].items()})
        self.stones = {}
        for value in self.data["items"].values():
            stone = value.get("megaStone")
            if isinstance(stone, dict):
                for base, form in stone.items():
                    self.stones.setdefault(base, {})[value["name"]] = form
            elif value.get("megaEvolves") and stone:
                self.stones.setdefault(value["megaEvolves"], {})[value["name"]] = stone

    def display(self, kind, value):
        key = canonical(value)
        if kind == "items" and key in ("", "none", "nothing", "empty"):
            return ""
        return self.names[kind].get(key)

    def all_species(self):
        return sorted(s["name"] for s in self.data["species"].values())

    def all_moves(self):
        return sorted(self.names["moves"].values())

    def all_items(self):
        return sorted(self.names["items"].values())

    def species(self, name):
        return self.data["species"].get(canonical(name))

    def move(self, name):
        return self.data["moves"].get(canonical(name))

    def legal_move(self, species, move):
        info = self.species(species)
        return bool(info and canonical(move) in info["learnset"])

    def stats(self, member):
        info = self.species(member["species"])
        if info is None:
            raise ValueError(f"Unknown species: {member['species']}")
        nature = self.data["natures"][member["nature"].title()]
        # Champions uses SP directly, without IVs. Parity-tested against statModify.
        stats = {k: info["base"][k] + member.get("sp", {}).get(k, 0) + (75 if k == "hp" else 20) for k in STATS}
        for key in STATS[1:]:
            if nature["plus"] == key:
                stats[key] = stats[key] * 110 // 100
            elif nature["minus"] == key:
                stats[key] = stats[key] * 90 // 100
        if canonical(member["species"]) == "shedinja":
            stats["hp"] = 1
        return stats

    def effectiveness(self, attack_type, defender_types):
        value = 1.0
        for typ in defender_types:
            code = self.data["types"][typ].get(attack_type, 0)
            value *= {0: 1, 1: 2, 2: .5, 3: 0}[code]
        return value

    def resolve_form(self, name, preview_types=(), preview=()):
        if not name:
            return name
        exact = self.display("species", name) or name
        options = [s for s in preview if s and (canonical(s).startswith(canonical(exact)) or canonical(exact).startswith(canonical(s)))]
        return exact if exact in options or len(options) != 1 else options[0]


_catalog = None


def get():
    global _catalog
    if _catalog is None:
        _catalog = Catalog()
    return _catalog
