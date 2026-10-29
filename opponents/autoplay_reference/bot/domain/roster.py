"""Validated, visible starting loadouts. Battle state never writes through this object."""
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

from opponents.autoplay_reference.bot.domain import catalog
from opponents.autoplay_reference.bot.runtime import config


def team_path():
    return Path(config.DATA) / "team.json"


def identity(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_member(member):
    c = catalog.get()
    errors = []
    info = c.species(member.get("species"))
    if not info:
        return ["unknown species/form"]
    if member.get("nature") not in catalog.NATURES:
        errors.append("unknown nature")
    if member.get("ability") not in info["possibleAbilities"]:
        errors.append("unknown/invalid ability")
    sp = member.get("sp", {})
    if set(sp) != set(catalog.STATS) or any(type(v) is not int or not 0 <= v <= 32 for v in sp.values()) or sum(sp.values()) > 66:
        errors.append("invalid stat points")
    else:
        expected = c.stats(member)
        stats = member.get("stats")
        if stats is None and len(member.get("shown", [])) == 6:
            stats = dict(zip(catalog.STATS, member["shown"]))
        if stats != expected:
            errors.append(f"displayed stats inconsistent: {stats}; expected {expected}")
    item = member.get("item")
    if item is None or (item != "" and c.display("items", item) != item):
        errors.append("unverified/unknown equipped item")
    moves = member.get("moves", [])
    if len(moves) != 4 or len({catalog.canonical(m.get('name')) for m in moves if isinstance(m, dict)}) != 4:
        errors.append("four distinct moves required")
    for move in moves:
        if not isinstance(move, dict):
            errors.append("move must have a name and PP")
            continue
        meta = c.move(move.get("name"))
        if not meta or type(move.get("pp")) is not int or move["pp"] != meta["max_pp"]:
            errors.append(f"unknown move or inconsistent maximum PP: {move}")
    return errors


class Roster:
    def __init__(self, data, expected_members=6):
        self.data = copy.deepcopy(data)
        members = self.data.get("team", [])
        errors = []
        if len(members) != expected_members or len({m.get('species') for m in members}) != len(members):
            errors.append(f"expected {expected_members} distinct team members")
        for member in members:
            errors.extend(f"{member.get('species', '?')}: {e}" for e in validate_member(member))
        if errors:
            raise ValueError("Invalid saved roster: " + "; ".join(errors))
        for m in members:
            m["stats"] = m.get("stats") or dict(zip(catalog.STATS, m["shown"]))
        self.members = {m["species"]: m for m in members}
        self.identity = identity(self.data)

    @classmethod
    def load(cls, path=None):
        return cls(json.loads(Path(path or team_path()).read_text(encoding="utf-8")))

    def view(self, name, *, mega=False, hero=False):
        m = copy.deepcopy(self.members[name])
        form = None
        if mega:
            form = catalog.get().stones.get(name, {}).get(m.get("item"))
        if hero and m["ability"] == "Zero to Hero":
            form = name + "-Hero"
        if form:
            m["species"] = form
            m["stats"] = catalog.get().stats(m)
            m["ability"] = catalog.get().species(form)["possibleAbilities"][0]
        m["types"] = catalog.get().species(m["species"])["types"]
        return m

    def reminder(self, path=None):
        return ("Verify that your saved team matches the game. Last Box read: "
                f"{self.data.get('last_box_read_at', 'unknown')}. After team or item changes, "
                f"run `python -m bot readbox` and check `{Path(path or team_path()).resolve()}`.")


def atomic_save(data, path=None, backup=True):
    path = Path(path or team_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    live_roster = path.resolve() == team_path().resolve()
    if backup and path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = Path(config.runtime_path("roster_backups")) if live_roster else path.parent
        backup_dir.mkdir(parents=True, exist_ok=True)
        (backup_dir / f"team.backup_{stamp}.json").write_bytes(path.read_bytes())
    temporary_dir = Path(config.runtime_path("tmp")) if live_roster else path.parent
    temporary_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=temporary_dir, prefix=".team-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def correct_items(roster, corrections, path=None):
    """Verified preview observations; update disk and the active roster before search."""
    target = Path(path or team_path())
    data = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else copy.deepcopy(roster.data)
    for m in data["team"]:
        if m["species"] in corrections:
            m["item"] = corrections[m["species"]]
    checked = Roster(data)
    if checked.identity != roster.identity:
        atomic_save(data, path)
        roster.__dict__.update(checked.__dict__)
    return roster
