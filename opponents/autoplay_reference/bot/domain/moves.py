"""Move facts from the pinned catalog; no battle effects are applied here."""
from opponents.autoplay_reference.bot.domain import catalog


def offensive_boosts(name, types=()):
    """Potential self Atk/SpA increases, including callback-based moves in singles."""
    move = catalog.get().move(name) or {}
    if move.get("category") != "Status":
        return ()
    key = catalog.canonical(name)
    if key == "curse":
        return () if "Ghost" in types else ("atk",)
    if key == "bellydrum":
        return ("atk",)
    if key == "acupressure":
        return ("atk", "spa")
    changes = move.get("selfBoosts") or (move.get("boosts") if move.get("target") in ("self", "allies") else {}) or {}
    return tuple(stat for stat in ("atk", "spa") if changes.get(stat, 0) > 0)


def damaging(name):
    return (catalog.get().move(name) or {}).get("category") in ("Physical", "Special")


def setup(name, types=()):
    """Known self-stat setup, including defensive boosts and callback-based offensive moves."""
    move = catalog.get().move(name) or {}
    if move.get('category') != 'Status':
        return False
    changes = move.get('selfBoosts') or (move.get('boosts') if move.get('target') in ('self','allies') else {}) or {}
    return bool(offensive_boosts(name, types) or any(v > 0 for v in changes.values())
                or catalog.canonical(name) == 'stockpile')
