"""Intersect screen-established choices with the hydrated simulator request."""
from opponents.autoplay_reference.bot.domain import catalog


def intersect(available, legal):
    accepted, rejected = [], []
    for action in available:
        match = next((row for row in legal if
            action.kind == "move" and row["kind"] == "move" and
            catalog.canonical(action.name) == row["name"] and action.mega == row["mega"] or
            action.kind in ("send", "switch") and row["kind"] == "switch" and action.species == row["identity"]), None)
        if match:
            accepted.append({**match, "id": action.id})
        else:
            rejected.append({"action": action.to_dict(), "reason": "not legal in hydrated Showdown state"})
    return accepted, rejected
