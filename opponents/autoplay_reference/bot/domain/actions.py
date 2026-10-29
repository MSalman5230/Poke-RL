"""Screen-addressable actions and the only failure policy: established availability."""
from dataclasses import asdict, dataclass
from itertools import permutations

from opponents.autoplay_reference.bot.domain import catalog
from opponents.autoplay_reference.bot.state_support import mechanics
from opponents.autoplay_reference.bot.decision.rules import mega_rules


@dataclass(frozen=True)
class Action:
    kind: str
    slot: int | None = None
    mega: bool = False
    species: str | None = None
    row: int | None = None
    picks: tuple[str, ...] = ()
    rows: tuple[int, ...] = ()
    name: str | None = None

    @property
    def id(self):
        if self.kind == "move":
            return f"fight_{self.slot}" + ("_mega" if self.mega else "")
        if self.kind in ("switch", "send"):
            return f"{'send' if self.kind == 'send' else 'switch'}_{self.row}"
        return "preview_" + "_".join(map(str, self.rows))

    def to_dict(self):
        return {"id": self.id, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        return cls(**{key: tuple(data) if key in ("picks", "rows") else data
                      for key, data in value.items() if key in cls.__dataclass_fields__})


class Decision(dict):
    def __init__(self, action, available, source, diagnostics):
        self.action, self.available, self.source, self.diagnostics = action, tuple(available), source, diagnostics
        super().__init__(action=action.to_dict() if action else None,
                         available_actions=[a.to_dict() for a in available], diagnostics=diagnostics,
                         choice=action.id if action else None, source=source,
                         options=[a.to_dict() for a in available], search=diagnostics,
                         latency=diagnostics.get("timings", {}).get("total", 0))
        if action and action.kind == "preview":
            self.update(picks=list(action.picks), rows=list(action.rows), lead=action.picks[0])


def current_item(ctx, roster, name):
    observed = ctx.get("item_state", {}).get("me", {}).get(name, {})
    if observed.get("certainty") == "unknown":
        return None
    if observed.get("certainty") in ("held", "empty"):
        return observed["item"]
    return "" if name in ctx.get("consumed", []) else roster.members.get(name, {}).get("item")


def trapped(ctx, roster):
    me = ctx.get("me") or ctx.get("outgoing")
    if me not in roster.members:
        return False
    m = roster.members[me]
    types = ctx.get("types_override", {}).get("me", {}).get(me) or catalog.get().species(me)["types"]
    item = current_item(ctx, roster, me)
    vol = ctx.get("vol", {}).get("me", {})
    return bool(ctx.get("trapped") or vol.get("partiallytrapped") or
                {"noretreat", "trapped", "ingrain"}.intersection(vol.get("effects", {}))) and "Ghost" not in types and item != "Shed Shell"


def available(ctx, roster, kind="turn"):
    if kind == "preview":
        eligible = [(r["species"], r["row"]) for r in ctx.get("ours", []) if r.get("eligible", True)
                    and r.get("species") in roster.members and type(r.get("row")) is int and 1 <= r["row"] <= 6]
        if len(set(s for s, _ in eligible)) != len(eligible) or len(set(r for _, r in eligible)) != len(eligible):
            return []
        prefix=tuple(ctx.get('selected_prefix',()))
        return [Action("preview", picks=tuple(s for s, _ in order), rows=tuple(r for _, r in order))
                for order in permutations(sorted(eligible, key=lambda x: x[1]), 3)
                if tuple(s for s,_ in order[:len(prefix)]) == prefix]
    out = []
    me = ctx.get("me") or ctx.get("outgoing")
    confirmed = {m['slot']:m for m in ctx.get('confirmed_moves', []) if m.get('verified') and m.get('pp', 0) > 0}
    if kind == "turn" and me in roster.members and (confirmed or
            isinstance(ctx.get("my_hp"), (int, float)) and ctx["my_hp"] > 0):
        m = roster.members[me]
        v = ctx.get("vol", {}).get("me", {})
        lock = (v.get("charging") or v.get("encore") or v.get("rampage") or {}).get("move")
        item = current_item(ctx, roster, me)
        if (item in mechanics.CHOICE_ITEMS or item is None) and not v.get("choice_lock_reset"):
            lock = lock or v.get("choice_lock") or v.get("last_move")
        disabled = {catalog.canonical(x) for x in ctx.get("disabled_moves", [])}
        if v.get("disabled"):
            disabled.add(catalog.canonical(v["disabled"]["move"]))
        slots = [(slot, row) for slot,row in sorted(confirmed.items())] if confirmed else list(enumerate(m['moves'], 1))
        for slot, move in slots:
            if v.get("transformed") and not confirmed:
                continue  # Copied move slots have not been read from a supported own-move menu.
            pp = move['pp'] if confirmed else ctx.get("pp_left", {}).get(move["name"])
            meta = catalog.get().move(move["name"])
            if ctx.get("field_effects", {}).get("gravity") and meta.get("flags", {}).get("gravity"):
                continue
            if pp is None or (pp <= 0 and not v.get("charging")) or catalog.canonical(move["name"]) in disabled:
                continue
            if lock and catalog.canonical(lock) != catalog.canonical(move["name"]):
                continue
            if v.get("taunt") and meta["category"] == "Status":
                continue
            if "torment" in v.get("effects", {}) and move["name"] == v.get("last_move"):
                continue
            out.append(Action("move", slot=slot, name=move["name"]))
            if not ctx.get("flags", {}).get("mega_used") and item in catalog.get().stones.get(me, {}):
                out.append(Action("move", slot=slot, name=move["name"], mega=True))
    if kind == "send" or not trapped(ctx, roster):
        for row in sorted(ctx.get("bench", []), key=lambda x: x.get("row", 99)):
            if (row.get("species") in roster.members and row["species"] != me and not row.get("fainted")
                    and isinstance(row.get("hp"), (int, float)) and row["hp"] > 0 and type(row.get("row")) is int and 1 <= row["row"] <= 3):
                out.append(Action("send" if kind == "send" else "switch", species=row["species"], row=row["row"]))
    return mega_rules.force(out)
