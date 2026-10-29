"""BE-24: deterministic tactical recovery independent of prediction and Showdown."""
import hashlib
import json
from opponents.autoplay_reference.bot.domain import catalog, moves

POLICY = 'confirmed-tactical-v1'


def fingerprint(ctx, battle, kind):
    ignored = {'hud_reading', 'game_clock', 'send_tries', 'previous_order_context', 'previous_damage_evidence',
               'availability_verified_at'}
    def relevant(value):
        if isinstance(value,dict):
            return {k:relevant(v) for k,v in value.items() if k not in
                    ('evidence','captures','source','reason','event_id','timestamp','prompt')}
        return [relevant(v) for v in value] if isinstance(value,(tuple,list)) else value
    facts = {name: relevant(fact.snapshot())
             for name, fact in battle['sim'].beliefs.items()}
    observed = {k:v for k,v in ctx.items() if k not in ignored}
    if 'confirmed_moves' in observed:
        observed['confirmed_moves'] = [{k:m.get(k) for k in ('slot','name','pp','verified')}
                                       for m in observed['confirmed_moves']]
    value = [kind, battle.get('prompt_id'), observed, facts, relevant(vars(battle['sim'].field))]
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def choose(ctx, roster, available, *, enemy_moves=(), enemy_form=None):
    """Returns an action/reasons only. No live-state mutation, I/O or simulated confidence."""
    c = catalog.get()
    me = ctx.get('me') or ctx.get('outgoing')
    info = c.species(enemy_form or ctx.get('opp')) or {}
    foe_types = ctx.get('opp_types_override') or info.get('types', [])
    if isinstance(foe_types, dict):
        foe_types = foe_types.get(ctx.get('opp'), info.get('types', []))
    profiles = {}
    for action in available:
        name = action.species if action.kind in ('switch', 'send') else me
        if name not in roster.members:
            continue
        flags = ctx.get('flags', {})
        profiles[action.id] = roster.view(name,
            mega=action.mega or bool(flags.get('mega_used') and flags.get('mega_species') == name),
            hero=bool(flags.get('hero') and name == 'Palafin'))

    def attack_value(action):
        meta, member = c.move(action.name) or {}, profiles.get(action.id, {})
        species = c.species(member.get('species')) or {}
        own_types = ctx.get('types_override', {}).get('me', {}).get(me) or species.get('types', [])
        accuracy = 1 if meta.get('accuracy') is True else (meta.get('accuracy') or 0) / 100
        stage = ctx.get('my_boosts', {}).get('atk' if meta.get('category') == 'Physical' else 'spa', 0)
        boost = (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
        copied = ctx.get('vol',{}).get('me',{}).get('transformed')
        stats = copied.get('stats',{}) if copied else member.get('stats',{})
        stat = stats.get('atk' if meta.get('category') == 'Physical' else 'spa', 1)
        return (meta.get('bp', 0) * accuracy * stat * boost
                * (1.5 if meta.get('type') in own_types else 1)
                * c.effectiveness(meta['type'], foe_types)) if meta.get('type') else 0

    rows = []
    attacks = [a for a in available if a.kind == 'move' and (c.move(a.name) or {}).get('category') != 'Status']
    for a in attacks:
        rows.append({'id': a.id, 'kind': 'move', 'utility': attack_value(a)})
    positive = [a for a in attacks if attack_value(a) > 0]
    if positive:
        best = max(positive, key=lambda a: (attack_value(a), -(a.slot or 99)))
        return best, {'policy': POLICY, 'rule': 'BE-24', 'reason': 'best known damaging move', 'candidates': rows}

    entries = [a for a in available if a.kind in ('send', 'switch')]
    if entries:
        types = [(c.move(m) or {}).get('type') for m in enemy_moves
                 if (c.move(m) or {}).get('category') != 'Status']
        types = [t for t in types if t] or foe_types
        def entry_value(a):
            member = profiles.get(a.id, {})
            own_types = (c.species(member.get('species')) or {}).get('types', [])
            weakness = max((c.effectiveness(t, own_types) for t in types), default=1)
            bench = next((b for b in ctx.get('bench', []) if b.get('species') == a.species), {})
            hp = bench.get('hp_pct')
            if hp is None:
                hp = 100 * (bench.get('hp') or 0) / max(1, member.get('stats', {}).get('hp', 1))
            return -weakness, hp, -(a.row or 99)
        best = max(entries, key=entry_value)
        rows.extend({'id': a.id, 'kind': a.kind, 'rank': entry_value(a)} for a in entries)
        return best, {'policy': POLICY, 'rule': 'BE-24', 'reason': 'known attack resistance then living HP', 'candidates': rows}

    status = [a for a in available if a.kind == 'move' and a not in attacks]
    if status:
        best = min(status, key=lambda a: (moves.setup(a.name), a.slot or 99))
    else:
        best = min(attacks, key=lambda a: a.slot or 99) if attacks else None
    return best, {'policy': POLICY, 'rule': 'BE-24', 'reason': 'only remaining confirmed action', 'candidates': rows,
                  'forced_setup_exception': bool(best and moves.offensive_boosts(best.name))}
