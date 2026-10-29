def remaining_pp(b, species, moves):
    used = b.get('pp_used', {}).get(species, {})
    result = {m['name']: max(0, m['pp'] - used.get(m['name'], 0)) for m in moves}
    for name, reading in b.get('pp_observed', {}).get(species, {}).items():
        result[name] = max(0, reading['pp'] - max(0, used.get(name, 0) - reading['used']))
    return result

def volatiles(b, side, turn):
    """Observed volatile effects for that side's active Pokemon at the start of `turn`."""
    v = (b.get('vol') or {}).get(side) or {}
    out = []
    trap = v.get('partiallytrapped')
    if trap:
        out.append({'id': 'partiallytrapped', **trap})
    if v.get('substitute'):
        out.append({'id': 'substitute'})
    if v.get('leechseed'):
        out.append({'id': 'leechseed'})
    if v.get('confusion'):
        c = v['confusion'] if isinstance(v['confusion'], dict) else {}
        out.append({'id': 'confusion', 'elapsed': c.get('elapsed', 0), 'source': c.get('source')})
    tracked = v.get('effects', {})
    if v.get('taunt') and 'taunt' not in tracked and (turn - v['taunt'] <= 3):
        out.append({'id': 'taunt', 'elapsed': max(0, min(2, turn - 1 - v['taunt']))})
    enc = v.get('encore')
    if enc and enc.get('move') and ('encore' not in tracked) and (turn - enc['turn'] <= 3):
        out.append({'id': 'encore', 'move': enc['move'], 'elapsed': max(0, min(2, turn - 1 - enc['turn']))})
    if v.get('destinybond'):
        out.append({'id': 'destinybond'})
    if v.get('protect_turn') == turn - 1:
        out.append({'id': 'stall'})
    rp = v.get('rampage')
    if rp and rp['count'] < 3:
        out.append({'id': 'lockedmove', 'move': rp['move'], 'elapsed': rp['count']})
    present = {entry['id'] for entry in out}
    out.extend(({'id': key, **record} for key, record in tracked.items() if key not in present))
    return out
