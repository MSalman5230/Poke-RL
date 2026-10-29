"""BE-16/20: charging attacks stay shortlisted; pivots use fixed receiver plans."""
from opponents.autoplay_reference.bot.domain import catalog

PIVOTS = frozenset(('flipturn', 'uturn', 'voltswitch'))


def is_pivot(action):
    return action['kind'] == 'move' and action.get('name') in PIVOTS


def is_charging_attack(action):
    """Pinned move metadata identifies two-turn attacks without a separate move list."""
    if action['kind'] != 'move':
        return False
    move = catalog.get().move(action.get('name')) or {}
    return move.get('category') in ('Physical', 'Special') and bool(move.get('flags', {}).get('charge'))


def receivers(before, pauses, party_rows):
    """Only actual pivot requests establish which living members can receive the move."""
    names = {a['identity'] for row in pauses if row for p in row['pending'] if p['side'] == 0
             for a in p['options']}
    order = {p['identity']: i for i, p in enumerate(before['sides'][0]['pokemon'])}
    return sorted(names, key=lambda name: party_rows.get(name, order[name]))


def replacement(input_state):
    """An observed pivot cause, rather than a guessed move list, enables next-turn entry scoring."""
    phase = input_state['sides'][0]['replacement']
    return input_state['replacement_phase'] == 'pivot' and phase.get('kind') == 'pivot' and \
        catalog.canonical(phase.get('move')) in PIVOTS


def pending(row, action, actor):
    """Match a real self-switch prompt, excluding faints, forced departures and entry chains."""
    if not is_pivot(action) or row['after']['sides'][0]['active'] != actor:
        return False
    mon = row.get('root', {}).get('sides', [{}])[0].get('pokemon', [{}])[0]
    return mon.get('hp', 0) > 0 and mon.get('switchFlag') == action['name']
