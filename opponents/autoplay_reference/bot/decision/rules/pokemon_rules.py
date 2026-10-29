"""BE-19: Zero-form Palafin takes a reliable KO first, otherwise prefers a working exit."""
from opponents.autoplay_reference.bot.decision.rules import move_rules

PARAMETERS = {'move': 'flipturn', 'preference': 'ko-then-exit', 'minimum_exit_successes': 1}


def eligibility(result):
    before, options = result['before'], result['options']
    own = before['sides'][0]
    actor = own['active']
    mon = next(p for p in own['pokemon'] if p['identity'] == actor)
    data = {'eligible': False, 'actor': actor, 'reason': 'not-zero-palafin', 'options': [], 'candidates': {}}
    if before.get('request') != 'move' or mon.get('species') != 'Palafin' or mon['hp'] <= 0:
        return data
    if any(form != 'Palafin' for form in result.get('active_forms', [mon['species']])):
        return {**data, 'reason': 'mixed-active-forms'}
    if not any(move_rules.is_pivot(a) and a['name'] == 'flipturn' for a in options.values()):
        return {**data, 'reason': 'flip-turn-unavailable'}
    if not any(p['identity'] != actor and p['hp'] > 0 for p in own['pokemon']):
        return {**data, 'reason': 'no-living-teammate'}
    return {**data, 'eligible': True, 'reason': None}


def exit_facts(samples, actor, indices=None):
    indices = range(len(samples)) if indices is None else indices
    endings = [next(p for p in samples[i]['after']['sides'][0]['pokemon'] if p['identity'] == actor) for i in indices]
    successes = sum(p['hp'] > 0 and p.get('species') == 'Palafin-Hero' for p in endings)
    return {'successes': successes, 'total': len(endings),
            'probability': successes / len(endings) if endings else None, 'viable': successes > 0}


def palafin_exits(result, entries):
    data = eligibility(result)
    if not data['eligible']:
        return data
    for key, samples in entries.items():
        action = result['options'][key]
        if action['kind'] != 'switch' and not (move_rules.is_pivot(action) and action['name'] == 'flipturn'):
            continue
        facts = exit_facts(samples, data['actor'], result.get('trial_indices'))
        data['candidates'][key] = facts
        if facts['viable']:
            data['options'].append(key)
    data['reason'] = None if data['options'] else 'no-working-exit'
    return data


def receiver_choices(result, key, paths):
    """Filter working receiver paths before ranking; never hide a death in an averaged path."""
    data = eligibility(result)
    choices = list(paths)
    if data['eligible'] and result['options'][key].get('name') == 'flipturn':
        working = [name for name, path in paths.items()
                   if exit_facts(path['entries'], data['actor'], result.get('trial_indices'))['viable']]
        return working or choices
    return choices
