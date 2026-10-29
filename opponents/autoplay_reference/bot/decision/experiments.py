"""Shared bounded mechanics batches and Python-owned pending replacement choices."""
from opponents.autoplay_reference.bot.decision.rules import move_rules
from opponents.autoplay_reference.bot.team_selection import replacements


def ranks_for(row, prompt, ranks):
    side = prompt['side']
    ordinary = ranks.get('opp' if side else 'me', {})
    if not ranks.get('by_forms'):
        return ordinary
    summary = row['after']['sides']
    roots = row.get('root', {}).get('sides', [])
    def form(index, identity):
        mon = next((p for p in summary[index]['pokemon'] if p['identity'] == identity), {})
        root = next((p for p in roots[index]['pokemon'] if p.get('m', {}).get('identity') == identity), {}) if roots else {}
        current = mon.get('species', identity)
        potential = root.get('canMegaEvo')
        return potential if isinstance(potential, str) and potential else current
    foe = form(1 - side, prompt['opponent'])
    result = {}
    for option in prompt['options']:
        name = option['identity']
        own = form(side, name)
        a, b = (foe, own) if side else (own, foe)
        result[name] = {prompt['opponent']: ranks['by_forms'].get(a, {}).get(b, 0) * (-1 if side else 1)}
    return result


def resolve(client, deadline, root, cases, ranks, explicit=None, *, pivots=None):
    tasks = {case['id']: case for case in cases}
    trace = {key: [] for key in tasks}
    rounds = {key: (0, 1) for key in tasks}
    explicit = dict(explicit or {})
    completed = {}
    while cases:
        deadline.check()
        response = client.request('step', timeout=deadline.remaining(), root=root, cases=cases)
        if response['errors']:
            raise RuntimeError(response['errors'][0])
        if len(response['rows']) != len(cases):
            raise RuntimeError('Incomplete mechanics batch')
        cases = []
        for row in response['rows']:
            key = row['id']
            if not row['pending']:
                row['replacement_choices'] = trace[key]
                completed[key] = row
                continue
            count, last_side = rounds[key]
            pending = [p for p in row['pending'] if p['options']]
            if not pending:
                raise ValueError('Unresolved replacement request')
            prompt = ([p for p in pending if p['side'] > last_side] or pending)[0]
            side = prompt['side']
            if pivots is not None and not side and key not in pivots and tasks[key]['kind'] == 'turn':
                original = tasks[key].get('root') or root
                actor = original['sides'][0]['pokemon'][0]['m']['identity']
                if move_rules.pending(row, tasks[key]['action'], actor):
                    pivots[key] = {**row, 'replacement_choices': list(trace[key])}
            count += side <= last_side
            if count > 12:
                raise ValueError('Excessive replacement chain')
            rounds[key] = count, side
            target = explicit.get(key) if not side else None
            action = replacements.choose(prompt['options'], prompt['opponent'], ranks_for(row, prompt, ranks), explicit=target)
            if not side:
                explicit.pop(key, None)
            trace[key].append({'rule': 'explicit-candidate' if target else 'TS-09', 'side': side,
                              'opponent': prompt['opponent'], 'action': action})
            cases.append({'id': key, 'kind': 'resume', 'root': row['root'], 'side': side,
                          'choice': action['choice'], 'trace': row.get('measurements'),
                          'measure': tasks[key].get('measure', True),
                          'retain_root': tasks[key].get('retain_root', False)})
    return completed
