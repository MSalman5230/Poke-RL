"""Fixed reply with genuine queued-action contexts for exchange-dependent moves."""
from opponents.autoplay_reference.bot.decision.rules import mega_rules
from opponents.autoplay_reference.bot.decision import experiments
from opponents.autoplay_reference.bot.runtime.budget import Deadline

POLICY = "highest-immediate-damage-v3"
CONTEXTUAL = frozenset(('suckerpunch', 'upperhand', 'counter', 'mirrorcoat', 'metalburst', 'comeuppance',
                       'focuspunch', 'shelltrap', 'beakblast', 'payback', 'avalanche', 'revenge', 'assurance',
                       'mefirst', 'copycat', 'mirrormove'))


def usage_weights(data, name, prediction):
    weights = {}
    for species in dict.fromkeys([prediction.battle_form, name]):
        for key, source in data.ranked(species, "Moves"):
            weights.setdefault(key, source["share"])
    return weights


def predict(client, root, seeds, weights, *, timeout, own_actions=None, ranks=None, reuse=None):
    deadline = Deadline(timeout)
    if not seeds:
        raise ValueError('Opponent probes require saved seeds')
    prepared = client.request('probe_options', timeout=deadline.remaining(), root=root)
    options = mega_rules.force(prepared['legal'][1])
    legal = prepared['legal'][0]
    references = mega_rules.force(legal if own_actions is None else [a for a in own_actions if a['kind'] == 'move' and
        any(x['choice'] == a['choice'] for x in legal)]) if prepared['can_exchange'] else []
    defender_mega = next((a for a in references if a.get('mega')), None)
    ordinary = [a for a in options if a['name'] not in CONTEXTUAL]
    measured = client.request('probe', timeout=deadline.remaining(), root=root, seeds=seeds, options=ordinary,
                              defender_mega_action=defender_mega)['rows'] if ordinary else []
    if len(measured) != len(ordinary) or any((a['name'], a['mega']) != (b['name'], b['mega'])
                                          for a, b in zip(ordinary, measured)):
        raise ValueError('Incomplete opponent probe comparison')
    by_choice = {a['choice']: {**a, 'mode': 'isolated', 'samples': len(seeds),
                             'defender_mega_action': defender_mega} for a in measured}
    contextual = [a for a in options if a['name'] in CONTEXTUAL]
    cases = [{'id': f'{j}:{k}:{i}', 'kind': 'turn', 'seed': seed, 'action': action, 'reply': reply,
              'retain_root': reuse is not None}
             for j, reply in enumerate(contextual) for k, action in enumerate(references) for i, seed in enumerate(seeds)]
    pivots = {} if reuse is not None else None
    completed = experiments.resolve(client, deadline, root, cases, ranks or {}, pivots=pivots) if cases else {}
    if len(completed) != len(cases):
        raise ValueError('Incomplete contextual opponent comparison')
    actor = prepared['summary']['sides'][1]['active']
    for j, reply in enumerate(contextual):
        contexts = []
        for k, action in enumerate(references):
            trials = []
            for i, seed in enumerate(seeds):
                row = completed[f'{j}:{k}:{i}']
                hits = [h for h in row['measurements'].get('inflicted', [])
                        if h['actor'] == actor and h['side'] == 1 and h['target_side'] == 0]
                trials.append({'seed': seed, 'damage': sum(h['amount'] for h in hits), 'hits': hits,
                               'replacement_choices': row['replacement_choices']})
            contexts.append({'action': action, 'damage': sum(t['damage'] for t in trials) / len(seeds),
                             'samples': len(trials), 'trials': trials})
        best = max(contexts, key=lambda row: row['damage']) if contexts else None
        by_choice[reply['choice']] = {**reply, 'mode': 'contextual', 'samples': len(seeds),
            'damage': best['damage'] if best else 0, 'contexts': contexts,
            'reference': best['action'] if best else None,
            'inapplicable': None if contexts else 'no established legal move context at this phase'}
    rows = [by_choice[a['choice']] for a in options]
    deadline.check()
    chosen = choose(rows, weights) if rows else None
    if reuse is not None:
        reuse.clear()
        if chosen and chosen['name'] in CONTEXTUAL:
            j = next(i for i, action in enumerate(contextual) if action['choice'] == chosen['choice'])
            reuse.update(root=root, ranks=ranks or {}, reply=chosen['choice'], rows={
                (action['choice'], tuple(seed)): (completed[f'{j}:{k}:{i}'], pivots.get(f'{j}:{k}:{i}'))
                for k, action in enumerate(references) for i, seed in enumerate(seeds)})
    return rows, chosen


def pending_reply(input_state, reply):
    """OA-07: observed pending moves precede predictions; completed actions stay completed."""
    if input_state["replacement_phase"] == "faint" or input_state["sides"][0]["replacement"].get("opponent_acted") is True or not reply:
        return None
    return {"name": input_state.get("pending_opponent_move") or reply["name"], "mega": bool(reply.get("mega"))}


def choose(rows, move_weights):
    if not rows:
        raise ValueError("No legal opponent attack/reply")
    best = max(rows, key=lambda row: row["damage"])
    if best["damage"] <= 0:
        best = max(rows, key=lambda row: move_weights.get(row["name"], 0))
    # Context evidence lives in rows; do not copy it into every mechanical turn task.
    return {key: value for key, value in best.items() if key in ('kind', 'name', 'choice', 'mega', 'damage')}
