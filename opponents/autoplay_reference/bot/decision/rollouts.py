"""Bounded mechanical experiments; every strategic choice has one Python owner."""
import json
from opponents.autoplay_reference.bot.decision import evaluation, opponent_action, experiments
from opponents.autoplay_reference.bot.decision.rules import move_rules, pokemon_rules, mega_rules
from opponents.autoplay_reference.bot.domain import moves
from opponents.autoplay_reference.bot.runtime import budget as budgets
from opponents.autoplay_reference.bot.team_selection import replacements


class Rollouts:
    def __init__(self, client, timeout, epoch_ms=None):
        self.client = client
        self.deadline = budgets.Deadline(timeout, epoch_ms)
        self.reply_cache = {}
        self.probe_work = {}
        self.partial = None

    def _resolve(self, root, cases, ranks, explicit=None, *, pivots=None):
        return experiments.resolve(self.client, self.deadline, root, cases, ranks, explicit, pivots=pivots)

    def _pivot_entries(self, request, result, starts, *, retain_root):
        """Branch actual saved pivot prompts; a fixed receiver is shared by all 32 trials."""
        for key, data in result['pivot_evaluations'].items():
            baseline = result['rows'][key]
            data['baseline_entries'] = baseline
            paths = data['receivers']
            for receiver in move_rules.receivers(result['before'], data['paused'], request.get('own_party_rows', {})):
                path = {'receiver': receiver, 'entries': list(baseline), 'complete': False,
                        'pivot_successes': 0, 'reused_samples': 0, 'unavailable_samples': 0,
                        'applicable_trials': list(result.get('trial_indices', range(len(baseline)))),
                        'unavailable_trials': []}
                paths[receiver] = path
                cases = []
                for i, pause in enumerate(data['paused']):
                    if pause is None:
                        continue
                    prompt = next(p for p in pause['pending'] if p['side'] == 0)
                    option = next((a for a in prompt['options'] if a['identity'] == receiver), None)
                    if option is None:
                        path['unavailable_samples'] += 1
                        path['unavailable_trials'].append(i)
                        continue
                    at = len(pause['replacement_choices'])
                    history = baseline[i]['replacement_choices']
                    if at < len(history) and history[at]['side'] == 0 and history[at]['action']['identity'] == receiver:
                        path['entries'][i] = {**baseline[i], 'replacement_choices': history[:at] + [
                            {**history[at], 'rule': 'BE-20', 'reused_automatic': True}] + history[at + 1:]}
                        path['reused_samples'] += 1
                        continue
                    cases.append({'id': str(i), 'kind': 'resume', 'root': pause['root'], 'side': 0,
                                  'choice': option['choice'], 'trace': pause['measurements'],
                                  'retain_root': retain_root})
                completed = self._resolve(None, cases, request.get('ranks', {})) if cases else {}
                if len(completed) != len(cases):
                    raise RuntimeError('Incomplete pivot receiver batch')
                for i, pause in enumerate(data['paused']):
                    if str(i) not in completed:
                        continue
                    row = completed[str(i)]
                    row['replacement_choices'] = pause['replacement_choices'] + [
                        {'rule': 'BE-20', 'side': 0, 'opponent': pause['after']['sides'][1]['active'],
                         'action': next(a for p in pause['pending'] if p['side'] == 0
                                        for a in p['options'] if a['identity'] == receiver)}] + row['replacement_choices']
                    path['entries'][i] = self._outcome(starts[i], row, request['seeds'][i])
                path.update(outcomes=path['entries'], pivot_successes=len(completed) + path['reused_samples'], complete=True)
                path['first_turn_score'] = path['score'] = evaluation._mean(path['entries'], False)
                if pokemon_rules.eligibility(result)['eligible']:
                    path['palafin_exit'] = pokemon_rules.exit_facts(path['entries'], result['before']['sides'][0]['active'],
                                                                  result.get('trial_indices'))
            data.update(complete=True, horizon_turns=1)
            if paths:
                receiver = evaluation.choose_receiver(result, key, paths)
                data['selected_receiver'] = receiver
                data['selected_score'] = paths[receiver]['score']
                result['rows'][key] = paths[receiver]['outcomes']

    def _continue_pivot(self, request, result, key):
        """Use the common continuation for each receiver before selecting its full mean."""
        data = result['pivot_evaluations'][key]
        data.update(complete=False, horizon_turns=2)
        if not data['receivers']:
            self._continue_entry(request, result, key, next_turn_only=True)
        else:
            for path in data['receivers'].values():
                branch = {'before': result['before'], 'rows': {key: path['entries']},
                          'plans': {}, 'plan_evaluations': {}}
                path['continuation_evaluation'] = branch
                path['complete'] = False
                self._continue_entry(request, branch, key, next_turn_only=True)
                path.update(outcomes=branch['rows'][key], complete=True,
                            score=evaluation._mean(branch['rows'][key], False))
            receiver = evaluation.choose_receiver(result, key, data['receivers'])
            data['selected_receiver'] = receiver
            data['selected_score'] = data['receivers'][receiver]['score']
            branch = data['receivers'][receiver]['continuation_evaluation']
            result['rows'][key] = branch['rows'][key]
            plan = branch['plans'][key]
            result['plans'][key] = {**plan, 'receiver': receiver} if plan else None
            if key in branch['plan_evaluations']:
                result['plan_evaluations'][key] = {**branch['plan_evaluations'][key], 'receiver': receiver}
        data['complete'] = True

    @staticmethod
    def _outcome(before, row, seed):
        result = {'seed': seed, 'after': row['after'], 'components': evaluation.components(before, row['after']),
                  'protocol': row.get('protocol', []), 'replacement_choices': row.get('replacement_choices', []),
                  'measurements': row.get('measurements')}
        for key in ('root', 'legal'):
            if key in row:
                result[key] = row[key]
        return result

    def _reply(self, request, row):
        root = row['root']
        foe = row['after']['sides'][1]['active']
        weights = request.get('move_weights_by_species', {}).get(foe, request.get('move_weights', {}))
        # Probes reset their PRNG on a clone; these fields cannot affect probe mechanics.
        state = {k: v for k, v in root.items() if k not in ('prng', 'log', 'inputLog', 'sentLogPos')}
        own_actions = list(self._options([row], request).values())
        # Legal own contexts also determine the ordinary probe's defender Mega reference.
        key = json.dumps([state, request['seeds'], weights, own_actions, request.get('ranks', {})], sort_keys=True, separators=(',', ':'))
        if key not in self.reply_cache:
            self.reply_cache[key] = opponent_action.predict(self.client, root, request['seeds'], weights,
                timeout=self.deadline.remaining(), own_actions=own_actions, ranks=request.get('ranks', {}))
        probes, reply = self.reply_cache[key]
        if reply is None:
            raise ValueError('No legal opponent continuation')
        return {'options': probes, 'reply': reply}

    @staticmethod
    def _options(prefix, request, *, attacks_only=False):
        known = {p['identity']: [m['id'] for m in p['moves']] for p in request['input']['sides'][0]['pokemon']}
        options, seen = [], set()
        for row in prefix:
            actor = row['after']['sides'][0]['active']
            for action in mega_rules.force(row.get('legal', [[], []])[0]):
                if action['kind'] == 'move' and action['name'] not in known.get(actor, set()):
                    continue
                if attacks_only and (action['kind'] != 'move' or not moves.damaging(action['name'])):
                    continue
                identity = (action['kind'], action.get('name'), action.get('mega'), action.get('identity'))
                if identity not in seen:
                    seen.add(identity)
                    if action['kind'] == 'move':
                        order = (0, known[actor].index(action['name']), action['mega'])
                    else:
                        order = (1, request.get('own_party_rows', {}).get(action['identity'], list(known).index(action['identity'])))
                    options.append((order, action))
        # A move disabled in the first surviving trial must not lose its original slot tie-break.
        return {str(i): {**action, 'id': str(i)} for i, (_, action) in enumerate(sorted(options, key=lambda row: row[0]))}

    @staticmethod
    def _legal_action(action, legal):
        return next((option for option in legal if option['kind'] == action['kind'] and
                     (option.get('name'), option.get('mega'), option.get('identity')) ==
                     (action.get('name'), action.get('mega'), action.get('identity'))), None)

    def _turn(self, request, prefix=None, options=None, *, allow_setup=True, allow_plans=True,
              trial_indices=None, record=None):
        seeds = request['seeds']
        initial = prefix is None
        reuse = self.probe_work if (initial and self.probe_work.get('root') == request['root'] and
            self.probe_work.get('reply') == (request.get('reply') or {}).get('choice') and
            self.probe_work.get('ranks') == request.get('ranks', {})) else {}
        indices = tuple(range(len(seeds))) if trial_indices is None else tuple(trial_indices)
        before = self.client.request('inspect', timeout=self.deadline.remaining(), root=request['root'])['summary'] if initial else prefix[indices[0]]['after']
        options = options if options is not None else {a['id']: a for a in mega_rules.force(request['candidates'])}
        result = {'before': before, 'options': options, 'rows': {key: [] for key in options},
                  'emergency': bool(request.get('emergency')),
                  'reused_probe_exchanges': 0,
                  'setup_evaluations': {}, 'completed_batches': 0, 'errors': [], 'plans': {}, 'plan_evaluations': {},
                  'pivot_evaluations': {key: {'rule': 'BE-20', 'paused': [], 'receivers': {},
                                            'selected_receiver': None, 'complete': False}
                                        for key, a in options.items() if move_rules.is_pivot(a)}}
        if trial_indices is not None:
            result['trial_indices'] = indices
        if not initial:
            result['active_forms'] = [evaluation.member(prefix[i]['after'], 0,
                prefix[i]['after']['sides'][0]['active']).get('species') for i in indices]
        if initial:
            self.partial = result
        if record is not None:
            record['continuation'] = result
        replies = []
        if not initial and options:
            result['reply_predictions'] = replies
            for i, row in enumerate(prefix):
                replies.append(self._reply(request, row) if i in indices and row.get('root') and not row['after']['ended'] else None)
        starts = []
        for i, seed in enumerate(seeds[:request.get('max_batches', len(seeds))]):
            self.deadline.check()
            start = before if initial else prefix[i]['after']
            starts.append(start)
            root = request['root'] if initial else prefix[i].get('root')
            cases, skipped, paused = [], {}, {}
            legal = None if initial else prefix[i].get('legal', [[], []])[0]
            for key, action in options.items():
                supplied = action if initial else self._legal_action(action, legal)
                if i not in indices or root is None or start['ended'] or supplied is None:
                    skipped[key] = {'after': start, 'measurements': None, 'unavailable': True}
                    continue
                cached = reuse.get('rows', {}).get((supplied['choice'], tuple(seed))) if reuse else None
                if cached:
                    skipped[key], pivot = cached
                    if pivot:
                        paused[key] = pivot
                    result['reused_probe_exchanges'] += 1
                    continue
                task = {'id': key, 'kind': 'turn', 'root': root, 'action': supplied,
                        'reply': request.get('reply') if initial else replies[i]['reply'],
                        'retain_root': allow_plans or allow_setup and bool(moves.offensive_boosts(action.get('name')))}
                if initial:
                    task['seed'] = seed
                cases.append(task)
            completed = self._resolve(None, cases, request.get('ranks', {}), pivots=paused) if cases else {}
            completed.update(skipped)
            for key, data in result['pivot_evaluations'].items():
                data['paused'].append(paused.get(key))
            batch = {key: self._outcome(start, completed[key], seed) for key in options}
            for key, row in batch.items():
                result['rows'][key].append(row)
            result['completed_batches'] += 1
        if result['completed_batches'] == evaluation.TRIALS and options:
            self._pivot_entries(request, result, starts, retain_root=allow_plans)
            exits = evaluation.palafin_exits(result)['options']
            result['setup_pending'] = allow_setup and not exits and any(
                evaluation.is_setup(a, before, result['rows'][key]) for key, a in options.items())
            result['switch_pending'] = allow_plans and any(a['kind'] == 'switch' for a in options.values())
            result['ordinary_pending'] = allow_plans
            selected = None
            if result['setup_pending'] or result['ordinary_pending']:
                selected, _ = evaluation.select(result)
            if result['setup_pending'] and selected is None:
                self._setups(request, result)
                result['setup_pending'] = False
                if result['ordinary_pending']:
                    selected, _ = evaluation.select(result)
            result['setup_pending'] = False
            if result['ordinary_pending'] and selected is None:
                result['ordinary_candidates'] = evaluation.ordinary_options(result)
                for key in result['ordinary_candidates']:
                    if key in result['pivot_evaluations']:
                        self._continue_pivot(request, result, key)
                    else:
                        self._continue_entry(request, result, key, next_turn_only=True)
            result['switch_pending'] = False
            result['ordinary_pending'] = False
        result['choice'], result['selection'] = evaluation.select(result)
        return result

    def _setups(self, request, result):
        for key, action in result['options'].items():
            if not evaluation.is_setup(action, result['before'], result['rows'][key]):
                continue
            samples = result['rows'][key]
            gate = evaluation.setup_gate(action, result['before'], samples)
            if not gate['eligible']:
                result['setup_evaluations'][key] = {**gate, 'qualifies': False}
                continue
            actor, prefix = gate['actor'], []
            for row in samples:
                usable = (evaluation.member(row['after'], 0, actor)['hp'] > 0 and
                          row['after']['sides'][0]['active'] == actor and not row['after']['ended'])
                prefix.append(row if usable else {'after': row['after'], 'legal': [[], []]})
            options = self._options(prefix, request, attacks_only=True)
            progress = {**gate, 'qualifies': False, 'reason': 'incomplete'}
            result['setup_evaluations'][key] = progress
            following = self._turn(request, prefix, options, allow_setup=False, allow_plans=False, record=progress)
            if following['completed_batches'] != evaluation.TRIALS:
                raise ValueError('Incomplete setup continuation')
            data = evaluation.setup_result(action, result['before'], samples, following)
            data['continuation'] = following
            result['setup_evaluations'][key] = data

    def _continue_entry(self, request, result, key, *, next_turn_only=False):
        """Share continuation scoring for staying, switching and faint replacement."""
        prefix = result['rows'][key]
        if all(row['after']['ended'] for row in prefix):
            result['plans'][key] = None
            return
        progress = {'entries': prefix, 'complete': False, 'reason': 'incomplete',
                    'next_turn_only': next_turn_only}
        result['plan_evaluations'][key] = progress
        groups = {}
        if next_turn_only:
            for i, row in enumerate(prefix):
                if not row['after']['ended']:
                    if not row.get('root'):
                        raise ValueError('Missing next-turn continuation root')
                    groups.setdefault(row['after']['sides'][0]['active'], []).append(i)
        else:
            groups[prefix[0]['after']['sides'][0]['active']] = list(range(len(prefix)))
        progress['groups'] = groups
        plans, final = {}, list(prefix)
        for actor, indices in groups.items():
            options = self._options([prefix[i] for i in indices], request)
            if request.get('emergency'):
                options = {k: a for k, a in options.items() if a['kind'] != 'move' or not moves.offensive_boosts(a.get('name'))}
            if next_turn_only:
                options = {name: action for name, action in options.items()
                           if all(self._legal_action(action, prefix[i]['legal'][0]) for i in indices)}
            record = progress if len(groups) == 1 else progress.setdefault('continuations', {}).setdefault(actor, {})
            following = self._turn(request, prefix, options, allow_setup=not next_turn_only, allow_plans=False,
                                   trial_indices=indices if next_turn_only else None, record=record)
            choice = following['choice']
            if choice is None:
                raise ValueError('No complete continuation after entry')
            plans[actor] = following['options'][choice]
            selected = evaluation.chosen_rows(following, choice)
            for i in indices:
                entry, row = prefix[i], selected[i]
                value = self._outcome(result['before'], row, row['seed'])
                value['replacement_choices'] = entry['replacement_choices'] + row['replacement_choices']
                final[i] = value
        result['rows'][key] = final
        result['plans'][key] = {**({'action': next(iter(plans.values()))} if len(plans) == 1 else {'actions': plans}),
                                'mean': evaluation._mean(final, False)}
        progress.update(complete=True, reason=None)

    @staticmethod
    def _send_case(key, request, seed):
        return {'id': key, 'kind': 'send', 'seed': seed, 'retain_root': True,
                'residual': request['input']['replacement_phase'] != 'faint',
                'pending_action': opponent_action.pending_reply(request['input'], request.get('reply'))}

    def _send(self, request):
        before = self.client.request('inspect', timeout=self.deadline.remaining(), root=request['root'])['summary']
        options = {a['id']: a for a in request['candidates']}
        result = {'before': before, 'options': options, 'rows': {k: [] for k in options}, 'completed_batches': 0,
                  'errors': [], 'plans': {}, 'plan_evaluations': {}, 'emergency': bool(request.get('emergency'))}
        self.partial = result
        for seed in request['seeds'][:request.get('max_batches', len(request['seeds']))]:
            cases = [self._send_case(key, request, seed) for key in options]
            rows = self._resolve(request['root'], cases, request.get('ranks', {}), {k: a['identity'] for k, a in options.items()})
            for key in options:
                result['rows'][key].append(self._outcome(before, rows[key], seed))
            result['completed_batches'] += 1
        pivot = move_rules.replacement(request['input'])
        if (request['input']['replacement_phase'] == 'faint' or pivot and not request.get('emergency')) and result['completed_batches'] == evaluation.TRIALS:
            for key in options:
                self._continue_entry(request, result, key, next_turn_only=pivot or bool(request.get('emergency')))
        result['pivot_replacement'] = pivot
        result['choice'], result['selection'] = evaluation.select(result)
        return result

    def simulate(self, request):
        result = {'rows': {a['id']: [] for a in request['candidates']}, 'completed_batches': 0,
                  'plans': {}, 'plan_evaluations': {}, 'errors': [], 'choice': None}
        try:
            self.deadline.check()
            result = self._send(request) if request['input']['kind'] == 'send' else self._turn(
                request, allow_setup=not request.get('emergency'), allow_plans=not request.get('emergency'))
        except Exception as exc:
            result = self.partial or result
            result['errors'].append(str(exc))
            result['choice'] = None
        result['deadline_passed'] = not self.deadline.remaining() or any('deadline' in e.lower() for e in result['errors'])
        if result['deadline_passed']:
            result['choice'] = None
            if not result['errors']:
                result['errors'].append('Decision deadline passed')
        if result.get('choice') is None and len(request['seeds']) == evaluation.TRIALS and not result['errors']:
            result['errors'].append('No qualified complete 32-trial decision')
        return result


def simulate(client, *, timeout=None, probe_work=None, **request):
    if timeout is None:
        timeout = budgets.ceiling(request['input']['kind'])
    runner = Rollouts(client, timeout, request.get('deadline_ms'))
    runner.probe_work = probe_work or {}
    return runner.simulate(request)
