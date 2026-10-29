"""One-turn greedy policy over exact Showdown mechanics and explicit hypotheses."""
from dataclasses import asdict
import copy
import random
import time
from opponents.autoplay_reference.bot.domain import actions, catalog, state_contract, moves
from opponents.autoplay_reference.bot.domain.types import DecisionInput, ObservedBattleState
from opponents.autoplay_reference.bot.opponent.data import datasets
from opponents.autoplay_reference.bot.opponent import estimation
from opponents.autoplay_reference.bot.domain.forms import own_forms
from opponents.autoplay_reference.bot.simulator import bridge, state
from opponents.autoplay_reference.bot.simulator.actions import intersect
from opponents.autoplay_reference.bot.team_selection.coverage import TeamSelector
from opponents.autoplay_reference.bot.team_selection import preview
from opponents.autoplay_reference.bot.team_selection.matchups import Matchups
from opponents.autoplay_reference.bot.runtime import budget as budgets
from opponents.autoplay_reference.bot.decision import evaluation, opponent_action, rollouts, identity, recovery as tactical

POLICY = "greedy-showdown-v2"
ROLLOUTS = evaluation.TRIALS


def verify():
    catalog.get()
    datasets()
    return bridge.get().verify()


def prepare_preview(roster, *, progress=None, stop_flag=None):
    from opponents.autoplay_reference.bot.team_selection.prepare import prepare
    return prepare(roster, progress=progress, stop_flag=stop_flag)


def decide(ctx, battle, kind="turn", remaining=None, budget=None, seed=None, *, emergency=False):
    started = time.monotonic()
    allowance = budgets.allowance(kind, remaining, budget)
    hard_deadline = started + allowance
    deadline = hard_deadline if emergency or kind == "preview" else started + allowance * (1 - budgets.EMERGENCY_RESERVE_FRACTION)
    context, roster = battle["sim"], battle["sim"].roster
    ctx = copy.deepcopy(ctx)
    if context.field.unrepresented:
        ctx["field_effects"] = copy.deepcopy(context.field.unrepresented)
    available = actions.available(ctx, roster, kind)
    seed = seed if seed is not None else random.SystemRandom().getrandbits(64)
    rng = random.Random(seed)
    seeds = tuple(tuple(rng.randrange(65536) for _ in range(4)) for _ in range(ROLLOUTS))
    diag = {"schema_version": 2, "backend": "pokemon-showdown", "policy": POLICY, "kind": kind,
            "strategy_identity": identity.current(),
            "selection_policy": "BE-23" if emergency else evaluation.POLICY, "opponent_action_policy": opponent_action.POLICY,
            "emergency_mode": emergency, "hard_deadline_seconds": allowance,
            "rule_parameters": evaluation.PARAMETERS,
            "state_builder": state_contract.BUILDER, "observed_state": copy.deepcopy(ctx),
            "roster": copy.deepcopy(roster.data), "roster_identity": roster.identity, "datasets": context.datasets.identities,
            "catalog": catalog.get().identity, "seed": seed, "rollout_seeds": seeds,
            "clocks": {"remaining_seconds": remaining, "execution_reserve": budgets.EXECUTION_RESERVE if kind == "turn" else 0,
                       "decision_allowance": allowance},
            "available_actions": [a.to_dict() for a in available], "errors": [], "assumptions": [],
            "rejected_actions": [], "fallback": False, "deadline_passed": False, "timings": {"budget": allowance}}
    diag['prompt_id'] = battle.get('prompt_id')
    failure_key = tactical.fingerprint(ctx, battle, kind) if kind != 'preview' else None
    cached_failure = (battle.get('decision_failure') or {}).get('key') == failure_key and failure_key is not None
    if cached_failure:
        diag['reused_failure'] = copy.deepcopy(battle['decision_failure'])
    selected, stage = None, 'availability'
    try:
        if not available:
            raise ValueError("Action availability is not established")
        if ctx.get('tactical_reason'):
            stage = 'reconstruction'
            raise ValueError(ctx['tactical_reason'])
        if time.monotonic() >= deadline:
            raise TimeoutError("Decision deadline passed")
        if cached_failure:
            raise ValueError('unchanged failed input; reuse tactical recovery')
        if kind != "preview":
            stage = 'reconstruction'
            state.validate_observed(ctx, battle)
        stage = 'prediction'
        predictions = estimation.predict_builds(battle, ctx, allow_defaults=kind != 'preview')
        diag["predictions"] = {n: asdict(p) for n, p in predictions.items()}
        request = DecisionInput(ObservedBattleState(kind, copy.deepcopy(ctx), battle.get("turn", 0)),
                                copy.deepcopy(roster.data), tuple(predictions.values()), tuple(available), seeds, deadline)
        if kind == "preview":
            eligible = [r["species"] for r in ctx["ours"] if r.get("eligible", True) and r.get("species") in roster.members]
            selection = preview.select(roster, predictions, eligible, context.datasets, deadline=deadline,
                                       selected=ctx.get('selected_prefix',()))
            diag["team_selection"] = asdict(selection)
            selected = next((a for a in available if a.picks == selection.picks), None)
        else:
            party, guessed = estimation.predict_party(battle, ctx, predictions, allow_defaults=True)
            diag['party_predictions'] = {n: asdict(p) for n, p in party.items()}
            diag["opponent_party"] = asdict(guessed)
            stage = 'reconstruction'
            hydration, assumptions = state.build(ctx, battle, kind, party)
            diag["hydration_input"] = hydration
            diag["assumptions"].extend(assumptions)
            client = bridge.get()
            stage = 'simulator'
            if client.identity is None:
                client.verify(timeout=max(0, deadline-time.monotonic()))
            diag["simulator_identity"] = client.identity
            prepared = client.request("prepare", timeout=max(0, deadline - time.monotonic()), input=hydration)
            diag["simulator_root"] = prepared["root"]
            diag["hydrated_state"] = prepared["summary"]
            diag["assumptions"].extend(prepared["assumptions"])
            candidates, rejected = intersect(available, prepared["legal"][0])
            if emergency:
                candidates = [a for a in candidates if a['kind'] != 'move' or not moves.offensive_boosts(a.get('name'))]
            diag["rejected_actions"].extend(rejected)
            if not candidates:
                raise ValueError("No established action is legal in the reconstructed simulator")
            diag["timings"]["preparation"] = time.monotonic() - started
            active = hydration["sides"][1]["names"][0]
            weights = opponent_action.usage_weights(context.datasets, active, party[active])
            ranks = Matchups(context.datasets).replacement_ranks(battle["picks"], party, own_forms(roster, battle.get('flags')))
            probe_work = {}
            probes, reply = opponent_action.predict(client, prepared["root"], seeds, weights,
                timeout=max(0, deadline - time.monotonic()), own_actions=candidates, ranks=ranks, reuse=probe_work)
            if reply is None and kind != "send":
                raise ValueError("No legal opponent reply at the move prompt")
            diag["opponent_replies"] = probes
            diag["predicted_reply"] = reply
            payload = {"root": prepared["root"], "input": hydration, "candidates": candidates, "reply": reply,
                       "seeds": request.seeds, "move_weights": weights,
                       "own_party_rows": {name: i for i, name in enumerate(battle["picks"], 1)},
                       "move_weights_by_species": {n: opponent_action.usage_weights(context.datasets, n, p) for n, p in party.items()},
                       "ranks": ranks, "emergency": emergency}
            diag["simulation_request"] = copy.deepcopy(payload)
            result = rollouts.simulate(client, timeout=max(0, deadline - time.monotonic()), probe_work=probe_work, **payload)
            diag["outcomes"] = result
            diag["errors"].extend({"error": e, 'stage': 'simulator'} for e in result["errors"])
            diag["deadline_passed"] |= result["deadline_passed"]
            choice, diag["rule_evaluation"] = evaluation.emergency_select(result) if emergency else evaluation.select(result)
            _, diag["ranked_actions"] = evaluation.rank(result["rows"])
            selected = next((a for a in available if a.id == choice), None)
    except Exception as exc:
        diag["errors"].append({"error": repr(exc), 'stage': 'deadline' if isinstance(exc, TimeoutError) else stage,
                               'operation': stage})
        diag["deadline_passed"] |= isinstance(exc, TimeoutError) or time.monotonic() >= deadline
    if selected is None and stage == 'simulator' and not emergency and not cached_failure and kind != 'preview' and available and time.monotonic() < hard_deadline:
        restarted = False
        if stage == 'simulator':
            # Recovery gets one fresh process/identity, inside the original clock.
            try:
                bridge.get().close()
                bridge.get().verify(timeout=max(0, hard_deadline-time.monotonic()))
                diag['worker_restarted'] = True
                restarted = True
            except Exception as exc:
                diag['errors'].append({'stage': 'simulator', 'error': repr(exc), 'recovery': 'worker restart'})
        if restarted and time.monotonic() < hard_deadline:
            recovery = decide(ctx, battle, kind, remaining=remaining,
                              budget=hard_deadline-time.monotonic(), seed=seed, emergency=True)
            diag['emergency'] = recovery.diagnostics
            selected = recovery.action
            diag['emergency_used'] = selected is not None and recovery.source == 'emergency_scored'
            if recovery.source == 'heuristic_recovery':
                diag['tactical_recovery'] = recovery.diagnostics.get('tactical_recovery')
    if selected is None and kind != 'preview' and available:
        belief = context.beliefs.get(ctx.get('opp'))
        selected, reasons = tactical.choose(ctx, roster, available,
                                            enemy_moves=belief.revealed_moves if belief else (),
                                            enemy_form=belief.form if belief else None)
        diag['tactical_recovery'] = reasons
    if diag.get('tactical_recovery'):
        diag['fallback'] = True
        if not cached_failure:
            battle['decision_failure'] = {'key': failure_key, 'errors': diag['errors']}
    elif selected is not None:
        battle.pop('decision_failure', None)
    diag["evidence"] = {n: belief.snapshot() for n, belief in context.beliefs.items()}
    diag["selected_action"] = selected.to_dict() if selected else None
    diag["timings"]["total"] = time.monotonic() - started
    diag["timings"].setdefault("preparation", diag["timings"]["total"])
    diag["completed_result_used"] = selected is not None and not diag["fallback"]
    diag["engine_failed"] = bool(diag["errors"])
    source = ('heuristic_recovery' if diag.get('tactical_recovery') else
              "emergency_scored" if emergency or diag.get('emergency_used') else "greedy_showdown") if selected else "unavailable"
    return actions.Decision(selected, available, source, diag)


def decide_turn(ctx, battle, move_time=None, budget=None, *, emergency=False):
    return decide(ctx, battle, "turn", remaining=move_time, budget=budget, emergency=emergency)


def decide_send(ctx, battle, remaining=None):
    return decide(ctx, battle, "send", remaining=remaining)


def decide_preview(ours, opp, db, roster, budget=None, selected_prefix=()):
    from opponents.autoplay_reference.bot.runtime.context import BattleContext
    context = BattleContext(db, roster)
    rows = [row if isinstance(row, dict) else {"species": row, "row": i} for i, row in enumerate(ours, 1)]
    ctx = {"ours": rows, "opponents": copy.deepcopy(opp), "flags": {}}
    if selected_prefix:
        ctx['selected_prefix']=list(selected_prefix)
    return decide(ctx, {"sim": context, "opp": [o.get("species") for o in opp], "turn": 0}, "preview", budget=budget)
