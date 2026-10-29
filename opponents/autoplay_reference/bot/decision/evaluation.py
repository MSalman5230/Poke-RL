"""One owner for outcome scores, probability gates and action preferences."""
from dataclasses import asdict, dataclass
from functools import reduce
from operator import add
from opponents.autoplay_reference.bot.domain import catalog, moves
from opponents.autoplay_reference.bot.decision.rules import move_rules, pokemon_rules

POLICY = "setup-priority-v3"
TRIALS = 32
THRESHOLD = .70
BOOST_LIMIT = 3
SETUP_SURVIVAL = 1.0
SETUP_KO = .85
TERMINAL_LOSS = -10000
PARAMETERS = {"trials": TRIALS, "threshold": THRESHOLD, "boost_limit_before_move": BOOST_LIMIT,
              "setup_survival_threshold": SETUP_SURVIVAL, "setup_ko_threshold": SETUP_KO,
              "setup_threshold_inclusive": True, "setup_ko_denominator": "all-trials", "entry_followup_decisions": 1,
              "mega_policy": "always-when-legal-v1",
              "ordinary_followup_turns": 1, "sacrifice_comparison": True,
              "all_loss_tiebreak": "first-turn-damage",
              "palafin_zero": pokemon_rules.PARAMETERS,
              "pivot_moves": sorted(move_rules.PIVOTS), "pivot_receivers": "best-fixed-mean"}


def probability(successes, total, *, threshold=THRESHOLD, inclusive=False):
    return {"successes": successes, "total": total, "probability": successes / total if total else None,
            "threshold": threshold, "passes": total > 0 and (successes / total >= threshold if inclusive else successes / total > threshold)}


def member(summary, side, name):
    return next(p for p in summary["sides"][side]["pokemon"] if p["identity"] == name)


def boost_stats(action, before, samples=()):
    own = before["sides"][0]
    mon = member(before, 0, own["active"])
    potential = moves.offensive_boosts(action.get("name"), mon["types"])
    if action.get("name") == "curse" and not potential and any(
            member(row["after"], 0, mon["identity"])["boosts"].get("atk", 0) > mon["boosts"].get("atk", 0) for row in samples):
        potential = ("atk",)
    return tuple(stat for stat in potential
                 if mon["boosts"].get(stat, 0) < BOOST_LIMIT)


def is_setup(action, before, samples=()):
    mon = member(before, 0, before["sides"][0]["active"])
    potential = moves.offensive_boosts(action.get("name"), mon["types"])
    # Curse can change behavior after a faster type-changing move in the same turn.
    changed = action.get("name") == "curse" and any(
        member(row["after"], 0, mon["identity"])["boosts"].get(stat, 0) > mon["boosts"].get(stat, 0)
        for row in samples for stat in ("atk", "spa"))
    return action["kind"] == "move" and bool(potential or changed)


def attack_facts(action, outcome, actor):
    trace = outcome.get("measurements") or {}
    hits = [h for h in trace.get("damage", []) if h["actor"] == actor and h["side"] == 0 and
            h["target_side"] == 1 and h["move"] == action.get("name")]
    knockout = any(h["knockout"] for h in hits)
    priority = False
    if knockout:
        for queue in trace.get("queues", []):
            own = next((m for m in queue if m["side"] == 0 and m["actor"] == actor and m["move"] == action.get("name")), None)
            foe = next((m for m in queue if m["side"] == 1), None)
            if own and foe:
                ordinary_later = foe["priority"] > 0 or foe["priority"] == 0 and own["speed"] < foe["speed"]
                priority |= own["priority"] > 0 and ordinary_later and queue.index(own) < queue.index(foe)
    return sum(h["amount"] for h in hits), knockout, priority


@dataclass(frozen=True)
class CandidateMeasurement:
    id: str
    score: float
    damage: float
    knockout: dict
    priority_knockout: dict
    setup: bool
    samples: int
    first_turn_score: float
    applicable_samples: int


def choose_ko(entries):
    """One priority preference for immediate attacks and fixed setup follow-ups."""
    preferred = [entry for entry in entries if entry["priority_knockout"]["passes"]]
    return max(preferred or entries, key=lambda entry: entry["score"]), bool(preferred)


def measure(rows, options, before, *, score_rows=None, trial_indices=None):
    actor = before["sides"][0]["active"]
    result = []
    for key, samples in rows.items():
        action = options[key]
        applicable = [samples[i] for i in trial_indices] if trial_indices is not None else samples
        facts = [attack_facts(action, sample, actor) for sample in applicable]
        first_score = _mean(samples, False)
        final = score_rows[key] if score_rows is not None else samples
        result.append(CandidateMeasurement(key, first_score if final is samples else _mean(final, False),
            sum(f[0] for f in facts) / len(facts) if facts else 0,
            probability(sum(f[1] for f in facts), len(facts)), probability(sum(f[2] for f in facts), len(facts)),
            is_setup(action, before, applicable), len(samples), first_score, len(applicable)))
    return result


def first_rows(result):
    """Continuation scores must never turn a next-turn KO into an immediate KO."""
    plans = result.get("plan_evaluations", {})
    return {key: plans.get(key, {}).get("entries", rows) for key, rows in result["rows"].items()}


def palafin_exits(result):
    """Compatibility entry point; the Pokemon rule owns all Palafin decisions."""
    return pokemon_rules.palafin_exits(result, first_rows(result))


def _ordinary(metrics, options):
    attacks = [m for m in metrics if options[m.id]["kind"] == "move" and moves.damaging(options[m.id].get("name"))]
    pivots = [m for m in attacks if move_rules.is_pivot(options[m.id])]
    ordinary = [m for m in attacks if m not in pivots]
    others = [m for m in metrics if not m.setup and m not in attacks]
    others.extend(pivots)
    others.extend(m for m in ordinary if move_rules.is_charging_attack(options[m.id]))
    if ordinary:
        others.append(max(ordinary, key=lambda m: (m.damage, m.first_turn_score)))
    return [m for m in metrics if m in others]


def choose_receiver(result, key, paths):
    """BE-19/20: one fixed receiver by full mean, after Pokemon-specific eligibility."""
    choices = list(paths) if result.get('emergency') else pokemon_rules.receiver_choices(result, key, paths)
    return rank({name: paths[name]['outcomes'] for name in choices})[0]


def ordinary_options(result):
    exits = palafin_exits(result)["options"]
    if exits:
        return exits
    metrics = measure(first_rows(result), result["options"], result["before"], trial_indices=result.get("trial_indices"))
    return [m.id for m in _ordinary(metrics, result["options"])]


def setup_gate(action, before, samples):
    actor = before["sides"][0]["active"]
    stats = boost_stats(action, before, samples)
    survival = probability(sum(member(s["after"], 0, actor)["hp"] > 0 for s in samples), len(samples),
                           threshold=SETUP_SURVIVAL, inclusive=True)
    reason = "stage-limit" if not stats else "survival" if not survival["passes"] else None
    return {"actor": actor, "stats": stats, "survival": survival, "reason": reason,
            "eligible": len(samples) == TRIALS and reason is None}


def setup_result(action, before, prefix, following):
    """Evaluate a fixed follow-up across all surviving setup trials, never a per-seed oracle."""
    gate = setup_gate(action, before, prefix)
    result = {**gate, "qualifies": False, "followups": {}}
    if not gate["eligible"]:
        return result
    if following.get('errors') or following.get('deadline_passed') or any(
            len(rows) != TRIALS for rows in following['rows'].values()):
        return {**result, 'reason': 'incomplete'}
    actor, stats = gate["actor"], gate["stats"]
    start = member(before, 0, actor)["boosts"]
    survivors = [i for i, row in enumerate(prefix) if member(row["after"], 0, actor)["hp"] > 0]
    choices = []
    for key, samples in following["rows"].items():
        option = following["options"][key]
        wins = 0
        priority_wins = 0
        outcomes = []
        for i, row in enumerate(samples):
            useful = any(h.get("offensive") and h["knockout"] and h["actor"] == actor and h["target_side"] == 1 and
                h["move"] == option["name"] and h["offensive"]["owner"] == actor and
                h["offensive"]["stat"] in stats and
                h["offensive"]["stage"] > start.get(h["offensive"]["stat"], 0)
                for h in (row.get("measurements") or {}).get("damage", []))
            valid = i in survivors and prefix[i]["after"]["sides"][0]["active"] == actor and useful
            wins += valid
            priority_wins += valid and attack_facts(option, row, actor)[2]
            outcomes.append({**row, "components": components(before, row["after"])})
        ko = probability(wins, len(prefix), threshold=SETUP_KO, inclusive=True)
        priority = probability(priority_wins, len(prefix))
        data = {"action": option, "ko": ko, "priority_ko": priority, "score": _mean(outcomes, False)}
        result["followups"][key] = data
        if ko["passes"]:
            choices.append((key, data, outcomes))
    if choices:
        picked, _ = choose_ko([{"id": key, "score": data["score"], "priority_knockout": data["priority_ko"]}
                               for key, data, _ in choices])
        key, chosen, outcomes = next(entry for entry in choices if entry[0] == picked["id"])
        result.update(qualifies=True, followup_id=key, followup=chosen["action"], ko=chosen["ko"],
                      score=chosen["score"], outcomes=outcomes, reason=None)
    else:
        result["reason"] = "boosted-ko"
    return result


def chosen_rows(result, choice):
    setup = result.get("setup_evaluations", {}).get(choice, {})
    return setup.get("outcomes", result["rows"][choice])


def select(result):
    """BE-12..20. Reused unchanged for turns, replacement continuations and replay."""
    rows, options = result.get("rows", {}), result.get("options", {})
    if result.get('emergency'):
        return emergency_select(result)
    if result.get("errors") or result.get("deadline_passed") or result.get("completed_batches") != TRIALS or not rows or any(len(r) != TRIALS for r in rows.values()) or any(
            not p['complete'] or any(not path['complete'] or len(path['entries']) != TRIALS or
                                    len(path['outcomes']) != TRIALS for path in p.get('receivers', {}).values())
            for p in result.get('pivot_evaluations', {}).values()):
        return None, {"rule": "incomplete", "reason": "Every comparison requires 32 completed trials", "candidates": []}
    metrics = measure(first_rows(result), options, result["before"], score_rows=rows,
                      trial_indices=result.get("trial_indices"))
    exits = palafin_exits(result)
    diagnostics = {"rule": None, "parameters": PARAMETERS, "candidates": [asdict(m) for m in metrics],
                   "palafin_zero": exits,
                   "setup": {key: {k: v for k, v in data.items() if k not in ("outcomes", "continuation")}
                             for key, data in result.get("setup_evaluations", {}).items()}}
    attacks = [m for m in metrics if moves.damaging(options[m.id].get("name")) and options[m.id]["kind"] == "move"]
    immediate = [m for m in attacks if m.knockout["passes"]]
    if immediate:
        chosen, priority = choose_ko([asdict(m) for m in immediate])
        diagnostics["rule"] = "priority-ko" if priority else "immediate-ko"
        return chosen["id"], diagnostics
    if exits["options"]:
        if result.get("switch_pending") or result.get("ordinary_pending"):
            return None, {**diagnostics, "rule": "incomplete", "reason": "Palafin exit comparison has not completed"}
        chosen = max((m for m in metrics if m.id in exits["options"]), key=lambda m: m.score)
        diagnostics["rule"] = "palafin-zero-exit"
        return chosen.id, diagnostics
    setups = [(key, data) for key, data in result.get("setup_evaluations", {}).items() if data.get("qualifies")]
    if setups:
        chosen, _ = max(setups, key=lambda entry: entry[1]["score"])
        diagnostics["rule"] = "offensive-setup"
        return chosen, diagnostics
    if result.get("setup_pending"):
        return None, {**diagnostics, "rule": "incomplete", "reason": "Setup comparison has not completed"}
    if result.get("switch_pending"):
        return None, {**diagnostics, "rule": "incomplete", "reason": "Switch continuation comparison has not completed"}
    if result.get("ordinary_pending"):
        return None, {**diagnostics, "rule": "incomplete", "reason": "Stay/switch continuation comparison has not completed"}
    candidates = _ordinary(metrics, options)
    if not candidates:
        return None, {**diagnostics, "rule": "no-qualified-action"}
    diagnostics["rule"] = "ordinary-hp-ko"
    if all(m.score == TERMINAL_LOSS for m in candidates):
        chosen = max(candidates, key=lambda m: m.damage)
        diagnostics["tie_break"] = {"rule": "all-loss-first-turn-damage",
                                    "candidates": [m.id for m in candidates], "selected_damage": chosen.damage}
        return chosen.id, diagnostics
    return max(candidates, key=lambda m: m.score).id, diagnostics


def score(components):
    terminal = components.get("terminal", 0)
    return terminal or 100 * (components["opponent_knockouts"] - components["own_knockouts"]) + components["hp_change"]


def components(before, after):
    """Extract and score the same outcome for turns, continuation plans and replay."""
    result = {"own_knockouts": 0, "opponent_knockouts": 0, "hp_change": 0, "terminal": 0}
    for side, team in enumerate(before["sides"]):
        final = {member["identity"]: member for member in after["sides"][side]["pokemon"]}
        for member in team["pokemon"]:
            if member["identity"] not in final:
                raise ValueError("Simulation lost party identity")
            now = final[member["identity"]]
            if member["hp"] > 0 and not now["hp"]:
                result["opponent_knockouts" if side else "own_knockouts"] += 1
            result["hp_change"] += (-1 if side else 1) * 100 * (now["hp"] / now["maxhp"] - member["hp"] / member["maxhp"])
    if after["ended"]:
        result["terminal"] = {"Us": 10000, "Them": TERMINAL_LOSS}.get(after["winner"], 0)
    result["score"] = score(result)
    return result


def _mean(outcomes, sequential):
    values = (score(row["components"]) for row in outcomes)
    # Preserve the old worker's left-to-right continuation totals and their exact ties.
    total = reduce(add, values, 0) if sequential else sum(values)
    return total / len(outcomes)


def rank(rows, *, sequential=False):
    """Input order is the stable tie-breaker; only equally completed batches are compared."""
    counts = {len(outcomes) for outcomes in rows.values()}
    if len(counts) != 1 or not counts or not next(iter(counts)):
        return None, []
    ranked = [{"id": name, "score": _mean(outcomes, sequential),
               "samples": len(outcomes)} for name, outcomes in rows.items()]
    return max(ranked, key=lambda row: row["score"])["id"], ranked


def emergency_select(result):
    """BE-23: complete one-exchange HP/KO ranking, without setup or preference gates."""
    rows = result.get('rows', {})
    if (result.get('errors') or result.get('deadline_passed') or result.get('completed_batches') != TRIALS
            or not rows or set(rows) != set(result.get('options', {})) or any(len(r) != TRIALS for r in rows.values())
            or any(not p.get('complete') for p in result.get('plan_evaluations', {}).values())
            or any(not p.get('complete') or any(not path.get('complete') or len(path.get('entries', [])) != TRIALS
                or len(path.get('outcomes', [])) != TRIALS for path in p.get('receivers', {}).values())
                for p in result.get('pivot_evaluations', {}).values())):
        return None, {'rule': 'BE-23', 'reason': 'incomplete', 'candidates': []}
    choice, ranked = rank(rows)
    return choice, {'rule': 'BE-23', 'horizon': 'one exchange', 'candidates': ranked}
