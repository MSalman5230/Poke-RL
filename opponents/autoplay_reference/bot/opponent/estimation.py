"""Opponent estimates own hidden-build and party assumptions, never observations."""
from dataclasses import replace
from opponents.autoplay_reference.bot.domain.forms import own_forms
from opponents.autoplay_reference.bot.opponent.prediction import BuildEstimator, Opponent
from opponents.autoplay_reference.bot.team_selection.coverage import TeamSelector
from opponents.autoplay_reference.bot.opponent import mega


def predict_builds(battle, observed, *, names=None, allow_defaults=False):
    context = battle["sim"]
    preview_order = list(dict.fromkeys([n for n in battle.get("opp", []) if n] + battle.get("opp_seen", []) +
                                      ([observed["opp"]] if observed.get("opp") else [])))
    names = list(dict.fromkeys(preview_order if names is None else names))
    estimator = BuildEstimator(context.datasets)
    priors = {name: mega.prior(context.belief(name), context.datasets) for name in names}
    owner_order = [name for name in preview_order if name in names] + [name for name in names if name not in preview_order]
    selected = mega.owner(owner_order, priors, active=observed.get('opp'), used=context.mega_used_by_opp)
    predictions = {name: estimator.predict(context.belief(name), mega_selected=name == selected,
                                           allow_defaults=allow_defaults) for name in names}
    if allow_defaults:
        fainted = set(battle.get('opp_fainted', [])) | set(observed.get('opp_fainted', []))
        for name in [n for n in names if n in fainted and not predictions[n].complete]:
            predictions[name] = fainted_placeholder(estimator, name, predictions[name])
    return predictions


def fainted_placeholder(estimator, name, incomplete):
    """BP-14: a fainted opponent is hydrated at 0 HP only, so its unresolved build cannot block play."""
    blank = Opponent(name)
    blank.mega_allowed = False
    placeholder = estimator.predict(blank, mega_selected=False, allow_defaults=True)
    reason = 'fainted opponent hydrated at 0 HP; observation-free base build'
    return replace(placeholder, fields={**placeholder.fields, 'fainted_placeholder': {
        'source': 'BP-14', 'replaced_errors': list(incomplete.errors), 'assumption': reason,
        'weight': None, 'share': None}}, assumptions=(*placeholder.assumptions, f'{name}: {reason}; facts unchanged'))


def predict_party(battle, observed, predictions, *, allow_defaults=False):
    context = battle["sim"]
    available = [name for name, prediction in predictions.items() if prediction.complete]
    required = list(dict.fromkeys(battle.get("opp_seen", []) + ([observed["opp"]] if observed.get("opp") else [])))
    if any(name not in available for name in required):
        raise ValueError("Revealed opponent has an incomplete build")
    priors = {name: prediction.fields.get('mega', {}) for name, prediction in predictions.items()}
    potential = {name: (priors[name].get('form') if priors[name].get('eligible') and not context.mega_used_by_opp else None)
                 or prediction.battle_form for name, prediction in predictions.items()}
    selection = TeamSelector(context.datasets).select(available, list(context.roster.members), required=required,
        forms=potential, opposing_forms=own_forms(context.roster, battle.get('flags')),
        mega_priors=priors, mega_active=observed.get('opp'), mega_used=context.mega_used_by_opp)
    party = predict_builds(battle, observed, names=selection.picks, allow_defaults=allow_defaults)
    if any(not prediction.complete for prediction in party.values()):
        raise ValueError('Selected opponent party has an incomplete scoped build')
    return party, selection
