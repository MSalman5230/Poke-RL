"""BP-10/11: family usage and one battle-local Mega hypothesis, never observations."""
from opponents.autoplay_reference.bot.domain import catalog


def prior(facts, data):
    forms = list(dict.fromkeys(facts.stone_form.values()))
    names = [facts.species, *forms]
    usages, cutoff = {}, None
    for candidate in (1500, 0):
        table = getattr(data, 'tables', {}).get(candidate, {})
        values = {name: max(0, table.get(catalog.canonical(name), {}).get('usage', 0) or 0) for name in names}
        if sum(values.values()) > 0:
            usages, cutoff = values, candidate
            break
    total = sum(usages.values())
    mega_total = sum(usages.get(name, 0) for name in forms)
    form = max(forms, key=lambda name: usages.get(name, 0)) if mega_total else None
    observed_stone = facts.stone_form.get(facts.item_known)
    observed_mega = facts.form if facts.form in forms else None
    excluded = None
    if facts.item_known is not None and not observed_stone and not observed_mega:
        excluded = 'observed non-Mega item'
    elif facts.item_replaced_unknown:
        excluded = 'current acquired item unknown'
    elif facts.mega_declined and not observed_stone and not observed_mega:
        excluded = 'no evolution at first confirmed opportunity'
    probability = 1.0 if observed_stone or observed_mega else mega_total / total if total else 0.0
    return {'source': 'smogon-family-usage', 'cutoff': cutoff, 'family': facts.species,
            'usages': usages, 'usage_probability': mega_total / total if total else 0.0,
            'probability': probability, 'form': observed_mega or observed_stone or form,
            'observed_stone': bool(observed_stone), 'observed_mega': bool(observed_mega),
            'excluded': excluded, 'eligible': bool(forms and (observed_stone or observed_mega or mega_total)) and not excluded,
            'weight': mega_total, 'share': probability,
            'assumption': 'family-normalized usage; one most-probable Mega hypothesis',
            'first_opportunity': facts.first_mega_opportunity, 'declined': facts.mega_declined}


def owner(names, priors, *, active=None, used=None):
    names = list(dict.fromkeys(names))
    if used:
        return used if used in names else None
    eligible = [name for name in names if priors.get(name, {}).get('eligible')]
    if active in eligible and priors[active].get('observed_stone'):
        return active
    return max(eligible, key=lambda name: (priors[name].get('observed_mega', False),
               priors[name].get('observed_stone', False), priors[name]['probability'])) if eligible else None
