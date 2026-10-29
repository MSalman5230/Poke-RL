"""Catalog-backed form identities; potential forms never activate live state."""
from opponents.autoplay_reference.bot.domain import catalog


def loadout_family(species):
    """Usage aliases for a temporary battle form, not permanent regional/appliance forms."""
    info = catalog.get().species(species) or {}
    base = info.get('baseSpecies')
    return tuple(dict.fromkeys([species, *([base] if info.get('battleOnly') and base
        and catalog.get().species(base) else [])]))


def usage_forms(*names):
    return tuple(dict.fromkeys(form for name in names if name for form in loadout_family(name)))


def matchup_form(species, item, *, observed=None, mega_allowed=True, hero=False):
    if observed:
        return observed
    mega = catalog.get().stones.get(species, {}).get(item) if mega_allowed else None
    return mega or (species + "-Hero" if hero else species)


def own_forms(roster, flags=None):
    flags = flags or {}
    return {name: matchup_form(name, member["item"],
            mega_allowed=not flags.get('mega_used') or name == flags.get('mega_species'),
            hero=member["ability"] == "Zero to Hero")
            for name, member in roster.members.items()}
