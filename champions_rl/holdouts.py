"""Prospective core holdouts for new training, beyond exact-team exclusion."""
from .knowledge import canonical

CORES=(('Slowbro','Sylveon','Corviknight'),('Pelipper','Basculegion','Meowscarada'),('Torkoal','Charizard','Venusaur'))


def held_out_core(team):
    names={canonical(s['species']) for s in team}
    return any(set(map(canonical,core))<=names for core in CORES)
