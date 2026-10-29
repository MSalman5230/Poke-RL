"""Difficulty sampling and retention of strong historical policies."""
from .arena import STYLES


def style_weights(outcomes):
    weights=[]
    for name in STYLES:
        rows=[r for r in outcomes if r['opponent']==name and not r.get('draw')]
        # Shrink small samples toward 50%; all substantive styles retain mass.
        failures=sum(not r['win'] for r in rows)
        difficulty=(failures+6)/(len(rows)+12)
        weights.append(.025 if name=='random' else .06+difficulty**2)
    total=sum(weights)
    return [v/total for v in weights]


def retain_league(names,ratings,limit=12,recent=4):
    if len(names)<=limit:return names
    # Keep the imitation anchor, difficult older opponents, and recent policies.
    newest=names[-recent:]
    anchor=names[0]
    older=[n for n in names[:-recent] if n!=anchor]
    strongest=sorted(older,key=lambda n:ratings.get(n,.5),reverse=True)[:limit-recent-1]
    keep={anchor,*strongest,*newest}
    return [n for n in names if n in keep]
