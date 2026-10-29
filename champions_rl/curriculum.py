"""Difficulty sampling and retention of strong historical policies."""
from .policies import TRAINING_STYLES


def style_weights(outcomes):
    weights={}
    for name in TRAINING_STYLES:
        rows=[r for r in outcomes if r['opponent']==name and not r.get('draw')]
        # Shrink small samples toward 50%; all substantive styles retain mass.
        failures=sum(not r['win'] for r in rows)
        difficulty=(failures+6)/(len(rows)+12)
        weights[name]=.025 if name=='random' else .06+difficulty**2
    total=sum(weights.values())
    return {k:v/total for k,v in weights.items()}


def retain_league(names,ratings,limit=12,recent=4):
    if len(names)<=limit:return names
    # Keep the imitation anchor, difficult older opponents, and recent policies.
    newest=names[-recent:]
    anchors={names[0],*(n for n in names if n.startswith('seeded_'))}
    older=[n for n in names[:-recent] if n not in anchors]
    strongest=sorted(older,key=lambda n:ratings.get(n,.5),reverse=True)[:max(0,limit-len(set(newest)|anchors))]
    keep={*anchors,*strongest,*newest}
    return [n for n in names if n in keep]
