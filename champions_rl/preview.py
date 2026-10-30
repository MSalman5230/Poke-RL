"""Publicly known mechanics that make a preview ordering meaningful."""
from .knowledge import canonical


def disguise_targets(obs,picks):
    """Potential initial disguise for each selected Illusion user, by roster slot."""
    result=[]
    for slot in picks:
        mon=obs['own'][slot]
        if canonical(mon.get('ability',mon.get('set',{}).get('ability')))=='illusion':
            result.append((slot,next(other for other in reversed(picks) if other!=slot)))
    return tuple(sorted(result))


def preview_key(obs,picks):
    return (picks[0],tuple(sorted(picks[1:])),disguise_targets(obs,picks))
