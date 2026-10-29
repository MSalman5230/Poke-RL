"""BE-21/OA-10: evolve whenever a legal Mega move is available; never search both."""
from opponents.autoplay_reference.bot.domain import catalog


def force(actions):
    """Preserve order and types for screen Actions and simulator action dictionaries."""
    actions = list(actions)
    def value(action, key, default=None):
        return action.get(key, default) if isinstance(action, dict) else getattr(action, key, default)
    def key(action):
        return catalog.canonical(value(action, 'name')) or value(action, 'slot')
    mega = {key(a) for a in actions if value(a, 'kind') == 'move' and value(a, 'mega', False)}
    return [a for a in actions if value(a, 'kind') != 'move' or value(a, 'mega', False) or key(a) not in mega]
