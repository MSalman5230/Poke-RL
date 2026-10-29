from opponents.autoplay_reference.bot.domain import catalog

class Field:

    def __init__(self):
        self.weather, self.weather_turns = (None, 0)
        self.terrain, self.terrain_turns = (None, 0)
        self.trick_room = 0
        self.sides = {s: {'sr': False, 'spikes': 0, 'tspikes': 0, 'web': False, 'reflect': 0, 'light_screen': 0, 'aurora_veil': 0, 'tailwind': 0, 'safeguard': 0, 'healing_wish': 0, 'lunar_dance': 0, 'mist': 0, 'lucky_chant': 0, 'crafty_shield': 0, 'mat_block': 0, 'quick_guard': 0, 'wide_guard': 0} for s in ('me', 'opp')}
        self.durations = {}
        self.pending_abilities = {}
        self.unrepresented = {}

    def predicted_duration(self, key, fallback, members, diagnostics):
        record = self.durations.get(key)
        if not record:
            return fallback
        total = record['base']
        if total == -1:
            return -1
        extender = record.get('extender')
        if extender:
            known = record.get('item_known', False)
            item = record.get('item') if known else members.get((record.get('side'), record.get('source')), {}).get('item')
            if catalog.canonical(item) == catalog.canonical(extender):
                total = 8
            if not known:
                diagnostics.append({'fact': 'field duration hypothesis', 'effect': key, 'value': record, 'assumed_item': item, 'remaining': max(0, total - record['elapsed']), 'reason': 'unrevealed duration extender; hypothesis only'})
        return max(0, total - record['elapsed'])
