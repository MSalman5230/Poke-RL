"""Battle-local observation context; simulated futures never update these facts."""
from opponents.autoplay_reference.bot.domain import catalog
from opponents.autoplay_reference.bot.opponent.data import datasets
from opponents.autoplay_reference.bot.opponent.prediction import Opponent
from opponents.autoplay_reference.bot.state_support.field import Field


class BattleContext:
    def __init__(self, db=None, roster=None):
        if roster is None:
            from opponents.autoplay_reference.bot.domain.roster import Roster
            roster = Roster.load()
        self.db, self.roster = db, roster
        self.c = catalog.get()
        self.datasets = datasets()
        self.field = Field()
        self.beliefs = {}
        self.mega_used_by_opp = None

    def belief(self, species):
        if species not in self.beliefs:
            self.beliefs[species] = Opponent(species)
            if self.mega_used_by_opp and self.mega_used_by_opp != species:
                self.beliefs[species].mega_allowed = False
        return self.beliefs[species]

    def opp_mega_used(self, species):
        self.mega_used_by_opp = species
        for name, opponent in self.beliefs.items():
            opponent.mega_allowed = name == species

    def observe_mega_prompt(self, species, turn, *, captures=()):
        facts = self.belief(species)
        if not self.mega_used_by_opp and facts.stone_form and not facts.form and not facts.first_mega_opportunity:
            facts.mega_choice = {'turn': turn, 'captures': list(captures), 'source': 'observed fight prompt'}

    def note_mega_opportunity(self, species, turn, *, event_id=None, captures=()):
        facts = self.belief(species)
        if (self.mega_used_by_opp or not facts.stone_form or facts.form or facts.first_mega_opportunity or
                not facts.mega_choice or facts.mega_choice['turn'] != turn):
            return
        facts.first_mega_opportunity = {'turn': turn, 'event_id': event_id, 'captures': list(captures),
                                        'prompt': facts.mega_choice, 'source': 'observed native move', 'pending': True}

    def finish_mega_opportunities(self):
        for facts in self.beliefs.values():
            opportunity = facts.first_mega_opportunity
            if not opportunity or not opportunity.get('pending'):
                continue
            opportunity['pending'] = False
            if facts.form in facts.stone_form.values() or facts.item_known in facts.stone_form:
                continue
            facts.mega_declined = {**opportunity, 'reason': 'no Mega observed at first confirmed move opportunity',
                                  'assumption': True}
            facts.record('non-Mega hypothesis', False, **facts.mega_declined)

    def own(self, species, hp=None, mega=False, hero=False, consumed=()):
        member = self.roster.view(species, mega=mega, hero=hero)
        if species in consumed:
            member["item"] = ""
        if hp is not None:
            member["curHP"] = hp
        return member

