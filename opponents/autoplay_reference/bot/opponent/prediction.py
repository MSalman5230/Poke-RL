"""A single usage-based build, constrained by independently stored observations."""
import copy
from opponents.autoplay_reference.bot.domain import catalog, moves
from opponents.autoplay_reference.bot.domain.types import BuildPrediction
from opponents.autoplay_reference.bot.domain.forms import matchup_form, loadout_family, usage_forms
from opponents.autoplay_reference.bot.opponent.data import datasets
from opponents.autoplay_reference.bot.opponent import mega

CHOICE = {"Choice Band", "Choice Specs", "Choice Scarf"}


class Opponent:
    """Battle-local facts only. Prediction never writes through to this object."""
    def __init__(self, species, db=None):
        self.species = species
        self.form = None
        self.item_known = None
        self.ability_known = None
        self.original_ability = None
        self.temporary_ability = False
        self.revealed_moves = []
        self.pp_used = {}
        self.moves_this_stint = []
        self.evidence = []
        self.disguise_busted = False
        self.mega_allowed = True
        self.hero_ready = False
        self.item_replaced_unknown = False
        self.mega_choice = None
        self.first_mega_opportunity = None
        self.mega_declined = None
        self.mega_activated = False
        self.stone_form = catalog.get().stones.get(species, {})

    @property
    def info(self):
        return catalog.get().species(self.form or self.species)

    @property
    def ability(self):
        return self.ability_known

    def record(self, kind, accepted, **fields):
        self.evidence.append({"kind": kind, "accepted": accepted, **fields})

    def reveal_move(self, move):
        name = catalog.get().display("moves", move)
        if name and name not in self.revealed_moves:
            self.revealed_moves.append(name)
            self.record("move", True, move=name)

    def note_move_used(self, move, spends_pp=True, *, called_by=None, pp_cost=None):
        if not called_by or called_by == "Sleep Talk":
            self.reveal_move(move)
        charged = called_by if called_by else move
        cost = (int(spends_pp) if pp_cost is None else pp_cost) if not called_by else (pp_cost or 0)
        self.pp_used[charged] = self.pp_used.get(charged, 0) + cost
        if not called_by:
            self.moves_this_stint.append(move)
        else:
            self.record("called move", True, move=move, called_by=called_by)

    def reveal_item(self, item):
        name = catalog.get().display("items", item)
        if name is not None:
            self.item_known = name
            self.item_replaced_unknown = False
            self.record("item", True, item=name)

    def acquire_item(self, item):
        self.reveal_item(item)
        self.record("item acquisition", True, item=item)

    def acquire_item_unknown(self):
        self.item_known = None
        self.item_replaced_unknown = True
        self.record("item acquisition", False, reason="incoming battle item unreadable")

    def reveal_ability(self, ability, temporary=False):
        name = catalog.get().display("abilities", ability)
        if name is None:
            return
        if temporary and not self.temporary_ability:
            self.original_ability = self.ability_known
        self.ability_known = name
        self.temporary_ability |= temporary
        self.record("ability", True, ability=name, temporary=temporary)

    def on_switch_out(self):
        if self.temporary_ability:
            self.ability_known = self.original_ability
            self.original_ability = None
            self.temporary_ability = False

    def on_switch_in(self):
        self.on_switch_out()
        self.moves_this_stint = []
        self.mega_choice = None
        if self.hero_ready and self.species == "Palafin" and not self.form:
            self.set_form("Palafin-Hero")

    def set_form(self, form):
        if not catalog.get().species(form):
            raise ValueError(f"Unknown observed form {form}")
        self.form = form
        self.ability_known = catalog.get().species(form)["possibleAbilities"][0]
        self.temporary_ability = False
        self.record("form", True, form=form)

    def record_damage(self, direction, move, low, high, template, **context):
        self.record("damage", False, direction=direction, move=move, observed_interval=[low, high],
                    simulator_input=copy.deepcopy(template), context=context,
                    reason="retained for future prediction rules; no damage filtering in v1")

    def snapshot(self):
        return copy.deepcopy({"species": self.species, "form": self.form, "moves": self.revealed_moves,
                              "item": self.item_known, "ability": self.ability_known,
                              "pp_used": self.pp_used, "moves_this_stint": self.moves_this_stint,
                              'item_replaced_unknown': self.item_replaced_unknown,
                              'temporary_ability': self.temporary_ability, 'original_ability': self.original_ability,
                              "mega_allowed": self.mega_allowed, "mega_choice": self.mega_choice,
                              "first_mega_opportunity": self.first_mega_opportunity,
                              "mega_declined": self.mega_declined,
                              "mega_activated": self.mega_activated,
                              "evidence": self.evidence})


class BuildEstimator:
    def __init__(self, data=None):
        self.data = data or datasets()
        self.catalog = catalog.get()

    def predict(self, facts, *, mega_selected=None, allow_defaults=False):
        c, data = self.catalog, self.data
        species = facts.species
        errors, assumptions, fields, recovered = [], [], {}, []
        if c.species(species) is None:
            return BuildPrediction(species, None, None, errors=("unknown species",))
        current = facts.form or species
        form_prior = mega.prior(facts, data)
        hypothesis = None
        if facts.mega_activated and not facts.form:
            # The activation is observed; only its suffix (X/Y, Z) is missing. Active play predicts
            # the most-used candidate form (catalog order without usage); facts keep the form unknown.
            forms = sorted(set(facts.stone_form.values()), key=catalog.canonical)
            hypothesis = (form_prior['form'] if form_prior['form'] in forms else forms[0] if forms else None) if allow_defaults else None
            if hypothesis is None:
                errors.append('observed Mega activation has unresolved form')
            else:
                current = hypothesis
                reason = ('observed Mega activation with unresolved form; ' +
                          ('most-used compatible Mega form' if form_prior['form'] == hypothesis else 'first catalog Mega form'))
                recovered.append(reason)
                assumptions.append(f'form: {reason}; simulation assumption only')
                fields['form'] = {'source': 'baseline-default', 'value': hypothesis, 'assumption': reason,
                                  'candidates': forms, 'usages': form_prior.get('usages')}
        if c.species(current) is None:
            return BuildPrediction(species, None, None, errors=('unknown observed form',))
        natural = c.species(current)["possibleAbilities"]
        def ranked(name, trait):
            try:
                return list(data.ranked(name, trait))
            except Exception as exc:
                if not allow_defaults:
                    raise
                recovered.append(f'{name} {trait}: {exc!r}')
                return []
        def default(field, value, reason):
            recovered.append(reason)
            assumptions.append(f'{field}: {reason}; simulation assumption only')
            fields[field] = {'source': 'baseline-default', 'value': value, 'assumption': reason}
            return value
        if mega_selected is None:
            mega_selected = facts.mega_allowed and form_prior['eligible']
        fields['mega'] = {**form_prior, 'selected': bool(mega_selected)}
        item = facts.item_known
        if item is None and hypothesis:
            item = next(stone for stone, form in facts.stone_form.items() if form == hypothesis)
            fields['item'] = {'source': 'baseline-default', 'value': item,
                              'assumption': 'stone implied by the Mega-form hypothesis'}
        if item is None and facts.item_replaced_unknown and not allow_defaults:
            errors.append("current acquired item is unknown")
        if item is None and facts.item_replaced_unknown and allow_defaults:
            item = default('item', '', 'current acquired item unknown; simulate no held item')
        if item is None and not errors:
            if mega_selected and form_prior['form']:
                item = next((stone for stone, form in facts.stone_form.items() if form == form_prior['form']), None)
                fields['item'] = {**form_prior, 'source': 'smogon-family-Mega-hypothesis', 'value': item}
            item_rows = []
            pool = loadout_family(current) if facts.form else [species]
            for form in pool:
                item_rows.extend((key, source) for key, source in ranked(form, "Items")
                                 if c.display("items", key) is not None
                                 and c.display("items", key) not in facts.stone_form
                                 and (len(set(facts.moves_this_stint)) <= 1 or c.display("items", key) not in CHOICE))
            for cutoff in (1500, 0):
                rows = [r for r in item_rows if r[1]["cutoff"] == cutoff]
                if item is None and rows:
                    key, source = min(rows, key=lambda r: (-r[1]["weight"], r[0]))
                    item = c.display("items", key)
                    fields["item"] = source
                    break
        elif facts.item_known is not None:
            fields["item"] = {"source": "observed", "value": item}
        if item is None:
            if allow_defaults:
                item = default('item', '', 'no compatible item data; simulate no held item')
            else:
                errors.append("no compatible item data")
        battle_form = matchup_form(species, item, observed=facts.form or hypothesis,
                                   mega_allowed=facts.mega_allowed and bool(mega_selected), hero=species == "Palafin")
        usage_form = facts.stone_form.get(item) or battle_form
        def choose(trait, convert, valid):
            for name in usage_forms(usage_form, current, species, *facts.stone_form.values()):
                for key, source in ranked(name, trait):
                    value = convert(key)
                    if value is not None and valid(value):
                        return value, source
            return None, None
        ability = facts.ability_known
        if hypothesis and ability not in natural and not facts.temporary_ability:
            ability = None      # a pre-Evolution reveal is not the hypothesized form's ability
        if ability is None:
            ability, source = choose("Abilities", lambda key: c.display("abilities", key), lambda a: a in natural)
            if ability is None and len(natural) == 1:
                ability, source = natural[0], {"source": "sole legal ability"}
            if ability is not None:
                fields["ability"] = source
        else:
            fields["ability"] = {"source": "observed", "value": ability}
        if ability is None:
            if allow_defaults and natural:
                ability = default('ability', sorted(natural, key=catalog.canonical)[0], 'no compatible ability usage; catalog ability')
            else:
                errors.append("no compatible current ability")
        def spread(key):
            try:
                nature, points = key.split(":")
                points = [int(x) for x in points.split("/")]
                if nature in catalog.NATURES and len(points) == 6 and all(0 <= p <= 32 for p in points) and sum(points) <= 66:
                    return nature, dict(zip(catalog.STATS, points))
            except (ValueError, TypeError):
                pass
            return None
        selected_spread, source = choose("Spreads", spread, lambda _: True)
        if selected_spread is None:
            if allow_defaults:
                nature, sp = default('spread', ('Serious', dict.fromkeys(catalog.STATS, 0)),
                                     'no compatible spread; Serious nature and zero stat points')
            else:
                errors.append("no compatible nature/SP spread")
                nature, sp = "Serious", {}
        else:
            nature, sp = selected_spread
            fields["spread"] = source
        selected_moves = list(dict.fromkeys(facts.revealed_moves))
        fields["moves"] = {m: {"source": "observed"} for m in selected_moves}
        if len(selected_moves) > 4:
            errors.append("more than four observed native moves")
        if any(not c.legal_move(current, move) for move in selected_moves):
            errors.append("observed native move is incompatible with the current species/form")
        for name in usage_forms(usage_form, current, species, *facts.stone_form.values()):
            for key, source in ranked(name, "Moves"):
                move = c.display("moves", key)
                if len(selected_moves) >= 4:
                    break
                if move and move not in selected_moves and c.legal_move(current, move):
                    selected_moves.append(move)
                    fields["moves"][move] = source
        legal = c.species(current)["learnset"]
        required_moves = min(4, len(legal))
        if len(legal) < 4:
            for move_id in legal:
                move = c.display("moves", move_id)
                if move and move not in selected_moves:
                    selected_moves.append(move)
                    fields["moves"][move] = {"source": "complete legal learnset"}
        if len(selected_moves) != required_moves:
            if allow_defaults and len(selected_moves) < required_moves:
                types = c.species(current)['types']
                def order(move_id):
                    m = c.move(move_id)
                    accuracy = 1 if m.get('accuracy') is True else (m.get('accuracy') or 0) / 100
                    power = m.get('bp', 0) * accuracy * (1.5 if m['type'] in types else 1)
                    return (m['category'] == 'Status', moves.setup(m['name'], types), -power, move_id)
                for move_id in sorted(legal, key=order):
                    move = c.display('moves', move_id)
                    if move and move not in selected_moves:
                        selected_moves.append(move)
                        fields['moves'][move] = {'source': 'baseline-default', 'assumption': 'catalog-legal move, not observed'}
                        if len(selected_moves) == required_moves:
                            break
                recovered.append('insufficient usage moves; completed from current-form legal catalog')
                assumptions.append('missing moves completed from legal catalog; not observations')
            if len(selected_moves) != required_moves:
                errors.append("insufficient legal move data")
        original_ability = facts.original_ability if facts.temporary_ability else ability
        if facts.temporary_ability and original_ability is None:
            original_ability, source = choose("Abilities", lambda key: c.display("abilities", key), lambda a: a in natural)
            if original_ability is None and len(natural) == 1:
                original_ability, source = natural[0], {"source": "sole legal ability"}
            if original_ability is None:
                if allow_defaults and natural:
                    original_ability = default('original_ability', sorted(natural, key=catalog.canonical)[0],
                                               'original temporary ability unknown; catalog natural ability')
                else:
                    errors.append("original ability before temporary change is unknown")
            else:
                fields["original_ability"] = source
                assumptions.append("original ability before temporary change is unknown; natural ability predicted")
        member = None
        if not errors:
            member = {"species": current, "item": item, "ability": ability, "nature": nature, "sp": sp,
                      "moves": [{"name": m, "pp": max(0, c.move(m)["max_pp"] - facts.pp_used.get(m, 0))}
                                for m in selected_moves], "battle_form": battle_form,
                      "original_ability": original_ability}
            member["stats"] = c.stats(member)
        assumptions.append("independent usage marginals; not an observed joint set")
        if recovered:
            fields['recovery'] = {'source': 'BP-14', 'issues': recovered, 'assumption': 'missing fields only; observations unchanged'}
        if facts.form is None and not hypothesis and battle_form != species:
            assumptions.append(f"{battle_form} is a potential battle form, not an observed activation")
        sources = [v for k, v in fields.items() if k != "moves"] + list(fields["moves"].values())
        for source in sources:
            source.setdefault("weight", None)
            source.setdefault("share", None)
            source.setdefault("assumption", None if source.get("source") == "observed" else "sole legal current ability")
        return BuildPrediction(species, member, battle_form, fields, tuple(assumptions), tuple(errors))
