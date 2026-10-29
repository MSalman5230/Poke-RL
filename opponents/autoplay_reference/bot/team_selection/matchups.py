"""One directional matchup calculation shared by preview, party guesses and replacements."""
from opponents.autoplay_reference.bot.domain import catalog
from opponents.autoplay_reference.bot.opponent.data import datasets


class Matchups:
    def __init__(self, data=None):
        self.data = data or datasets()

    def pair(self, ours, theirs):
        a, b = catalog.canonical(ours), catalog.canonical(theirs)
        for cutoff in (1500, 0):
            rows = self.data.tables[cutoff]
            ar = {catalog.canonical(k): v for k, v in rows.get(a, {}).get("Checks and Counters", {}).items()}
            br = {catalog.canonical(k): v for k, v in rows.get(b, {}).get("Checks and Counters", {}).items()}
            loss, win = ar.get(b), br.get(a)
            if loss and win and min(loss["n"], win["n"]) > 0:
                n = min(loss["n"], win["n"])
                value = (win["p"] - loss["p"]) * n / (n + 20)
                return value, {"ours": ours, "opponent": theirs, "cutoff": cutoff,
                               "our_losses": loss, "their_losses": win, "score": value}
        return 0.0, {"ours": ours, "opponent": theirs, "score": 0.0, "missing": True}

    def replacement_ranks(self, ours, predictions, forms):
        matrix = {name: {foe: self.pair(forms.get(name, name), prediction.battle_form or foe)[0]
                         for foe, prediction in predictions.items()} for name in ours}
        c = catalog.get()
        own_forms = set(ours) | set(forms.values()) | {form for name in ours for form in c.stones.get(name, {}).values()}
        enemy_forms = set(predictions) | {p.battle_form for p in predictions.values() if p.battle_form} | {
            form for name in predictions for form in c.stones.get(name, {}).values()}
        if 'Ditto' in predictions:
            enemy_forms |= own_forms
        by_forms = {a: {b: self.pair(a, b)[0] for b in sorted(enemy_forms)} for a in sorted(own_forms)}
        return {"me": matrix, "opp": {foe: {name: -matrix[name][foe] for name in matrix} for foe in predictions},
                'by_forms': by_forms}
