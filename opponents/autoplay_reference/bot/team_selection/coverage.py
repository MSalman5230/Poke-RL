"""Exhaustive three-of-six coverage using directional Smogon matchup indices."""
from itertools import combinations
from opponents.autoplay_reference.bot.domain.types import TeamSelection
from opponents.autoplay_reference.bot.team_selection.matchups import Matchups
from opponents.autoplay_reference.bot.opponent import mega


def best_trio(candidates):
    return max(candidates, key=lambda row: tuple(row.get('rank_key',(row["score"], row["backup"]))))


def lead_order(picks, lead_scores, allowed=None):
    eligible = [name for name in picks if allowed is None or name in allowed]
    lead = max(eligible, key=lead_scores.get)
    return (lead, *(name for name in picks if name != lead))


class TeamSelector(Matchups):

    def select(self, ours, opponents, *, forms=None, opposing_forms=None, required=(),
               mega_priors=None, mega_active=None, mega_used=None, selected=()):
        names = list(dict.fromkeys(ours))
        selected=tuple(selected)
        if len(set(selected))!=len(selected):
            raise ValueError('Observed selection prefix contains duplicate identities')
        required=tuple(dict.fromkeys((*required,*selected)))
        foes = list(dict.fromkeys(n for n in opponents if n))
        if len(names) < 3 or len(required) > 3 or not set(required).issubset(names):
            raise ValueError("Three eligible identities and consistent required party members are needed")
        forms, opposing_forms = forms or {}, opposing_forms or {}
        carriers = [name for name in names if '-Mega' in (forms.get(name) or '')]
        enemy_carriers = [name for name in foes if '-Mega' in (opposing_forms.get(name) or '')]
        enemy_owner = enemy_carriers[0] if enemy_carriers else None
        opposing_forms = {name: (opposing_forms.get(name) or name) if name == enemy_owner or
                          name not in enemy_carriers else name for name in foes}
        evidence, cache = [], {}
        def matrix_for(owner):
            result = {}
            for name in names:
                form = (forms.get(name) or name) if name == owner or name not in carriers else name
                result[name] = {}
                for foe in foes:
                    other = opposing_forms.get(foe) or foe
                    if (form, other) not in cache:
                        value, row = self.pair(form, other)
                        cache[form, other] = value
                        evidence.append(row)
                    result[name][foe] = cache[form, other]
            return result
        candidates = []
        for trio in combinations(names, 3):
            if not set(required).issubset(trio):
                continue
            if mega_priors is not None:
                owners = [mega.owner(trio, mega_priors, active=mega_active, used=mega_used)]
            else:
                owners = [name for name in trio if name in carriers] or [None]
            configurations = []
            for owner in owners:
                matrix = matrix_for(owner)
                sorted_scores = [sorted((matrix[n][f] for n in trio), reverse=True) for f in foes]
                configurations.append({'mega_owner': owner,
                    'forms': {n: (forms.get(n) or n) if n == owner or n not in carriers else n for n in trio},
                    'allowed_leads': tuple(n for n in trio if n not in carriers or n == owner or mega_used),
                    'score': sum(s[0] for s in sorted_scores) / max(1, len(foes)),
                    'backup': sum(s[1] for s in sorted_scores) / max(1, len(foes))})
            if selected:
                configurations=[c for c in configurations if selected[0] in c['allowed_leads']]
            if not configurations:
                continue
            configuration = max(configurations, key=lambda row: (row['score'], row['backup']))
            candidates.append({'picks': trio, **configuration, 'configurations': configurations})
        best = best_trio(candidates)
        matrix = matrix_for(best['mega_owner'])
        weights = {f: getattr(self.data, 'family_lead', self.data.lead)(opposing_forms.get(f) or f) for f in foes}
        total = sum(weights.values())
        weights = {f: (v / total if total else 1 / max(1, len(foes))) for f, v in weights.items()}
        lead_scores = {n: sum(weights[f] * matrix[n][f] for f in foes) for n in best["picks"]}
        picks = lead_order(best["picks"], lead_scores, best['allowed_leads'])
        picks = (*selected,*(n for n in picks if n not in selected))
        return TeamSelection(picks, matrix, tuple(candidates), tuple(evidence), lead_scores,
                             best['mega_owner'], best['forms'], best['allowed_leads'])
