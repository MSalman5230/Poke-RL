"""TS-13..16: robust complementary preview coverage from prepared pair profiles."""
from dataclasses import replace
from itertools import combinations, permutations, product
import time
from opponents.autoplay_reference.bot.domain import catalog
from opponents.autoplay_reference.bot.domain.types import TeamSelection
from opponents.autoplay_reference.bot.domain.forms import own_forms
from opponents.autoplay_reference.bot.simulator import bridge
from opponents.autoplay_reference.bot.team_selection import cache, coverage, duels
from opponents.autoplay_reference.bot.team_selection.prepare import own_variants, enemy_variants

POLICY='damage-speed-coverage-v1'


def assignment(matrix,trio,foes):
    choices=[{'pairs':list(zip(trio,order)),'score':sum(matrix[n][f] for n,f in zip(trio,order))/3}
             for order in permutations(foes)]
    return max(choices,key=lambda row:row['score'])


def rank(roster,predictions,eligible,entries,weights, *, rebuild=False, weights_normalized=False, selected=()):
    names=list(dict.fromkeys(eligible))
    selected=tuple(selected)
    if len(selected)>3 or len(set(selected))!=len(selected) or not set(selected).issubset(names):
        raise ValueError('Observed selection prefix is inconsistent with eligible identities')
    foes=list(predictions)
    if len(names)<3 or len(foes)!=6 or any(not p.complete for p in predictions.values()):
        raise ValueError('A complete six-opponent preview is required for damage coverage')
    owners=[n for n,p in predictions.items() if '-Mega' in (p.battle_form or '')]
    if len(owners)>1:
        raise ValueError('At most one enemy Mega hypothesis is allowed')
    # TS-13: their Mega is in every trio they can bring; six without a Mega form (B38) can bring any three
    enemy_owner=owners[0] if owners else None
    enemy_trios=([(enemy_owner,*pair) for pair in combinations([n for n in foes if n!=enemy_owner],2)] if enemy_owner
                 else list(combinations(foes,3)))
    own=own_variants(roster)
    enemy={n:enemy_variants(n,p) for n,p in predictions.items()}
    evidence={}
    pair_cache={}
    def recorded(task):
        key=duels.sha(task)
        row=entries[key]
        if row['input']!=task:
            raise ValueError('Saved preview input mismatch')
        if key not in evidence:
            duels.validate(row['profile'],task)
            evidence[key]={'key':key,'input':task,'profile':row['profile'],'evaluation':row['evaluation']}
            for dependency in duels.activation_tasks(task).values():
                recorded(dependency)
        return row
    def pair(name,form,foe, *, opening=False):
        memo=(name,form,foe,opening)
        if memo in pair_cache:
            return pair_cache[memo]
        options=list(enemy[foe].values())
        if opening and len(options)>1:
            options=options[:1]
        values=[]
        for other in options:
            task=duels.task(own[name][form],other,opening=opening)
            key=duels.sha(task)
            row=recorded(task)
            value=duels.score(row['profile'],task,activation_lookup=lambda t:recorded(t)['profile']) if rebuild else row['evaluation']
            if value is None:
                raise KeyError('Complete preview race/activation evaluation is not prepared')
            evidence[key]={'key':key,'input':task,'profile':row['profile'],'evaluation':value}
            values.append(value['score'])
        pair_cache[memo]=min(values)
        return pair_cache[memo]
    candidates=[]
    carriers={n:catalog.get().stones.get(n,{}).get(roster.members[n]['item']) for n in names}
    for trio in combinations(names,3):
        if not set(selected).issubset(trio):
            continue
        configurations=[]
        for owner in ([n for n in trio if carriers[n]] or [None]):
            modes=[list(own[n]) if roster.members[n]['ability']=='Zero to Hero' else
                   [carriers[n] if n==owner else roster.members[n]['species']] for n in trio]
            for variants in product(*modes):
                forms=dict(zip(trio,variants))
                matrix={n:{f:pair(n,forms[n],f) for f in foes} for n in trio}
                scores=[]
                for opposing in enemy_trios:
                    best=assignment(matrix,trio,opposing)
                    scores.append({'picks':opposing,**best})
                backup=sum(sorted((matrix[n][f] for n in trio),reverse=True)[1]
                           for opposing in enemy_trios for f in opposing)/(3*len(enemy_trios))
                worst=min(v['score'] for v in scores)
                average=sum(v['score'] for v in scores)/len(scores)
                configurations.append({'mega_owner':owner,'forms':forms,'matrix':matrix,
                    'allowed_leads':tuple(n for n in trio if not carriers[n] or n==owner),
                    'score':worst,'average':average,'backup':backup,'rank_key':[worst,average,backup],
                    'opposing_trios':scores})
        if selected:
            configurations=[c for c in configurations if selected[0] in c['allowed_leads']]
        if not configurations:
            continue
        configuration=max(configurations,key=lambda c:tuple(c['rank_key']))
        candidates.append({'picks':trio,**configuration,'configurations':configurations})
    best=coverage.best_trio(candidates)
    normalized={n:weights.get(n,0) for n in foes}
    total=sum(normalized.values())
    if not weights_normalized:
        normalized={n:v/total if total else 1/len(foes) for n,v in normalized.items()}
    opening={n:{f:pair(n,roster.members[n]['species'] if roster.members[n]['ability']=='Zero to Hero'
                       else best['forms'][n],f,opening=True) for f in foes} for n in best['picks']}
    lead_scores={n:sum(normalized[f]*opening[n][f] for f in foes) for n in best['picks']}
    picks=coverage.lead_order(best['picks'],lead_scores,best['allowed_leads'])
    picks=(*selected,*(n for n in picks if n not in selected))
    return TeamSelection(picks,best['matrix'],tuple(candidates),tuple(evidence.values()),lead_scores,
        best['mega_owner'],best['forms'],best['allowed_leads'],POLICY,
        {'complete':True,'enemy_mega_owner':enemy_owner,'opponent_trios':enemy_trios,
         'lead_weights':normalized,'opening_matrix':opening,'score_identity':duels.scoring_identity(),
         'assumptions':duels.ASSUMPTIONS,'observed_prefix':selected})


def select(roster,predictions,eligible,data, *, deadline=None, selected=()):
    started=time.monotonic()
    try:
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('Preview deadline exhausted')
        store=cache.get(roster,data)
        weights={n:data.family_lead(p.battle_form or n) for n,p in predictions.items()}
        selection=rank(roster,predictions,eligible,store.pairs,weights,selected=selected)
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('Complete preview ranking exceeded the deadline')
        return replace(selection,details={**selection.details,'cache_identity':store.identity,
            'cache_path':str(store.path),'pair_profiles':len(selection.evidence),
            'simulator_identity':bridge.get().identity,
            'timing_seconds':time.monotonic()-started})
    except (KeyError,ValueError,OSError,TimeoutError) as exc:
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('No complete preview result before deadline') from exc
        selection=coverage.TeamSelector(data).select(eligible,list(predictions),forms=own_forms(roster),
            opposing_forms={n:p.battle_form for n,p in predictions.items()},selected=selected)
        return replace(selection,details={'requested_policy':POLICY,'complete':True,
            'fallback_reason':repr(exc),'timing_seconds':time.monotonic()-started})
